from django.shortcuts import render, redirect
from django.http import HttpRequest, HttpResponse
from .forms import TrainForm
from .utils.env import GridWorld, to_symbol_grid
from .utils.policy import PolicyNet
from .utils.reinforce import (
    reinforce_train,
    evaluate_policy,
    sample_trajectories,
)
from .utils.reward import RewardNet, fit_bradley_terry
from .utils.rlhf import rlhf_finetune
import random


# -------- In-memory state for demo/grading (reset on server restart) --------
SESSION = {
    # Task 1 (baseline training)
    "trained": False,
    "policy_state_dict": None,
    "last_return_mean": None,
    "last_return_std": None,
    "last_seed": 0,
    "baseline_state_dict": None,  # snapshot of the first trained baseline

    # Task 2 (preferences)
    "trajectories": [],   # list[(states, actions, rewards)]
    "prefs": [],          # list[(idx_a, idx_b, winner)]

    # Task 3 (reward model + RLHF)
    "reward_state_dict": None,  # RewardNet weights
    "rlhf_done": False,

    # Metrics history for the mini chart
    # keys: "after_train", "after_rlhf" -> {"mean": float, "std": float}
    "metrics": {
        "after_train": None,
        "after_rlhf": None,
    },
}


def _chart_points():
    """Return [(label, mean)] from SESSION.metrics, skipping missing."""
    pts = []
    if SESSION["metrics"]["after_train"] is not None:
        pts.append(("After Train", SESSION["metrics"]["after_train"]["mean"]))
    if SESSION["metrics"]["after_rlhf"] is not None:
        pts.append(("After RLHF", SESSION["metrics"]["after_rlhf"]["mean"]))
    return pts


def _render_index(request: HttpRequest, seed: int, form: TrainForm, show_results_inline: bool):
    """
    Render the single-page UI:
      - grid preview (randomized by seed)
      - train form
      - inline training results (if available)
      - Task 2 status (prefs count)
      - Task 3 status (reward fitted / RLHF done)
      - Mini chart data
    """
    env = GridWorld(seed=seed)
    grid = to_symbol_grid(env.state)

    chart_pts = _chart_points()
    chart_max = max([abs(v) for _, v in chart_pts], default=1.0)  # scale bars
    return render(request, "project5/index.html", {
        # grid + controls
        "grid": grid,
        "seed": seed,

        # Task 1 form/results
        "form": form,
        "show_results": show_results_inline and SESSION["trained"],
        "mean_return": SESSION["last_return_mean"],
        "std_return": SESSION["last_return_std"],

        # Task 2 status
        "trained": SESSION["trained"],
        "num_prefs": len(SESSION["prefs"]),

        # Task 3 status
        "reward_fit": SESSION["reward_state_dict"] is not None,
        "rlhf_done": SESSION["rlhf_done"],

        # Chart
        "chart_points": chart_pts,   # list of (label, mean)
        "chart_absmax": chart_max,   # scale factor
    })


# =========================
# Task 1 — Baseline training
# =========================
def index(request: HttpRequest):
    """Show the grid BEFORE training and the Train form."""
    grid_seed = int(request.GET.get("seed", SESSION.get("last_seed", 0)))  # for display grid only
    form = TrainForm(initial={
        "episodes": 300,
        "gamma": 0.99,
        "lr": 0.05,
        "max_steps": 50,
        "seed": 0,  # 👈 keep training seed fixed at 0
    })
    return _render_index(request, grid_seed, form, show_results_inline=True)


def train(request: HttpRequest):
    """Train the baseline policy with REINFORCE and show results inline on the same page."""
    if request.method != "POST":
        return redirect("project5:index")

    form = TrainForm(request.POST)
    if not form.is_valid():
        seed = int(request.POST.get("seed", SESSION.get("last_seed", 0)))
        return _render_index(request, seed, form, show_results_inline=False)

    episodes = form.cleaned_data["episodes"]
    gamma    = form.cleaned_data["gamma"]
    lr       = form.cleaned_data["lr"]
    max_steps= form.cleaned_data["max_steps"]
    seed     = form.cleaned_data["seed"]

    # Train baseline
    env = GridWorld(seed=seed)
    policy = PolicyNet()
    reinforce_train(env, policy, episodes=episodes, gamma=gamma, lr=lr, max_steps=max_steps)

    # Evaluate for reporting
    ret_mean, ret_std = evaluate_policy(env, policy, n_episodes=50, max_steps=max_steps, gamma=gamma)

    # Save session stats
    sd = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
    SESSION["trained"] = True
    SESSION["policy_state_dict"] = sd
    SESSION["last_return_mean"] = float(ret_mean)
    SESSION["last_return_std"]  = float(ret_std)
    SESSION["last_seed"] = int(seed)

    # Snapshot baseline once (first time training completes)
    if SESSION["baseline_state_dict"] is None:
        SESSION["baseline_state_dict"] = {k: v.clone() for k, v in sd.items()}

    # Update metrics history for the chart
    SESSION["metrics"]["after_train"] = {"mean": float(ret_mean), "std": float(ret_std)}

    # Rebuild form with same values for display
    filled_form = TrainForm(initial={
        "episodes": episodes, "gamma": gamma, "lr": lr, "max_steps": max_steps, "seed": seed,
    })

    return _render_index(request, seed, filled_form, show_results_inline=True)


# =========================
# Task 2 — Preferences
# =========================
def sample(request: HttpRequest):
    """Sample a fresh batch of trajectories (new random env seed every time)."""
    if not SESSION["trained"]:
        return HttpResponse("Train the model first.", status=400)

    policy = PolicyNet()
    policy.load_state_dict(SESSION["policy_state_dict"])

    # fresh random seed per batch so stats vary
    new_seed = random.SystemRandom().randint(0, 2**31 - 1)
    env = GridWorld(seed=new_seed)

    trajs = sample_trajectories(env, policy, n_trajectories=6, max_steps=50)
    SESSION["trajectories"] = trajs
    SESSION["prefs"].clear()
    SESSION["last_sample_seed"] = new_seed
    return redirect("project5:compare")

def compare(request: HttpRequest):
    """
    Show two random different trajectories and collect a preference.
    """
    trajs = SESSION.get("trajectories", [])
    if len(trajs) < 2:
        return redirect("project5:sample")

    idx_a, idx_b = random.sample(range(len(trajs)), 2)

    if request.method == "POST":
        choice = request.POST.get("preference")
        if choice in ["A", "B"]:
            winner = idx_a if choice == "A" else idx_b
            SESSION["prefs"].append((idx_a, idx_b, winner))
        return redirect("project5:compare")

    def summarize(traj):
        states, actions, rewards = traj
        return {
            "length": len(actions),
            "return": sum(rewards),
            "cheese": sum(1 for r in rewards if r >= 9.9),
            "traps": sum(1 for r in rewards if r <= -49.0),
        }

    A = summarize(trajs[idx_a])
    B = summarize(trajs[idx_b])

    return render(request, "project5/compare.html", {
        "A": A, "B": B, "num_prefs": len(SESSION["prefs"]),
    })

# =========================
# Task 3 — Reward fit + RLHF
# =========================
def fit_reward(request: HttpRequest):
    """Fit a RewardNet from collected preferences (Bradley–Terry-style objective)."""
    trajs = SESSION.get("trajectories", [])
    prefs = SESSION.get("prefs", [])
    if not SESSION["trained"]:
        return HttpResponse("Train the model first.", status=400)
    if len(trajs) < 2 or len(prefs) == 0:
        return HttpResponse("Collect preferences first (sample trajectories, compare pairs).", status=400)

    rnet = RewardNet()
    _ = fit_bradley_terry(rnet, trajs, prefs, steps=400, lr=1e-2, gamma=0.99)
    SESSION["reward_state_dict"] = {k: v.detach().cpu().clone() for k, v in rnet.state_dict().items()}
    SESSION["rlhf_done"] = False
    return redirect("project5:index")


def rlhf_retrain(request: HttpRequest):
    """Fine-tune the policy using the learned reward and a KL penalty to the baseline."""
    if not SESSION["trained"]:
        return HttpResponse("Train the model first.", status=400)
    if SESSION.get("reward_state_dict") is None:
        return HttpResponse("Fit the reward model first.", status=400)

    # load current policy
    policy = PolicyNet()
    policy.load_state_dict(SESSION["policy_state_dict"])

    # load / freeze baseline
    baseline = PolicyNet()
    if SESSION.get("baseline_state_dict") is None:
        SESSION["baseline_state_dict"] = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
    baseline.load_state_dict(SESSION["baseline_state_dict"])
    for p in baseline.parameters():
        p.requires_grad_(False)

    # load / freeze reward model
    rnet = RewardNet()
    rnet.load_state_dict(SESSION["reward_state_dict"])
    for p in rnet.parameters():
        p.requires_grad_(False)

    env = GridWorld(seed=SESSION["last_seed"])

    # Hyperparams from form; defaults here
    beta = float(request.POST.get("kl_beta", "0.01"))
    epochs = int(request.POST.get("epochs", "50"))
    lr = float(request.POST.get("lr", "1e-3"))
    max_steps = int(request.POST.get("max_steps", "50"))

    new_pol = rlhf_finetune(
        env, policy, rnet, baseline,
        beta=beta, epochs=epochs, lr=lr, max_steps=max_steps, gamma=0.99
    )

    # Evaluate & persist
    ret_mean, ret_std = evaluate_policy(env, new_pol, n_episodes=50, max_steps=max_steps, gamma=0.99)
    SESSION["policy_state_dict"] = {k: v.detach().cpu().clone() for k, v in new_pol.state_dict().items()}
    SESSION["last_return_mean"] = float(ret_mean)
    SESSION["last_return_std"]  = float(ret_std)
    SESSION["rlhf_done"] = True

    # Update metrics history for chart
    SESSION["metrics"]["after_rlhf"] = {"mean": float(ret_mean), "std": float(ret_std)}

    return redirect("project5:index")


# =========================
# Utility — Reset everything
# =========================
def reset(request: HttpRequest):
    """Clear all session state and go back to the index."""
    if request.method != "POST":
        return redirect("project5:index")

    for k in list(SESSION.keys()):
        SESSION[k] = None  # wipe values
    # reinitialize structure
    SESSION.update({
        "trained": False,
        "policy_state_dict": None,
        "last_return_mean": None,
        "last_return_std": None,
        "last_seed": 0,
        "baseline_state_dict": None,
        "trajectories": [],
        "prefs": [],
        "reward_state_dict": None,
        "rlhf_done": False,
        "metrics": {"after_train": None, "after_rlhf": None},
    })
    return redirect("project5:index")


from django.urls import reverse
import random

def reshuffle(request: HttpRequest):
    """Pick a new random seed for the grid preview only."""
    new_seed = random.randint(0, 2**31 - 1)
    return redirect(f"{reverse('project5:index')}?seed={new_seed}")

