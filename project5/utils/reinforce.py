import torch
import torch.optim as optim
import numpy as np

def _rollout(env, policy, max_steps=50, gamma=0.99, require_grad=False):
    """
    When require_grad=True, the collected logps will carry gradients.
    """
    obs = env.reset()
    logps, rewards = [], []
    for _ in range(max_steps):
        a, logp = policy.act(obs, require_grad=require_grad)
        obs, r = env.step(a)
        logps.append(logp)
        rewards.append(r)

    # returns-to-go
    G, rtg = 0.0, []
    for r in reversed(rewards):
        G = r + gamma * G
        rtg.append(G)
    rtg.reverse()

    returns = torch.tensor(rtg, dtype=torch.float32)
    if require_grad:
        logps = torch.stack(logps)  # (T,) with grad
    else:
        # keep dtype consistent; no grad needed in eval
        logps = torch.stack([lp.detach() for lp in logps])

    return logps, returns, float(sum(rewards))

def reinforce_train(env, policy, episodes=300, gamma=0.99, lr=1e-3, max_steps=50):
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    for ep in range(episodes):
        logps, returns, ep_return = _rollout(
            env, policy, max_steps=max_steps, gamma=gamma, require_grad=True
        )

        # normalize returns for stability
        if returns.std().item() > 1e-6:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # policy gradient: maximize sum(logpi * Gt)  => minimize negative
        loss = -(logps * returns.detach()).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (ep + 1) % 50 == 0:
            print(f"[REINFORCE] Episode {ep+1}/{episodes} | last_ep_return={ep_return:.2f}")

def evaluate_policy(env, policy, n_episodes=20, max_steps=50, gamma=0.99):
    returns = []
    for _ in range(n_episodes):
        _, _, ep_return = _rollout(
            env, policy, max_steps=max_steps, gamma=gamma, require_grad=False
        )
        returns.append(ep_return)
    returns = np.array(returns, dtype=float)
    return returns.mean(), returns.std()


def sample_trajectories(env, policy, n_trajectories=10, max_steps=50, gamma=0.99):
    trajs = []
    for _ in range(n_trajectories):
        obs = env.reset()
        states, actions, rewards = [], [], []
        for t in range(max_steps):
            a, logp = policy.act(obs, require_grad=False)
            obs, r = env.step(a)
            states.append(obs)
            actions.append(a)
            rewards.append(r)
        trajs.append((states, actions, rewards))
    return trajs
