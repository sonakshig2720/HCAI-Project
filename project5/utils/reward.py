# project5/utils/reward.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class RewardNet(nn.Module):
    """Simple linear reward model over flattened obs (6*5*5 = 150 -> 1)."""
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(6 * 5 * 5, 1)

    def forward(self, obs):
        # obs: (6,5,5) or (B,6,5,5)
        if obs.dim() == 3:
            x = obs.reshape(1, -1)  # (1, 150)
        else:
            x = obs.reshape(obs.size(0), -1)  # (B, 150)
        r = self.lin(x).squeeze(-1)  # (B,) or (1,)
        return r

def learned_return(model: nn.Module, traj, gamma=0.99):
    """Sum of discounted learned rewards over a trajectory."""
    states, actions, rewards = traj
    G = 0.0
    for obs in reversed(states):
        with torch.no_grad():
            rhat = model(obs).item()
        G = rhat + gamma * G
    return G

def fit_bradley_terry(model: nn.Module, trajectories, prefs, steps=400, lr=1e-2, gamma=0.99, l2=1e-4):
    """
    Maximize preference likelihood: P(A > B) = sigmoid(G_A - G_B)
    where G_* are discounted sums of learned rewards along each trajectory.
    """
    if not prefs:
        return 0.0

    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    last_loss = 0.0
    for it in range(steps):
        opt.zero_grad()
        losses = []
        for ia, ib, winner in prefs:
            states_a, _, _ = trajectories[ia]
            states_b, _, _ = trajectories[ib]

            # compute learned returns with gradient through model
            # stack to do it faster
            Sa = torch.stack([s for s in states_a], dim=0)  # (Ta,6,5,5)
            Sb = torch.stack([s for s in states_b], dim=0)  # (Tb,6,5,5)

            # discounted sum on learned rewards
            Ra = model(Sa)  # (Ta,)
            Rb = model(Sb)  # (Tb,)

            Ga = 0.0
            for r in reversed(Ra):
                Ga = r + gamma * Ga
            Gb = 0.0
            for r in reversed(Rb):
                Gb = r + gamma * Gb

            z = Ga - Gb
            p = torch.sigmoid(z)
            y = torch.tensor(1.0 if winner == ia else 0.0, dtype=torch.float32, device=p.device)
            loss = F.binary_cross_entropy(p, y)
            losses.append(loss)

        total = torch.stack(losses).mean()
        total.backward()
        opt.step()
        last_loss = float(total.item())
    return last_loss
