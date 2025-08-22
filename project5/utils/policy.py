import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNet(nn.Module):
    """
    Conv(6->16, 3x3, same) -> ReLU -> Flatten -> Linear -> ReLU -> Linear (4 actions)
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )

    def forward(self, x):
        # x: (B,6,5,5)
        logits = self.net(x)               # (B, 4)
        probs  = F.softmax(logits, dim=-1) # (B, 4)
        return probs

    def act(self, obs, require_grad: bool = False):
        """
        obs: (6,5,5) tensor.
        If require_grad=True, returns a logp that participates in autograd.
        """
        if require_grad:
            probs = self.forward(obs.unsqueeze(0)).squeeze(0)
            dist = torch.distributions.Categorical(probs=probs)
            a = dist.sample()
            logp = dist.log_prob(a)  # requires grad
            return int(a.item()), logp
        else:
            with torch.no_grad():
                probs = self.forward(obs.unsqueeze(0)).squeeze(0)
                dist = torch.distributions.Categorical(probs=probs)
                a = dist.sample()
                logp = dist.log_prob(a)
                return int(a.item()), logp
