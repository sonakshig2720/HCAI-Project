# project5/utils/rlhf.py
import torch
import torch.optim as optim
import torch.nn.functional as F

def categorical_from_policy(policy, obs_batch):
    # obs_batch: (B,6,5,5) tensor
    probs = policy(obs_batch)  # (B,4)
    return probs

def kl_mean(p, q):
    # p, q: (B,4) probabilities
    # KL(p || q)
    p = torch.clamp(p, 1e-8, 1.0)
    q = torch.clamp(q, 1e-8, 1.0)
    return (p * (p.log() - q.log())).sum(dim=-1).mean()

def rlhf_finetune(env, policy, reward_model, baseline_policy,
                  beta=0.01, epochs=50, lr=1e-3, max_steps=50, gamma=0.99):
    """
    KL-regularized policy gradient using learned reward_model.
    Loss = -(sum_t logpi(a_t|s_t) * G_t) + beta * KL(π || π_baseline) on visited states.
    """
    opt = optim.Adam(policy.parameters(), lr=lr)

    for ep in range(epochs):
        obs = env.reset()
        logps = []
        learned_rs = []
        states_batch = []

        for _ in range(max_steps):
            # sample action with grad
            probs = policy(obs.unsqueeze(0)).squeeze(0)    # (4,)
            dist = torch.distributions.Categorical(probs=probs)
            a = dist.sample()
            logps.append(dist.log_prob(a))
            states_batch.append(obs)

            # learned reward (no need to backprop into reward model here)
            with torch.no_grad():
                r_hat = reward_model(obs)
            learned_rs.append(r_hat)

            obs, _ = env.step(int(a.item()))

        # returns from learned rewards
        G = 0.0
        returns = []
        for r in reversed(learned_rs):
            G = float(r) + gamma * G
            returns.append(G)
        returns.reverse()
        returns = torch.tensor(returns, dtype=torch.float32)

        # normalize returns
        if returns.std().item() > 1e-6:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # policy loss
        logps_t = torch.stack(logps)
        pg_loss = -(logps_t * returns.detach()).sum()

        # KL penalty on states we actually visited
        with torch.no_grad():
            obs_batch = torch.stack(states_batch, dim=0)  # (T,6,5,5)
        p_curr = categorical_from_policy(policy, obs_batch)            # grad flows to policy
        with torch.no_grad():
            p_base = categorical_from_policy(baseline_policy, obs_batch)  # frozen
        kl = kl_mean(p_curr, p_base)

        loss = pg_loss + beta * kl

        opt.zero_grad()
        loss.backward()
        opt.step()

    return policy
