
import numpy as np, json

def returns_from_traj(traj, gamma=0.99):
    _,_,rewards = traj
    G = 0.0; out = []
    for r in reversed(rewards):
        G = r + gamma*G
        out.append(G)
    out.reverse()
    return sum(out)

class RewardNet:
    # linear reward over observation features for simplicity
    def __init__(self, dim=6*5*5):
        self.w = np.zeros((dim,), dtype=float)

    def reward(self, obs, action):
        # ignore action for simplicity, reward = w^T x
        x = obs.reshape(-1)
        return float(x @ self.w)

    def to_json(self):
        return json.dumps({"w": self.w.tolist()})

    @staticmethod
    def from_json(s):
        obj = json.loads(s)
        r = RewardNet()
        r.w = np.array(obj["w"], dtype=float)
        return r

def fit_bradley_terry(model, trajectories, prefs, steps=200, lr=0.1, gamma=0.99):
    # pairwise preference on total (learned) return
    def learned_return(traj):
        states,actions,rewards = traj
        G = 0.0
        for obs,a,r in reversed(list(zip(states,actions,rewards))):
            G = model.reward(obs,a) + gamma * G
        return G

    for _ in range(steps):
        grad = np.zeros_like(model.w)
        loss = 0.0
        for (ia, ib, winner) in prefs:
            Ga = learned_return(trajectories[ia])
            Gb = learned_return(trajectories[ib])
            # probability A preferred over B: sigma(Ga - Gb)
            z = Ga - Gb
            p = 1.0 / (1.0 + np.exp(-z))
            y = 1.0 if winner == ia else 0.0
            loss += -(y*np.log(p+1e-8) + (1-y)*np.log(1-p+1e-8))
            # gradient wrt w is (y - p) * (dGa/dw - dGb/dw)
            dGa = sum(obs.reshape(-1) for obs in trajectories[ia][0])
            dGb = sum(obs.reshape(-1) for obs in trajectories[ib][0])
            grad += (y - p) * (dGa - dGb)
        model.w += lr * grad / max(1, len(prefs))
