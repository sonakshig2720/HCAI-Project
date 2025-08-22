from dataclasses import dataclass
import numpy as np
import torch

GRID = 5
EMPTY, MOUSE, CHEESE, TRAP, WALL, ORGANIC = 0, 1, 2, 3, 4, 5
ACTIONS = [(-1,0),(1,0),(0,-1),(0,1)]  # up,down,left,right

@dataclass
class State:
    grid: np.ndarray
    pos: tuple

class GridWorld:
    """
    5x5 grid.
    Rewards:
      +10 entering CHEESE or ORGANIC
      -50 entering TRAP
      -0.2 empty move or bump into a wall
    Observation: 6x5x5 one-hot planes (mouse is rendered in its own channel).
    """
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.state = None
        self.reset()

    def _empty_cell(self, g):
        while True:
            x, y = self.rng.integers(0, GRID, 2)
            if g[x, y] == EMPTY:
                return (x, y)

    def reset(self):
        g = np.zeros((GRID, GRID), dtype=np.int32)
        # walls
        for _ in range(2):
            x, y = self._empty_cell(g)
            g[x, y] = WALL
        # traps
        for _ in range(2):
            x, y = self._empty_cell(g)
            g[x, y] = TRAP
        # cheese
        cx, cy = self._empty_cell(g)
        g[cx, cy] = CHEESE
        # organic cheese
        ox, oy = self._empty_cell(g)
        g[ox, oy] = ORGANIC
        # mouse
        mx, my = self._empty_cell(g)
        self.state = State(g, (mx, my))
        return self.obs()

    def obs(self):
        g = np.zeros((6, GRID, GRID), dtype=np.float32)
        grid = self.state.grid
        # render static objects
        for k, val in enumerate([EMPTY, MOUSE, CHEESE, TRAP, WALL, ORGANIC]):
            mask = (grid == val).astype(np.float32)
            if val == MOUSE:
                mask = np.zeros_like(mask, dtype=np.float32)
                x, y = self.state.pos
                mask[x, y] = 1.0
            g[k] = mask
        return torch.from_numpy(g)

    def step(self, a: int):
        dx, dy = ACTIONS[a]
        x, y = self.state.pos
        nx, ny = x + dx, y + dy
        reward = -0.2

        # bump or external wall
        if nx < 0 or ny < 0 or nx >= GRID or ny >= GRID or self.state.grid[nx, ny] == WALL:
            nx, ny = x, y
            reward = -0.2
        else:
            cell = self.state.grid[nx, ny]
            if cell == TRAP:
                reward = -50.0
            elif cell in (CHEESE, ORGANIC):
                reward = +10.0
                # consume cheese so episode can continue
                self.state.grid[nx, ny] = EMPTY

        self.state.pos = (nx, ny)
        return self.obs(), reward



def _cell_class_and_char(grid, pos, i, j):
    EMPTY, MOUSE, CHEESE, TRAP, WALL, ORGANIC = 0,1,2,3,4,5
    if (i, j) == pos:
        return "tile-mouse", "🐭"
    v = grid[i, j]
    if v == WALL:
        return "tile-wall", "🧱"
    if v == TRAP:
        return "tile-trap", "☠️"
    if v == CHEESE:
        return "tile-cheese", "🧀"
    if v == ORGANIC:
        return "tile-organic", "🧀"
    return "tile-empty", ""

def to_symbol_grid(state):
    """Return a 5x5 list of dicts: [{cls:'tile-...', char:'emoji'}, ...]"""
    g = state.grid
    x, y = state.pos
    out = []
    for i in range(g.shape[0]):
        row = []
        for j in range(g.shape[1]):
            cls, char = _cell_class_and_char(g, (x, y), i, j)
            row.append({"cls": cls, "char": char})
        out.append(row)
    return out

