from __future__ import annotations
import random
from pathlib import Path
from typing import Callable, Dict, List, Tuple
import numpy as np
import torch
from torch import tensor
from scipy.integrate import solve_ivp

from src.data_handlers.BaseDataHandlerParams import BaseDataHandlerParams
from src.data_handlers.base_data_handler import BaseDataHandler


# ============ Base 3D founders ============
def lorenz_rhs(t, x, p):
    sigma, rho, beta = p["sigma"], p["rho"], p["beta"]
    X, Y, Z = x
    return np.array([sigma*(Y - X), X*(rho - Z) - Y, X*Y - beta*Z], dtype=float)

def rossler_rhs(t, x, p):
    a, b, c = p["a"], p["b"], p["c"]
    X, Y, Z = x
    return np.array([-Y - Z, X + a*Y, b + Z*(X - c)], dtype=float)

def chen_rhs(t, x, p):
    a, b, c = p["a"], p["b"], p["c"]
    X, Y, Z = x
    return np.array([a*(Y - X), (c - a)*X - X*Z + c*Y, X*Y - b*Z], dtype=float)

FOUNDERS = [
    ("Lorenz63", 3, lorenz_rhs, dict(sigma=10.0, rho=28.0, beta=8/3), np.array([1.0, 1.0, 1.0])),
    ("Rossler",  3, rossler_rhs, dict(a=0.2, b=0.2, c=5.7),           np.array([0.1, 0.0, 0.0])),
    ("Chen",     3, chen_rhs,    dict(a=35.0, b=3.0, c=28.0),         np.array([-10.0, 0.0, 37.0])),
]

_rng = np.random.default_rng()


# ============ Utilities ============
def mutate_params(p: Dict[str, float], sigma: float = 0.15) -> Dict[str, float]:
    """Multiplicative jitter of parameters (keeps signs)."""
    q = {}
    for k, v in p.items():
        if v == 0: q[k] = 0.0
        else:
            s = np.exp(_rng.normal(0.0, sigma))
            q[k] = float(np.sign(v) * s * abs(v))
    return q

def make_rhs(f: Callable, params: Dict[str, float]) -> Callable:
    return lambda t, x: f(t, x, params)


def integrate(rhs, x0, tmax, dt, atol=1e-8, rtol=1e-6, radius=None):
    import math
    # exact count: include both t=0 and t=tmax
    n_total = max(1, int(math.floor(tmax / dt)) + 1)
    t_eval = dt * np.arange(n_total)

    events = None
    if radius is not None:
        def _rad_ev(t, y):
            return radius - np.linalg.norm(y)
        _rad_ev.terminal = True
        _rad_ev.direction = -1
        events = _rad_ev

    sol = solve_ivp(lambda t, x: rhs(t, x),
                    (0.0, t_eval[-1]),
                    x0,
                    t_eval=t_eval,
                    method="DOP853",
                    atol=atol, rtol=rtol,
                    events=events)

    if (not sol.success) or (sol.y.size == 0) or (sol.t.shape[0] < n_total):
        X = np.full((n_total, x0.shape[0]), np.nan, dtype=float)
        if sol.y.size:
            X[:sol.y.shape[1]] = sol.y.T
        return t_eval, X
    return sol.t, sol.y.T


def reject_bad(X: np.ndarray, radius=1e6, min_var=1e-6) -> bool:
    """Cull blow-ups and near-constant trajectories."""
    if not np.isfinite(X).all(): return True
    if np.max(np.linalg.norm(X, axis=1)) > radius: return True
    if np.sum(np.var(X, axis=0)) < min_var: return True
    return False

def skew_product(rhs_a: Callable, rhs_b: Callable, s_a: float, s_b: float) -> Callable:
    """6D skew system: A (driver) weakly drives B (response)."""
    def child(t, z):
        x = z[:3]; y = z[3:]
        fx = rhs_a(t, x)
        fy = rhs_b(t, y).copy()
        fy[0] += 0.05 * x[0]  # weak coupling
        return np.concatenate([s_a*fx, s_b*fy])
    return child


# ============ Sample ONE system (used for all splits) ============
def sample_random_skew_system(output_dim: int):
    """Pick founders, mutate params once, build a single skew-product system."""
    i, j = _rng.integers(0, len(FOUNDERS), size=2)
    (na, _, fa, pa0, xa0), (nb, _, fb, pb0, xb0) = FOUNDERS[i], FOUNDERS[j]
    pa, pb = mutate_params(pa0), mutate_params(pb0)
    rhs_a, rhs_b = make_rhs(fa, pa), make_rhs(fb, pb)

    # Fixed scales (fast; skip per-attempt RMS integration)
    s_a = 1.0
    s_b = 1.0
    rhs_child = skew_product(rhs_a, rhs_b, s_a, s_b)

    dim = 6
    z0_base = np.concatenate([xa0, xb0]).astype(float)

    def project(X):
        return X[:, :3] if output_dim == 3 else X

    meta = dict(parents=(na, nb), params=(pa, pb), scales=(s_a, s_b))
    return rhs_child, z0_base, dim, project, meta


def simulate_trajectory(
    rhs,
    z0_base: np.ndarray,
    dim: int,
    T: int,
    dt: float,
    warmup_frac: float,
    project: Callable[[np.ndarray], np.ndarray],
    ic_scale: float = 0.05,
    max_attempts: int = 6,
    min_var: float = 1e-8,
    radius: float = 1e6,
) -> np.ndarray:
    """
    Integrate the fixed skew system with different jittered ICs until a "good"
    trajectory is found (or give up after max_attempts).

    Good = finite, bounded (||x|| <= radius), and not near-constant
    (sum of per-dim variances >= min_var), both overall and on the
    post-warmup window.

    Returns:
        Array of shape [T, D] after projection (D=6 or 3), ready for tensor().
    """
    # choose tmax so that after warmup we still have exactly T samples
    tmax = T * dt / (1.0 - float(warmup_frac))

    for attempt in range(max_attempts):
        # jitter initial condition
        z0 = z0_base + ic_scale * _rng.normal(size=dim)

        # integrate
        _, X = integrate(rhs, z0, tmax=float(tmax), dt=float(dt), atol=1e-6, rtol=1e-4, radius=radius)


        # reject if the whole path already looks bad (NaNs, blow-up, or static)
        if reject_bad(X, radius=radius, min_var=min_var):
            continue

        # trim warmup and ensure we have enough samples
        w = int(warmup_frac * len(X))
        if w + T > len(X):
            # not enough points after warmup; try another IC
            continue
        Xw = X[w:w + T]

        # also check the post-warmup segment
        if reject_bad(Xw, radius=radius, min_var=min_var):
            continue

        # success
        return project(Xw)

    raise RuntimeError(f"Bad trajectory after {max_attempts} IC resamples.")



# ============ Data handler ============
class RandomSkewDataHandler(BaseDataHandler):
    """
    Train/Val/Test come from the SAME skew system (sampled once),
    differing only by initial conditions. Returns list of tensors shaped
    [1, T, D], where D = output_dim (6 by default, 3 if requested).
    """
    def __init__(self, hyperparams: BaseDataHandlerParams, experiment_name: str, is_eval_data: bool):
        super().__init__(hyperparams)

        self.warmup_frac   = float(hyperparams.get("warmup_frac", 0.1))
        self.output_dim    = int(hyperparams.get("output_dim", 6))   # 6 (driver+response) or 3 (driver only)
        self.noise_pct     = float(hyperparams.get("noise_pct", 0.0))# measurement noise

        seed = hyperparams.get("seed", 42)
        random.seed(seed); np.random.seed(seed)

        self._traj_dir = Path(self.save_dir) / f"{experiment_name}" / "trajectories"

        # Sample ONE system once; reuse for all splits
        self._rhs, self._z0_base, self._dim, self._project, self._meta = \
            sample_random_skew_system(self.output_dim)

        approach = hyperparams.get("split_approach", "trajectory")
        if approach != "trajectory":
            raise ValueError("Use split_approach='trajectory' with this handler.")

        # Build splits: same system, new IC per trajectory
        self.train_trajectories = self._build_set_same_system(self.num_trajectories)
        self.val_trajectories   = self._build_set_same_system(self.num_trajectories)
        self.test_trajectories  = self._build_set_same_system(self.num_trajectories)

        if is_eval_data:
            self.save_trajectories_to_file(self._traj_dir / "train", self.train_trajectories)
            self.save_trajectories_to_file(self._traj_dir / "val",   self.val_trajectories)
            self.save_trajectories_to_file(self._traj_dir / "test",  self.test_trajectories)

    def _build_set_same_system(self, count: int) -> List[torch.Tensor]:
        out: List[torch.Tensor] = []
        T, dt = len(self.t_eval), float(self.dt)
        for _ in range(count):
            X = simulate_trajectory(self._rhs, self._z0_base, self._dim,
                                    T, dt, self.warmup_frac, self._project, ic_scale=0.1)
            
            traj = tensor(X, dtype=torch.float32, device=self.device).unsqueeze(0)  # [1, T, D]

            # measurement noise (like other handlers), per-trajectory std
            if self.noise_pct > 0.0:
                scale = traj.std(dim=1, keepdim=True)                  # [1, 1, D]
                noise = torch.randn_like(traj) * (self.noise_pct * scale)
                traj = traj + noise

            out.append(traj)
        return out