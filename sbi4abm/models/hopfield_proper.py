from typing import Optional, Any

import numpy as np
import torch

from sbi4abm.models.model import Model


DEFAULT_PARAMS = {
    "rho": 1.0,
    "eps": 0.8,
    "lmbda": 0.5,
}


def _simulate(
    rho: float,
    eps: float,
    lmbda: float,
    s: np.ndarray,
    w: np.ndarray,
    seed: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:

    if not (seed is None):
        np.random.seed(seed)

    n = w.shape[1]
    nmin1 = n - 1
    oneminlmbda = 1.0 - lmbda
    nk = s[0].shape
    k = s.shape[-1]

    for t in range(w.shape[0] - 1):
        p = w[t].dot(s[t]) / nmin1
        pi = 1.0 / (1 + np.exp(-rho * p))
        s[t + 1] = 2 * (pi > 0.5 + eps * (np.random.random(nk) - 0.5)) - 1
        w[t + 1] = w[t] * oneminlmbda + lmbda * s[t + 1].dot(s[t + 1].T) / k
        np.fill_diagonal(w[t + 1], 0.0)

    return w, s


class Hopfield(Model):

    def __init__(
        self, n: int = 100, k: int = 2, initial_state: Optional[np.ndarray] = None
    ):

        self.n = n
        self.k = k
        assert not (initial_state is None), "Must specify an initial state"
        self.initial_state = initial_state

    def simulate(
        self,
        t_max: int = 50,
        parameters: Optional[dict[str, Any]] = None,
        seed: Optional[int] = None,
        **kwargs
    ):

        pars = parameters if not (parameters is None) else DEFAULT_PARAMS

        rho = float(pars["rho"])
        eps = float(pars["eps"])
        lmbda = float(pars["lmbda"])

        if t_max <= 0:
            raise ValueError("T must be positive int")

        s = np.ones((t_max + 1, self.n, self.k))
        s[0, :] = self.initial_state[:, -2:]
        w = np.zeros((t_max + 1, self.n, self.n))
        w[0] = self.initial_state[:, :-2]  # np.random.random((self._N, self._N))
        # np.fill_diagonal(w[0], 0.)
        w, s = _simulate(rho, eps, lmbda, s, w, seed)
        y = torch.from_numpy(np.concatenate((w, s), axis=-1)).float()
        return y
