"""
cdf_eri_optimizer_multibackend_v5_standalone.py

Multi-backend CDF ERI optimizer for fitting

    eri[p,q,r,s] ~= sum_t U[t,p,k] U[t,q,k] Z[t,k,l] U[t,r,l] U[t,s,l]

Backends currently supported:
  - MLX   : good default on Apple Silicon
  - Torch : good default on CUDA clusters and usable on Apple MPS

JAX is intentionally left as a future extension point.

Major additions relative to cdf_eri_optimizer_multibackend_v2.py
---------------------------------------------------------------
1. backend={"auto", "mlx", "torch"}, device={"auto", "gpu", "cuda", "mps", "cpu"}.
2. fix_U0 flag:
     fix_U0=True  => U[0] fixed, trainable rotations are exp(X[0]), ..., exp(X[ndf-1]); nfac=ndf+1.
     fix_U0=False => all U[t]=exp(X[t]) are trainable; nfac=ndf.
3. init={"random", "lchol"}; lchol init diagonalizes selected Cholesky vectors.
4. LChol initialization supports both fixed-U0 and free-U0 layouts.
5. Torch/MPS fix: avoids torch.linalg.matrix_exp, which is not implemented
   on MPS in many PyTorch releases, by using a differentiable scaling-and-
   squaring Padé expm for skew matrices.

Torch/MPS usage notes
---------------------
Use backend="torch", device="mps", dtype="float32". Do not rely on
PYTORCH_ENABLE_MPS_FALLBACK for this optimizer unless you intentionally want
unsupported operations to run on CPU; v3 keeps the skew-matrix exponential on
MPS using primitive differentiable Torch ops. Torch/MPS does not reliably
support float64, so the code forces float32 for device="mps". For serious
FeMoCo convergence, prefer backend="torch", device="cuda", dtype="float64"
when available.

This file is self-contained except for optional MLX/PyTorch installs.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Sequence
import json
import time
import warnings

import numpy as np
import scipy.linalg

try:  # optional
    import mlx.core as mx  # type: ignore
    from mlx.optimizers import Adam as MLXAdam  # type: ignore
    _HAS_MLX = True
except Exception:  # pragma: no cover
    mx = None  # type: ignore
    MLXAdam = None  # type: ignore
    _HAS_MLX = False

try:  # optional
    import torch  # type: ignore
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    _HAS_TORCH = False



def _torch_expm_pade13_scaling_squaring(A):
    """Differentiable matrix exponential using [13/13] Padé + scaling/squaring.

    This is used for Torch/MPS because torch.linalg.matrix_exp is not
    implemented on MPS in many PyTorch releases. The implementation uses only
    primitive Torch operations (matmul, additions, torch.linalg.solve), so
    autograd can differentiate through it.

    Parameters
    ----------
    A : torch.Tensor, shape (..., n, n)
        Matrix or batch of matrices.

    Returns
    -------
    expA : torch.Tensor, shape (..., n, n)
        Approximation to exp(A).
    """
    if not _HAS_TORCH:
        raise ImportError("Torch is required for _torch_expm_pade13_scaling_squaring.")
    if A.shape[-1] != A.shape[-2]:
        raise ValueError("A must be square in its last two dimensions.")

    import math

    n = int(A.shape[-1])
    dtype = A.dtype
    device = A.device

    I = torch.eye(n, dtype=dtype, device=device)
    if A.ndim > 2:
        I = I.expand(*A.shape[:-2], n, n)

    # Higham theta_13 bound for the 1-norm.
    theta13 = 5.371920351148152
    norm1 = torch.amax(torch.sum(torch.abs(A), dim=-2))
    norm1_float = float(norm1.detach().cpu())
    if norm1_float == 0.0:
        return I.clone()

    s = max(0, int(math.ceil(math.log2(norm1_float / theta13))))
    A = A / float(2 ** s)

    # Padé [13/13] coefficients from Higham/Al-Mohy.
    b = [
        64764752532480000.0,
        32382376266240000.0,
        7771770303897600.0,
        1187353796428800.0,
        129060195264000.0,
        10559470521600.0,
        670442572800.0,
        33522128640.0,
        1323241920.0,
        40840800.0,
        960960.0,
        16380.0,
        182.0,
        1.0,
    ]
    b = [torch.as_tensor(x, dtype=dtype, device=device) for x in b]

    A2 = A @ A
    A4 = A2 @ A2
    A6 = A4 @ A2

    U = A @ (
        A6 @ (b[13] * A6 + b[11] * A4 + b[9] * A2)
        + b[7] * A6
        + b[5] * A4
        + b[3] * A2
        + b[1] * I
    )
    V = (
        A6 @ (b[12] * A6 + b[10] * A4 + b[8] * A2)
        + b[6] * A6
        + b[4] * A4
        + b[2] * A2
        + b[0] * I
    )

    R = torch.linalg.solve(V - U, V + U)
    for _ in range(s):
        R = R @ R
    return R

try:  # optional project-local MLX helper; some MLX versions lack mx.linalg.expm
    from MLX_functions import expm_skew as _PROJECT_MLX_EXPM_SKEW  # type: ignore
except Exception:  # pragma: no cover
    _PROJECT_MLX_EXPM_SKEW = None


# =============================================================================
# NumPy initialization helpers
# =============================================================================


def _extract_system_arrays(system: Any) -> Tuple[np.ndarray, Optional[np.ndarray], float, Optional[Tuple[int, int]], int]:
    """Accept either an object with h1/eri/e_nuc/nelec/norb or a dict.

    Dict form must contain the user-facing keys:
        {"h1", "eri", "e_nuc", "nelec"}

    Optional dict key:
        "norb" -- if omitted, inferred from h1.shape[0].
    """
    if isinstance(system, dict):
        required = {"h1", "eri", "e_nuc", "nelec"}
        missing = required.difference(system.keys())
        if missing:
            raise ValueError(
                "system dict must contain keys {'h1', 'eri', 'e_nuc', 'nelec'}; "
                f"missing {sorted(missing)}."
            )
        h1 = np.asarray(system["h1"], dtype=np.float64)
        eri = np.asarray(system["eri"], dtype=np.float64) if system.get("eri") is not None else None
        e_nuc = float(system["e_nuc"])
        nelec = tuple(int(x) for x in system["nelec"])
        norb = int(system.get("norb", h1.shape[0]))
    else:
        if not hasattr(system, "h1") or not hasattr(system, "eri") or not hasattr(system, "e_nuc"):
            raise ValueError("system object must have h1, eri, and e_nuc attributes.")
        h1 = np.asarray(system.h1, dtype=np.float64)
        eri = np.asarray(system.eri, dtype=np.float64) if getattr(system, "eri") is not None else None
        e_nuc = float(system.e_nuc)
        nelec = tuple(system.nelec) if hasattr(system, "nelec") else None
        norb = int(system.norb) if hasattr(system, "norb") else h1.shape[0]
    if h1.shape != (norb, norb):
        raise ValueError(f"h1 must have shape {(norb, norb)}, got {h1.shape}.")
    if eri is not None and eri.shape != (norb, norb, norb, norb):
        raise ValueError(f"eri must have shape {(norb, norb, norb, norb)}, got {eri.shape}.")
    return h1, eri, e_nuc, nelec, norb


def make_SO_matrix(Q: np.ndarray) -> np.ndarray:
    Q = np.array(Q, dtype=np.float64, copy=True)
    if np.linalg.det(Q) < 0:
        Q[:, -1] *= -1.0
    return Q


def orthogonal_to_skew_log(Q: np.ndarray, imag_tol: float = 1e-8) -> np.ndarray:
    Q = make_SO_matrix(Q)
    X = scipy.linalg.logm(Q)
    max_im = float(np.max(np.abs(np.imag(X)))) if X.size else 0.0
    if max_im > imag_tol:
        warnings.warn(f"orthogonal_to_skew_log: logm imaginary part = {max_im:.3e}")
    X = np.real(X)
    X = 0.5 * (X - X.T)
    return np.ascontiguousarray(X, dtype=np.float64)


def lchol_guess_params(
    Lchol: np.ndarray,
    U0_fixed: Optional[np.ndarray] = None,
    n_lchol_keep: Optional[int] = None,
    sort_by_norm: bool = True,
    symmetrize: bool = True,
    fix_U0: bool = True,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Initialize CDF X/Z from Cholesky vectors.

    If fix_U0=True:
        U[0] is fixed externally. Z[0]=0. LChol factors fill U[1:], Z[1:].
        X.shape=(nkeep,norb,norb), Z.shape=(nkeep+1,norb,norb).

    If fix_U0=False:
        all U[t] are trainable. LChol factors fill U[:], Z[:].
        X.shape=(nkeep,norb,norb), Z.shape=(nkeep,norb,norb).
    """
    Lchol = np.asarray(Lchol, dtype=np.float64)
    if Lchol.ndim != 3 or Lchol.shape[1] != Lchol.shape[2]:
        raise ValueError("Lchol must have shape (nchol, norb, norb).")
    nchol, norb, _ = Lchol.shape
    if fix_U0:
        if U0_fixed is None:
            raise ValueError("U0_fixed is required when fix_U0=True.")
        U0_fixed = np.asarray(U0_fixed, dtype=np.float64)
        if U0_fixed.shape != (norb, norb):
            raise ValueError(f"U0_fixed must have shape {(norb, norb)}.")

    if n_lchol_keep is None:
        n_lchol_keep = nchol
    n_lchol_keep = int(n_lchol_keep)
    if n_lchol_keep < 1 or n_lchol_keep > nchol:
        raise ValueError("n_lchol_keep must satisfy 1 <= n_lchol_keep <= nchol.")

    if sort_by_norm:
        weights = np.linalg.norm(Lchol.reshape(nchol, -1), axis=1)
        order = np.argsort(weights)[::-1]
    else:
        order = np.arange(nchol)
    keep = np.ascontiguousarray(order[:n_lchol_keep], dtype=np.int64)

    X = np.empty((n_lchol_keep, norb, norb), dtype=np.float64)
    nfac = n_lchol_keep + 1 if fix_U0 else n_lchol_keep
    Z = np.zeros((nfac, norb, norb), dtype=np.float64)
    z_offset = 1 if fix_U0 else 0

    for out_t, chol_t in enumerate(keep):
        L = Lchol[int(chol_t)]
        if symmetrize:
            L = 0.5 * (L + L.T)
        evals, evecs = np.linalg.eigh(L)
        idx = np.argsort(np.abs(evals))[::-1]
        evals = evals[idx]
        evecs = evecs[:, idx]
        evecs = make_SO_matrix(evecs)
        X[out_t] = orthogonal_to_skew_log(evecs)
        Z[out_t + z_offset] = np.outer(evals, evals)

    return {"X": np.ascontiguousarray(X), "Z": np.ascontiguousarray(Z)}, keep


# =============================================================================
# Backend/device selection
# =============================================================================


def _select_backend_device(backend: str, device: str) -> Tuple[str, str]:
    backend = str(backend).lower()
    device = str(device).lower()
    if backend == "jax":
        raise NotImplementedError("JAX backend is reserved for a future implementation.")
    if backend not in ("auto", "mlx", "torch"):
        raise ValueError("backend must be 'auto', 'mlx', or 'torch'.")

    if backend == "auto":
        # Prefer CUDA for cluster users, then MLX GPU on local Mac, then torch MPS, then CPU.
        if _HAS_TORCH and torch.cuda.is_available():
            return "torch", "cuda"
        if _HAS_MLX:
            # MLX exists primarily on Apple platforms; try GPU default.
            return "mlx", "gpu"
        if _HAS_TORCH and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "torch", "mps"
        if _HAS_TORCH:
            return "torch", "cpu"
        raise ImportError("No supported backend found. Install mlx or torch.")

    if backend == "mlx":
        if not _HAS_MLX:
            raise ImportError("MLX backend requested but mlx is not installed.")
        if device == "auto":
            device = "gpu"
        if device in ("gpu", "mps", "cuda"):
            device = "gpu"
        elif device != "cpu":
            raise ValueError("For MLX, device must be 'auto', 'gpu', or 'cpu'.")
        return "mlx", device

    if backend == "torch":
        if not _HAS_TORCH:
            raise ImportError("Torch backend requested but torch is not installed.")
        if device == "auto" or device == "gpu":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        if device not in ("cuda", "mps", "cpu"):
            raise ValueError("For torch, device must be 'auto', 'gpu', 'cuda', 'mps', or 'cpu'.")
        return "torch", device

    raise AssertionError("unreachable")


def _dtype_for_backend(backend: str, device: str, dtype: str):
    dtype = str(dtype).lower()
    if dtype == "auto":
        # float64 for CUDA/CPU by default; float32 for Apple GPU paths.
        if device in ("mps", "gpu"):
            dtype = "float32"
        else:
            dtype = "float64"
    if backend == "mlx":
        if dtype == "float32":
            return mx.float32
        if dtype == "float64":
            return mx.float64
    elif backend == "torch":
        if dtype == "float32":
            return torch.float32
        if dtype == "float64":
            # Torch MPS historically has limited fp64; force fp32 to avoid failures.
            if device == "mps":
                warnings.warn("torch/mps does not reliably support float64; using float32.")
                return torch.float32
            return torch.float64
    raise ValueError("dtype must be 'auto', 'float32', or 'float64'.")


@dataclass
class CDFOptimizerConfig:
    ndf: int
    norb: int
    seed: Optional[int] = None
    x_init_scale: float = 1.0
    z_init_scale: float = 1.0
    fix_U0: bool = True
    include_identity: bool = True  # alias retained for compatibility
    base_rotation: str = "h1_eigh"
    contraction_mode: str = "pair_batched"
    batch_t: int = 4
    symmetrize_z: bool = True
    backend: str = "auto"
    device: str = "auto"
    dtype: str = "auto"
    init: str = "random"
    n_lchol_keep: Optional[int] = None


class CDFERIOptimizer:
    """Fit a compact full-core density-density ERI factorization using MLX or PyTorch."""

    def __init__(
        self,
        system: Any,
        ndf: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        x_init_scale: float = 1.0,
        z_init_scale: float = 1.0,
        include_identity: Optional[bool] = None,
        fix_U0: bool = True,
        base_rotation: str = "h1_eigh",
        U0: Optional[Any] = None,
        contraction_mode: str = "pair_batched",
        batch_t: int = 4,
        symmetrize_z: bool = True,
        backend: str = "auto",
        device: str = "auto",
        dtype: str = "auto",
        init: str = "random",
        Lchol: Optional[np.ndarray] = None,
        n_lchol_keep: Optional[int] = None,
        sort_lchol_by_norm: bool = True,
        symmetrize_lchol: bool = True,
        verbose_backend: bool = True,
    ):
        if include_identity is not None:
            # Old API compatibility. include_identity=True means fix_U0=True.
            fix_U0 = bool(include_identity)
        self.system = system
        h1_np, eri_np, e_nuc, nelec, norb = _extract_system_arrays(system)
        if eri_np is None:
            raise ValueError("system.eri is required for CDF fitting.")
        self.h1_np = h1_np
        self.eri_np = eri_np
        self.e_nuc = e_nuc
        self.nelec = nelec
        self.norb = int(norb)
        self.fix_U0 = bool(fix_U0)
        self.include_identity = self.fix_U0

        self.backend, self.device = _select_backend_device(backend, device)
        self.dtype_name = str(dtype).lower()
        self.dtype = _dtype_for_backend(self.backend, self.device, dtype)
        self.contraction_mode = str(contraction_mode)
        self.batch_t = int(batch_t)
        self.symmetrize_z_default = bool(symmetrize_z)
        self.base_rotation = str(base_rotation)
        self.history: List[Dict[str, float]] = []
        self.last_loss: Optional[float] = None
        self.keep_lchol: Optional[np.ndarray] = None

        # Device setup
        if self.backend == "mlx":
            if self.device == "gpu":
                try:
                    mx.set_default_device(mx.gpu)
                except Exception:
                    warnings.warn("Could not set MLX default GPU; continuing with MLX default device.")
            elif self.device == "cpu":
                mx.set_default_device(mx.cpu)
        elif self.backend == "torch":
            self.torch_device = torch.device(self.device)

        if verbose_backend:
            print(f"CDFERIOptimizer backend={self.backend} device={self.device} dtype={self.dtype}")

        self.h1 = self._array(h1_np)
        self.eri = self._array(eri_np)
        self.eri_norm2 = self._sum(self.eri * self.eri)
        self._eval(self.eri_norm2)

        # Base rotation for fixed-U0 mode.
        self.U0 = None
        if self.fix_U0:
            self.U0 = self._make_base_rotation(U0=U0, base_rotation=self.base_rotation)
            self._eval(self.U0)

        init = str(init).lower()
        if init not in ("random", "lchol"):
            raise ValueError("init must be 'random' or 'lchol'.")

        if params is not None:
            # Infer ndf from params if needed.
            X_np = np.asarray(params["X"])
            if ndf is None:
                ndf = int(X_np.shape[0])
        elif init == "lchol":
            if Lchol is None:
                raise ValueError("Lchol is required for init='lchol'.")
            nchol = int(np.asarray(Lchol).shape[0])
            if n_lchol_keep is None:
                n_lchol_keep = int(ndf) if ndf is not None else nchol
            ndf = int(n_lchol_keep)
        else:
            if ndf is None:
                raise ValueError("ndf is required when params is None and init='random'.")
            ndf = int(ndf)

        self.ndf = int(ndf)
        self.ntrain_u = self.ndf
        self.nfac = self.ndf + 1 if self.fix_U0 else self.ndf

        self.config = CDFOptimizerConfig(
            ndf=self.ndf,
            norb=self.norb,
            seed=seed,
            x_init_scale=float(x_init_scale),
            z_init_scale=float(z_init_scale),
            fix_U0=self.fix_U0,
            include_identity=self.fix_U0,
            base_rotation=str(base_rotation),
            contraction_mode=self.contraction_mode,
            batch_t=self.batch_t,
            symmetrize_z=self.symmetrize_z_default,
            backend=self.backend,
            device=self.device,
            dtype=self.dtype_name,
            init=init,
            n_lchol_keep=n_lchol_keep,
        )

        if params is None:
            if init == "lchol":
                U0_np = self.get_U0_numpy() if self.fix_U0 else None
                params_np, keep = lchol_guess_params(
                    Lchol=np.asarray(Lchol, dtype=np.float64),
                    U0_fixed=U0_np,
                    n_lchol_keep=int(n_lchol_keep) if n_lchol_keep is not None else self.ndf,
                    sort_by_norm=bool(sort_lchol_by_norm),
                    symmetrize=bool(symmetrize_lchol),
                    fix_U0=self.fix_U0,
                )
                self.keep_lchol = keep
                self.params = self._normalize_params(params_np)
            else:
                self.params = self._init_params(seed=seed, x_scale=x_init_scale, z_scale=z_init_scale)
        else:
            self.params = self._normalize_params(params)
        self._materialize_params()

    # ------------------------------------------------------------------
    # Backend ops
    # ------------------------------------------------------------------

    def _array(self, x: Any):
        if self.backend == "mlx":
            return mx.array(x, dtype=self.dtype)
        return torch.as_tensor(x, dtype=self.dtype, device=self.torch_device)

    def _zeros(self, shape: Sequence[int]):
        if self.backend == "mlx":
            return mx.zeros(tuple(shape), dtype=self.dtype)
        return torch.zeros(tuple(shape), dtype=self.dtype, device=self.torch_device)

    def _eye(self, n: int):
        if self.backend == "mlx":
            return mx.eye(n, dtype=self.dtype)
        return torch.eye(n, dtype=self.dtype, device=self.torch_device)

    def _sum(self, x, axis=None):
        if self.backend == "mlx":
            return mx.sum(x, axis=axis)
        return torch.sum(x, dim=axis) if axis is not None else torch.sum(x)

    def _abs(self, x):
        return mx.abs(x) if self.backend == "mlx" else torch.abs(x)

    def _sqrt(self, x):
        return mx.sqrt(x) if self.backend == "mlx" else torch.sqrt(x)

    def _log(self, x):
        return mx.log(x) if self.backend == "mlx" else torch.log(x)

    def _einsum(self, subs: str, *args):
        return mx.einsum(subs, *args) if self.backend == "mlx" else torch.einsum(subs, *args)

    def _stack(self, xs, axis=0):
        return mx.stack(xs, axis=axis) if self.backend == "mlx" else torch.stack(xs, dim=axis)

    def _swapaxes(self, x, a: int, b: int):
        return mx.swapaxes(x, a, b) if self.backend == "mlx" else torch.swapaxes(x, a, b)

    def _transpose4(self, x, perm):
        return mx.transpose(x, perm) if self.backend == "mlx" else x.permute(*perm)

    def _eval(self, *args):
        if self.backend == "mlx":
            mx.eval(*args)
        # torch eager: no-op, synchronize not needed here.

    def _to_numpy(self, x) -> np.ndarray:
        if self.backend == "mlx":
            self._eval(x)
            return np.asarray(x)
        return x.detach().cpu().numpy()

    def _item_float(self, x) -> float:
        if self.backend == "mlx":
            self._eval(x)
            return float(x.item())
        return float(x.detach().cpu().item())

    def _randn(self, shape, seed: Optional[int], scale: float):
        if self.backend == "mlx":
            if seed is not None:
                mx.random.seed(int(seed))
            return float(scale) * mx.random.normal(tuple(shape), dtype=self.dtype)
        gen = None
        if seed is not None:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(seed))
        arr = torch.randn(tuple(shape), generator=gen, dtype=self.dtype, device=self.torch_device)
        return float(scale) * arr

    def _expm_skew(self, X):
        """Return exp(A), where A is the skew-symmetric part of X.

        Torch CUDA/CPU use torch.linalg.matrix_exp. Torch MPS uses the local
        differentiable Padé scaling-and-squaring implementation because
        torch.linalg.matrix_exp is not implemented on MPS in many PyTorch
        releases. MLX prefers the project-local expm_skew implementation when
        available; otherwise it falls back to the existing MLX compatibility path.
        """
        A = 0.5 * (X - self._swapaxes(X, -1, -2))
        if self.backend == "mlx":
            # Prefer an exact/project-local implementation when available.
            if _PROJECT_MLX_EXPM_SKEW is not None:
                return _PROJECT_MLX_EXPM_SKEW(X)
            # Legacy MLX compatibility path retained from v2 for environments
            # without the project-local MLX scaling/squaring helper.
            I = self._eye(self.norb)
            return mx.linalg.solve(I - 0.5 * A, I + 0.5 * A)

        if self.backend == "torch":
            if self.device == "mps":
                return _torch_expm_pade13_scaling_squaring(A)
            return torch.linalg.matrix_exp(A)

        raise ValueError(f"Unsupported backend {self.backend}")

    # ------------------------------------------------------------------
    # Base rotation / params
    # ------------------------------------------------------------------

    def _make_base_rotation(self, U0: Optional[Any], base_rotation: str):
        if U0 is not None:
            return self._array(U0)
        if base_rotation == "identity":
            return self._eye(self.norb)
        if base_rotation == "h1_eigh":
            # Use NumPy for deterministic base rotation; avoid backend eig quirks.
            _, U = np.linalg.eigh(self.h1_np)
            return self._array(make_SO_matrix(U))
        if base_rotation == "custom":
            raise ValueError("base_rotation='custom' requires passing U0=...")
        raise ValueError("base_rotation must be 'h1_eigh', 'identity', or 'custom'.")

    def get_U0_numpy(self) -> Optional[np.ndarray]:
        if self.U0 is None:
            return None
        return self._to_numpy(self.U0)

    def set_base_rotation(self, U0: Any) -> None:
        if not self.fix_U0:
            raise ValueError("set_base_rotation is only meaningful when fix_U0=True.")
        U0_np = np.asarray(U0, dtype=np.float64)
        if U0_np.shape != (self.norb, self.norb):
            raise ValueError(f"U0 must have shape {(self.norb, self.norb)}, got {U0_np.shape}")
        self.U0 = self._array(U0_np)
        self.base_rotation = "custom"
        self._eval(self.U0)

    def _init_params(self, seed: Optional[int], x_scale: float, z_scale: float) -> Dict[str, Any]:
        X = self._randn((self.ntrain_u, self.norb, self.norb), seed, x_scale)
        Z = self._randn((self.nfac, self.norb, self.norb), None if seed is None else int(seed) + 1000003, z_scale)
        return {"X": X, "Z": Z}

    def _normalize_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        X = self._array(params["X"])
        Z = self._array(params["Z"])
        if tuple(X.shape) != (self.ntrain_u, self.norb, self.norb):
            raise ValueError(f"X must have shape {(self.ntrain_u, self.norb, self.norb)}, got {tuple(X.shape)}")
        if tuple(Z.shape) != (self.nfac, self.norb, self.norb):
            raise ValueError(f"Z must have shape {(self.nfac, self.norb, self.norb)}, got {tuple(Z.shape)}")
        return {"X": X, "Z": Z}

    def _materialize_params(self) -> None:
        if self.backend == "mlx":
            mx.eval(self.params["X"], self.params["Z"])
            self.params = {"X": mx.array(self.params["X"]), "Z": mx.array(self.params["Z"])}
            mx.eval(self.params["X"], self.params["Z"])
        else:
            self.params = {
                "X": self.params["X"].detach().clone().to(self.torch_device).to(self.dtype),
                "Z": self.params["Z"].detach().clone().to(self.torch_device).to(self.dtype),
            }

    # ------------------------------------------------------------------
    # Constructors / persistence
    # ------------------------------------------------------------------

    @classmethod
    def from_mol_obj(cls, mol_obj: Any, ndf: int, **kwargs):
        from NOCI import NOCISystem
        return cls(NOCISystem.from_mol_obj(mol_obj), ndf=ndf, **kwargs)

    @classmethod
    def load(cls, path: str | Path, system: Any, backend: Optional[str] = None, device: Optional[str] = None, **overrides):
        path = Path(path)
        data = np.load(path, allow_pickle=True)
        config = json.loads(str(data["config_json"]))
        params = {"X": data["X"], "Z": data["Z"]}
        U0 = data["U0"] if "U0" in data and data["U0"].size else None
        if backend is not None:
            config["backend"] = backend
        if device is not None:
            config["device"] = device
        config.update(overrides)
        obj = cls(
            system=system,
            ndf=int(config["ndf"]),
            params=params,
            seed=config.get("seed", None),
            x_init_scale=float(config.get("x_init_scale", 1.0)),
            z_init_scale=float(config.get("z_init_scale", 1.0)),
            fix_U0=bool(config.get("fix_U0", config.get("include_identity", True))),
            base_rotation=str(config.get("base_rotation", "h1_eigh")) if U0 is None else "custom",
            U0=U0,
            contraction_mode=str(config.get("contraction_mode", "pair_batched")),
            batch_t=int(config.get("batch_t", 4)),
            symmetrize_z=bool(config.get("symmetrize_z", True)),
            backend=str(config.get("backend", "auto")),
            device=str(config.get("device", "auto")),
            dtype=str(config.get("dtype", "auto")),
            verbose_backend=bool(config.get("verbose_backend", False)),
        )
        if "history_json" in data:
            obj.history = json.loads(str(data["history_json"]))
        if "last_loss" in data and not np.isnan(float(data["last_loss"])):
            obj.last_loss = float(data["last_loss"])
        if "keep_lchol" in data:
            obj.keep_lchol = data["keep_lchol"]
        return obj

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        self._materialize_params()
        payload = {
            "X": self.get_params_numpy()["X"],
            "Z": self.get_params_numpy()["Z"],
            "U0": np.array([]) if self.U0 is None else self.get_U0_numpy(),
            "config_json": np.array(json.dumps(asdict(self.config))),
            "history_json": np.array(json.dumps(self.history)),
            "last_loss": np.array(np.nan if self.last_loss is None else self.last_loss),
            "keep_lchol": np.array([]) if self.keep_lchol is None else self.keep_lchol,
        }
        np.savez(path, **payload)
        return path

    @classmethod
    def from_previous(
        cls,
        previous: "CDFERIOptimizer",
        new_ndf: int,
        seed: Optional[int] = None,
        x_new_scale: float = 1e-2,
        z_new_scale: float = 1e-3,
        zero_new_z: bool = True,
        **kwargs,
    ) -> "CDFERIOptimizer":
        if new_ndf < previous.ndf:
            raise ValueError("new_ndf must be >= previous.ndf for warm-start expansion.")
        old = previous.get_params_numpy()
        norb = previous.norb
        new_nfac = int(new_ndf) + 1 if previous.fix_U0 else int(new_ndf)
        X = np.zeros((int(new_ndf), norb, norb), dtype=old["X"].dtype)
        Z = np.zeros((new_nfac, norb, norb), dtype=old["Z"].dtype)
        X[: previous.ndf] = old["X"]
        Z[: previous.nfac] = old["Z"]
        rng = np.random.default_rng(seed)
        nadd_x = int(new_ndf) - previous.ndf
        if nadd_x > 0:
            X[previous.ndf:] = float(x_new_scale) * rng.normal(size=(nadd_x, norb, norb))
        nadd_z = new_nfac - previous.nfac
        if nadd_z > 0 and not zero_new_z:
            Z[previous.nfac:] = float(z_new_scale) * rng.normal(size=(nadd_z, norb, norb))
        opts = dict(
            system=previous.system,
            ndf=int(new_ndf),
            params={"X": X, "Z": Z},
            seed=seed,
            x_init_scale=previous.config.x_init_scale,
            z_init_scale=previous.config.z_init_scale,
            fix_U0=previous.fix_U0,
            base_rotation=previous.base_rotation,
            U0=previous.get_U0_numpy(),
            contraction_mode=previous.contraction_mode,
            batch_t=previous.batch_t,
            symmetrize_z=previous.symmetrize_z_default,
            backend=previous.backend,
            device=previous.device,
            dtype=previous.dtype_name,
        )
        opts.update(kwargs)
        return cls(**opts)

    # ------------------------------------------------------------------
    # Model / contractions
    # ------------------------------------------------------------------

    def X_to_U(self, params: Optional[Dict[str, Any]] = None):
        if params is None:
            params = self.params
        Us = []
        if self.fix_U0:
            Us.append(self.U0)
        for j in range(int(params["X"].shape[0])):
            Us.append(self._expm_skew(params["X"][j]))
        return self._stack(Us, axis=0)

    def effective_Z(self, Z, symmetrize_z: bool = True):
        if symmetrize_z:
            return 0.5 * (Z + self._swapaxes(Z, 1, 2))
        return Z

    def _cdf_eri_direct(self, U, Z):
        return self._einsum("tpk,tqk,tkl,trl,tsl->pqrs", U, U, Z, U, U)

    def _cdf_eri_pair_batched(self, U, Z, batch_t: int):
        norb = int(U.shape[1])
        nf = int(U.shape[0])
        eri_fit = self._zeros((norb, norb, norb, norb))
        bt = max(1, int(batch_t))
        for t0 in range(0, nf, bt):
            t1 = min(t0 + bt, nf)
            Ub = U[t0:t1]
            Zb = Z[t0:t1]
            A = self._einsum("tpk,tqk->tpqk", Ub, Ub)
            eri_fit = eri_fit + self._einsum("tpqk,tkl,trsl->pqrs", A, Zb, A)
        return eri_fit

    def _cdf_eri_factor_loop(self, U, Z):
        norb = int(U.shape[1])
        eri_fit = self._zeros((norb, norb, norb, norb))
        for t in range(int(U.shape[0])):
            A = self._einsum("pk,qk->pqk", U[t], U[t])
            eri_fit = eri_fit + self._einsum("pqk,kl,rsl->pqrs", A, Z[t], A)
        return eri_fit

    def reconstruct_eri(
        self,
        params: Optional[Dict[str, Any]] = None,
        contraction_mode: Optional[str] = None,
        batch_t: Optional[int] = None,
        symmetrize_z: Optional[bool] = None,
    ):
        if params is None:
            params = self.params
        if contraction_mode is None:
            contraction_mode = self.contraction_mode
        if batch_t is None:
            batch_t = self.batch_t
        if symmetrize_z is None:
            symmetrize_z = self.symmetrize_z_default
        U = self.X_to_U(params)
        Z = self.effective_Z(params["Z"], bool(symmetrize_z))
        if contraction_mode == "direct":
            return self._cdf_eri_direct(U, Z)
        if contraction_mode == "pair_batched":
            return self._cdf_eri_pair_batched(U, Z, int(batch_t))
        if contraction_mode == "factor_loop":
            return self._cdf_eri_factor_loop(U, Z)
        raise ValueError("contraction_mode must be 'direct', 'pair_batched', or 'factor_loop'.")

    def z_l1_from_Z(self, Z):
        return self._sum(self._abs(Z))

    def z_l2_from_Z(self, Z):
        return self._sum(Z * Z)

    def factor_entropy_from_Z(self, Z, eps: float = 1e-12):
        wt = self._sum(self._abs(Z), axis=(1, 2) if self.backend == "torch" else [1, 2])
        pt = wt / (self._sum(wt) + float(eps))
        return -self._sum(pt * self._log(pt + float(eps)))

    def loss_components(
        self,
        params: Optional[Dict[str, Any]] = None,
        lambda_z1: float = 1e-1,
        lambda_dist: float = 1e-1,
        lambda_z2: float = 0.0,
        lambda_norm: float = 0.0,
        contraction_mode: Optional[str] = None,
        batch_t: Optional[int] = None,
        symmetrize_z: Optional[bool] = None,
    ) -> Dict[str, Any]:
        if params is None:
            params = self.params
        if symmetrize_z is None:
            symmetrize_z = self.symmetrize_z_default
        eri_fit = self.reconstruct_eri(params, contraction_mode, batch_t, symmetrize_z)
        diff = self.eri - eri_fit
        fit = self._sum(diff * diff)
        rel_fit = self._sqrt(fit / (self.eri_norm2 + 1e-30))
        Zeff = self.effective_Z(params["Z"], bool(symmetrize_z))
        z1 = self.z_l1_from_Z(Zeff)
        z2 = self.z_l2_from_Z(Zeff)
        ent = self.factor_entropy_from_Z(Zeff)
        norm_mismatch = (z2 - self.eri_norm2) / (self.eri_norm2 + 1e-30)
        norm_penalty = norm_mismatch * norm_mismatch
        total = fit + float(lambda_z1) * z1 + float(lambda_dist) * ent + float(lambda_z2) * z2 + float(lambda_norm) * norm_penalty
        return {
            "total": total,
            "fit": fit,
            "rel_fit": rel_fit,
            "z1": z1,
            "z2": z2,
            "dist": ent,
            "norm_penalty": norm_penalty,
            "z_norm2": z2,
            "eri_norm2": self.eri_norm2,
        }

    def _loss(self, params, lambda_z1, lambda_dist, lambda_z2, lambda_norm, contraction_mode, batch_t, symmetrize_z):
        return self.loss_components(params, lambda_z1, lambda_dist, lambda_z2, lambda_norm, contraction_mode, batch_t, symmetrize_z)["total"]

    # ------------------------------------------------------------------
    # Optimization helpers
    # ------------------------------------------------------------------

    def _clip_grads(self, grads: Dict[str, Any], max_grad_norm: Optional[float]) -> Dict[str, Any]:
        if max_grad_norm is None or max_grad_norm <= 0:
            return grads
        gx, gz = grads["X"], grads["Z"]
        norm = self._sqrt(self._sum(gx * gx) + self._sum(gz * gz))
        if self.backend == "mlx":
            scale = mx.minimum(mx.array(1.0, dtype=self.dtype), float(max_grad_norm) / (norm + 1e-12))
        else:
            scale = torch.minimum(torch.tensor(1.0, dtype=self.dtype, device=self.torch_device), torch.tensor(float(max_grad_norm), dtype=self.dtype, device=self.torch_device) / (norm + 1e-12))
        return {"X": gx * scale, "Z": gz * scale}

    def _freeze_grads(self, grads: Dict[str, Any], optimize_X: bool, optimize_Z: bool) -> Dict[str, Any]:
        if self.backend == "mlx":
            gx = grads["X"] if optimize_X else mx.zeros_like(grads["X"])
            gz = grads["Z"] if optimize_Z else mx.zeros_like(grads["Z"])
        else:
            gx = grads["X"] if optimize_X else torch.zeros_like(grads["X"])
            gz = grads["Z"] if optimize_Z else torch.zeros_like(grads["Z"])
        return {"X": gx, "Z": gz}

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------

    def optimize(
        self,
        learning_rate: float = 1e-2,
        maxiter: int = 4000,
        lambda_z1: float = 1e-1,
        lambda_dist: float = 1e-1,
        lambda_z2: float = 0.0,
        lambda_norm: float = 0.0,
        print_every: int = 500,
        reset_history: bool = False,
        reset_optimizer: bool = True,
        return_history: bool = True,
        contraction_mode: Optional[str] = None,
        batch_t: Optional[int] = None,
        symmetrize_z: Optional[bool] = None,
        max_grad_norm: Optional[float] = None,
        optimize_X: bool = True,
        optimize_Z: bool = True,
        materialize_each_step: bool = True,
        print_initial: bool = True,
    ) -> List[Dict[str, float]]:
        if reset_history:
            self.history = []
        if contraction_mode is None:
            contraction_mode = self.contraction_mode
        else:
            self.contraction_mode = str(contraction_mode)
        if batch_t is None:
            batch_t = self.batch_t
        else:
            self.batch_t = int(batch_t)
        if symmetrize_z is None:
            symmetrize_z = self.symmetrize_z_default
        else:
            self.symmetrize_z_default = bool(symmetrize_z)
        self.config.contraction_mode = self.contraction_mode
        self.config.batch_t = self.batch_t
        self.config.symmetrize_z = self.symmetrize_z_default

        self._materialize_params()
        run_history: List[Dict[str, float]] = []
        t0 = time.time()

        if self.backend == "mlx":
            params = {"X": mx.array(self.params["X"]), "Z": mx.array(self.params["Z"])}
            mx.eval(params["X"], params["Z"])
            optimizer = MLXAdam(learning_rate=float(learning_rate))
            _ = reset_optimizer

            def loss_fn_raw(p):
                return self._loss(p, float(lambda_z1), float(lambda_dist), float(lambda_z2), float(lambda_norm), str(contraction_mode), int(batch_t), bool(symmetrize_z))

            loss_fn = mx.compile(loss_fn_raw)
            grad_fn = mx.grad(loss_fn, argnums=0)

            if print_initial:
                self._print_components("Initial", self.loss_components(params, lambda_z1, lambda_dist, lambda_z2, lambda_norm, contraction_mode, batch_t, symmetrize_z))

            for step in range(int(maxiter)):
                grads = grad_fn(params)
                grads = self._freeze_grads(grads, optimize_X, optimize_Z)
                grads = self._clip_grads(grads, max_grad_norm)
                loss = loss_fn(params)
                params = optimizer.apply_gradients(grads, params)
                if materialize_each_step:
                    mx.eval(params["X"], params["Z"], loss)
                else:
                    mx.eval(loss)
                rec = self._maybe_record(step, maxiter, print_every, params, learning_rate, lambda_z1, lambda_dist, lambda_z2, lambda_norm, batch_t, max_grad_norm, optimize_X, optimize_Z, contraction_mode, symmetrize_z, t0, run_history)
                if rec is not None and print_every is not None and int(print_every) > 0 and (step % int(print_every) == 0):
                    self._print_record(step, rec)
            mx.eval(params["X"], params["Z"])
            self.params = {"X": mx.array(params["X"]), "Z": mx.array(params["Z"])}
            mx.eval(self.params["X"], self.params["Z"])
        else:
            X = self.params["X"].detach().clone().requires_grad_(True)
            Z = self.params["Z"].detach().clone().requires_grad_(True)
            params = {"X": X, "Z": Z}
            opt_params = []
            if optimize_X:
                opt_params.append(X)
            if optimize_Z:
                opt_params.append(Z)
            optimizer = torch.optim.Adam(opt_params, lr=float(learning_rate)) if opt_params else None

            if print_initial:
                with torch.no_grad():
                    self._print_components("Initial", self.loss_components(params, lambda_z1, lambda_dist, lambda_z2, lambda_norm, contraction_mode, batch_t, symmetrize_z))

            for step in range(int(maxiter)):
                if optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)
                loss = self._loss(params, float(lambda_z1), float(lambda_dist), float(lambda_z2), float(lambda_norm), str(contraction_mode), int(batch_t), bool(symmetrize_z))
                loss.backward()
                if max_grad_norm is not None and max_grad_norm > 0 and opt_params:
                    torch.nn.utils.clip_grad_norm_(opt_params, max_norm=float(max_grad_norm))
                if optimizer is not None:
                    optimizer.step()
                if materialize_each_step and self.device == "cuda":
                    torch.cuda.synchronize()
                with torch.no_grad():
                    rec = self._maybe_record(step, maxiter, print_every, params, learning_rate, lambda_z1, lambda_dist, lambda_z2, lambda_norm, batch_t, max_grad_norm, optimize_X, optimize_Z, contraction_mode, symmetrize_z, t0, run_history)
                    if rec is not None and print_every is not None and int(print_every) > 0 and (step % int(print_every) == 0):
                        self._print_record(step, rec)
            self.params = {"X": X.detach().clone(), "Z": Z.detach().clone()}

        final_comps = self.loss_components(self.params, lambda_z1, lambda_dist, lambda_z2, lambda_norm, contraction_mode, batch_t, symmetrize_z)
        self.last_loss = self._item_float(final_comps["total"])
        print("************************************")
        print("Final Loss: ", self.last_loss)
        print("Final Fit:  ", self._item_float(final_comps["fit"]))
        print("Final Rel:  ", self._item_float(final_comps["rel_fit"]))
        print("Elapsed Time: ", time.time() - t0, " s")
        return run_history if return_history else []

    fit = optimize

    def _print_components(self, label: str, comps: Dict[str, Any]) -> None:
        print(
            f"{label}: "
            f"loss={self._item_float(comps['total']):.8e}  "
            f"fit={self._item_float(comps['fit']):.8e}  "
            f"rel={self._item_float(comps['rel_fit']):.8e}  "
            f"z1={self._item_float(comps['z1']):.8e}  "
            f"z2={self._item_float(comps['z2']):.8e}  "
            f"dist={self._item_float(comps['dist']):.8e}"
        )

    def _print_record(self, step: int, rec: Dict[str, float]) -> None:
        print(
            f"Step {step:6d}: "
            f"loss={rec['loss']:.8e}  fit={rec['fit']:.8e}  rel={rec['rel_fit']:.8e}  "
            f"z1={rec['z1']:.8e}  z2={rec['z2']:.8e}  dist={rec['dist']:.8e}"
        )

    def _maybe_record(self, step, maxiter, print_every, params, learning_rate, lambda_z1, lambda_dist, lambda_z2, lambda_norm, batch_t, max_grad_norm, optimize_X, optimize_Z, contraction_mode, symmetrize_z, t0, run_history):
        do_print = print_every is not None and int(print_every) > 0 and (step % int(print_every) == 0)
        is_last = step == int(maxiter) - 1
        if not (do_print or is_last):
            return None
        comps = self.loss_components(params, lambda_z1, lambda_dist, lambda_z2, lambda_norm, contraction_mode, batch_t, symmetrize_z)
        rec = {
            "step": float(step),
            "loss": self._item_float(comps["total"]),
            "fit": self._item_float(comps["fit"]),
            "rel_fit": self._item_float(comps["rel_fit"]),
            "z1": self._item_float(comps["z1"]),
            "z2": self._item_float(comps["z2"]),
            "dist": self._item_float(comps["dist"]),
            "norm_penalty": self._item_float(comps["norm_penalty"]),
            "learning_rate": float(learning_rate),
            "lambda_z1": float(lambda_z1),
            "lambda_dist": float(lambda_dist),
            "lambda_z2": float(lambda_z2),
            "lambda_norm": float(lambda_norm),
            "batch_t": float(batch_t),
            "max_grad_norm": float(max_grad_norm) if max_grad_norm is not None else -1.0,
            "optimize_X": float(bool(optimize_X)),
            "optimize_Z": float(bool(optimize_Z)),
            "elapsed": float(time.time() - t0),
            "backend": 0.0 if self.backend == "mlx" else 1.0,
        }
        run_history.append(rec)
        self.history.append(rec)
        return rec

    # ------------------------------------------------------------------
    # Diagnostics / exports
    # ------------------------------------------------------------------

    def relative_eri_error(self, **kwargs) -> float:
        # Force pure fit error by default, while allowing other loss_components kwargs
        # such as contraction_mode, batch_t, and symmetrize_z.
        kwargs.pop("lambda_z1", None)
        kwargs.pop("lambda_dist", None)
        comps = self.loss_components(self.params, lambda_z1=0.0, lambda_dist=0.0, **kwargs)
        return self._item_float(comps["rel_fit"])

    def get_cdf_arrays(self, symmetrize_z: Optional[bool] = None):
        if symmetrize_z is None:
            symmetrize_z = self.symmetrize_z_default
        U = self.X_to_U(self.params)
        Z = self.effective_Z(self.params["Z"], bool(symmetrize_z))
        self._eval(U, Z)
        return U, Z

    # compatibility aliases
    def get_cdf_mx(self, symmetrize_z: Optional[bool] = None):
        if self.backend != "mlx":
            raise RuntimeError("get_cdf_mx is only available for backend='mlx'. Use get_cdf_numpy().")
        return self.get_cdf_arrays(symmetrize_z=symmetrize_z)

    def get_cdf_torch(self, symmetrize_z: Optional[bool] = None):
        if self.backend != "torch":
            raise RuntimeError("get_cdf_torch is only available for backend='torch'. Use get_cdf_numpy().")
        return self.get_cdf_arrays(symmetrize_z=symmetrize_z)

    def get_cdf_numpy(self, symmetrize_z: Optional[bool] = None) -> Tuple[np.ndarray, np.ndarray]:
        U, Z = self.get_cdf_arrays(symmetrize_z=symmetrize_z)
        return self._to_numpy(U), self._to_numpy(Z)

    def get_params_numpy(self) -> Dict[str, np.ndarray]:
        self._materialize_params()
        return {"X": self._to_numpy(self.params["X"]), "Z": self._to_numpy(self.params["Z"])}

    def set_params(self, params: Dict[str, Any]) -> None:
        self.params = self._normalize_params(params)
        self._materialize_params()
        self.last_loss = None

    def reinitialize(self, seed: Optional[int] = None, x_init_scale: Optional[float] = None, z_init_scale: Optional[float] = None, init: str = "random", Lchol: Optional[np.ndarray] = None, n_lchol_keep: Optional[int] = None) -> None:
        if x_init_scale is None:
            x_init_scale = self.config.x_init_scale
        if z_init_scale is None:
            z_init_scale = self.config.z_init_scale
        if init == "lchol":
            if Lchol is None:
                raise ValueError("Lchol is required for init='lchol'.")
            U0_np = self.get_U0_numpy() if self.fix_U0 else None
            params_np, keep = lchol_guess_params(Lchol, U0_np, n_lchol_keep or self.ndf, fix_U0=self.fix_U0)
            self.keep_lchol = keep
            self.params = self._normalize_params(params_np)
        else:
            self.params = self._init_params(seed=seed, x_scale=float(x_init_scale), z_scale=float(z_init_scale))
        self._materialize_params()
        self.history = []
        self.last_loss = None

    def symmetry_errors(self) -> Dict[str, float]:
        G = self.reconstruct_eri(self.params)
        e_pq = self._item_float(torch.max(torch.abs(G - torch.swapaxes(G, 0, 1))) if self.backend == "torch" else mx.max(mx.abs(G - mx.swapaxes(G, 0, 1))))
        e_rs = self._item_float(torch.max(torch.abs(G - torch.swapaxes(G, 2, 3))) if self.backend == "torch" else mx.max(mx.abs(G - mx.swapaxes(G, 2, 3))))
        pair_perm = G.permute(2, 3, 0, 1) if self.backend == "torch" else mx.transpose(G, (2, 3, 0, 1))
        e_pair = self._item_float(torch.max(torch.abs(G - pair_perm)) if self.backend == "torch" else mx.max(mx.abs(G - pair_perm)))
        return {"pq": e_pq, "rs": e_rs, "pair": e_pair}

    def summary(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "device": self.device,
            "dtype": str(self.dtype),
            "norb": self.norb,
            "ndf": self.ndf,
            "nfac": self.nfac,
            "fix_U0": self.fix_U0,
            "base_rotation": self.base_rotation,
            "contraction_mode": self.contraction_mode,
            "batch_t": self.batch_t,
            "symmetrize_z": self.symmetrize_z_default,
            "last_loss": self.last_loss,
            "keep_lchol": None if self.keep_lchol is None else self.keep_lchol.tolist(),
        }


__all__ = [
    "CDFERIOptimizer",
    "CDFOptimizerConfig",
    "make_SO_matrix",
    "orthogonal_to_skew_log",
    "lchol_guess_params",
]


# =============================================================================
# v4 extension layer (inlined; no import dependency)
# =============================================================================

_V3CDFERIOptimizer = CDFERIOptimizer
def _as_bool_bliss(use_bliss_beta: bool, lambda_hbi_fro: float, lambda_beta_l2: float) -> bool:
    return bool(use_bliss_beta) or float(lambda_hbi_fro) != 0.0 or float(lambda_beta_l2) != 0.0


class CDFERIOptimizer(_V3CDFERIOptimizer):
    """v4 optimizer with Torch L-BFGS and optional BLISS beta objective."""

    def __init__(
        self,
        *args,
        beta: Optional[Any] = None,
        beta_init_scale: float = 0.0,
        eta_electrons: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if eta_electrons is None:
            if self.nelec is None:
                eta_electrons = 0.0
            else:
                eta_electrons = float(sum(int(x) for x in self.nelec))
        self.eta_electrons = float(eta_electrons)

        # h_core[p,q] = h[p,q] - 0.5 * sum_r g[p,r,r,q]
        self.h_core_np = self.h1_np - 0.5 * np.einsum("prrq->pq", self.eri_np)
        self.h_core = self._array(self.h_core_np)

        if beta is not None:
            B_np = np.asarray(beta, dtype=np.float64)
            if B_np.shape != (self.norb, self.norb):
                raise ValueError(f"beta/B must have shape {(self.norb, self.norb)}, got {B_np.shape}")
            self.B = self._array(B_np)
        elif beta_init_scale and float(beta_init_scale) != 0.0:
            self.B = self._randn((self.norb, self.norb), None if self.config.seed is None else int(self.config.seed) + 2000003, float(beta_init_scale))
        else:
            self.B = self._zeros((self.norb, self.norb))
        self._eval(self.B, self.h_core)

    # ------------------------------------------------------------------
    # BLISS beta / hBI helpers
    # ------------------------------------------------------------------

    def _get_B_from_params(self, params: Optional[Dict[str, Any]] = None):
        if params is not None and "B" in params:
            return params["B"]
        return self.B

    def beta_from_params(self, params: Optional[Dict[str, Any]] = None):
        B = self._get_B_from_params(params)
        return 0.5 * (B + self._swapaxes(B, 0, 1))

    def bliss_beta_shift_tensor(self, beta):
        """Return 0.5 * (beta_pq delta_rs + delta_pq beta_rs)."""
        I = self._eye(self.norb)
        return 0.5 * (beta[:, :, None, None] * I[None, None, :, :] + I[:, :, None, None] * beta[None, None, :, :])

    def hbi_matrix(self, beta=None, params: Optional[Dict[str, Any]] = None):
        if beta is None:
            beta = self.beta_from_params(params)
        return self.h_core + 0.5 * float(self.eta_electrons) * beta

    def hbi_traceless_fro_penalty(self, beta=None, params: Optional[Dict[str, Any]] = None):
        A = self.hbi_matrix(beta=beta, params=params)
        I = self._eye(self.norb)
        tr = self._sum(A * I) / float(self.norb)
        A0 = A - tr * I
        return self._sum(A0 * A0)

    def beta_l2_penalty(self, beta=None, params: Optional[Dict[str, Any]] = None):
        if beta is None:
            beta = self.beta_from_params(params)
        return self._sum(beta * beta)

    def hbi_eig_abs_numpy(self, beta: Optional[np.ndarray] = None, remove_alpha1: bool = True) -> Dict[str, Any]:
        """Diagnostic true spectral/eigenvalue 1-norm for hBI, computed in NumPy."""
        if beta is None:
            beta = self.get_beta_numpy()
        A = self.h_core_np + 0.5 * float(self.eta_electrons) * np.asarray(beta, dtype=np.float64)
        evals = np.linalg.eigvalsh(0.5 * (A + A.T))
        alpha1 = float(np.median(evals)) if remove_alpha1 else 0.0
        return {
            "alpha1": alpha1,
            "eigvals": evals,
            "eig_abs_sum": float(np.sum(np.abs(evals - alpha1))),
            "eig_abs_sum_no_alpha1": float(np.sum(np.abs(evals))),
            "fro_traceless": float(np.sum((A - np.trace(A) / A.shape[0] * np.eye(A.shape[0])) ** 2)),
        }

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def loss_components(
        self,
        params: Optional[Dict[str, Any]] = None,
        lambda_z1: float = 1e-1,
        lambda_dist: float = 1e-1,
        lambda_z2: float = 0.0,
        lambda_norm: float = 0.0,
        contraction_mode: Optional[str] = None,
        batch_t: Optional[int] = None,
        symmetrize_z: Optional[bool] = None,
        use_bliss_beta: bool = False,
        lambda_hbi_fro: float = 0.0,
        lambda_beta_l2: float = 0.0,
    ) -> Dict[str, Any]:
        if params is None:
            params = self.params
        if symmetrize_z is None:
            symmetrize_z = self.symmetrize_z_default

        eri_fit = self.reconstruct_eri(params, contraction_mode, batch_t, symmetrize_z)
        beta = self.beta_from_params(params)
        use_bliss = _as_bool_bliss(use_bliss_beta, lambda_hbi_fro, lambda_beta_l2)
        if use_bliss:
            shift = self.bliss_beta_shift_tensor(beta)
            diff = self.eri - shift - eri_fit
        else:
            diff = self.eri - eri_fit

        fit = self._sum(diff * diff)
        rel_fit = self._sqrt(fit / (self.eri_norm2 + 1e-30))
        Zeff = self.effective_Z(params["Z"], bool(symmetrize_z))
        z1 = self.z_l1_from_Z(Zeff)
        z2 = self.z_l2_from_Z(Zeff)
        ent = self.factor_entropy_from_Z(Zeff)
        norm_mismatch = (z2 - self.eri_norm2) / (self.eri_norm2 + 1e-30)
        norm_penalty = norm_mismatch * norm_mismatch
        hbi_fro = self.hbi_traceless_fro_penalty(beta=beta) if (use_bliss and float(lambda_hbi_fro) != 0.0) else fit * 0.0
        beta_l2 = self.beta_l2_penalty(beta=beta) if (use_bliss and float(lambda_beta_l2) != 0.0) else fit * 0.0
        total = (
            fit
            + float(lambda_z1) * z1
            + float(lambda_dist) * ent
            + float(lambda_z2) * z2
            + float(lambda_norm) * norm_penalty
            + float(lambda_hbi_fro) * hbi_fro
            + float(lambda_beta_l2) * beta_l2
        )
        return {
            "total": total,
            "fit": fit,
            "rel_fit": rel_fit,
            "z1": z1,
            "z2": z2,
            "dist": ent,
            "norm_penalty": norm_penalty,
            "z_norm2": z2,
            "eri_norm2": self.eri_norm2,
            "hbi_fro": hbi_fro,
            "beta_l2": beta_l2,
            "use_bliss_beta": 1.0 if use_bliss else 0.0,
        }

    def _loss(
        self,
        params,
        lambda_z1,
        lambda_dist,
        lambda_z2,
        lambda_norm,
        contraction_mode,
        batch_t,
        symmetrize_z,
        use_bliss_beta=False,
        lambda_hbi_fro=0.0,
        lambda_beta_l2=0.0,
    ):
        return self.loss_components(
            params,
            lambda_z1,
            lambda_dist,
            lambda_z2,
            lambda_norm,
            contraction_mode,
            batch_t,
            symmetrize_z,
            use_bliss_beta=use_bliss_beta,
            lambda_hbi_fro=lambda_hbi_fro,
            lambda_beta_l2=lambda_beta_l2,
        )["total"]

    # ------------------------------------------------------------------
    # Gradient helpers generalized to optional B
    # ------------------------------------------------------------------

    def _clip_grads(self, grads: Dict[str, Any], max_grad_norm: Optional[float]) -> Dict[str, Any]:
        if max_grad_norm is None or max_grad_norm <= 0:
            return grads
        total = None
        for g in grads.values():
            term = self._sum(g * g)
            total = term if total is None else total + term
        norm = self._sqrt(total)
        if self.backend == "mlx":
            scale = mx.minimum(mx.array(1.0, dtype=self.dtype), float(max_grad_norm) / (norm + 1e-12))
        else:
            scale = torch.minimum(
                torch.tensor(1.0, dtype=self.dtype, device=self.torch_device),
                torch.tensor(float(max_grad_norm), dtype=self.dtype, device=self.torch_device) / (norm + 1e-12),
            )
        return {k: v * scale for k, v in grads.items()}

    def _freeze_grads(self, grads: Dict[str, Any], optimize_X: bool, optimize_Z: bool, optimize_beta: bool = False) -> Dict[str, Any]:
        out = {}
        for k, v in grads.items():
            keep = (k == "X" and optimize_X) or (k == "Z" and optimize_Z) or (k == "B" and optimize_beta)
            if keep:
                out[k] = v
            else:
                out[k] = mx.zeros_like(v) if self.backend == "mlx" else torch.zeros_like(v)
        return out

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------

    def optimize(
        self,
        learning_rate: float = 1e-2,
        maxiter: int = 4000,
        lambda_z1: float = 1e-1,
        lambda_dist: float = 1e-1,
        lambda_z2: float = 0.0,
        lambda_norm: float = 0.0,
        print_every: int = 500,
        reset_history: bool = False,
        reset_optimizer: bool = True,
        return_history: bool = True,
        contraction_mode: Optional[str] = None,
        batch_t: Optional[int] = None,
        symmetrize_z: Optional[bool] = None,
        max_grad_norm: Optional[float] = None,
        optimize_X: bool = True,
        optimize_Z: bool = True,
        optimize_beta: Optional[bool] = None,
        materialize_each_step: bool = True,
        print_initial: bool = True,
        optimizer: str = "adam",
        lbfgs_history_size: int = 20,
        lbfgs_max_iter: int = 20,
        lbfgs_line_search_fn: Optional[str] = "strong_wolfe",
        adam_warmup_iter: int = 1000,
        use_bliss_beta: bool = False,
        lambda_hbi_fro: float = 0.0,
        lambda_beta_l2: float = 0.0,
    ) -> List[Dict[str, float]]:
        if reset_history:
            self.history = []
        if contraction_mode is None:
            contraction_mode = self.contraction_mode
        else:
            self.contraction_mode = str(contraction_mode)
        if batch_t is None:
            batch_t = self.batch_t
        else:
            self.batch_t = int(batch_t)
        if symmetrize_z is None:
            symmetrize_z = self.symmetrize_z_default
        else:
            self.symmetrize_z_default = bool(symmetrize_z)

        if optimize_beta is None:
            optimize_beta = _as_bool_bliss(use_bliss_beta, lambda_hbi_fro, lambda_beta_l2)
        use_bliss = _as_bool_bliss(use_bliss_beta, lambda_hbi_fro, lambda_beta_l2)
        optimizer = str(optimizer).lower()
        if optimizer not in ("adam", "lbfgs", "adam_then_lbfgs"):
            raise ValueError("optimizer must be 'adam', 'lbfgs', or 'adam_then_lbfgs'.")
        if self.backend == "mlx" and optimizer != "adam":
            raise NotImplementedError("L-BFGS is currently implemented only for backend='torch'. Use optimizer='adam' for MLX.")

        self.config.contraction_mode = self.contraction_mode
        self.config.batch_t = self.batch_t
        self.config.symmetrize_z = self.symmetrize_z_default

        self._materialize_params()
        run_history: List[Dict[str, float]] = []
        t0 = time.time()

        if self.backend == "mlx":
            params = {"X": mx.array(self.params["X"]), "Z": mx.array(self.params["Z"]), "B": mx.array(self.B)}
            mx.eval(params["X"], params["Z"], params["B"])
            opt = MLXAdam(learning_rate=float(learning_rate))
            _ = reset_optimizer

            def loss_fn_raw(p):
                return self._loss(
                    p,
                    float(lambda_z1),
                    float(lambda_dist),
                    float(lambda_z2),
                    float(lambda_norm),
                    str(contraction_mode),
                    int(batch_t),
                    bool(symmetrize_z),
                    bool(use_bliss),
                    float(lambda_hbi_fro),
                    float(lambda_beta_l2),
                )

            loss_fn = mx.compile(loss_fn_raw)
            grad_fn = mx.grad(loss_fn, argnums=0)

            if print_initial:
                self._print_components_v4("Initial", self.loss_components(params, lambda_z1, lambda_dist, lambda_z2, lambda_norm, contraction_mode, batch_t, symmetrize_z, use_bliss, lambda_hbi_fro, lambda_beta_l2))

            for step in range(int(maxiter)):
                grads = grad_fn(params)
                grads = self._freeze_grads(grads, optimize_X, optimize_Z, bool(optimize_beta))
                grads = self._clip_grads(grads, max_grad_norm)
                loss = loss_fn(params)
                params = opt.apply_gradients(grads, params)
                if materialize_each_step:
                    mx.eval(params["X"], params["Z"], params["B"], loss)
                else:
                    mx.eval(loss)
                rec = self._maybe_record_v4(step, maxiter, print_every, params, learning_rate, lambda_z1, lambda_dist, lambda_z2, lambda_norm, batch_t, max_grad_norm, optimize_X, optimize_Z, bool(optimize_beta), contraction_mode, symmetrize_z, t0, run_history, use_bliss, lambda_hbi_fro, lambda_beta_l2, optimizer)
                if rec is not None and print_every is not None and int(print_every) > 0 and (step % int(print_every) == 0):
                    self._print_record_v4(step, rec)
            mx.eval(params["X"], params["Z"], params["B"])
            self.params = {"X": mx.array(params["X"]), "Z": mx.array(params["Z"])}
            self.B = mx.array(params["B"])
            mx.eval(self.params["X"], self.params["Z"], self.B)
        else:
            X = self.params["X"].detach().clone().requires_grad_(bool(optimize_X))
            Z = self.params["Z"].detach().clone().requires_grad_(bool(optimize_Z))
            B = self.B.detach().clone().requires_grad_(bool(optimize_beta))
            params = {"X": X, "Z": Z, "B": B}
            opt_params = []
            if optimize_X:
                opt_params.append(X)
            if optimize_Z:
                opt_params.append(Z)
            if optimize_beta:
                opt_params.append(B)

            def torch_loss():
                return self._loss(
                    params,
                    float(lambda_z1),
                    float(lambda_dist),
                    float(lambda_z2),
                    float(lambda_norm),
                    str(contraction_mode),
                    int(batch_t),
                    bool(symmetrize_z),
                    bool(use_bliss),
                    float(lambda_hbi_fro),
                    float(lambda_beta_l2),
                )

            if print_initial:
                with torch.no_grad():
                    self._print_components_v4("Initial", self.loss_components(params, lambda_z1, lambda_dist, lambda_z2, lambda_norm, contraction_mode, batch_t, symmetrize_z, use_bliss, lambda_hbi_fro, lambda_beta_l2))

            def adam_step_loop(nsteps: int, start_step: int = 0):
                adam = torch.optim.Adam(opt_params, lr=float(learning_rate)) if opt_params else None
                for local in range(int(nsteps)):
                    step = start_step + local
                    if adam is not None:
                        adam.zero_grad(set_to_none=True)
                    loss = torch_loss()
                    if opt_params:
                        loss.backward()
                        if max_grad_norm is not None and max_grad_norm > 0:
                            torch.nn.utils.clip_grad_norm_(opt_params, max_norm=float(max_grad_norm))
                        adam.step()
                    if materialize_each_step and self.device == "cuda":
                        torch.cuda.synchronize()
                    with torch.no_grad():
                        rec = self._maybe_record_v4(step, maxiter, print_every, params, learning_rate, lambda_z1, lambda_dist, lambda_z2, lambda_norm, batch_t, max_grad_norm, optimize_X, optimize_Z, bool(optimize_beta), contraction_mode, symmetrize_z, t0, run_history, use_bliss, lambda_hbi_fro, lambda_beta_l2, "adam")
                        if rec is not None and print_every is not None and int(print_every) > 0 and (step % int(print_every) == 0):
                            self._print_record_v4(step, rec)

            def lbfgs_step_loop(nsteps: int, start_step: int = 0):
                lbfgs = torch.optim.LBFGS(
                    opt_params,
                    lr=float(learning_rate),
                    max_iter=int(lbfgs_max_iter),
                    history_size=int(lbfgs_history_size),
                    line_search_fn=lbfgs_line_search_fn,
                ) if opt_params else None
                for local in range(int(nsteps)):
                    step = start_step + local
                    if lbfgs is not None:
                        def closure():
                            lbfgs.zero_grad(set_to_none=True)
                            loss = torch_loss()
                            loss.backward()
                            if max_grad_norm is not None and max_grad_norm > 0:
                                torch.nn.utils.clip_grad_norm_(opt_params, max_norm=float(max_grad_norm))
                            return loss
                        lbfgs.step(closure)
                    if materialize_each_step and self.device == "cuda":
                        torch.cuda.synchronize()
                    with torch.no_grad():
                        rec = self._maybe_record_v4(step, maxiter, print_every, params, learning_rate, lambda_z1, lambda_dist, lambda_z2, lambda_norm, batch_t, max_grad_norm, optimize_X, optimize_Z, bool(optimize_beta), contraction_mode, symmetrize_z, t0, run_history, use_bliss, lambda_hbi_fro, lambda_beta_l2, "lbfgs")
                        if rec is not None and print_every is not None and int(print_every) > 0 and (step % int(print_every) == 0):
                            self._print_record_v4(step, rec)

            if optimizer == "adam":
                adam_step_loop(int(maxiter), 0)
            elif optimizer == "lbfgs":
                lbfgs_step_loop(int(maxiter), 0)
            else:
                nwarm = min(int(adam_warmup_iter), int(maxiter))
                if nwarm > 0:
                    adam_step_loop(nwarm, 0)
                if int(maxiter) - nwarm > 0:
                    lbfgs_step_loop(int(maxiter) - nwarm, nwarm)

            self.params = {"X": X.detach().clone(), "Z": Z.detach().clone()}
            self.B = B.detach().clone()

        final_comps = self.loss_components(self.params | {"B": self.B} if isinstance(self.params, dict) else None, lambda_z1, lambda_dist, lambda_z2, lambda_norm, contraction_mode, batch_t, symmetrize_z, use_bliss, lambda_hbi_fro, lambda_beta_l2)
        self.last_loss = self._item_float(final_comps["total"])
        print("************************************")
        print("Final Loss: ", self.last_loss)
        print("Final Fit:  ", self._item_float(final_comps["fit"]))
        print("Final Rel:  ", self._item_float(final_comps["rel_fit"]))
        if use_bliss:
            print("Final hBI Fro: ", self._item_float(final_comps["hbi_fro"]))
            print("Final beta L2: ", self._item_float(final_comps["beta_l2"]))
        print("Elapsed Time: ", time.time() - t0, " s")
        return run_history if return_history else []

    fit = optimize

    def _print_components_v4(self, label: str, comps: Dict[str, Any]) -> None:
        print(
            f"{label}: "
            f"loss={self._item_float(comps['total']):.8e}  "
            f"fit={self._item_float(comps['fit']):.8e}  "
            f"rel={self._item_float(comps['rel_fit']):.8e}  "
            f"z1={self._item_float(comps['z1']):.8e}  "
            f"z2={self._item_float(comps['z2']):.8e}  "
            f"dist={self._item_float(comps['dist']):.8e}  "
            f"hBI_fro={self._item_float(comps['hbi_fro']):.8e}"
        )

    def _print_record_v4(self, step: int, rec: Dict[str, float]) -> None:
        print(
            f"Step {step:6d}: "
            f"loss={rec['loss']:.8e}  fit={rec['fit']:.8e}  rel={rec['rel_fit']:.8e}  "
            f"z1={rec['z1']:.8e}  z2={rec['z2']:.8e}  dist={rec['dist']:.8e}  "
            f"hBI_fro={rec['hbi_fro']:.8e}  opt={rec.get('optimizer_name','')}"
        )

    def _maybe_record_v4(self, step, maxiter, print_every, params, learning_rate, lambda_z1, lambda_dist, lambda_z2, lambda_norm, batch_t, max_grad_norm, optimize_X, optimize_Z, optimize_beta, contraction_mode, symmetrize_z, t0, run_history, use_bliss_beta, lambda_hbi_fro, lambda_beta_l2, optimizer_name):
        do_print = print_every is not None and int(print_every) > 0 and (step % int(print_every) == 0)
        is_last = step == int(maxiter) - 1
        if not (do_print or is_last):
            return None
        comps = self.loss_components(params, lambda_z1, lambda_dist, lambda_z2, lambda_norm, contraction_mode, batch_t, symmetrize_z, use_bliss_beta, lambda_hbi_fro, lambda_beta_l2)
        rec = {
            "step": float(step),
            "loss": self._item_float(comps["total"]),
            "fit": self._item_float(comps["fit"]),
            "rel_fit": self._item_float(comps["rel_fit"]),
            "z1": self._item_float(comps["z1"]),
            "z2": self._item_float(comps["z2"]),
            "dist": self._item_float(comps["dist"]),
            "norm_penalty": self._item_float(comps["norm_penalty"]),
            "hbi_fro": self._item_float(comps["hbi_fro"]),
            "beta_l2": self._item_float(comps["beta_l2"]),
            "learning_rate": float(learning_rate),
            "lambda_z1": float(lambda_z1),
            "lambda_dist": float(lambda_dist),
            "lambda_z2": float(lambda_z2),
            "lambda_norm": float(lambda_norm),
            "lambda_hbi_fro": float(lambda_hbi_fro),
            "lambda_beta_l2": float(lambda_beta_l2),
            "batch_t": float(batch_t),
            "max_grad_norm": float(max_grad_norm) if max_grad_norm is not None else -1.0,
            "optimize_X": float(bool(optimize_X)),
            "optimize_Z": float(bool(optimize_Z)),
            "optimize_beta": float(bool(optimize_beta)),
            "elapsed": float(time.time() - t0),
            "backend": 0.0 if self.backend == "mlx" else 1.0,
            "optimizer_name": str(optimizer_name),
        }
        run_history.append(rec)
        self.history.append(rec)
        return rec

    # ------------------------------------------------------------------
    # Postprocessing: alpha_1 and alpha_2^t median shifts
    # ------------------------------------------------------------------

    def get_beta_numpy(self) -> np.ndarray:
        return 0.5 * (self._to_numpy(self.B) + self._to_numpy(self.B).T)

    def optimal_alpha1(self, beta: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Return alpha_1 median solution for sum_i |eig_i(A)-alpha_1|."""
        return self.hbi_eig_abs_numpy(beta=beta, remove_alpha1=True)

    def alpha2_median_shifts(self, symmetrize_z: Optional[bool] = None) -> Dict[str, Any]:
        """Compute alpha_2^t = median_{kl} Z[t,k,l] after optional symmetrization."""
        if symmetrize_z is None:
            symmetrize_z = self.symmetrize_z_default
        Z = self.get_params_numpy()["Z"]
        if symmetrize_z:
            Z_eff = 0.5 * (Z + np.swapaxes(Z, 1, 2))
        else:
            Z_eff = Z.copy()
        a_t = np.median(Z_eff.reshape(Z_eff.shape[0], -1), axis=1)
        Z_shifted = Z_eff - a_t[:, None, None] * np.ones((1, self.norb, self.norb), dtype=Z_eff.dtype)
        return {
            "alpha2_t": a_t,
            "alpha2_total": float(np.sum(a_t)),
            "Z_shifted": Z_shifted,
            "z1_before": float(np.sum(np.abs(Z_eff))),
            "z1_after": float(np.sum(np.abs(Z_shifted))),
        }

    def apply_alpha2_median_shifts(self, symmetrize_z: Optional[bool] = None) -> Dict[str, Any]:
        """Apply per-factor median shifts to Z in-place and return shift metadata."""
        info = self.alpha2_median_shifts(symmetrize_z=symmetrize_z)
        params_np = self.get_params_numpy()
        params_np["Z"] = info["Z_shifted"]
        # Preserve current B.
        self.set_params({"X": params_np["X"], "Z": params_np["Z"]})
        return info

    def final_bliss_metadata(self, symmetrize_z: Optional[bool] = None) -> Dict[str, Any]:
        """Compute final alpha_1 and alpha_2^t metadata without mutating parameters."""
        beta = self.get_beta_numpy()
        a1 = self.optimal_alpha1(beta=beta)
        a2 = self.alpha2_median_shifts(symmetrize_z=symmetrize_z)
        return {
            "eta_electrons": self.eta_electrons,
            "alpha1": a1["alpha1"],
            "hbi_eig_abs_sum": a1["eig_abs_sum"],
            "hbi_eig_abs_sum_no_alpha1": a1["eig_abs_sum_no_alpha1"],
            "hbi_fro_traceless": a1["fro_traceless"],
            "alpha2_t": a2["alpha2_t"],
            "alpha2_total": a2["alpha2_total"],
            "z1_before_alpha2": a2["z1_before"],
            "z1_after_alpha2": a2["z1_after"],
            "beta": beta,
        }

    # ------------------------------------------------------------------
    # Persistence / diagnostics
    # ------------------------------------------------------------------

    def get_params_numpy(self) -> Dict[str, np.ndarray]:
        base = super().get_params_numpy()
        base["B"] = self._to_numpy(self.B)
        base["beta"] = self.get_beta_numpy()
        return base

    def set_params(self, params: Dict[str, Any]) -> None:
        B = params.get("B", params.get("beta", None))
        super().set_params({"X": params["X"], "Z": params["Z"]})
        if B is not None:
            self.B = self._array(B)
            self._eval(self.B)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        self._materialize_params()
        payload = {
            "X": super().get_params_numpy()["X"],
            "Z": super().get_params_numpy()["Z"],
            "B": self._to_numpy(self.B),
            "U0": np.array([]) if self.U0 is None else self.get_U0_numpy(),
            "config_json": np.array(json.dumps(self.summary())),
            "history_json": np.array(json.dumps(self.history)),
            "last_loss": np.array(np.nan if self.last_loss is None else self.last_loss),
            "keep_lchol": np.array([]) if self.keep_lchol is None else self.keep_lchol,
            "eta_electrons": np.array(self.eta_electrons),
        }
        np.savez(path, **payload)
        return path

    @classmethod
    def load(cls, path: str | Path, system: Any, backend: Optional[str] = None, device: Optional[str] = None, **overrides):
        path = Path(path)
        data = np.load(path, allow_pickle=True)
        config = json.loads(str(data["config_json"]))
        params = {"X": data["X"], "Z": data["Z"]}
        beta = data["B"] if "B" in data else None
        U0 = data["U0"] if "U0" in data and data["U0"].size else None
        if backend is not None:
            config["backend"] = backend
        if device is not None:
            config["device"] = device
        config.update(overrides)
        dtype_cfg = str(config.get("dtype_name", config.get("dtype", "auto")))
        if dtype_cfg.startswith("torch.") or dtype_cfg.startswith("mx."):
            dtype_cfg = dtype_cfg.split(".")[-1]

        obj = cls(
            system=system,
            ndf=int(config.get("ndf", params["X"].shape[0])),
            params=params,
            beta=beta,
            eta_electrons=float(data["eta_electrons"]) if "eta_electrons" in data else config.get("eta_electrons", None),
            fix_U0=bool(config.get("fix_U0", config.get("include_identity", True))),
            base_rotation=str(config.get("base_rotation", "h1_eigh")) if U0 is None else "custom",
            U0=U0,
            contraction_mode=str(config.get("contraction_mode", "pair_batched")),
            batch_t=int(config.get("batch_t", 4)),
            symmetrize_z=bool(config.get("symmetrize_z", True)),
            backend=str(config.get("backend", "auto")),
            device=str(config.get("device", "auto")),
            dtype=dtype_cfg,
            verbose_backend=bool(config.get("verbose_backend", False)),
        )
        if "history_json" in data:
            obj.history = json.loads(str(data["history_json"]))
        if "last_loss" in data and not np.isnan(float(data["last_loss"])):
            obj.last_loss = float(data["last_loss"])
        if "keep_lchol" in data:
            obj.keep_lchol = data["keep_lchol"]
        return obj

    def summary(self) -> Dict[str, Any]:
        s = super().summary()
        s.update({
            "version": "v4",
            "dtype_name": self.dtype_name,
            "eta_electrons": self.eta_electrons,
            "has_beta": True,
            "beta_norm_fro": float(np.linalg.norm(self.get_beta_numpy())),
        })
        return s


__all__ = [
    "CDFERIOptimizer",
    "CDFOptimizerConfig",
    "make_SO_matrix",
    "orthogonal_to_skew_log",
    "lchol_guess_params",
]


# =============================================================================
# v5 extension layer (inlined; no import dependency)
# =============================================================================

_V4CDFERIOptimizer = CDFERIOptimizer
def _np_project_so(U: np.ndarray) -> np.ndarray:
    """Nearest orthogonal projection by SVD, with det +1."""
    U = np.asarray(U, dtype=np.float64)
    W, _, Vt = np.linalg.svd(U, full_matrices=False)
    Q = W @ Vt
    if np.linalg.det(Q) < 0:
        W[:, -1] *= -1.0
        Q = W @ Vt
    return np.ascontiguousarray(Q)


class CDFERIOptimizer(_V4CDFERIOptimizer):
    """v5 CDF/BLISS optimizer with multiple orthogonal-rotation strategies."""

    def __init__(
        self,
        *args,
        rotation_param: str = "expm",
        hgh_theta_init_scale: float = 1e-4,
        hgh_v_init_scale: float = 1.0,
        free_u_from_current: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.version = "v5"
        self.rotation_param = str(rotation_param).lower()
        if self.rotation_param not in ("expm", "free", "hgh"):
            raise ValueError("rotation_param must be 'expm', 'free', or 'hgh'.")
        self.hgh_theta_init_scale = float(hgh_theta_init_scale)
        self.hgh_v_init_scale = float(hgh_v_init_scale)

        # Majorana/JW kappa core: h - 1/2 sum_r g_prrq + sum_r g_pqrr
        self.g_pqrr_np = np.einsum("pqrr->pq", self.eri_np)
        self.kappa_core_np = self.h_core_np + self.g_pqrr_np
        self.g_pqrr = self._array(self.g_pqrr_np)
        self.kappa_core = self._array(self.kappa_core_np)

        # Scalar BLISS shifts.  These are trainable if requested, but the
        # traceless Frobenius proxy is mostly insensitive to pure identity shifts.
        self.alpha1 = self._array(np.array(0.0, dtype=np.float64))
        self.alpha2 = self._array(np.array(0.0, dtype=np.float64))

        # Rotation state for non-expm modes.
        U_full_np = self._to_numpy(super().X_to_U(self.params))
        train_U_np = U_full_np[1:] if self.fix_U0 else U_full_np
        self.U_base_train = self._array(train_U_np)  # hgh base rotations
        self.U_free = self._array(train_U_np if free_u_from_current else np.eye(self.norb)[None].repeat(self.ntrain_u, axis=0))
        self._init_hgh_params(seed=self.config.seed)
        self._eval(self.kappa_core, self.g_pqrr, self.alpha1, self.alpha2, self.U_base_train, self.U_free)

    # ------------------------------------------------------------------
    # Robust scalar extraction.  v4 inherited x.item() can fail on some MLX paths.
    # ------------------------------------------------------------------

    def _item_float(self, x) -> float:
        if isinstance(x, (float, int, np.floating, np.integer)):
            return float(x)
        if self.backend == "mlx":
            self._eval(x)
            try:
                return float(x.item())
            except Exception:
                arr = np.asarray(x)
                if arr.size != 1:
                    raise TypeError(f"Expected scalar, got shape {arr.shape}")
                return float(arr.reshape(-1)[0])
        return float(x.detach().cpu().item())

    # ------------------------------------------------------------------
    # Basic backend helpers
    # ------------------------------------------------------------------

    def _backend_concat(self, xs, axis=0):
        return mx.concatenate(xs, axis=axis) if self.backend == "mlx" else torch.cat(xs, dim=axis)

    def _backend_cos(self, x):
        return mx.cos(x) if self.backend == "mlx" else torch.cos(x)

    def _backend_sin(self, x):
        return mx.sin(x) if self.backend == "mlx" else torch.sin(x)

    def _smooth_abs(self, x, eps: float = 1e-8):
        return self._sqrt(x * x + float(eps))

    def _scalar_from_params(self, params: Dict[str, Any], key: str, default):
        return params[key] if key in params else default

    # ------------------------------------------------------------------
    # Rotation parametrizations
    # ------------------------------------------------------------------

    def _init_hgh_params(self, seed: Optional[int] = None) -> None:
        rng = np.random.default_rng(seed)
        v = float(self.hgh_v_init_scale) * rng.normal(size=(self.ntrain_u, self.norb))
        theta_even = float(self.hgh_theta_init_scale) * rng.normal(size=(self.ntrain_u, self.norb // 2))
        theta_odd = float(self.hgh_theta_init_scale) * rng.normal(size=(self.ntrain_u, (self.norb - 1) // 2))
        self.hgh_v1 = self._array(v)
        self.hgh_v2 = self._array(v.copy())
        self.hgh_theta_even = self._array(theta_even)
        self.hgh_theta_odd = self._array(theta_odd)

    def _normalize_v(self, v, eps: float = 1e-12):
        norm2 = self._sum(v * v, axis=1 if self.backend == "torch" else [1])
        norm = self._sqrt(norm2 + float(eps))
        return v / norm[:, None]

    def _apply_householder_right(self, U, v):
        """Apply U <- U H(v) batchwise. U:(t,N,N), v:(t,N)."""
        vh = self._normalize_v(v)
        Uv = self._einsum("tij,tj->ti", U, vh)
        return U - 2.0 * Uv[:, :, None] * vh[:, None, :]

    def _replace_columns(self, U, i_idx: List[int], new_i, j_idx: List[int], new_j):
        """Functional column replacement for Torch/MLX autograd portability."""
        pieces = []
        imap = {int(c): k for k, c in enumerate(i_idx)}
        jmap = {int(c): k for k, c in enumerate(j_idx)}
        for col in range(self.norb):
            if col in imap:
                pieces.append(new_i[:, :, imap[col]:imap[col] + 1])
            elif col in jmap:
                pieces.append(new_j[:, :, jmap[col]:jmap[col] + 1])
            else:
                pieces.append(U[:, :, col:col + 1])
        return self._backend_concat(pieces, axis=2)

    def _apply_givens_layer(self, U, theta, parity: str):
        if parity == "even":
            i_idx = list(range(0, self.norb - 1, 2))
            j_idx = list(range(1, self.norb, 2))
        else:
            i_idx = list(range(1, self.norb - 1, 2))
            j_idx = list(range(2, self.norb, 2))
        if len(i_idx) == 0:
            return U
        th = theta[:, : len(i_idx)]
        c = self._backend_cos(th)
        s = self._backend_sin(th)
        Ui = U[:, :, i_idx]
        Uj = U[:, :, j_idx]
        new_i = c[:, None, :] * Ui - s[:, None, :] * Uj
        new_j = s[:, None, :] * Ui + c[:, None, :] * Uj
        return self._replace_columns(U, i_idx, new_i, j_idx, new_j)

    def _apply_hgh(self, U_base, v1, v2, theta_even, theta_odd):
        # Sandwich: H(v1) G_even G_odd H(v2), applied on the right.
        U = self._apply_householder_right(U_base, v1)
        U = self._apply_givens_layer(U, theta_even, "even")
        U = self._apply_givens_layer(U, theta_odd, "odd")
        U = self._apply_householder_right(U, v2)
        return U

    def _build_U_from_params(self, params: Optional[Dict[str, Any]] = None):
        if params is None:
            params = self.params
        mode = self.rotation_param
        if mode == "expm":
            return super().X_to_U(params)
        if mode == "free":
            Utrain = params.get("U", self.U_free)
        elif mode == "hgh":
            Ubase = params.get("U_base", self.U_base_train)
            Utrain = self._apply_hgh(
                Ubase,
                params.get("v1", self.hgh_v1),
                params.get("v2", self.hgh_v2),
                params.get("theta_even", self.hgh_theta_even),
                params.get("theta_odd", self.hgh_theta_odd),
            )
        else:
            raise ValueError(f"Unknown rotation_param={mode}")
        if self.fix_U0:
            return self._backend_concat([self.U0[None, :, :], Utrain], axis=0)
        return Utrain

    def X_to_U(self, params: Optional[Dict[str, Any]] = None):
        return self._build_U_from_params(params)

    def orthogonality_penalty(self, params: Optional[Dict[str, Any]] = None):
        if params is None:
            params = self.params
        if self.rotation_param != "free":
            # exact orthogonal modes: return zero-like scalar
            return self._sum(self.params["Z"] * 0.0)
        U = params.get("U", self.U_free)
        I = self._eye(self.norb)
        UtU = self._einsum("tpi,tpj->tij", U, U)
        D = UtU - I[None, :, :]
        return self._sum(D * D)

    def project_free_U(self) -> Dict[str, float]:
        """Project free trainable U factors to SO(N) in-place."""
        if self.rotation_param != "free":
            return {"orth_error_before": 0.0, "orth_error_after": 0.0}
        U = self._to_numpy(self.U_free)
        errs_before = []
        errs_after = []
        Qs = []
        I = np.eye(self.norb)
        for t in range(U.shape[0]):
            errs_before.append(float(np.linalg.norm(U[t].T @ U[t] - I)))
            Q = _np_project_so(U[t])
            errs_after.append(float(np.linalg.norm(Q.T @ Q - I)))
            Qs.append(Q)
        self.U_free = self._array(np.stack(Qs, axis=0))
        self._eval(self.U_free)
        return {"orth_error_before": float(max(errs_before)), "orth_error_after": float(max(errs_after))}

    def absorb_hgh_correction(self, reset: bool = True) -> None:
        """Absorb current HGH correction into U_base_train, then optionally reset C≈I."""
        if self.rotation_param != "hgh":
            return
        params = {
            "U_base": self.U_base_train,
            "v1": self.hgh_v1,
            "v2": self.hgh_v2,
            "theta_even": self.hgh_theta_even,
            "theta_odd": self.hgh_theta_odd,
        }
        Utrain = self._build_U_from_params(params)
        if self.fix_U0:
            Utrain = Utrain[1:]
        self.U_base_train = self._array(self._to_numpy(Utrain))
        if reset:
            self._init_hgh_params(seed=None)
        self._eval(self.U_base_train, self.hgh_v1, self.hgh_v2, self.hgh_theta_even, self.hgh_theta_odd)

    # ------------------------------------------------------------------
    # BLISS algebra and corrected kappa
    # ------------------------------------------------------------------

    def beta_from_params(self, params: Optional[Dict[str, Any]] = None):
        B = params["B"] if params is not None and "B" in params else self.B
        return 0.5 * (B + self._swapaxes(B, 0, 1))

    def alpha1_from_params(self, params: Optional[Dict[str, Any]] = None):
        return self._scalar_from_params(params or {}, "alpha1", self.alpha1)

    def alpha2_from_params(self, params: Optional[Dict[str, Any]] = None):
        return self._scalar_from_params(params or {}, "alpha2", self.alpha2)

    def bliss_shift_tensor(self, beta, alpha2=None):
        I = self._eye(self.norb)
        if alpha2 is None:
            alpha2 = self.alpha2
        return (
            0.5 * (beta[:, :, None, None] * I[None, None, :, :] + I[:, :, None, None] * beta[None, None, :, :])
            + alpha2 * I[:, :, None, None] * I[None, None, :, :]
        )

    def kappa_matrix(self, beta=None, alpha1=None, alpha2=None, params: Optional[Dict[str, Any]] = None):
        if beta is None:
            beta = self.beta_from_params(params)
        if alpha1 is None:
            alpha1 = self.alpha1_from_params(params)
        if alpha2 is None:
            alpha2 = self.alpha2_from_params(params)
        I = self._eye(self.norb)
        trB = self._sum(beta * I)
        # kappa = h_core + g_pqrr + (eta-N)/2 beta - (alpha1 + 0.5 TrB + N alpha2) I
        return (
            self.kappa_core
            + 0.5 * (float(self.eta_electrons) - float(self.norb)) * beta
            - (alpha1 + 0.5 * trB + float(self.norb) * alpha2) * I
        )

    def kappa_traceless_fro_penalty(self, beta=None, alpha1=None, alpha2=None, params: Optional[Dict[str, Any]] = None):
        K = self.kappa_matrix(beta=beta, alpha1=alpha1, alpha2=alpha2, params=params)
        I = self._eye(self.norb)
        tr = self._sum(K * I) / float(self.norb)
        K0 = K - tr * I
        return self._sum(K0 * K0)

    def cdf_z_norm(self, Z, smooth: bool = False, eps: float = 1e-8):
        """CDF two-body norm proxy: 1/2 |Z|_1 - 1/4 diagonal |Z|_1."""
        A = self._smooth_abs(Z, eps) if smooth else self._abs(Z)
        diag = self._einsum("tii->t", A)
        return 0.5 * self._sum(A) - 0.25 * self._sum(diag)

    # ------------------------------------------------------------------
    # Losses
    # ------------------------------------------------------------------

    def loss_components(
        self,
        params: Optional[Dict[str, Any]] = None,
        lambda_z1: float = 1e-1,
        lambda_dist: float = 1e-1,
        lambda_z2: float = 0.0,
        lambda_norm: float = 0.0,
        contraction_mode: Optional[str] = None,
        batch_t: Optional[int] = None,
        symmetrize_z: Optional[bool] = None,
        use_bliss_beta: bool = False,
        lambda_kappa_fro: float = 0.0,
        lambda_beta_l2: float = 0.0,
        lambda_orth: float = 0.0,
        lambda_correction: float = 0.0,
        use_cdf_z_norm: bool = True,
        smooth_l1_eps: float = 1e-8,
        objective_mode: str = "full",
    ) -> Dict[str, Any]:
        if params is None:
            params = self._current_working_params()
        if symmetrize_z is None:
            symmetrize_z = self.symmetrize_z_default
        objective_mode = str(objective_mode).lower()
        beta = self.beta_from_params(params)
        alpha1 = self.alpha1_from_params(params)
        alpha2 = self.alpha2_from_params(params)
        use_bliss = bool(use_bliss_beta) or float(lambda_kappa_fro) != 0.0 or float(lambda_beta_l2) != 0.0

        # BLISS-shifted tensor residual.
        shift = self.bliss_shift_tensor(beta, alpha2) if use_bliss else self._zeros((self.norb, self.norb, self.norb, self.norb))
        target_resid = self.eri - shift

        if objective_mode == "bliss_only":
            diff = target_resid
            fit = self._sum(self._smooth_abs(diff, smooth_l1_eps))
            rel_fit = fit / (self._sum(self._abs(self.eri)) + 1e-30)
            zero = fit * 0.0
            Zeff = self.effective_Z(params["Z"], bool(symmetrize_z)) if "Z" in params else None
            z1 = zero if Zeff is None else self.cdf_z_norm(Zeff, smooth=True, eps=smooth_l1_eps)
            z2 = zero if Zeff is None else self.z_l2_from_Z(Zeff)
            ent = zero if Zeff is None else self.factor_entropy_from_Z(Zeff)
        else:
            eri_fit = self.reconstruct_eri(params, contraction_mode, batch_t, symmetrize_z)
            diff = target_resid - eri_fit
            fit = self._sum(diff * diff)
            rel_fit = self._sqrt(fit / (self.eri_norm2 + 1e-30))
            Zeff = self.effective_Z(params["Z"], bool(symmetrize_z))
            z1 = self.cdf_z_norm(Zeff, smooth=False) if use_cdf_z_norm else self.z_l1_from_Z(Zeff)
            z2 = self.z_l2_from_Z(Zeff)
            ent = self.factor_entropy_from_Z(Zeff)

        norm_mismatch = (z2 - self.eri_norm2) / (self.eri_norm2 + 1e-30) if objective_mode != "bliss_only" else fit * 0.0
        norm_penalty = norm_mismatch * norm_mismatch
        kappa_fro = self.kappa_traceless_fro_penalty(beta=beta, alpha1=alpha1, alpha2=alpha2)
        beta_l2 = self._sum(beta * beta)
        orth = self.orthogonality_penalty(params) if self.rotation_param == "free" else fit * 0.0
        corr = self.correction_penalty(params) if self.rotation_param == "hgh" else fit * 0.0
        total = (
            fit
            + float(lambda_z1) * z1
            + float(lambda_dist) * ent
            + float(lambda_z2) * z2
            + float(lambda_norm) * norm_penalty
            + float(lambda_kappa_fro) * kappa_fro
            + float(lambda_beta_l2) * beta_l2
            + float(lambda_orth) * orth
            + float(lambda_correction) * corr
        )
        return {
            "total": total,
            "fit": fit,
            "rel_fit": rel_fit,
            "z1": z1,
            "z2": z2,
            "dist": ent,
            "norm_penalty": norm_penalty,
            "kappa_fro": kappa_fro,
            "beta_l2": beta_l2,
            "orth": orth,
            "correction": corr,
        }

    def correction_penalty(self, params: Dict[str, Any]):
        if self.rotation_param != "hgh":
            return self._sum(self.params["Z"] * 0.0)
        v1 = params.get("v1", self.hgh_v1)
        v2 = params.get("v2", self.hgh_v2)
        te = params.get("theta_even", self.hgh_theta_even)
        to = params.get("theta_odd", self.hgh_theta_odd)
        return self._sum((self._normalize_v(v1) - self._normalize_v(v2)) ** 2) + self._sum(te * te) + self._sum(to * to)

    def _loss(self, params, **kwargs):
        return self.loss_components(params, **kwargs)["total"]

    # ------------------------------------------------------------------
    # General optimizer
    # ------------------------------------------------------------------

    def _current_working_params(self) -> Dict[str, Any]:
        p: Dict[str, Any] = {"Z": self.params["Z"], "B": self.B, "alpha1": self.alpha1, "alpha2": self.alpha2}
        if self.rotation_param == "expm":
            p["X"] = self.params["X"]
        elif self.rotation_param == "free":
            p["U"] = self.U_free
        elif self.rotation_param == "hgh":
            p.update({"U_base": self.U_base_train, "v1": self.hgh_v1, "v2": self.hgh_v2, "theta_even": self.hgh_theta_even, "theta_odd": self.hgh_theta_odd})
        return p

    def _trainable_keys(self, optimize_rotation: bool, optimize_Z: bool, optimize_bliss: bool, optimize_alpha: bool) -> Dict[str, bool]:
        keys = {"Z": bool(optimize_Z), "B": bool(optimize_bliss), "alpha1": bool(optimize_alpha), "alpha2": bool(optimize_alpha)}
        if self.rotation_param == "expm":
            keys["X"] = bool(optimize_rotation)
        elif self.rotation_param == "free":
            keys["U"] = bool(optimize_rotation)
        elif self.rotation_param == "hgh":
            keys.update({"v1": bool(optimize_rotation), "v2": bool(optimize_rotation), "theta_even": bool(optimize_rotation), "theta_odd": bool(optimize_rotation), "U_base": False})
        return keys

    def _freeze_grads_dynamic(self, grads: Dict[str, Any], trainable: Dict[str, bool]) -> Dict[str, Any]:
        out = {}
        for k, g in grads.items():
            if trainable.get(k, False):
                out[k] = g
            else:
                out[k] = mx.zeros_like(g) if self.backend == "mlx" else torch.zeros_like(g)
        return out

    def _update_from_working_params(self, params: Dict[str, Any]) -> None:
        self.params["Z"] = self._array(self._to_numpy(params["Z"]))
        if "X" in params:
            self.params["X"] = self._array(self._to_numpy(params["X"]))
        self.B = self._array(self._to_numpy(params["B"]))
        self.alpha1 = self._array(np.array(self._item_float(params["alpha1"]), dtype=np.float64))
        self.alpha2 = self._array(np.array(self._item_float(params["alpha2"]), dtype=np.float64))
        if "U" in params:
            self.U_free = self._array(self._to_numpy(params["U"]))
        if "v1" in params:
            self.hgh_v1 = self._array(self._to_numpy(params["v1"]))
            self.hgh_v2 = self._array(self._to_numpy(params["v2"]))
            self.hgh_theta_even = self._array(self._to_numpy(params["theta_even"]))
            self.hgh_theta_odd = self._array(self._to_numpy(params["theta_odd"]))
        self._eval(*[v for v in self._current_working_params().values() if v is not None])

    def _record_v5(self, step, params, loss_kwargs, t0, hist, label=""):
        comps = self.loss_components(params, **loss_kwargs)
        rec = {
            "step": int(step),
            "label": label,
            "loss": self._item_float(comps["total"]),
            "fit": self._item_float(comps["fit"]),
            "rel_fit": self._item_float(comps["rel_fit"]),
            "z_norm": self._item_float(comps["z1"]),
            "entropy": self._item_float(comps["dist"]),
            "kappa_fro": self._item_float(comps["kappa_fro"]),
            "orth": self._item_float(comps["orth"]),
            "correction": self._item_float(comps["correction"]),
            "elapsed": time.time() - t0,
        }
        hist.append(rec)
        self.history.append(rec)
        return rec

    def optimize(
        self,
        learning_rate: float = 1e-2,
        maxiter: int = 4000,
        optimizer: str = "adam",
        optimize_strategy: str = "simultaneous",
        optimize_rotation: bool = True,
        optimize_Z: bool = True,
        optimize_bliss: bool = True,
        optimize_alpha: bool = True,
        use_bliss_beta: bool = True,
        lambda_z1: float = 1e-1,
        lambda_dist: float = 1e-1,
        lambda_z2: float = 0.0,
        lambda_norm: float = 0.0,
        lambda_kappa_fro: float = 0.0,
        lambda_beta_l2: float = 0.0,
        lambda_orth: float = 0.0,
        lambda_correction: float = 0.0,
        use_cdf_z_norm: bool = True,
        smooth_l1_eps: float = 1e-8,
        contraction_mode: Optional[str] = None,
        batch_t: Optional[int] = None,
        symmetrize_z: Optional[bool] = None,
        print_every: int = 500,
        print_initial: bool = True,
        reset_history: bool = False,
        max_grad_norm: Optional[float] = None,
        materialize_each_step: bool = True,
        project_every: Optional[int] = None,
        project_at_end: bool = False,
        lbfgs_history_size: int = 20,
        lbfgs_max_iter: int = 20,
        lbfgs_line_search_fn: Optional[str] = None,
        return_history: bool = True,
    ) -> List[Dict[str, Any]]:
        if reset_history:
            self.history = []
        if contraction_mode is None:
            contraction_mode = self.contraction_mode
        if batch_t is None:
            batch_t = self.batch_t
        if symmetrize_z is None:
            symmetrize_z = self.symmetrize_z_default
        strategy = str(optimize_strategy).lower()
        if strategy == "cdf_only":
            optimize_bliss = False
            optimize_alpha = False
            use_bliss_beta = True
        elif strategy == "bliss_only":
            optimize_rotation = False
            optimize_Z = False
            optimize_bliss = True
            optimize_alpha = True
            use_bliss_beta = True
        elif strategy != "simultaneous":
            raise ValueError("optimize_strategy must be 'simultaneous', 'cdf_only', or 'bliss_only'. Use optimize_alternating() for alternating cycles.")

        optimizer = str(optimizer).lower()
        if optimizer not in ("adam", "lbfgs"):
            raise ValueError("optimizer must be 'adam' or 'lbfgs'.")
        if self.backend == "mlx" and optimizer != "adam":
            raise NotImplementedError("MLX supports optimizer='adam' only in v5.")

        loss_kwargs = dict(
            lambda_z1=lambda_z1,
            lambda_dist=lambda_dist,
            lambda_z2=lambda_z2,
            lambda_norm=lambda_norm,
            contraction_mode=contraction_mode,
            batch_t=batch_t,
            symmetrize_z=symmetrize_z,
            use_bliss_beta=use_bliss_beta,
            lambda_kappa_fro=lambda_kappa_fro,
            lambda_beta_l2=lambda_beta_l2,
            lambda_orth=lambda_orth,
            lambda_correction=lambda_correction,
            use_cdf_z_norm=use_cdf_z_norm,
            smooth_l1_eps=smooth_l1_eps,
            objective_mode="bliss_only" if strategy == "bliss_only" else "full",
        )
        trainable = self._trainable_keys(optimize_rotation, optimize_Z, optimize_bliss, optimize_alpha)
        params0 = self._current_working_params()
        hist: List[Dict[str, Any]] = []
        t0 = time.time()

        if self.backend == "mlx":
            params = {k: mx.array(v) for k, v in params0.items()}
            mx.eval(*params.values())
            opt = MLXAdam(learning_rate=float(learning_rate))

            def loss_fn_raw(p):
                return self._loss(p, **loss_kwargs)

            # MLX CPU compile can be fragile; compile only on GPU.
            loss_fn = mx.compile(loss_fn_raw) if self.device != "cpu" else loss_fn_raw
            grad_fn = mx.grad(loss_fn, argnums=0)
            if print_initial:
                rec = self._record_v5(0, params, loss_kwargs, t0, hist, "initial")
                print(f"Initial: loss={rec['loss']:.8e} fit={rec['fit']:.8e} z={rec['z_norm']:.8e} kappa={rec['kappa_fro']:.8e} orth={rec['orth']:.8e}")
            for step in range(int(maxiter)):
                grads = grad_fn(params)
                grads = self._freeze_grads_dynamic(grads, trainable)
                grads = self._clip_grads(grads, max_grad_norm)
                loss = loss_fn(params)
                params = opt.apply_gradients(grads, params)
                if materialize_each_step:
                    mx.eval(*params.values(), loss)
                else:
                    mx.eval(loss)
                if self.rotation_param == "free" and project_every and int(project_every) > 0 and (step + 1) % int(project_every) == 0:
                    self._update_from_working_params(params)
                    self.project_free_U()
                    params = self._current_working_params()
                if print_every and ((step + 1) % int(print_every) == 0 or step == int(maxiter) - 1):
                    rec = self._record_v5(step + 1, params, loss_kwargs, t0, hist, "train")
                    print(f"Step {step+1:6d}: loss={rec['loss']:.8e} fit={rec['fit']:.8e} z={rec['z_norm']:.8e} kappa={rec['kappa_fro']:.8e} orth={rec['orth']:.8e}")
            self._update_from_working_params(params)
        else:
            params = {}
            opt_params = []
            for k, v in params0.items():
                ten = v.detach().clone().requires_grad_(bool(trainable.get(k, False)))
                params[k] = ten
                if trainable.get(k, False):
                    opt_params.append(ten)
            if print_initial:
                with torch.no_grad():
                    rec = self._record_v5(0, params, loss_kwargs, t0, hist, "initial")
                    print(f"Initial: loss={rec['loss']:.8e} fit={rec['fit']:.8e} z={rec['z_norm']:.8e} kappa={rec['kappa_fro']:.8e} orth={rec['orth']:.8e}")

            if optimizer == "adam":
                opt = torch.optim.Adam(opt_params, lr=float(learning_rate)) if opt_params else None
                for step in range(int(maxiter)):
                    if opt is not None:
                        opt.zero_grad(set_to_none=True)
                    loss = self._loss(params, **loss_kwargs)
                    if opt_params:
                        loss.backward()
                        if max_grad_norm is not None and max_grad_norm > 0:
                            torch.nn.utils.clip_grad_norm_(opt_params, max_norm=float(max_grad_norm))
                        opt.step()
                    if materialize_each_step and self.device == "cuda":
                        torch.cuda.synchronize()
                    if self.rotation_param == "free" and project_every and int(project_every) > 0 and (step + 1) % int(project_every) == 0:
                        self._update_from_working_params(params)
                        self.project_free_U()
                        params = self._current_working_params()
                        opt_params = []
                        for k, v in params.items():
                            ten = v.detach().clone().requires_grad_(bool(trainable.get(k, False)))
                            params[k] = ten
                            if trainable.get(k, False):
                                opt_params.append(ten)
                        opt = torch.optim.Adam(opt_params, lr=float(learning_rate)) if opt_params else None
                    if print_every and ((step + 1) % int(print_every) == 0 or step == int(maxiter) - 1):
                        with torch.no_grad():
                            rec = self._record_v5(step + 1, params, loss_kwargs, t0, hist, "train")
                            print(f"Step {step+1:6d}: loss={rec['loss']:.8e} fit={rec['fit']:.8e} z={rec['z_norm']:.8e} kappa={rec['kappa_fro']:.8e} orth={rec['orth']:.8e}")
            else:
                opt = torch.optim.LBFGS(opt_params, lr=float(learning_rate), max_iter=int(lbfgs_max_iter), history_size=int(lbfgs_history_size), line_search_fn=lbfgs_line_search_fn) if opt_params else None
                for step in range(int(maxiter)):
                    if opt is not None:
                        def closure():
                            opt.zero_grad(set_to_none=True)
                            loss = self._loss(params, **loss_kwargs)
                            loss.backward()
                            if max_grad_norm is not None and max_grad_norm > 0:
                                torch.nn.utils.clip_grad_norm_(opt_params, max_norm=float(max_grad_norm))
                            return loss
                        opt.step(closure)
                    if print_every and ((step + 1) % int(print_every) == 0 or step == int(maxiter) - 1):
                        with torch.no_grad():
                            rec = self._record_v5(step + 1, params, loss_kwargs, t0, hist, "lbfgs")
                            print(f"Step {step+1:6d}: loss={rec['loss']:.8e} fit={rec['fit']:.8e} z={rec['z_norm']:.8e} kappa={rec['kappa_fro']:.8e} orth={rec['orth']:.8e}")
            self._update_from_working_params(params)

        if self.rotation_param == "free" and project_at_end:
            info = self.project_free_U()
            print("Projection:", info)
        final = self.loss_components(self._current_working_params(), **loss_kwargs)
        self.last_loss = self._item_float(final["total"])
        print("************************************")
        print("Final Loss:", self.last_loss)
        print("Final Fit: ", self._item_float(final["fit"]))
        print("Elapsed Time:", time.time() - t0, "s")
        return hist if return_history else []

    fit = optimize

    def optimize_bliss_only(self, **kwargs) -> List[Dict[str, Any]]:
        kwargs.setdefault("optimize_strategy", "bliss_only")
        kwargs.setdefault("use_bliss_beta", True)
        kwargs.setdefault("lambda_kappa_fro", 1e-3)
        kwargs.setdefault("lambda_beta_l2", 0.0)
        kwargs.setdefault("lambda_z1", 0.0)
        kwargs.setdefault("lambda_dist", 0.0)
        return self.optimize(**kwargs)

    def optimize_alternating(
        self,
        cycles: int = 5,
        cdf_steps: int = 1000,
        bliss_steps: int = 250,
        absorb_hgh_each_cycle: bool = True,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Alternate CDF-only and BLISS-only subproblems."""
        all_hist: List[Dict[str, Any]] = []
        for c in range(int(cycles)):
            print(f"=== Alternating cycle {c+1}/{cycles}: CDF block ===")
            h1 = self.optimize(maxiter=int(cdf_steps), optimize_strategy="cdf_only", reset_history=False, **kwargs)
            all_hist.extend(h1)
            print(f"=== Alternating cycle {c+1}/{cycles}: BLISS block ===")
            h2 = self.optimize(maxiter=int(bliss_steps), optimize_strategy="bliss_only", reset_history=False, **kwargs)
            all_hist.extend(h2)
            if self.rotation_param == "hgh" and absorb_hgh_each_cycle:
                self.absorb_hgh_correction(reset=True)
        return all_hist

    # ------------------------------------------------------------------
    # Diagnostics and postprocessing
    # ------------------------------------------------------------------

    def get_rotation_numpy(self) -> np.ndarray:
        return self._to_numpy(self._build_U_from_params(self._current_working_params()))

    def kappa_numpy(self, beta: Optional[np.ndarray] = None, alpha1: Optional[float] = None, alpha2: Optional[float] = None) -> np.ndarray:
        if beta is None:
            beta = self.get_beta_numpy()
        if alpha1 is None:
            alpha1 = self._item_float(self.alpha1)
        if alpha2 is None:
            alpha2 = self._item_float(self.alpha2)
        N = self.norb
        trB = float(np.trace(beta))
        return self.kappa_core_np + 0.5 * (self.eta_electrons - N) * beta - (float(alpha1) + 0.5 * trB + N * float(alpha2)) * np.eye(N)

    def optimal_alpha1(self, beta: Optional[np.ndarray] = None, alpha2: Optional[float] = None) -> Dict[str, Any]:
        K0 = self.kappa_numpy(beta=beta, alpha1=0.0, alpha2=alpha2)
        evals = np.linalg.eigvalsh(0.5 * (K0 + K0.T))
        a1 = float(np.median(evals))
        return {"alpha1": a1, "eigvals": evals, "eig_abs_sum": float(np.sum(np.abs(evals - a1))), "eig_abs_sum_no_alpha1": float(np.sum(np.abs(evals)))}

    def alpha2_median_shifts(self, symmetrize_z: Optional[bool] = None) -> Dict[str, Any]:
        if symmetrize_z is None:
            symmetrize_z = self.symmetrize_z_default
        Z = self.get_params_numpy()["Z"]
        Z_eff = 0.5 * (Z + np.swapaxes(Z, 1, 2)) if symmetrize_z else Z.copy()
        a_t = np.median(Z_eff.reshape(Z_eff.shape[0], -1), axis=1)
        Z_shifted = Z_eff - a_t[:, None, None]
        def cdf_norm_np(A):
            return 0.5 * np.sum(np.abs(A)) - 0.25 * np.sum(np.abs(np.diagonal(A, axis1=1, axis2=2)))
        return {"alpha2_t": a_t, "alpha2_total": float(np.sum(a_t)), "Z_shifted": Z_shifted, "z_cdf_norm_before": float(cdf_norm_np(Z_eff)), "z_cdf_norm_after": float(cdf_norm_np(Z_shifted))}

    def final_norm_report(self, apply_alpha2_shift: bool = False) -> Dict[str, Any]:
        beta = self.get_beta_numpy()
        alpha2_val = self._item_float(self.alpha2)
        alpha1_info = self.optimal_alpha1(beta=beta, alpha2=alpha2_val)
        a2_info = self.alpha2_median_shifts()
        Z = a2_info["Z_shifted"] if apply_alpha2_shift else self.get_params_numpy()["Z"]
        z_norm = 0.5 * np.sum(np.abs(Z)) - 0.25 * np.sum(np.abs(np.diagonal(Z, axis1=1, axis2=2)))
        return {
            "lambda_one_body_kappa": alpha1_info["eig_abs_sum"],
            "lambda_two_body_cdf": float(z_norm),
            "lambda_total_proxy": float(alpha1_info["eig_abs_sum"] + z_norm),
            "alpha1_median": alpha1_info["alpha1"],
            "alpha2_global": alpha2_val,
            "alpha2_t_median": a2_info["alpha2_t"],
            "alpha2_total_median": a2_info["alpha2_total"],
            "rotation_param": self.rotation_param,
            "backend": self.backend,
            "device": self.device,
        }

    def get_params_numpy(self) -> Dict[str, np.ndarray]:
        out = super().get_params_numpy()
        out["B"] = self._to_numpy(self.B)
        out["beta"] = self.get_beta_numpy()
        out["alpha1"] = np.array(self._item_float(self.alpha1))
        out["alpha2"] = np.array(self._item_float(self.alpha2))
        out["U_effective"] = self.get_rotation_numpy()
        if self.rotation_param == "free":
            out["U_free_train"] = self._to_numpy(self.U_free)
        if self.rotation_param == "hgh":
            out["U_base_train"] = self._to_numpy(self.U_base_train)
            out["hgh_v1"] = self._to_numpy(self.hgh_v1)
            out["hgh_v2"] = self._to_numpy(self.hgh_v2)
            out["hgh_theta_even"] = self._to_numpy(self.hgh_theta_even)
            out["hgh_theta_odd"] = self._to_numpy(self.hgh_theta_odd)
        return out

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        p = self.get_params_numpy()
        payload = {k: v for k, v in p.items() if isinstance(v, np.ndarray)}
        payload.update({
            "config_json": np.array(json.dumps(self.summary())),
            "history_json": np.array(json.dumps(self.history)),
            "last_loss": np.array(np.nan if self.last_loss is None else self.last_loss),
        })
        np.savez(path, **payload)
        return path

    def summary(self) -> Dict[str, Any]:
        s = super().summary()
        s.update({"version": "v5", "rotation_param": self.rotation_param, "eta_electrons": self.eta_electrons, "alpha1": self._item_float(self.alpha1), "alpha2": self._item_float(self.alpha2)})
        return s


__all__ = [
    "CDFERIOptimizer",
    "CDFOptimizerConfig",
    "make_SO_matrix",
    "orthogonal_to_skew_log",
    "lchol_guess_params",
]
