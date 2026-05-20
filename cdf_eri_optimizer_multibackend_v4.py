"""
cdf_eri_optimizer_multibackend_v4.py

Refactor/extension of cdf_eri_optimizer_multibackend_v3.py.

Adds:
  1. Torch optimizer='lbfgs' and optimizer='adam_then_lbfgs'.
  2. Optional BLISS beta-shift objective on both MLX and Torch:

        g_target^BI[p,q,r,s]
          = g[p,q,r,s] - 0.5 * (beta[p,q] delta[r,s]
                                + delta[p,q] beta[r,s])

     with beta = 0.5 * (B + B.T), and optional traceless Frobenius control
     on the induced one-body block

        h_core = h1 - 0.5 * sum_r g[p,r,r,q]
        A      = h_core + 0.5 * eta * beta
        penalty = || A - Tr(A)/N I ||_F^2.

  3. Post-processing helpers for optimal alpha_1 and per-factor alpha_2^t
     median shifts:

        alpha_1 = median(eigvals(h_core + eta/2 beta))
        alpha_2^t = median_{kl} Z[t,k,l]

     The alpha_2^t shifts subtract alpha_2^t * ones(N,N) from each Z[t].

This file depends on v3 being importable from the same directory.  It keeps the
v3 API available while adding v4 options.  If no BLISS options are enabled and
optimizer='adam', behavior is intended to match v3.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import time

import numpy as np

from cdf_eri_optimizer_multibackend_v3 import (  # noqa: F401
    CDFERIOptimizer as _V3CDFERIOptimizer,
    CDFOptimizerConfig,
    make_SO_matrix,
    orthogonal_to_skew_log,
    lchol_guess_params,
    _HAS_MLX,
    _HAS_TORCH,
    mx,
    MLXAdam,
    torch,
)


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
