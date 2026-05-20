# CDF ERI Optimizer v4

`cdf_eri_optimizer_multibackend_v4.py` fits a compact compressed double-factorization (CDF) form of the electronic-repulsion tensor,

```text
g[p,q,r,s] ≈ Σ_t U[t,p,k] U[t,q,k] Z[t,k,l] U[t,r,l] U[t,s,l]
```

with MLX or Torch backends. v4 adds Torch L-BFGS, an optional BLISS/particle-number symmetry shift, and post-processing formulas for the optimal scalar shifts `alpha1` and `alpha2_t`.

Keep v3 and v4 in the same directory because v4 subclasses the v3 implementation:

```text
cdf_eri_optimizer_multibackend_v3.py
cdf_eri_optimizer_multibackend_v4.py
```

## Why use CDF?

A molecular Hamiltonian has a dense two-electron tensor `g[p,q,r,s]` with `O(N^4)` entries. CDF rewrites this tensor as a sum of rotated density-density factors. Instead of storing and manipulating the full ERI as unrelated four-index data, CDF exposes structure:

```text
full ERI tensor                 CDF ansatz
 g[p,q,r,s]      ≈      Σ_t  U_t ⊗ U_t   ·   Z_t   ·   U_t ⊗ U_t
 O(N^4) data              rotations + dense factor matrices
```

This is useful because many quantum simulation costs depend not only on reconstruction error, but also on norms of the factorized representation, especially `Σ_tkl |Z[t,k,l]|` and the induced one-body term. A good CDF fit can therefore trade a slightly more flexible factorization for lower simulation/resource-estimation cost.

Potential applications include:

- **Fault-tolerant quantum resource estimation**, where CDF/DF-like decompositions control block-encoding or product-formula costs.
- **Hamiltonian compression**, where a low-`Z`-norm representation is more valuable than a purely low-rank least-squares fit.
- **NOCI/SQD workflows**, where compact rotated density-density factors can accelerate approximate Hamiltonian builds, screening, or benchmarking.
- **Comparing factorization choices**, such as Cholesky/DF/CDF/THC-like fits under norm-aware objectives.

## Why BLISS is useful

The block-invariant symmetry shift (BLISS) uses number-conserving symmetry terms that vanish on the target particle-number sector but can reduce the norm of the representation. In v4, the implemented variational part is the `beta` shift:

```text
g_BI[p,q,r,s] = g[p,q,r,s]
                 - 0.5 * ( beta[p,q] delta[r,s] + delta[p,q] beta[r,s] )
```

The same `beta` changes the effective one-body block:

```text
h_BI = h_core + 0.5 * eta * beta
h_core[p,q] = h1[p,q] - 0.5 * Σ_r g[p,r,r,q]
eta = number of electrons
```

Intuition:

```text
                 choose beta
                     │
                     ▼
       move weight between two-body and one-body blocks
          │                                  │
          ▼                                  ▼
   lower ||Z||_1 target              avoid growing h_BI too much
```

The optimizer can use `beta` to make the two-body CDF target cheaper. To prevent it from simply hiding cost in the one-body term, v4 includes a backend-safe traceless Frobenius control:

```text
|| h_BI - Tr(h_BI)/N * I ||_F^2
```

This is not the exact eigenvalue 1-norm of the one-body term, but it is differentiable and works on MLX, Torch CUDA, Torch MPS, and CPU. The exact eigenvalue diagnostic can be computed after optimization.

## Installation/import

```python
from cdf_eri_optimizer_multibackend_v4 import CDFERIOptimizer
```

The `system` input is the same as v3. A dictionary must contain:

```python
system = {
    "h1": h1,          # shape (norb, norb)
    "eri": eri,        # shape (norb, norb, norb, norb)
    "e_nuc": e_nuc,
    "nelec": nelec,    # e.g. (nalpha, nbeta)
}
```

## Recommended workflows

### 1. Torch CUDA: Adam warmup + L-BFGS polish

This is the preferred path when a CUDA GPU is available.

```python
opt = CDFERIOptimizer(
    system,
    ndf=64,
    backend="torch",
    device="cuda",
    dtype="float64",
    init="lchol",
    Lchol=Lchol,
)

hist = opt.optimize(
    optimizer="adam_then_lbfgs",
    adam_warmup_iter=2000,
    maxiter=2050,            # 2000 Adam steps + 50 L-BFGS outer calls
    learning_rate=1e-2,
    lbfgs_max_iter=10,
    lbfgs_history_size=20,
    lambda_z1=1e-4,
    lambda_dist=1e-3,        # entropy regularizer from v3
    print_every=100,
)
```

Pure L-BFGS is also available for Torch:

```python
hist = opt.optimize(
    optimizer="lbfgs",
    maxiter=50,
    learning_rate=0.5,
    lbfgs_max_iter=20,
    lbfgs_history_size=20,
    lbfgs_line_search_fn="strong_wolfe",
    lambda_z1=1e-4,
    print_every=1,
)
```

### 2. Torch or MLX with BLISS beta

```python
hist = opt.optimize(
    optimizer="adam",
    use_bliss_beta=True,
    optimize_beta=True,
    lambda_z1=1e-4,
    lambda_dist=1e-3,        # keeps the entropy/distribution regularizer
    lambda_hbi_fro=1e-3,     # controls one-body growth from beta
    lambda_beta_l2=0.0,      # optional direct beta regularizer
    maxiter=3000,
    learning_rate=1e-2,
    print_every=100,
)
```

MLX example:

```python
opt = CDFERIOptimizer(
    system,
    ndf=64,
    backend="mlx",
    device="gpu",
    dtype="float32",
)

hist = opt.optimize(
    optimizer="adam",        # MLX L-BFGS is not implemented
    use_bliss_beta=True,
    lambda_hbi_fro=1e-3,
    lambda_z1=1e-4,
    lambda_dist=1e-3,
    maxiter=2000,
    print_every=100,
)
```

### 3. Post-process scalar BLISS shifts

After optimization, compute the analytic median shifts:

```python
meta = opt.final_bliss_metadata()

alpha1 = meta["alpha1"]
alpha2_t = meta["alpha2_t"]
alpha2_total = meta["alpha2_total"]
```

Definitions:

```text
alpha1 = median eigenvalue of h_core + 0.5 * eta * beta
alpha2_t = median_{k,l} Z[t,k,l]
alpha2_total = Σ_t alpha2_t
```

`alpha1` minimizes `Σ_i |eig_i(A) - alpha1|`. Each `alpha2_t` minimizes `Σ_kl |Z[t,k,l] - alpha2_t|`.

To apply the `alpha2_t` shifts to `Z` in-place:

```python
shift_info = opt.apply_alpha2_median_shifts()
print(shift_info["alpha2_total"])
print(shift_info["z1_before"], shift_info["z1_after"])
```

This performs:

```text
Z[t] <- Z[t] - alpha2_t * ones(N, N)
```

## Diagnostics

Compute the true one-body eigenvalue 1-norm diagnostic outside the training loop:

```python
hdiag = opt.hbi_eig_abs_numpy()
print(hdiag["alpha1"])
print(hdiag["eig_abs_sum"])
print(hdiag["eig_abs_sum_no_alpha1"])
```

This is useful for checking whether the BLISS beta shift reduced the two-body cost without making the one-body block too expensive.

Common metrics to compare before/after a run:

```python
meta = opt.final_bliss_metadata()
print("alpha1:", meta["alpha1"])
print("alpha2_total:", meta["alpha2_total"])
print("sum |alpha2_t|:", abs(meta["alpha2_t"]).sum())
print("hBI eig abs:", opt.hbi_eig_abs_numpy()["eig_abs_sum"])
```

## Saving and loading

```python
opt.save("cdf_v4_fit.npz")

opt2 = CDFERIOptimizer.load(
    "cdf_v4_fit.npz",
    system=system,
    backend="torch",
    device="cuda",
)
```

Saved data include `X`, `Z`, `B`, `U0`, history, and `eta_electrons`.

## Practical notes

- Use `backend="torch", device="cuda", dtype="float64"` for serious convergence when available.
- Use `dtype="float32"` on Torch MPS and MLX GPU.
- L-BFGS stores parameter history and can use much more memory than Adam; use `adam_then_lbfgs` for large fits.
- `lambda_dist` is the existing entropy/distribution regularizer and remains useful with BLISS.
- `lambda_hbi_fro` is a stable proxy, not the exact one-body eigenvalue 1-norm.
- `alpha2_t` post-processing changes `Z`; keep the returned metadata if downstream code needs the accumulated scalar shift.
