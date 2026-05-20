# CDF ERI Optimizer v4

`cdf_eri_optimizer_multibackend_v4.py` extends `cdf_eri_optimizer_multibackend_v3.py` with three experimental features:

1. **Torch L-BFGS** optimizer support.
2. **BLISS beta-shift objective** for both Torch and MLX.
3. **Median post-processing** for final optimal `alpha1` and per-factor `alpha2_t` shifts.

The v4 file imports v3 from the same directory, so keep both files together:

```text
cdf_eri_optimizer_multibackend_v3.py
cdf_eri_optimizer_multibackend_v4.py
```

## Basic import

```python
from cdf_eri_optimizer_multibackend_v4 import CDFERIOptimizer
```

The `system` input is the same as v3. A dict must contain:

```python
system = {
    "h1": h1,          # shape (norb, norb)
    "eri": eri,        # shape (norb, norb, norb, norb)
    "e_nuc": e_nuc,
    "nelec": nelec,    # e.g. (nalpha, nbeta)
}
```

## 1. Torch Adam usage

```python
opt = CDFERIOptimizer(
    system,
    ndf=32,
    backend="torch",
    device="cuda",      # or "cpu", "mps"
    dtype="float64",    # cuda/cpu; use float32 on mps
    init="lchol",
    Lchol=Lchol,
)

hist = opt.optimize(
    optimizer="adam",
    maxiter=2000,
    learning_rate=1e-2,
    lambda_z1=1e-4,
    lambda_dist=1e-3,
    print_every=100,
)
```

## 2. Torch L-BFGS usage

L-BFGS is only implemented for `backend="torch"`.

```python
hist = opt.optimize(
    optimizer="lbfgs",
    maxiter=50,              # outer L-BFGS calls
    learning_rate=0.5,
    lbfgs_max_iter=20,       # internal line-search iterations per outer call
    lbfgs_history_size=20,
    lbfgs_line_search_fn="strong_wolfe",
    lambda_z1=1e-4,
    lambda_dist=1e-3,
    print_every=1,
)
```

For most runs, a safer workflow is Adam warmup followed by L-BFGS polishing:

```python
hist = opt.optimize(
    optimizer="adam_then_lbfgs",
    adam_warmup_iter=2000,
    maxiter=2050,            # 2000 Adam steps + 50 L-BFGS outer steps
    learning_rate=1e-2,      # used for both Adam and L-BFGS lr
    lbfgs_max_iter=10,
    lbfgs_history_size=20,
    lambda_z1=1e-4,
    lambda_dist=1e-3,
    print_every=100,
)
```

## 3. BLISS beta-shift objective

v4 can fit the shifted ERI target

```text
g_BI[p,q,r,s] = g[p,q,r,s]
                 - 0.5 * (beta[p,q] delta[r,s] + delta[p,q] beta[r,s])
```

where `beta = 0.5 * (B + B.T)` is optimized as an additional variational parameter.

Enable this with:

```python
hist = opt.optimize(
    use_bliss_beta=True,
    optimize_beta=True,      # default is True when BLISS weights are active
    lambda_z1=1e-4,
    lambda_dist=1e-3,        # keep entropy regularizer if desired
    lambda_hbi_fro=1e-3,     # traceless Frobenius proxy for one-body cost
    lambda_beta_l2=0.0,      # optional direct beta regularizer
    maxiter=3000,
    learning_rate=1e-2,
    print_every=100,
)
```

The in-loop one-body control term is

```text
A = h_core + 0.5 * eta * beta
h_core[p,q] = h1[p,q] - 0.5 * sum_r eri[p,r,r,q]
penalty = || A - Tr(A)/N * I ||_F^2
```

This avoids eigensolver/autograd issues and is compatible with MLX, Torch CUDA, Torch MPS, and CPU.

### MLX BLISS example

```python
opt = CDFERIOptimizer(
    system,
    ndf=32,
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

## 4. Final median alpha shifts

After optimization, compute the final median solutions without mutating parameters:

```python
meta = opt.final_bliss_metadata()

alpha1 = meta["alpha1"]
alpha2_t = meta["alpha2_t"]
alpha2_total = meta["alpha2_total"]
```

`alpha1` is computed as the median of eigenvalues of

```text
h_core + 0.5 * eta * beta
```

because it minimizes

```text
sum_i | eig_i(A) - alpha1 |.
```

Each `alpha2_t` is computed as

```text
median_{k,l} Z[t,k,l]
```

because it minimizes

```text
sum_{k,l} | Z[t,k,l] - alpha2_t |.
```

To apply the per-factor `alpha2_t` shifts to `Z` in-place:

```python
shift_info = opt.apply_alpha2_median_shifts()
print(shift_info["alpha2_total"])
print(shift_info["z1_before"], shift_info["z1_after"])
```

This replaces each factor by

```text
Z[t] <- Z[t] - alpha2_t * ones(N, N)
```

and returns the accumulated `alpha2_total = sum_t alpha2_t`.

## 5. Diagnostics

The true eigenvalue/nuclear-norm-like one-body diagnostic is computed outside the training loop using NumPy:

```python
hdiag = opt.hbi_eig_abs_numpy()
print(hdiag["alpha1"])
print(hdiag["eig_abs_sum"])
print(hdiag["eig_abs_sum_no_alpha1"])
```

This is intended for logging/post-analysis, not necessarily as an in-loop objective.

## 6. Saving and loading

```python
opt.save("cdf_v4_fit.npz")

opt2 = CDFERIOptimizer.load(
    "cdf_v4_fit.npz",
    system=system,
    backend="torch",
    device="cuda",
)
```

The saved file includes `X`, `Z`, `B`, `U0`, history, and `eta_electrons`.

## Notes and caveats

- L-BFGS can use significantly more memory than Adam because it stores history vectors.
- For large `ndf`, prefer `optimizer="adam_then_lbfgs"` over pure L-BFGS from random initialization.
- On Torch MPS, use `dtype="float32"`; CUDA float64 is preferred for serious convergence.
- `lambda_hbi_fro` is a stable proxy for controlling one-body growth, not the exact eigenvalue 1-norm.
- The entropy regularizer from v3 is unchanged and remains controlled by `lambda_dist`.
