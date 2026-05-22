# CDF ERI Optimizer v5: CDF + BLISS + alternative rotation updates

`cdf_eri_optimizer_multibackend_v5.py` is an experimental optimizer for compressed double factorization (CDF) of molecular ERIs with BLISS-style symmetry shifts. It is designed to benchmark several optimization strategies without changing the older v3/v4 code paths.

The model fits

$$
g^{BI}_{pqrs}\approx \sum_t U^t_{pk}U^t_{qk}Z^t_{kl}U^t_{rl}U^t_{sl},
$$

where the BLISS-shifted target is

$$
g^{BI}_{pqrs}=g_{pqrs}-\frac12(B_{pq}\delta_{rs}+\delta_{pq}B_{rs})-\alpha_2\delta_{pq}\delta_{rs}.
$$

The Majorana/Jordan-Wigner one-body object used for norm control is

$$
\kappa_{pq}=h^{BI}_{pq}+\sum_r g^{BI}_{pqrr}
$$

or, after collecting the BLISS terms,

$$
\kappa
= h-\frac12\sum_r g_{prrq}+\sum_r g_{pqrr}
+\frac{\eta-N}{2}B
-\left(\alpha_1+\frac12\mathrm{Tr}(B)+N\alpha_2\right)I.
$$

During optimization v5 uses the backend-safe proxy

$$
\left\|\kappa-\frac{\mathrm{Tr}(\kappa)}{N}I\right\|_F^2,
$$

and final reporting computes the true eigenvalue 1-norm contribution

$$
\lambda_\kappa=\sum_i |\epsilon_i(\kappa-\alpha_1 I)|.
$$

The CDF two-body norm regularizer is

$$
\lambda_Z^{CDF}=\frac12\sum_{tkl}|Z^t_{kl}|-\frac14\sum_{tk}|Z^t_{kk}|.
$$

The entropy regularizer from earlier versions is retained.

---

## Backend recommendations

| Machine | Recommended backend | Optimizer | Notes |
|---|---|---|---|
| Apple Silicon local | `backend="mlx", device="gpu"` | Adam | Fastest Mac path. |
| Mac CPU comparison | `backend="torch", device="cpu"` | Adam or L-BFGS | Often faster/more stable than Torch MPS for this workload. |
| Torch MPS | `backend="torch", device="mps"` | Adam only | Compatibility/debug path; avoid L-BFGS and high-level linalg. |
| NVIDIA cluster | `backend="torch", device="cuda", dtype="float64"` | Adam or L-BFGS | Best serious large run path. |

MLX CPU is not recommended. Torch CPU is the safer CPU backend.

---

## Rotation parameterizations

Set with `rotation_param=` in the constructor.

### 1. `rotation_param="expm"`

Original exact manifold parameterization:

$$
U_t=\exp\left[\frac12(X_t-X_t^T)\right].
$$

Pros: full `SO(N)` expressivity, clean reference path.  
Cons: repeated matrix exponentials can dominate runtime.

### 2. `rotation_param="free"`

Directly optimize trainable `U_t` and add an orthogonality penalty:

$$
\lambda_{orth}\sum_t \|U_t^TU_t-I\|_F^2.
$$

You can project every `K` steps or only at the end.

Pros: fastest rough-search path.  
Cons: soft orthogonality can cheat; always monitor projected loss/orthogonality.

### 3. `rotation_param="hgh"`

Structured exact-orthogonal correction around a fixed base rotation:

$$
U_t^{eff}=U_t^{base}H(v_{t,1})G_{eo}(\theta_t)H(v_{t,2}).
$$

`G_eo` is one even layer plus one odd nearest-neighbor Givens layer. Initialization uses

$$
v_2=v_1,\qquad \theta\sim 10^{-4}\mathcal N(0,1),
$$

so the correction starts near identity but has nonzero gradients. After an inner optimization, call `absorb_hgh_correction()` to update

$$
U_t^{base}\leftarrow U_t^{eff}
$$

and reset the correction.

Pros: exact orthogonality, no matrix exponentials, low parameter count.  
Cons: structured local/global update; may need several outer cycles.

---

## Basic usage: MLX GPU with BLISS and CDF

```python
from cdf_eri_optimizer_multibackend_v5 import CDFERIOptimizer

opt = CDFERIOptimizer(
    ints,
    ndf=4,
    backend="mlx",
    device="gpu",
    dtype="float32",
    init="lchol",
    Lchol=Lchol,
    rotation_param="expm",   # "expm", "free", or "hgh"
)

hist = opt.optimize(
    optimizer="adam",
    optimize_strategy="simultaneous",
    learning_rate=1e-2,
    maxiter=2000,

    use_bliss_beta=True,
    lambda_kappa_fro=1e-3,
    lambda_beta_l2=0.0,

    lambda_z1=1e-2,
    lambda_dist=1e0,
    lambda_z2=0.0,
    lambda_norm=0.0,

    batch_t=6,
    contraction_mode="pair_batched",
    print_every=500,
)
```

Final norm report:

```python
report = opt.final_norm_report(apply_alpha2_shift=False)
report
```

---

## BLISS-only preconditioning

This optimizes only `B`, `alpha1`, and `alpha2`, using a smooth L1 norm on the shifted ERI and the traceless Frobenius proxy for `kappa`.

```python
hist_bliss = opt.optimize_bliss_only(
    maxiter=3000,
    learning_rate=1e-2,
    lambda_kappa_fro=1e-3,
    lambda_beta_l2=1e-8,
    smooth_l1_eps=1e-8,
    print_every=500,
)
```

After BLISS-only preconditioning, run CDF against the shifted target:

```python
hist_cdf = opt.optimize(
    optimize_strategy="cdf_only",
    maxiter=3000,
    learning_rate=3e-3,
    use_bliss_beta=True,
    lambda_z1=1e-2,
    lambda_dist=1e0,
    lambda_kappa_fro=1e-3,
)
```

---

## Alternating BLISS/CDF optimization

Alternating is safer than simultaneous optimization from a cold start because BLISS and CDF can otherwise compete for the same tensor weight.

```python
hist_alt = opt.optimize_alternating(
    cycles=5,
    cdf_steps=1000,
    bliss_steps=250,
    learning_rate=3e-3,
    use_bliss_beta=True,
    lambda_z1=1e-2,
    lambda_dist=1e0,
    lambda_kappa_fro=1e-3,
    lambda_beta_l2=1e-8,
    print_every=500,
)
```

For `rotation_param="hgh"`, `optimize_alternating(..., absorb_hgh_each_cycle=True)` absorbs the HGH correction after each cycle.

Recommended workflow:

```text
BLISS-only precondition
→ CDF-only fit
→ alternating BLISS/CDF
→ short simultaneous polish
```

---

## Free-U warm start with projection

```python
opt_free = CDFERIOptimizer(
    ints,
    ndf=4,
    backend="mlx",
    device="gpu",
    init="lchol",
    Lchol=Lchol,
    rotation_param="free",
)

hist_free = opt_free.optimize(
    optimizer="adam",
    maxiter=5000,
    learning_rate=1e-2,
    use_bliss_beta=True,
    lambda_z1=1e-2,
    lambda_dist=1e0,
    lambda_kappa_fro=1e-3,
    lambda_orth=1e0,
    project_every=500,      # project every K steps
    project_at_end=True,    # also project after optimization
    print_every=500,
)
```

Projection uses SVD/polar-style nearest-orthogonal projection and fixes determinant to `+1`.

Diagnostics to watch:

```python
report = opt_free.final_norm_report()
U = opt_free.get_rotation_numpy()
```

If projected loss gets much worse than free loss, increase `lambda_orth` or project more often.

---

## HGH refinement

```python
opt_hgh = CDFERIOptimizer(
    ints,
    ndf=4,
    backend="mlx",
    device="gpu",
    init="lchol",
    Lchol=Lchol,
    rotation_param="hgh",
    hgh_theta_init_scale=1e-4,
)

for outer in range(5):
    hist = opt_hgh.optimize(
        optimizer="adam",
        maxiter=1000,
        learning_rate=3e-3,
        use_bliss_beta=True,
        lambda_z1=1e-2,
        lambda_dist=1e0,
        lambda_kappa_fro=1e-3,
        lambda_correction=1e-4,
        print_every=500,
    )
    opt_hgh.absorb_hgh_correction(reset=True)
```

`lambda_correction` penalizes

$$
\|\hat v_1-\hat v_2\|^2+\|\theta_{even}\|^2+\|\theta_{odd}\|^2
$$

and acts like a trust region for each correction cycle.

---

## Torch CPU / CUDA L-BFGS

```python
opt_cpu = CDFERIOptimizer(
    ints,
    ndf=4,
    backend="torch",
    device="cpu",      # or "cuda"
    dtype="float64",
    init="lchol",
    Lchol=Lchol,
    rotation_param="expm",
)

hist = opt_cpu.optimize(
    optimizer="lbfgs",
    maxiter=100,
    learning_rate=1.0,
    lbfgs_max_iter=5,
    lbfgs_line_search_fn=None,
    use_bliss_beta=True,
    lambda_z1=1e-2,
    lambda_dist=1e0,
    lambda_kappa_fro=1e-3,
    print_every=10,
)
```

Use L-BFGS as a short polish, not as a long main optimizer.

---

## Important hyperparameters

| Argument | Meaning |
|---|---|
| `lambda_z1` | Weight on CDF two-body norm `0.5|Z|_1 - 0.25|diag Z|_1`. |
| `lambda_dist` | Entropy/distribution regularizer across factors. Positive values penalize high entropy and encourage concentration. |
| `lambda_kappa_fro` | Weight on traceless Frobenius proxy for the Majorana/JW one-body object `kappa`. |
| `lambda_beta_l2` | Direct Frobenius penalty on BLISS `B`. |
| `lambda_orth` | Free-U orthogonality penalty. Only used with `rotation_param="free"`. |
| `lambda_correction` | HGH trust-region penalty. Only used with `rotation_param="hgh"`. |
| `project_every` | Free-U projection interval. `None` means no projection during optimization. |
| `project_at_end` | Project free U after the last step. |
| `use_cdf_z_norm` | If `True`, uses CDF norm formula instead of raw `|Z|_1`. |
| `smooth_l1_eps` | Smooth absolute value epsilon for BLISS-only L1 minimization. |

---

## Performance notes

1. Matrix exponentials are often the bottleneck in the original `expm` mode.
2. `free` is fastest per step but can exploit nonorthogonality; use projection diagnostics.
3. `hgh` is exact-orthogonal and expm-free, but it is structured and may need several absorb/reset cycles.
4. Avoid forming extra full `N^4` tensors when possible. The current BLISS shift still materializes the shifted residual; future optimization should collapse BLISS inner products analytically.
5. For Mac: prefer MLX GPU for Adam and Torch CPU for comparison/L-BFGS.
6. For NVIDIA: prefer Torch CUDA float64.

---

## Suggested benchmark ladder

1. `rotation_param="expm"`, BLISS off: reproduce old v4/v3 behavior.
2. `rotation_param="expm"`, BLISS on: test corrected `kappa` penalty.
3. BLISS-only precondition → CDF-only fit.
4. `rotation_param="free"`, projection every 500–1000 steps.
5. `rotation_param="hgh"`, absorb/reset cycles.
6. Short simultaneous polish.
7. Optional Torch CPU/CUDA L-BFGS polish.

