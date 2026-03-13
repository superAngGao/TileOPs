# Gated DeltaNet WY Representation Bug Fix

Reference: [Gated Delta Networks (ICLR 2025)](https://arxiv.org/abs/2412.06464)

## Paper's Recurrence (Eq. 10)

```
S_t = S_{t-1} (α_t (I - β_t k_t k_t^T)) + β_t v_t k_t^T
o_t = S_t q_t
```

where `α_t = exp(g_t)` is the forget gate, `β_t` is the writing strength.

## Paper's WY Representation (Eq. 6-7, page 6)

### Ungated DeltaNet (Eq. 6-7)

```
A   = I + strictLower(diag(β) · K K^T)        ∈ R^{C×C}
T   = A^{-1} · diag(β)
W   = T · K,    U = T · V
```

### Gated DeltaNet (page 6)

```
A_g = I + strictLower(diag(β) · (Γ ⊙ K K^T))  ∈ R^{C×C}
T_g = A_g^{-1} · diag(β)
W̃  = T_g · K,   Ũ = T_g · V
```

where `Γ_{ij} = exp(g_cum_i - g_cum_j)`, `g_cum = cumsum(g)` within each chunk.

**BOTH W̃ and Ũ use the SAME gated matrix A_g.**

### Chunk output (page 6)

```
V_new = Ũ - W̃← · S̃→^T
O     = q̃← · S^T + (Q K^T ⊙ Γ_causal) · V_new
S_next = S̃→ + V_new^T · k̃→
```

with decay notations:

- `q̃← _r = exp(g_cum_r) · q_r`
- `W̃← _r = exp(g_cum_r) · W̃_r`
- `k̃→ _r = exp(g_cum_last - g_cum_r) · k_r`
- `S̃→   = exp(g_cum_last) · S`
- `Γ_causal[i,j] = exp(g_cum_i - g_cum_j)` for `i ≥ j`, else 0

## Bugs in TileOPs

### Bug 1: Gram matrix has extra β_j

| Location    | Current (wrong)           | Correct (paper)     |
| ----------- | ------------------------- | ------------------- |
| `Gram[i,j]` | `β_i · β_j · (k_i @ k_j)` | `β_i · (k_i @ k_j)` |

Current code computes `(k*β) @ (k*β)^T`, which gives `β_i β_j (k_i@k_j)`.
Paper requires `diag(β) · K K^T`, which gives `β_i (k_i@k_j)`.

### Bug 2: Sign of strictly lower triangular part is inverted

| Location       | Current (wrong)       | Correct (paper)       |
| -------------- | --------------------- | --------------------- |
| Matrix         | `I - tril(Gram, -1)`  | `I + strictLower(M)`  |
| Neumann P init | `P = +tril(Gram, -1)` | `P = -strictLower(M)` |

Current code computes `(I - P)^{-1}` with positive P.
Paper requires `(I + P)^{-1}` which needs P negated before Neumann expansion.

### Bug 3: Gate g is not cumulated

Current code uses raw per-step `g` directly. Paper requires `g_cum = cumsum(g)` within each chunk before computing the decay matrix `Γ`.

### Bug 4: Separate ungated Aw / gated Au design doesn't match paper

Current design uses ungated Aw for `w` and gated Au for `u`. The paper uses a **single gated** matrix `A_g` for both.

### Bug 5: kernel2 missing gate in intra-chunk attention

Current: `attn[i,j] = (q_i @ k_j)` for `i ≥ j` (causal only).
Paper: `attn[i,j] = exp(g_cum_i - g_cum_j) · (q_i @ k_j)` for `i ≥ j` (causal + decay).

### Bug 6: kernel2 v_new missing exp(g_last) factor on state

Current: `v_new = u - (w * exp(g_cum)) @ h`
Correct: `v_new = u - (w * exp(g_cum)) @ (exp(g_cum_last) * h)`

## Affected Files

### Forward (fix now)

| File                                   | What to fix                                   |
| -------------------------------------- | --------------------------------------------- |
| `fused_prepare_compute_w_u.py`         | Bugs 1-4: Gram, sign, cumsum, single matrix   |
| `prepare_wy_repr.py`                   | Bugs 1-3: same Gram/sign/cumsum issues        |
| `gated_deltanet_fwd.py` (kernel2)      | Bugs 5-6: attention gating, v_new state decay |
| `gated_deltanet_fwd.py` (forward fn)   | Bug 3: chunk-local cumsum of g before kernels |
| `tests/ops/test_gated_deltanet_fwd.py` | Reference implementation: all bugs            |

### Backward (fix later)

| File                                   | What to fix                         |
| -------------------------------------- | ----------------------------------- |
| `gated_deltanet_bwd.py`                | Same WY + kernel2 issues in reverse |
| `compute_w_u_bwd.py`                   | Depends on Aw/Au format change      |
| `tests/ops/test_gated_deltanet_bwd.py` | Reference implementation            |

## Verification

FLA's `chunk_gated_delta_rule` is the reference implementation (matches paper):

- FLA recurrent vs FLA chunk: cosine ≈ 1.0
- FLA vs paper step-by-step recurrence: cosine ≈ 1.0
- After fix, TileOPs forward should match FLA with tolerance ≤ 5e-2 (fp16).
