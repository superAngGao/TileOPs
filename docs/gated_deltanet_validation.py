"""Final validation: numerical alignment + performance vs FLA for fwd, bwd, fwd+bwd."""
import torch
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from tilelang.profiler import do_bench

from tileops.ops import GatedDeltaNetBwdOp, GatedDeltaNetFwdOp, GatedDeltaNetOp


def _to_fla(q, k, v, g, beta):
    return (
        q.permute(0, 2, 1, 3).contiguous(),
        k.permute(0, 2, 1, 3).contiguous(),
        v.permute(0, 2, 1, 3).contiguous(),
        g.permute(0, 2, 1).contiguous(),
        beta.permute(0, 2, 1).contiguous(),
    )


def cosine_sim(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()


def run_validation(B, H, S, DK, DV, BC, dtype):
    print(f"\n{'='*70}")
    print(f"B={B} H={H} S={S} DK={DK} DV={DV} BC={BC} dtype={dtype}")
    print(f"{'='*70}")

    torch.manual_seed(42)
    q = (torch.randn(B, H, S, DK, device="cuda", dtype=dtype) * 0.1).detach().requires_grad_(True)
    k = (torch.randn(B, H, S, DK, device="cuda", dtype=dtype) * 0.1).detach().requires_grad_(True)
    v = (torch.randn(B, H, S, DV, device="cuda", dtype=dtype) * 0.1).detach().requires_grad_(True)
    g = (-torch.rand(B, H, S, device="cuda", dtype=dtype)).detach().requires_grad_(True)
    beta = (torch.rand(B, H, S, device="cuda", dtype=dtype) * 0.5).detach().requires_grad_(True)
    do = torch.randn(B, H, S, DV, device="cuda", dtype=dtype) * 0.1

    scale = DK ** -0.5

    # ===================== FLA reference =====================
    q_f, k_f, v_f, g_f, beta_f = _to_fla(q.data, k.data, v.data, g.data, beta.data)
    q_f = q_f.detach().requires_grad_(True)
    k_f = k_f.detach().requires_grad_(True)
    v_f = v_f.detach().requires_grad_(True)
    g_f = g_f.detach().requires_grad_(True)
    beta_f = beta_f.detach().requires_grad_(True)
    do_f = do.permute(0, 2, 1, 3).contiguous()

    o_fla, _ = chunk_gated_delta_rule(q_f, k_f, v_f, g_f, beta_f, scale=scale)
    o_fla.backward(do_f)
    o_fla_bhsd = o_fla.permute(0, 2, 1, 3).contiguous()

    # ===================== TileOPs fwd =====================
    fwd_op = GatedDeltaNetFwdOp(B, H, S, DK, DV, BC, dtype)
    o_tile, S_fwd, Aw, Au = fwd_op.forward(q.data, k.data, v.data, g.data, beta.data)

    cos_fwd = cosine_sim(o_tile, o_fla_bhsd)
    print(f"\n[FWD] cosine(TileOPs, FLA) = {cos_fwd:.6f}")

    # ===================== TileOPs bwd =====================
    bwd_op = GatedDeltaNetBwdOp(B, H, S, DK, DV, BC, dtype)
    dq_t, dk_t, dv_t, dg_t, dbeta_t = bwd_op.forward(do, q.data, k.data, v.data, g.data, beta.data, S_fwd)

    dq_fla = q_f.grad.permute(0, 2, 1, 3).contiguous()
    dk_fla = k_f.grad.permute(0, 2, 1, 3).contiguous()
    dv_fla = v_f.grad.permute(0, 2, 1, 3).contiguous()
    dg_fla = g_f.grad.permute(0, 2, 1).contiguous()
    dbeta_fla = beta_f.grad.permute(0, 2, 1).contiguous()

    print(f"[BWD] cosine dq = {cosine_sim(dq_t, dq_fla):.6f}")
    print(f"[BWD] cosine dk = {cosine_sim(dk_t, dk_fla):.6f}")
    print(f"[BWD] cosine dv = {cosine_sim(dv_t, dv_fla):.6f}")
    print(f"[BWD] cosine dg = {cosine_sim(dg_t, dg_fla):.6f}")
    print(f"[BWD] cosine dbeta = {cosine_sim(dbeta_t, dbeta_fla):.6f}")

    # ===================== Performance =====================
    print(f"\n--- Performance (fp16, B={B} H={H} S={S}) ---")

    # FWD
    def tileops_fwd():
        return fwd_op.forward(q.data, k.data, v.data, g.data, beta.data)

    def fla_fwd():
        return chunk_gated_delta_rule(q_f.data, k_f.data, v_f.data, g_f.data, beta_f.data, scale=scale)

    lat_fwd_tile = do_bench(tileops_fwd, warmup=50, rep=100)
    lat_fwd_fla = do_bench(fla_fwd, warmup=50, rep=100)
    print(f"FWD:     TileOPs={lat_fwd_tile:.3f}ms  FLA={lat_fwd_fla:.3f}ms  ratio={lat_fwd_tile/lat_fwd_fla:.2f}x")

    # BWD (TileOPs: bwd only; FLA: fwd+bwd)
    def tileops_bwd():
        return bwd_op.forward(do, q.data, k.data, v.data, g.data, beta.data, S_fwd)

    q_f2 = q_f.data.detach().requires_grad_(True)
    k_f2 = k_f.data.detach().requires_grad_(True)
    v_f2 = v_f.data.detach().requires_grad_(True)
    g_f2 = g_f.data.detach().requires_grad_(True)
    beta_f2 = beta_f.data.detach().requires_grad_(True)

    def fla_fwdbwd():
        q_f2.grad = k_f2.grad = v_f2.grad = g_f2.grad = beta_f2.grad = None
        o, _ = chunk_gated_delta_rule(q_f2, k_f2, v_f2, g_f2, beta_f2, scale=scale)
        o.backward(do_f, retain_graph=True)

    lat_bwd_tile = do_bench(tileops_bwd, warmup=50, rep=100)
    lat_fla_fwdbwd = do_bench(fla_fwdbwd, warmup=50, rep=100)
    print(f"BWD:     TileOPs={lat_bwd_tile:.3f}ms  FLA(fwd+bwd)={lat_fla_fwdbwd:.3f}ms")

    # FWD+BWD
    op_combined = GatedDeltaNetOp(B, H, S, DK, DV, BC, dtype)
    q2 = q.data.detach().requires_grad_(True)
    k2 = k.data.detach().requires_grad_(True)
    v2 = v.data.detach().requires_grad_(True)
    g2 = g.data.detach().requires_grad_(True)
    beta2 = beta.data.detach().requires_grad_(True)

    def tileops_fwdbwd():
        q2.grad = k2.grad = v2.grad = g2.grad = beta2.grad = None
        o = op_combined(q2, k2, v2, g2, beta2)
        o.backward(do, retain_graph=True)

    lat_fwdbwd_tile = do_bench(tileops_fwdbwd, warmup=50, rep=100)
    print(f"FWD+BWD: TileOPs={lat_fwdbwd_tile:.3f}ms  FLA={lat_fla_fwdbwd:.3f}ms  ratio={lat_fwdbwd_tile/lat_fla_fwdbwd:.2f}x")


if __name__ == "__main__":
    for S in [256, 1024, 4096, 8192, 16384]:
        run_validation(2, 4, S, 64, 64, 64, torch.float16)
