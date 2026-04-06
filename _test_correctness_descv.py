"""Run all v2 correctness shapes with the desc-by-value hook enabled."""
import os
os.environ.setdefault("TILELANG_DISABLE_CACHE", "1")
os.environ["DESC_REWRITE"] = "1"

# Side-effects: registers patched cuda compile callback.
import _bench_ws_shfl  # noqa: F401
_bench_ws_shfl.enable_shfl_hook()

from _test_ws_fa3_v2 import test

print("=" * 60)
print("Correctness with desc-by-value + shfl hook enabled")
print("=" * 60)
ok = True
ok &= test(1, 256, 8, 4, 128, False)
ok &= test(4, 512, 64, 4, 128, False)
ok &= test(4, 512, 64, 4, 128, True)
ok &= test(1, 1024, 32, 8, 128, True)
ok &= test(2, 2048, 32, 8, 128, True)
print(f"\n{'All passed!' if ok else 'SOME FAILED!'}")
