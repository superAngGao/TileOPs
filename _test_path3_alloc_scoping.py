"""Try to get T.alloc_fragment to be SCOPED inside a T.ws(N) frame so the
lowered CUDA puts the declaration inside the if-then-else block.

Three approaches to try:
  M1: just use T.alloc_fragment inside T.ws(N) as before (control)
  M2: use @T.macro to encapsulate the WG body, including the alloc
  M3: use T.block explicitly inside T.ws(N) to push a Block frame
"""
import os
os.environ.setdefault("TILELANG_DISABLE_CACHE", "1")

import tilelang
import tilelang.language as T


def build_M1():
    @tilelang.jit(pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    })
    def func():
        @T.prim_func
        def main(A: T.Tensor((128,), "float32")):
            with T.Kernel(1, 1, 1, threads=384) as (bx, by, bz):
                with T.ws(1):
                    acc = T.alloc_fragment([64], "float")
                    T.clear(acc)
                    for i in T.Parallel(64):
                        acc[i] = i * 1.0
                    for i in T.Parallel(64):
                        A[i] = acc[i]
        return main
    return func


def build_M2():
    """Try @T.macro to encapsulate the WG body."""
    @T.macro
    def wg1_body(A):
        acc = T.alloc_fragment([64], "float")
        T.clear(acc)
        for i in T.Parallel(64):
            acc[i] = i * 1.0
        for i in T.Parallel(64):
            A[i] = acc[i]

    @tilelang.jit(pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    })
    def func():
        @T.prim_func
        def main(A: T.Tensor((128,), "float32")):
            with T.Kernel(1, 1, 1, threads=384) as (bx, by, bz):
                with T.ws(1):
                    wg1_body(A)
        return main
    return func


def build_M3():
    """Try explicit T.block inside T.ws(1)."""
    @tilelang.jit(pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    })
    def func():
        @T.prim_func
        def main(A: T.Tensor((128,), "float32")):
            with T.Kernel(1, 1, 1, threads=384) as (bx, by, bz):
                with T.ws(1):
                    with T.block("wg1_scope"):
                        T.reads(A[0])
                        T.writes(A[0])
                        acc = T.alloc_buffer([64], "float", scope="local.fragment")
                        for i in T.serial(64):
                            acc[i] = T.float32(i)
                        for i in T.serial(64):
                            A[i] = acc[i]
        return main
    return func


def try_build(name, builder):
    print(f"\n--- {name} ---")
    try:
        k = builder()()
        try:
            src = k.get_kernel_source()
            print(f"  Generated CUDA (acc-related lines):")
            for i, line in enumerate(src.splitlines()):
                if "acc" in line or "main_kernel" in line or " if " in line or "{" == line.strip() or "}" == line.strip():
                    print(f"    {i:3}: {line}")
        except Exception as e:
            print(f"  Couldn't dump: {e}")
    except Exception as e:
        print(f"  BUILD FAILED — {type(e).__name__}: {str(e)[:300]}")


if __name__ == "__main__":
    try_build("M1: bare T.alloc_fragment in T.ws", build_M1)
    try_build("M2: @T.macro encapsulation", build_M2)
    try_build("M3: explicit T.block inside T.ws", build_M3)
