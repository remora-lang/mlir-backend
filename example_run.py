#!/usr/bin/env python3
"""Compile and run a StableHLO module with the XLA bundled in JAX.

Random inputs are generated from the signature of @main, so any module works.
dot_general precision is a compile-time backend flag now (`mlir-backend
--precision default|high|highest`, or `compile x.fut --precision ...`), not
something this script rewrites after the fact -- pass --tol explicitly if
the module was compiled with --precision highest and the default tolerance
is too loose.

  python run.py main.mlir
  python run.py main.mlir --iters 100 --flops 34359738368 --check

Install (CUDA):  pip install "jax[cuda13]"   (or jax[cuda12])
"""
import argparse
import re
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# Helpers that don't need JAX
# ---------------------------------------------------------------------------

_DTYPES = {
    "f64": np.float64, "f32": np.float32, "f16": np.float16,
    "i64": np.int64, "i32": np.int32, "i16": np.int16, "i8": np.int8,
    "ui32": np.uint32, "ui8": np.uint8, "i1": np.bool_,
}


def _dtype(name):
    if name == "bf16":
        import ml_dtypes  # installed with jax
        return ml_dtypes.bfloat16
    if name not in _DTYPES:
        sys.exit(f"unsupported element type {name}")
    return _DTYPES[name]


def main_signature(text):
    """[(shape, dtype)] for each argument of @main."""
    m = re.search(r"func\.func\s+(?:public\s+)?@main\s*\((.*?)\)\s*(?:->|attributes|\{)", text, re.S)
    if not m:
        sys.exit("no @main function found (XLA requires the entry function to be named main)")
    sig = []
    for t in re.findall(r"%[\w.$-]+\s*:\s*tensor<([^>]*)>", m.group(1)):
        *dims, elt = t.split("x")
        sig.append((tuple(int(d) for d in dims), _dtype(elt)))
    return sig


def random_inputs(sig, seed):
    rng = np.random.default_rng(seed)
    out = []
    for shape, dt in sig:
        if dt == np.bool_:
            x = rng.integers(0, 2, shape).astype(dt)
        elif np.issubdtype(np.dtype(dt), np.integer):
            x = rng.integers(-10, 10, shape).astype(dt)
        else:
            x = rng.uniform(-1, 1, shape).astype(np.float32).astype(dt)
        out.append(x)
    return out


def stats_ms(ts):
    t = np.asarray(ts) * 1e3
    return dict(min=t.min(), median=np.median(t), mean=t.mean(),
                std=t.std(ddof=1) if len(t) > 1 else 0.0, max=t.max())


# ---------------------------------------------------------------------------
# JAX / XLA
# ---------------------------------------------------------------------------

def compile_on(platform, text):
    import jax.extend.backend as jex
    from jax._src.lib import xla_client as xc

    backend = jex.get_backend(platform)
    dev = backend.local_devices()[0]
    opts = jex.get_compile_options(num_replicas=1, num_partitions=1)
    if hasattr(backend, "compile_and_load"):           # jaxlib >= ~0.6
        try:
            devs = xc.DeviceList((dev,))
        except Exception:
            devs = [dev]
        exe = backend.compile_and_load(text, devs, opts)
    else:                                               # older jaxlib
        exe = backend.compile(text, opts)
    return exe, dev


def execute(exe, args):
    if hasattr(exe, "execute_sharded"):
        res = exe.execute_sharded(args)
        outs = [per_dev[0] for per_dev in res.disassemble_into_single_device_arrays()]
    else:
        outs = exe.execute(args)
    for o in outs:
        o.block_until_ready()
    return outs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("module", help="StableHLO .mlir file (entry function @main)")
    ap.add_argument("--platform", default="gpu", help="gpu (CUDA or ROCm) or cpu (default: gpu)")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--flops", type=float, help="FLOPs per run, to print GFLOP/s")
    ap.add_argument("--check", action="store_true", help="compare outputs against XLA:CPU")
    ap.add_argument("--tol", type=float, default=2e-3,
                    help="check tolerance on max|out-ref|/max|ref| (default 2e-3)")
    a = ap.parse_args()

    text = open(a.module).read()
    sig = main_signature(text)

    import jax
    jax.config.update("jax_enable_x64", True)  # keep f64/i64 inputs as-is; we never trace
    print(f"jax {jax.__version__}, platform {a.platform}")

    t0 = time.perf_counter()
    exe, dev = compile_on(a.platform, text)
    print(f"device: {dev.device_kind}   compiled in {time.perf_counter() - t0:.2f} s")

    host_in = random_inputs(sig, a.seed)
    print("inputs: " + ", ".join(f"{x.dtype}{list(x.shape)}" for x in host_in))
    dev_in = [jax.device_put(x, dev) for x in host_in]

    for _ in range(a.warmup):
        execute(exe, dev_in)
    ts = []
    for _ in range(a.iters):
        t = time.perf_counter()
        outs = execute(exe, dev_in)
        ts.append(time.perf_counter() - t)

    s = stats_ms(ts)
    print(f"timing over {a.iters} iters ({a.warmup} warm-up), wall-clock incl. launch:")
    for k in ("min", "median", "mean"):
        g = f"  ({a.flops / (s[k] * 1e-3) / 1e9:8.1f} GFLOP/s)" if a.flops else ""
        print(f"  {k:<6} {s[k]:9.3f} ms{g}")
    print(f"  std    {s['std']:9.3f} ms  ({100 * s['std'] / s['mean']:.1f}% of mean)")
    print(f"  max    {s['max']:9.3f} ms")
    print("outputs: " + ", ".join(f"{o.dtype}{list(o.shape)}" for o in outs))

    if a.check:
        cpu_exe, cpu_dev = compile_on("cpu", text)
        ref = execute(cpu_exe, [jax.device_put(x, cpu_dev) for x in host_in])
        ok = True
        for i, (o, r) in enumerate(zip(outs, ref)):
            o, r = np.asarray(o, np.float64), np.asarray(r, np.float64)
            err = np.abs(o - r).max() / max(np.abs(r).max(), 1e-30)
            ok &= bool(err <= a.tol)
            print(f"check output {i} vs XLA:CPU: max|out-ref|/max|ref| = {err:.3g} (tol {a.tol:g}) "
                  f"-> {'OK' if err <= a.tol else 'MISMATCH'}")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
