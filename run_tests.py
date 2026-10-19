#!/usr/bin/env python3
"""End-to-end test runner for the MLIR backend + JAX/XLA.

Each test is a `.fut` file carrying a Futhark-style test block:

    -- ==
    -- input { [1i32, 2i32, 3i32, 4i32] }
    -- output { 10i32 }

The runner compiles the file (`compile`, the Nix-provided wrapper around the
backend binary -- its stdout is the StableHLO module text, no separate
artifact file), reads `@main`'s signature from that text (same regex as
`example_run.py`'s `main_signature`), binds the `input` values to `@main`'s
arguments positionally (there is no more derived-size indirection: dynamic-
size programs are `Undefined()` before they ever reach here), runs the
compiled module via JAX/XLA on `--platform cpu` (reusing `example_run.py`'s
`compile_on`/`execute`), and checks the result against `output`. Comparison
is on the flat sequence of numeric values, ignoring type suffixes and
formatting.

A file may instead carry a lone `-- undefined` comment line (no `input`/
`output` blocks needed): the test then passes iff the file fails to compile
(i.e. hits `Undefined()`), and fails if it compiles successfully.

Supported literals: signed-int and float scalars and rectangular arrays
(1D/2D) with explicit type suffixes (`i32`/`i64`/`f32`/...); floats are
compared with a tolerance since backends format them differently (`2.0` vs
`2`). Anything we can't handle (booleans, `empty(...)`) is reported as a
failure -- it means the program isn't handled -- not silently skipped.
"""

import glob
import math
import os
import re
import subprocess
import sys

import numpy as np
import jax

import example_run as er

# Tests live under tests/static_shapes/ and tests/dynamic_shapes/ (the
# latter can never pass under this backend -- XLA requires static shapes --
# so it's kept separate rather than mixed in with the suite that's actually
# expected to make progress). The glob is recursive so the no-args default
# still covers both; pass a directory (e.g. `run-tests tests/static_shapes`)
# to run just one in isolation.
TESTS_GLOB = "tests/**/*.fut"

_COLOR = sys.stdout.isatty()


def _c(s, code):
    return f"\033[{code}m{s}\033[0m" if _COLOR else s


class Unsupported(Exception):
    pass


# --- Futhark literal parsing -------------------------------------------------

def _tokenize(s):
    return re.findall(r"\[|\]|,|[^\s\[\],]+", s)


def _parse_values(tokens):
    """Parse a whitespace-separated sequence of Futhark values.

    Returns a list; each element is a scalar token (str) or a nested list.
    """
    pos = 0

    def parse():
        nonlocal pos
        tok = tokens[pos]
        if tok == "[":
            pos += 1
            arr = []
            while tokens[pos] != "]":
                if tokens[pos] == ",":
                    pos += 1
                    continue
                arr.append(parse())
            pos += 1  # consume ]
            return arr
        if tok in (",", "]"):
            raise Unsupported(f"unexpected token {tok!r}")
        pos += 1
        return tok

    out = []
    while pos < len(tokens):
        if tokens[pos] == ",":
            pos += 1
            continue
        out.append(parse())
    return out


def _strip_suffix(tok):
    m = re.match(r"^(-?\d+(?:\.\d+)?)(?:[iuf]\d+|bool)?$", tok)
    if not m:
        raise Unsupported(f"cannot parse scalar {tok!r}")
    return m.group(1)


def _elem_type(tok):
    m = re.search(r"([iuf]\d+|bool)$", tok)
    if m:
        return m.group(1)
    raise Unsupported(f"array element {tok!r} has no type suffix")


def _leaves(value):
    if isinstance(value, list):
        for v in value:
            yield from _leaves(v)
    else:
        yield value


def _shape(value):
    dims = []
    v = value
    while isinstance(v, list):
        dims.append(len(v))
        v = v[0]
    return dims


def _to_array(value):
    """One parsed Futhark value -> a numpy array (0-d for a scalar)."""
    if not isinstance(value, list):
        dt = er._DTYPES.get(_elem_type(value))
        if dt is None:
            raise Unsupported(f"unsupported element type in {value!r}")
        return np.array(_strip_suffix(value), dtype=dt)
    leaves = list(_leaves(value))
    dt = er._DTYPES.get(_elem_type(leaves[0]))
    if dt is None:
        raise Unsupported(f"unsupported element type in {leaves[0]!r}")
    flat = np.array([_strip_suffix(t) for t in leaves], dtype=dt)
    return flat.reshape(_shape(value))


# --- expected-output parsing / comparison ------------------------------------

def _expected_values(block):
    cleaned = re.sub(r"[\[\],{}]", " ", block)
    return [_strip_suffix(t) for t in cleaned.split()]


def _flatten_outputs(outs):
    vals = []
    for o in outs:
        arr = np.asarray(o)
        vals.extend(str(x) for x in arr.reshape(-1))
    return vals


def _values_match(actual, expected, rel_tol=1e-5, abs_tol=1e-6):
    """Compare flat value lists element-wise.

    Numeric tokens are compared with a tolerance (floats print differently
    across backends, e.g. `2.0` vs `2`); anything non-numeric falls back to an
    exact string match.
    """
    if len(actual) != len(expected):
        return False
    for a, e in zip(actual, expected):
        try:
            fa, fe = float(a), float(e)
        except ValueError:
            if a != e:
                return False
            continue
        if not math.isclose(fa, fe, rel_tol=rel_tol, abs_tol=abs_tol):
            return False
    return True


# --- test-block extraction ---------------------------------------------------

def _read_cases(path):
    """Return list of (input_str, output_str) pairs from a file's test blocks."""
    comment = []
    for line in open(path):
        s = line.strip()
        if s.startswith("--"):
            comment.append(s[2:].strip())
    text = " ".join(comment)
    inputs = re.findall(r"input\s*\{(.*?)\}", text)
    outputs = re.findall(r"output\s*\{(.*?)\}", text)
    return list(zip(inputs, outputs))


def _is_undefined_test(path):
    """A lone `-- undefined` comment line marks a file that must fail to
    compile (with Undefined()) to pass -- no input/output blocks needed."""
    for line in open(path):
        s = line.strip()
        if s.startswith("--") and s[2:].strip().lower() == "undefined":
            return True
    return False


# --- driver ------------------------------------------------------------------

def _compile(path):
    """Compile `path` via the `compile` Nix wrapper, returning the emitted
    StableHLO module text, or raising Unsupported on failure."""
    proc = subprocess.run(["compile", path], capture_output=True, text=True)
    if proc.returncode != 0:
        tail = proc.stderr.strip().splitlines()[-5:]
        raise Unsupported("compile failed:\n    " + "\n    ".join(tail))
    return proc.stdout


def _run_file(path):
    """Yield (case_name, ok, msg) for every case in `path`."""
    if _is_undefined_test(path):
        try:
            _compile(path)
        except Unsupported:
            yield "compile (expected Undefined())", True, None
            return
        yield ("compile (expected Undefined())", False,
               "compiled successfully, but this test expects Undefined()")
        return

    cases = _read_cases(path)
    if not cases:
        return

    try:
        text = _compile(path)
        sig = er.main_signature(text)
        exe, dev = er.compile_on("cpu", text)
    except Unsupported as e:
        for input_str, _ in cases:
            yield f"input {{{input_str.strip()}}}", False, f"unsupported: {e}"
        return
    except (Exception, SystemExit) as e:
        for input_str, _ in cases:
            yield f"input {{{input_str.strip()}}}", False, f"compile/load failed: {e}"
        return

    for input_str, output_str in cases:
        name = f"input {{{input_str.strip()}}}"
        try:
            values = _parse_values(_tokenize(input_str))
            if len(values) != len(sig):
                raise Unsupported(
                    f"@main expects {len(sig)} args, test supplies {len(values)}"
                )
            host_in = [_to_array(v) for v in values]
            dev_in = [jax.device_put(x, dev) for x in host_in]
            outs = er.execute(exe, dev_in)
            actual = _flatten_outputs(outs)
            expected = _expected_values(output_str)
            if _values_match(actual, expected):
                yield name, True, None
            else:
                yield name, False, f"expected {expected}, got {actual}"
        except Unsupported as e:
            yield name, False, f"unsupported: {e}"
        except Exception as e:
            yield name, False, f"run failed: {e}"


def main():
    jax.config.update("jax_enable_x64", True)  # keep f64/i64 inputs as-is
    args = sys.argv[1:] or sorted(glob.glob(TESTS_GLOB, recursive=True))
    files = []
    for a in args:
        if os.path.isdir(a):
            files.extend(sorted(glob.glob(os.path.join(a, "*.fut"))))
        else:
            files.append(a)
    passed = failed = 0
    for path in files:
        for case_name, ok, msg in _run_file(path):
            label = f"{path}  {case_name}"
            if ok:
                print(f"{_c('PASS', '32')} {label}")
                passed += 1
            else:
                print(f"{_c('FAIL', '31')} {label}\n     {msg}")
                failed += 1
    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
