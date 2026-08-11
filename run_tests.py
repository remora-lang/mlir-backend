#!/usr/bin/env python3
"""End-to-end test runner for the MLIR backend + IREE.

Each test is a `.fut` file carrying a Futhark-style test block:

    -- ==
    -- input { [1i32, 2i32, 3i32, 4i32] }
    -- output { 10i32 }

The runner marshals the `input` values into `iree-run-module` flags (deriving
array shapes and the leading dynamic-dimension `i64` sizes from the literals),
runs the file through `run-iree`, and checks the printed result matches
`output`. Comparison is on the flat sequence of numeric values, ignoring type
suffixes and formatting.

Supported literals: signed-int scalars and rectangular int arrays (1D/2D) with
explicit type suffixes (`i32`/`i64`/...). Anything we can't handle (floats,
booleans, `empty(...)`) is reported as a failure -- it means the program isn't
handled -- not silently skipped.
"""

import glob
import re
import subprocess
import sys

TESTS_GLOB = "tests/*.fut"

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


def _marshal(value):
    """One Futhark value -> list of iree-run-module `--input` argument strings."""
    if not isinstance(value, list):
        return [_strip_suffix(value)]
    dims = _shape(value)
    leaves = list(_leaves(value))
    etype = _elem_type(leaves[0])
    flat = " ".join(_strip_suffix(t) for t in leaves)
    tensor = "x".join(str(d) for d in dims) + f"x{etype}=" + flat
    # Dynamic dimensions are passed as explicit leading i64 args, then the array.
    return [str(d) for d in dims] + [tensor]


# --- IREE output parsing -----------------------------------------------------

def _iree_output_values(stdout):
    """Flat list of numeric-value tokens across all `result[i]:` lines."""
    out = []
    lines = stdout.splitlines()
    i = 0
    while i < len(lines):
        m = re.match(r"\s*result\[\d+\]:\s*(.*)", lines[i])
        if m:
            payload = m.group(1)
            if payload.startswith("hal.buffer_view"):
                i += 1
                payload = lines[i].strip()
            if "=" in payload:
                payload = payload.split("=", 1)[1]
            payload = payload.replace("[", " ").replace("]", " ")
            out.extend(payload.split())
        i += 1
    return out


def _expected_values(block):
    cleaned = re.sub(r"[\[\],{}]", " ", block)
    return [_strip_suffix(t) for t in cleaned.split()]


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


# --- driver ------------------------------------------------------------------

def _run_case(path, input_str, output_str):
    values = _parse_values(_tokenize(input_str))
    flags = []
    for v in values:
        flags += [f"--input={a}" for a in _marshal(v)]
    proc = subprocess.run(
        ["run-iree", path, *flags],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        tail = proc.stderr.strip().splitlines()[-3:]
        return False, "run-iree failed:\n    " + "\n    ".join(tail)
    actual = _iree_output_values(proc.stdout)
    expected = _expected_values(output_str)
    if actual != expected:
        return False, f"expected {expected}, got {actual}"
    return True, None


def main():
    files = sys.argv[1:] or sorted(glob.glob(TESTS_GLOB))
    passed = failed = 0
    for path in files:
        for input_str, output_str in _read_cases(path):
            name = f"{path}  input {{{input_str.strip()}}}"
            try:
                ok, msg = _run_case(path, input_str, output_str)
            except Unsupported as e:
                # We couldn't handle this program's annotations -- that's a
                # failure of the backend/runner, not something to skip.
                ok, msg = False, f"unsupported: {e}"
            if ok:
                print(f"{_c('PASS', '32')} {name}")
                passed += 1
            else:
                print(f"{_c('FAIL', '31')} {name}\n     {msg}")
                failed += 1
    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
