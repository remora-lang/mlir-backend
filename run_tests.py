#!/usr/bin/env python3
"""End-to-end test runner for the MLIR backend + IREE.

Each test is a `.fut` file carrying a Futhark-style test block:

    -- ==
    -- input { [1i32, 2i32, 3i32, 4i32] }
    -- output { 10i32 }

The runner compiles the file (`run-iree --compile-only`), reads `entry_main`'s
signature and its `futhark.*` ABI attributes from the emitted `out/$name.mlir`,
binds the `input` values to those parameters (resolving leading `i64` size args
by symbol from the array shapes), then runs the built module (`run-iree
--run-only`) and checks the printed result matches `output`. Comparison is on
the flat sequence of numeric values, ignoring type suffixes and formatting.

Supported literals: signed-int and float scalars and rectangular arrays (1D/2D)
with explicit type suffixes (`i32`/`i64`/`f32`/...); floats are compared with a
tolerance since backends format them differently (`2.0` vs `2`). Anything we
can't handle (booleans, `empty(...)`) is reported as a failure -- it means the
program isn't handled -- not silently skipped.
"""

import glob
import math
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


def _payload(value):
    """One Futhark value -> its `iree-run-module` payload string (no size args)."""
    if not isinstance(value, list):
        return _strip_suffix(value)
    dims = _shape(value)
    leaves = list(_leaves(value))
    etype = _elem_type(leaves[0])
    flat = " ".join(_strip_suffix(t) for t in leaves)
    return "x".join(str(d) for d in dims) + f"x{etype}=" + flat


# --- entry-point signature (ABI reflection) ----------------------------------
#
# The compiled `out/$name.mlir` carries `entry_main`'s parameter list. The
# backend annotates each tensor (array) parameter with a `futhark.size_args`
# attribute: one entry per dimension giving the *parameter index* that supplies
# that dimension's extent, or -1 for a statically-sized dimension. Example:
#   %arg0: i64, %arg1: i64, %arg2: i64,
#   %arg3: tensor<?x?xf32> {futhark.size_args = [0, 1]},  # dims from args 0 and 1
#   %arg4: tensor<?x?xf32> {futhark.size_args = [1, 2]}    # (arg1 shared w/ arg3)
# Size parameters are always prepended, so we classify by arity rather than by
# "is this arg referenced": the leading (num_params - num_test_values) params
# are the derived sizes, resolved from the array shapes via `size_args`; the
# rest consume the test inputs in order. This is exactly what lets us tell
# `f [n] (x: [n])` (n derived) from `f (n: i64) (x: [n])` (n supplied), which
# have identical signatures.

_BRACKETS = {"(": ")", "[": "]", "{": "}", "<": ">"}
_OPENERS, _CLOSERS = set(_BRACKETS), set(_BRACKETS.values())


def _match_paren(text, start):
    """Index of the `)` matching the `(` at `text[start]`."""
    depth = 0
    for j in range(start, len(text)):
        c = text[j]
        if c in _OPENERS:
            depth += 1
        elif c in _CLOSERS:
            depth -= 1
            if depth == 0:
                return j
    raise Unsupported("unterminated entry_main parameter list")


def _split_top_level(s):
    """Split on commas that are not nested inside any bracket."""
    parts, depth, cur = [], 0, ""
    for c in s:
        if c in _OPENERS:
            depth += 1
        elif c in _CLOSERS:
            depth -= 1
        if c == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += c
    parts.append(cur)
    return [p.strip() for p in parts if p.strip()]


def _classify_param(param):
    """Map one entry_main parameter string to ('array', size_args) | ('scalar', None).

    `size_args` is the list of parameter indices feeding each dimension (-1 for a
    static dimension). Integer type suffixes (`: i64`) are ignored.
    """
    m = re.search(r"futhark\.size_args\s*=\s*(\[[^\]]*\]|array<[^>]*>)", param)
    if not m:
        return "scalar", None
    refs = [int(x) for x in re.findall(r"(?<![A-Za-z0-9])-?\d+", m.group(1))]
    return "array", refs


def _entry_params(mlir_path):
    """Ordered list of ('array', size_args) | ('scalar', None) for entry_main."""
    text = open(mlir_path).read()
    key = "@entry_main("
    i = text.find(key)
    if i < 0:
        raise Unsupported(f"no entry_main in {mlir_path}")
    lparen = i + len(key) - 1
    inner = text[lparen + 1 : _match_paren(text, lparen)]
    return [_classify_param(p) for p in _split_top_level(inner)]


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


# --- driver ------------------------------------------------------------------

def _out_mlir(path):
    """The `out/$name.mlir` path run-iree writes when compiling `path`."""
    name = re.sub(r"\.fut(_gpu)?$", "", path.rsplit("/", 1)[-1])
    return f"out/{name}.mlir"


def _fail(proc, prog):
    tail = proc.stderr.strip().splitlines()[-3:]
    return False, f"{prog} failed:\n    " + "\n    ".join(tail)


def _build_flags(params, values):
    """Bind Futhark `values` to entry_main `params` -> ordered --input flags.

    Sizes are always prepended, so the leading `len(params) - len(values)`
    parameters are the derived sizes; the rest consume the test values in order.
    Each derived size is resolved from the shape of an array whose `size_args`
    references its parameter index.
    """
    n_sizes = len(params) - len(values)
    if n_sizes < 0:
        raise Unsupported(
            f"entry_main has {len(params)} params but test supplies "
            f"{len(values)} values"
        )
    # The trailing params consume the test values positionally.
    bound = dict(zip(range(n_sizes, len(params)), values))
    # Resolve each size param's value from an array that references it.
    size_val = {}
    for i, (kind, refs) in enumerate(params):
        if kind != "array" or i not in bound:
            continue
        for dim, ref in enumerate(refs):
            if ref >= 0:
                size_val[ref] = _shape(bound[i])[dim]
    flags = []
    for i, (kind, _) in enumerate(params):
        if i < n_sizes:
            if i not in size_val:
                raise Unsupported(f"could not resolve size for param {i}")
            flags.append(f"--input={size_val[i]}")
        else:
            flags.append(f"--input={_payload(bound[i])}")
    return flags


def _run_case(path, input_str, output_str):
    values = _parse_values(_tokenize(input_str))
    # Phase 1: compile so entry_main's signature is available on disk.
    comp = subprocess.run(
        ["run-iree", "--compile-only", path],
        capture_output=True, text=True,
    )
    if comp.returncode != 0:
        return _fail(comp, "compile")
    flags = _build_flags(_entry_params(_out_mlir(path)), values)
    # Phase 2: run the already-built module with the bound inputs.
    proc = subprocess.run(
        ["run-iree", "--run-only", path, *flags],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return _fail(proc, "run-iree")
    actual = _iree_output_values(proc.stdout)
    expected = _expected_values(output_str)
    if not _values_match(actual, expected):
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
