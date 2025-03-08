"""Math answer verifier: extract-then-compare with sympy fallback.

Comparison order:
1. String equality after normalization (fast path).
2. sympy equality on parsed expressions (for e.g. 1/2 == 0.5, sqrt(4) == 2).
3. Numeric closeness with a small tolerance (for irrational rounding).

Returns a bool and a short reason string, which is useful in the reward
audit log we dump alongside each rollout.
"""
from __future__ import annotations

from dataclasses import dataclass

import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

from src.data.math_bench import extract_boxed
from src.verifiers.latex_norm import normalize_latex

_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)

# things that will always come out of the pred; strip before parsing
_STOP_TOKENS = ("\\displaystyle", "\\!", "\\,", " ")


@dataclass
class VerifierResult:
    ok: bool
    reason: str


def _clean(x: str) -> str:
    for t in _STOP_TOKENS:
        x = x.replace(t, "")
    return x


def _parse(x: str) -> sp.Expr | None:
    try:
        return parse_expr(x, transformations=_TRANSFORMS)
    except Exception:
        return None


def _sympy_equal(a: str, b: str) -> bool:
    ea, eb = _parse(a), _parse(b)
    if ea is None or eb is None:
        return False
    try:
        diff = sp.simplify(ea - eb)
        if diff == 0:
            return True
        # numeric fallback for irrational tails
        return abs(float(diff.evalf())) < 1e-6
    except Exception:
        return False


def verify(pred_response: str, gold_boxed: str) -> VerifierResult:
    pred = extract_boxed(pred_response)
    if not pred:
        return VerifierResult(False, "no boxed answer")

    p = _clean(normalize_latex(pred))
    g = _clean(normalize_latex(gold_boxed))

    if not p or not g:
        return VerifierResult(False, "empty after normalize")

    if p == g:
        return VerifierResult(True, "string match")

    if _sympy_equal(p, g):
        return VerifierResult(True, "sympy match")

    return VerifierResult(False, f"mismatch: {p!r} vs {g!r}")
