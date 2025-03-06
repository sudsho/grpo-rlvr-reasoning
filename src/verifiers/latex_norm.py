"""LaTeX -> sympy-friendly string normalization.

Handles the common cases the MATH benchmark throws at extract-then-compare:
- \\frac{a}{b} -> (a)/(b)
- \\sqrt{a}   -> sqrt(a)
- \\pi        -> pi
- \\cdot, \\times -> *
- \\left \\right stripped
- \\text{...} stripped
- percentages, commas in numbers, trailing periods
- dfrac / tfrac are aliases for frac
"""
from __future__ import annotations

import re

_TEXT_WRAPPERS = [r"\\text", r"\\mathrm", r"\\mathit", r"\\mathbf"]


def _strip_wrappers(s: str) -> str:
    for w in _TEXT_WRAPPERS:
        # \\text{foo} -> foo
        s = re.sub(w + r"\s*\{([^{}]*)\}", r"\1", s)
    return s


def _frac(s: str) -> str:
    pat = re.compile(r"\\(?:d|t)?frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
    prev = None
    while prev != s:
        prev = s
        s = pat.sub(r"((\1)/(\2))", s)
    return s


def _sqrt(s: str) -> str:
    return re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", s)


def normalize_latex(s: str) -> str:
    if s is None:
        return ""
    s = s.strip()
    # strip surrounding $, $$, and any \\(...\\) wrappers
    s = s.strip("$").strip()
    s = re.sub(r"\\left|\\right", "", s)
    s = _strip_wrappers(s)
    s = _frac(s)
    s = _sqrt(s)
    s = s.replace("\\pi", "pi")
    s = s.replace("\\cdot", "*").replace("\\times", "*")
    s = s.replace("^", "**")
    # percentages: "50\%" or "50%" -> "(50/100)"
    s = re.sub(r"(-?\d+(?:\.\d+)?)\s*\\?%", r"(\1/100)", s)
    # commas as thousands separators
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)
    # collapse whitespace
    s = re.sub(r"\s+", "", s)
    # drop trailing period
    if s.endswith("."):
        s = s[:-1]
    return s
