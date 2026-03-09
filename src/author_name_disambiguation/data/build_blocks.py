from __future__ import annotations

import re
import unicodedata

import pandas as pd

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
_SURNAME_PARTICLES = {"da", "de", "del", "della", "der", "di", "du", "la", "le", "van", "von"}
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def _ascii_lower(text: str) -> str:
    norm = unicodedata.normalize("NFKD", text)
    ascii_text = norm.encode("ascii", "ignore").decode("ascii")
    return ascii_text.lower().strip()


def _clean_token(token: str) -> str:
    return _NON_ALNUM_RE.sub("", token.lower())


def create_block_key(author_name: str) -> str:
    """Create block key as first initial + last name (paper-compatible).

    Examples:
    - "Frenkel, Josif" -> "j.frenkel"
    - "Josif Frenkel" -> "j.frenkel"
    """
    if author_name is None:
        return "unknown"

    raw = _ascii_lower(str(author_name))
    if not raw:
        return "unknown"

    if "," in raw:
        left, right = raw.split(",", 1)
        last_part = left.strip()
        first_part = right.strip()
    else:
        parts = [p for p in raw.split() if p]
        if not parts:
            return "unknown"
        # ADS often stores names as "Lastname [particles] I[...]" without comma.
        # Detect trailing initials and treat all preceding tokens as surname phrase.
        if len(parts) >= 2:
            trailing_initials = 0
            for token in reversed(parts):
                cleaned = _clean_token(token)
                if 0 < len(cleaned) <= 3:
                    trailing_initials += 1
                else:
                    break
            if 0 < trailing_initials < len(parts):
                last_part = " ".join(parts[:-trailing_initials]).strip()
                first_part = parts[-trailing_initials]
            else:
                last_part = parts[-1]
                first_part = parts[0]
        else:
            last_part = parts[0]
            first_part = parts[0]

    first_initial = ""
    for ch in first_part:
        if ch.isalpha() or ch.isdigit():
            first_initial = ch
            break

    last_tokens = [t for t in re.split(r"\s+", last_part) if t]
    if last_tokens and _clean_token(last_tokens[-1]) in _SUFFIXES and len(last_tokens) > 1:
        last_tokens = last_tokens[:-1]
    # For particle-based surnames, use the lexical surname token for blocking.
    if len(last_tokens) > 1 and _clean_token(last_tokens[0]) in _SURNAME_PARTICLES:
        last_tokens = last_tokens[1:]
    last_name = _clean_token(last_tokens[-1]) if last_tokens else ""

    if not first_initial or not last_name:
        return "unknown"

    return f"{first_initial}.{last_name}"


def add_block_key(df: pd.DataFrame, author_col: str = "author_raw", output_col: str = "block_key") -> pd.DataFrame:
    out = df.copy()
    out[output_col] = out[author_col].map(create_block_key)
    return out
