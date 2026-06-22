"""
Evaluation Metrics
==================
Token-level F1 (SQuAD-style multiset overlap) and text normalisation
helpers used by all retrieval evaluation methods.

Tokenisation uses the **same HuggingFace tokenizer** (GPT-2) as the
rest of the pipeline so that token counts are consistent.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import List, Tuple

from chunking_utils import get_tokenizer


# ── Text normalisation ───────────────────────────────────
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_text(text: str) -> str:
    """Lower-case, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = text.translate(_PUNCT_TABLE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_normalized(text: str) -> List[str]:
    """Normalise *text*, then tokenise with the project tokenizer.

    Returns a list of **string tokens** (decoded back from IDs) so that
    multiset overlap is computed in the token space the pipeline uses.
    """
    tok = get_tokenizer()
    normed = normalize_text(text)
    if not normed:
        return []
    ids = tok.encode(normed, add_special_tokens=False)
    return [tok.decode([tid]).strip() for tid in ids if tok.decode([tid]).strip()]


def count_tokens(text: str) -> int:
    """Count tokens in *text* (unnormalised) using the project tokenizer."""
    tok = get_tokenizer()
    if not text or not text.strip():
        return 0
    return len(tok.encode(text, add_special_tokens=False))


# ── Token-level F1 (SQuAD-style) ─────────────────────────
def token_precision_recall_f1(prediction: str, reference: str) -> Tuple[float, float, float]:
    """Compute token precision, recall, and F1 for two text strings.

    Both strings are normalised and tokenised.  Multiset intersection
    is used (same as the original SQuAD evaluation script).

    All three values are 0.0 when either normalized side is empty or
    when there is no token overlap.
    """
    pred_tokens = tokenize_normalized(prediction)
    ref_tokens = tokenize_normalized(reference)

    if not pred_tokens or not ref_tokens:
        return 0.0, 0.0, 0.0

    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0, 0.0, 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)

    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def token_f1(prediction: str, reference: str) -> float:
    """Backward-compatible token-level F1 convenience wrapper."""
    return token_precision_recall_f1(prediction, reference)[2]
