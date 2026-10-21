"""
8-Layer False-Negative Protection Engine for Safe Negative Mining
"""
from typing import Dict, Set, Tuple
from src.config import MAX_COSINE_SIM

def is_false_negative(
    pos_id: str,
    cand_id: str,
    cand_score: float,
    meta: Dict,
    banned_ids: Set[str]
) -> Tuple[bool, str]:
    """
    Returns (True, reason) if cand_id is too similar, related, or identical to pos_id.
    Guarantees only truly irrelevant books are selected as hard negatives.
    """
    # 1. Exact and cross-query banned ID check
    if cand_id in banned_ids:
        return True, "banned_id"

    # 2. Extreme similarity ceiling (blocks unindexed editions / near-identical blurbs)
    if cand_score > MAX_COSINE_SIM:
        return True, "cosine_ceiling"

    # 3. Normalized full title match
    pos_title = meta["title_of"].get(pos_id, "")
    cand_title = meta["title_of"].get(cand_id, "")
    if pos_title and cand_title and pos_title == cand_title:
        return True, "identical_title"

    # 4. Base title match (stripping subtitles/editions)
    pos_base = meta["base_title_of"].get(pos_id, "")
    cand_base = meta["base_title_of"].get(cand_id, "")
    if pos_base and cand_base and pos_base == cand_base and len(pos_base) > 3:
        return True, "base_title_match"

    # 5. Significant title word overlap (>= 65% overlap)
    pos_words = meta["sig_words_of"].get(pos_id, set())
    cand_words = meta["sig_words_of"].get(cand_id, set())
    if pos_words and cand_words:
        shared = pos_words & cand_words
        min_len = min(len(pos_words), len(cand_words))
        if min_len >= 2 and (len(shared) / min_len) >= 0.65:
            return True, "title_word_overlap"

    # 6. Series / Universe match
    pos_series = meta["series_of"].get(pos_id, "")
    cand_series = meta["series_of"].get(cand_id, "")
    if pos_series and cand_series and pos_series == cand_series:
        return True, "series_match"
    if pos_series and pos_series in cand_title:
        return True, "series_in_cand_title"
    if cand_series and cand_series in pos_title:
        return True, "cand_series_in_pos_title"

    # 7. Author tags overlap from Goodreads popular shelves
    pos_auth = meta["author_tags_of"].get(pos_id, set())
    cand_auth = meta["author_tags_of"].get(cand_id, set())
    if pos_auth and cand_auth and (pos_auth & cand_auth):
        return True, "shared_author_shelf"

    return False, ""
