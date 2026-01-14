from __future__ import annotations

from typing import Dict, List, Set, Tuple
from nlhe_sim.cards import Card, rank_value


def hole_to_notation(c1: Card, c2: Card) -> str:
    """Return e.g. 'AKs', 'QJo', '77'."""
    r1, r2 = c1.r, c2.r
    if rank_value(r2) > rank_value(r1):
        c1, c2 = c2, c1
        r1, r2 = r2, r1
    if r1 == r2:
        return f"{r1}{r2}"
    suited = (c1.s == c2.s)
    return f"{r1}{r2}{'s' if suited else 'o'}"


def expand_range(tokens: List[str]) -> Set[str]:
    return set(tokens)


# Simplified, rake-aware-ish 1/3 charts (edit freely)
OPEN_RANGES: Dict[str, Set[str]] = {
    "UTG": expand_range(["AA","KK","QQ","JJ","TT","99","88",
                        "AKs","AQs","AJs","ATs","KQs","KJs","QJs","JTs",
                        "AKo","AQo"]),
    "MP":  expand_range(["AA","KK","QQ","JJ","TT","99","88","77","66",
                        "AKs","AQs","AJs","ATs","A9s","KQs","KJs","KTs","QJs","QTs","JTs","T9s",
                        "AKo","AQo","AJo","KQo"]),
    "CO":  expand_range(["AA","KK","QQ","JJ","TT","99","88","77","66","55","44",
                        "AKs","AQs","AJs","ATs","A9s","A8s","A7s","KQs","KJs","KTs","K9s",
                        "QJs","QTs","Q9s","JTs","J9s","T9s","98s","87s",
                        "AKo","AQo","AJo","ATo","KQo","KJo","QJo"]),
    "BTN": expand_range(["AA","KK","QQ","JJ","TT","99","88","77","66","55","44","33","22",
                        "AKs","AQs","AJs","ATs","A9s","A8s","A7s","A6s","A5s","A4s","A3s","A2s",
                        "KQs","KJs","KTs","K9s","K8s","QJs","QTs","Q9s","Q8s",
                        "JTs","J9s","J8s","T9s","T8s","98s","87s","76s",
                        "AKo","AQo","AJo","ATo","KQo","KJo","QJo","JTo"]),
    "SB":  expand_range(["AA","KK","QQ","JJ","TT","99","88","77","66","55","44","33","22",
                        "AKs","AQs","AJs","ATs","A9s","A8s","A7s","A6s","A5s","A4s","A3s","A2s",
                        "KQs","KJs","KTs","K9s","QJs","QTs","JTs","T9s","98s",
                        "AKo","AQo","AJo","ATo","KQo","KJo","QJo"]),
}

THREEBET_RANGES: Dict[Tuple[str, str], Set[str]] = {
    ("BTN","CO"): expand_range(["AA","KK","QQ","JJ","TT","AKs","AQs","AJs","KQs","AKo","AQo","A5s","A4s"]),
    ("BTN","MP"): expand_range(["AA","KK","QQ","JJ","TT","AKs","AQs","AKo","AQo","A5s"]),
    ("BTN","UTG"):expand_range(["AA","KK","QQ","JJ","AKs","AQs","AKo","A5s"]),
    ("CO","MP"):  expand_range(["AA","KK","QQ","JJ","TT","AKs","AQs","AKo","A5s"]),
    ("CO","UTG"): expand_range(["AA","KK","QQ","JJ","AKs","AKo","A5s"]),
    ("SB","BTN"): expand_range(["AA","KK","QQ","JJ","TT","99","AKs","AQs","AJs","KQs","AKo","AQo","A5s","KTs"]),
    ("SB","CO"):  expand_range(["AA","KK","QQ","JJ","TT","AKs","AQs","KQs","AKo","AQo","A5s"]),
    ("SB","MP"):  expand_range(["AA","KK","QQ","JJ","AKs","AQs","AKo","A5s"]),
    ("BB","BTN"): expand_range(["AA","KK","QQ","JJ","TT","99","AKs","AQs","AJs","KQs","AKo","AQo","A5s","KTs"]),
    ("BB","CO"):  expand_range(["AA","KK","QQ","JJ","TT","AKs","AQs","KQs","AKo","AQo","A5s"]),
}

CALL_VS_OPEN: Dict[Tuple[str, str], Set[str]] = {
    ("BTN","CO"): expand_range(["99","88","77","66","55",
                               "AQs","AJs","ATs","KQs","KJs","QJs","JTs","T9s","98s",
                               "AQo","AJo","KQo"]),
    ("BTN","MP"): expand_range(["99","88","77","AQs","AJs","KQs","QJs","JTs","T9s","AQo"]),
    ("CO","MP"):  expand_range(["99","88","77","AQs","AJs","KQs","QJs","JTs","T9s","AQo"]),
    ("CO","UTG"): expand_range(["99","88","AQs","AJs","KQs","QJs","JTs"]),
    ("BB","BTN"): expand_range(["99","88","77","66","55","44","33","22",
                               "AJs","ATs","A9s","A8s","KQs","KJs","KTs","QJs","QTs","JTs","T9s","98s","87s",
                               "AJo","ATo","KQo","QJo","JTo"]),
    ("BB","CO"):  expand_range(["99","88","77","66","55","44","33","22",
                               "ATs","A9s","A8s","KQs","KJs","QJs","JTs","T9s","98s","87s",
                               "AJo","KQo"]),
    ("SB","BTN"): expand_range(["TT","99","88","77","66","55",
                               "AQs","AJs","ATs","KQs","KJs","QJs","JTs","T9s","98s",
                               "AQo","AJo","KQo"]),
}
