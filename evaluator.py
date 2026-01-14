from __future__ import annotations

from typing import List, Optional, Tuple
from collections import Counter
from nlhe_sim.cards import Card, rank_value

# Higher is better
HAND_CATS = {
    "HIGH": 0,
    "PAIR": 1,
    "TWO_PAIR": 2,
    "TRIPS": 3,
    "STRAIGHT": 4,
    "FLUSH": 5,
    "FULL_HOUSE": 6,
    "QUADS": 7,
    "STRAIGHT_FLUSH": 8,
    "ROYAL_FLUSH": 9,  # explicitly included
}


def _is_straight(vals: List[int]) -> Optional[int]:
    """Return high card of straight (5..14), or None."""
    uniq = sorted(set(vals), reverse=True)
    if 14 in uniq:
        uniq.append(1)  # wheel A-5
    for i in range(len(uniq) - 4):
        w = uniq[i : i + 5]
        if w[0] - w[4] == 4 and len(w) == 5:
            return w[0] if w[0] != 1 else 5
    return None


def eval_7(cards: List[Card]) -> Tuple[int, List[int]]:
    """
    Returns comparable hand strength tuple: (category, kickers...)
    Higher is better.
    """
    vals = [rank_value(c.r) for c in cards]
    suits = [c.s for c in cards]
    val_counts = Counter(vals)
    suit_counts = Counter(suits)

    flush_suit = None
    for s, cnt in suit_counts.items():
        if cnt >= 5:
            flush_suit = s
            break

    straight_high = _is_straight(vals)

    # Straight flush / Royal flush
    if flush_suit:
        flush_vals = sorted([rank_value(c.r) for c in cards if c.s == flush_suit], reverse=True)
        sf_high = _is_straight(flush_vals)
        if sf_high:
            if sf_high == 14:
                return (HAND_CATS["ROYAL_FLUSH"], [14])
            return (HAND_CATS["STRAIGHT_FLUSH"], [sf_high])

    # Quads
    quads = [v for v, c in val_counts.items() if c == 4]
    if quads:
        q = max(quads)
        kicker = max([v for v in vals if v != q])
        return (HAND_CATS["QUADS"], [q, kicker])

    # Full house
    trips = sorted([v for v, c in val_counts.items() if c == 3], reverse=True)
    pairs = sorted([v for v, c in val_counts.items() if c == 2], reverse=True)
    if trips and (pairs or len(trips) >= 2):
        t = trips[0]
        p = max(pairs) if pairs else trips[1]
        return (HAND_CATS["FULL_HOUSE"], [t, p])

    # Flush
    if flush_suit:
        flush_vals = sorted([rank_value(c.r) for c in cards if c.s == flush_suit], reverse=True)[:5]
        return (HAND_CATS["FLUSH"], flush_vals)

    # Straight
    if straight_high:
        return (HAND_CATS["STRAIGHT"], [straight_high])

    # Trips
    if trips:
        t = trips[0]
        kickers = sorted([v for v in vals if v != t], reverse=True)[:2]
        return (HAND_CATS["TRIPS"], [t] + kickers)

    # Two pair
    if len(pairs) >= 2:
        p1, p2 = pairs[0], pairs[1]
        kicker = max([v for v in vals if v != p1 and v != p2])
        return (HAND_CATS["TWO_PAIR"], [p1, p2, kicker])

    # One pair
    if len(pairs) == 1:
        p = pairs[0]
        kickers = sorted([v for v in vals if v != p], reverse=True)[:3]
        return (HAND_CATS["PAIR"], [p] + kickers)

    # High card
    return (HAND_CATS["HIGH"], sorted(vals, reverse=True)[:5])
