from __future__ import annotations

from typing import Tuple, Optional, Dict, List
import random

from nlhe_sim.ranges import hole_to_notation, OPEN_RANGES, THREEBET_RANGES, CALL_VS_OPEN
from nlhe_sim.postflop import board_texture

def decide_action(
    rng: random.Random,
    *,
    street: str,
    hole,                       # List[Card]
    legal_actions: List[str],
    to_call: int,
    pot: int,
    current_bet: int,
    committed: int,
    role: str,                  # 'PFR'/'CALLER'
    bucket_dist: Dict[str, float],
    board,                      # List[Card]
    pos: str,                   # 'UTG'/'MP'/'CO'/'BTN'/'SB'/'BB'
    facing_open: bool,
    opener_pos: Optional[str],
    preflop_current: int,
    human_fold_to_cbet: float,  # 0..1 estimate
) -> Tuple[str, int]:
    """
    Returns (action, amount).
    - BET:  amount = bet_to_total_commitment
    - RAISE: amount = raise_to_total_commitment
    """
    # -------- PREFLOP charts --------
    if street == "PREFLOP":
        hand = hole_to_notation(hole[0], hole[1])

        # First-in (approx): no open yet and facing only blind/straddle
        if (not facing_open) and preflop_current <= 3 and to_call <= 3:
            if pos in OPEN_RANGES and hand in OPEN_RANGES[pos]:
                open_to = 9 if pos in ("UTG", "MP") else 8
                # If straddle made current bet bigger, open 3x current
                if preflop_current > 3:
                    open_to = preflop_current * 3
                if "BET" in legal_actions:
                    return ("BET", committed + open_to)
                if "RAISE" in legal_actions:
                    return ("RAISE", open_to)
            return ("CHECK", 0) if "CHECK" in legal_actions else ("FOLD", 0)

        # Facing open/raise
        if facing_open and opener_pos:
            key = (pos, opener_pos)
            three = THREEBET_RANGES.get(key, set())
            call = CALL_VS_OPEN.get(key, set())

            if hand in three and "RAISE" in legal_actions:
                ip = (pos in ("BTN", "CO"))
                raise_to = max(current_bet + 3, current_bet * (3 if ip else 4))
                return ("RAISE", raise_to)

            if hand in call and "CALL" in legal_actions and to_call <= max(12, current_bet):
                return ("CALL", 0)

            if pos == "BB" and "CALL" in legal_actions and to_call <= 6 and rng.random() < 0.15:
                return ("CALL", 0)

            return ("FOLD", 0)

        # fallback
        if "CALL" in legal_actions and to_call <= 3 and rng.random() < 0.3:
            return ("CALL", 0)
        return ("FOLD", 0)

    # -------- POSTFLOP coarse bucket logic --------
    tex = board_texture(board) if len(board) >= 3 else "UNKNOWN"
    strong = bucket_dist.get("STRONG", 0.25)
    medium = bucket_dist.get("MEDIUM", 0.25)
    draw = bucket_dist.get("DRAW", 0.25)
    air = bucket_dist.get("AIR", 0.25)

    bluff_boost = min(0.12, max(0.0, human_fold_to_cbet - 0.50))

    def pick_size(street_: str) -> Tuple[str, float]:
        if street_ == "FLOP":
            sizes = [("SMALL", 0.33), ("MED", 0.66), ("OVER", 1.25)]
        else:
            sizes = [("SMALL", 0.50), ("MED", 1.00), ("OVER", 1.50)]

        if tex in ("WET", "MONOTONE"):
            weights = [0.25, 0.45, 0.30]
        else:
            weights = [0.45, 0.40, 0.15]
        if air > 0.45:
            weights = [weights[0] + 0.15, weights[1] - 0.10, weights[2] - 0.05]

        size = rng.choices([s[0] for s in sizes], weights=weights, k=1)[0]
        frac = dict(sizes)[size]
        return size, frac

    # Facing bet
    if to_call > 0:
        fold_p = max(0.05, 0.55 * air + 0.25 * medium - 0.10 * strong - 0.10 * draw)
        raise_p = min(0.25, 0.20 * strong + 0.10 * draw)
        r = rng.random()
        if "FOLD" in legal_actions and r < fold_p:
            return ("FOLD", 0)
        if "RAISE" in legal_actions and r < fold_p + raise_p:
            _, frac = pick_size(street)
            raise_to = current_bet + max(1, int(pot * frac))
            return ("RAISE", raise_to)
        return ("CALL", 0) if "CALL" in legal_actions else ("FOLD", 0)

    # No bet faced
    cbet_bias = 0.10 if role == "PFR" else -0.05
    base_bet = 0.30 + cbet_bias + 0.35*strong + 0.20*draw - 0.15*air + bluff_boost
    base_bet = max(0.10, min(0.85, base_bet))

    if "BET" in legal_actions and rng.random() < base_bet:
        _, frac = pick_size(street)
        bet_amt = max(1, int(pot * frac))
        return ("BET", committed + bet_amt)

    return ("CHECK", 0)
