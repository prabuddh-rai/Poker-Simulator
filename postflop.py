from __future__ import annotations

from typing import Dict, List
from nlhe_sim.cards import Card, rank_value

BUCKETS = ["STRONG", "MEDIUM", "DRAW", "AIR"]


def board_texture(board: List[Card]) -> str:
    """Coarse: DRY / WET / MONOTONE / PAIRED."""
    if len(board) < 3:
        return "UNKNOWN"
    suits = [c.s for c in board[:3]]
    ranks = sorted([rank_value(c.r) for c in board[:3]], reverse=True)

    paired = len(set(ranks)) < 3
    monotone = len(set(suits)) == 1
    gaps = max(ranks) - min(ranks)
    connected = gaps <= 4
    two_tone = len(set(suits)) == 2

    if monotone:
        return "MONOTONE"
    if paired:
        return "PAIRED"
    if connected or two_tone:
        return "WET"
    return "DRY"


def initial_bucket_dist(role: str, texture: str) -> Dict[str, float]:
    """role: PFR or CALLER"""
    if role == "PFR":
        base = {"STRONG": 0.18, "MEDIUM": 0.32, "DRAW": 0.20, "AIR": 0.30}
    else:
        base = {"STRONG": 0.16, "MEDIUM": 0.30, "DRAW": 0.22, "AIR": 0.32}

    if texture == "DRY":
        base["AIR"] += 0.05
        base["DRAW"] -= 0.05
    elif texture == "WET":
        base["DRAW"] += 0.06
        base["AIR"] -= 0.06
    elif texture == "MONOTONE":
        base["DRAW"] += 0.05
        base["MEDIUM"] -= 0.03
        base["AIR"] -= 0.02
    elif texture == "PAIRED":
        base["STRONG"] += 0.03
        base["AIR"] -= 0.03

    s = sum(base.values())
    return {k: max(0.0, v / s) for k, v in base.items()}


def update_bucket_on_action(dist: Dict[str, float], action: str, size_class: str, texture: str) -> Dict[str, float]:
    """Update coarse distribution based on action + sizing + texture."""
    d = dist.copy()

    def bump(up: str, down: str, amt: float):
        d[up] += amt
        d[down] = max(0.0, d[down] - amt)

    if action in ("BET", "RAISE", "ALLIN"):
        if size_class == "SMALL":
            bump("MEDIUM", "AIR", 0.06)
            bump("DRAW", "AIR", 0.04 if texture in ("WET", "MONOTONE") else 0.02)
        elif size_class == "MED":
            bump("STRONG", "AIR", 0.06)
            bump("DRAW", "AIR", 0.06 if texture in ("WET", "MONOTONE") else 0.03)
        else:  # OVER
            bump("STRONG", "AIR", 0.10)
            bump("DRAW", "MEDIUM", 0.05 if texture in ("WET", "MONOTONE") else 0.02)

    elif action == "CALL":
        bump("MEDIUM", "AIR", 0.05)
        bump("DRAW", "AIR", 0.05 if texture in ("WET", "MONOTONE") else 0.02)

    elif action == "CHECK":
        bump("AIR", "STRONG", 0.05)

    s = sum(d.values())
    if s <= 0:
        return {"STRONG": 0.25, "MEDIUM": 0.25, "DRAW": 0.25, "AIR": 0.25}
    return {k: v / s for k, v in d.items()}
