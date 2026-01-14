from __future__ import annotations

from dataclasses import dataclass
import random
from typing import List

RANKS = "23456789TJQKA"
SUITS = "shdc"  # spades hearts diamonds clubs
RANK_TO_VAL = {r: i + 2 for i, r in enumerate(RANKS)}


def rank_value(r: str) -> int:
    return RANK_TO_VAL[r]


@dataclass(frozen=True)
class Card:
    r: str
    s: str

    def __str__(self) -> str:
        return f"{self.r}{self.s}"


class Deck:
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.cards: List[Card] = [Card(r, s) for r in RANKS for s in SUITS]
        self.rng.shuffle(self.cards)

    def deal(self) -> Card:
        return self.cards.pop()
