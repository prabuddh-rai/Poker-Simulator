from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Generator, Any
import random
from pathlib import Path

from nlhe_sim.cards import Card, Deck
from nlhe_sim.evaluator import eval_7
from nlhe_sim.postflop import BUCKETS, initial_bucket_dist, board_texture, update_bucket_on_action
from nlhe_sim import bots

HAND_SEPARATOR = "-" * 60  # exact HH format prefix


@dataclass
class Player:
    name: str
    stack: int
    seat: int
    is_human: bool = False
    in_hand: bool = True
    folded: bool = False
    all_in: bool = False

    # committed is CURRENT STREET committed (reset to 0 when street is locked into hs.pots)
    committed: int = 0
    hole: List[Card] = field(default_factory=list)

    role: str = "UNKNOWN"  # PFR / CALLER / UNKNOWN
    bucket_dist: Dict[str, float] = field(default_factory=lambda: {b: 0.25 for b in BUCKETS})

    stats: Dict[str, float] = field(default_factory=lambda: {
        "hands": 0.0,
        "fold_to_cbet": 0.0,
        "opps_fold_to_cbet": 0.0,
    })


@dataclass
class Pot:
    amount: int
    eligible: List[int]  # seats eligible to win this pot


@dataclass
class HandState:
    hand_id: int
    button: int
    sb: int = 1
    bb: int = 3
    rake_pct: float = 0.05
    rake_cap: int = 5

    street: str = "PREFLOP"
    board: List[Card] = field(default_factory=list)

    # Locked pot(s) from completed streets
    pots: List[Pot] = field(default_factory=list)

    took_flop: bool = False

    current_bet: int = 0
    last_raiser: Optional[int] = None
    preflop_open_seat: Optional[int] = None

    action_log: List[Dict] = field(default_factory=list)


class NLHESim:
    """
    Gradio/GUI-friendly step-based NLHE simulator.

    IMPORTANT invariants:
      - hs.pots holds locked pots from previous completed betting rounds.
      - player.committed holds chips committed in the CURRENT street only.
      - At end of a betting round/street, we lock commitments into hs.pots (side-pot aware),
        then reset all player.committed to 0.

    Public step API:
      - start_new_hand()
      - run_until_human_or_hand_end()
      - apply_human_action(action_code)
    """

    def __init__(
        self,
        n_players: int = 6,
        starting_stack: int = 300,
        sb: int = 1,
        bb: int = 3,
        rake_pct: float = 0.05,
        rake_cap: int = 5,
        rng_seed: Optional[int] = None,
        straddles_enabled: bool = True,
        straddle_amount: int = 6,
        allow_restraddle: bool = False,  # retained for clarity; logic posts max 1 straddle
    ):
        if not (2 <= n_players <= 9):
            raise ValueError("n_players must be 2..9")

        self.rng = random.Random(rng_seed)
        self.seed = rng_seed

        self.sb, self.bb = sb, bb
        self.rake_pct, self.rake_cap = rake_pct, rake_cap

        self.straddles_enabled = straddles_enabled
        self.straddle_amount = straddle_amount
        self.allow_restraddle = allow_restraddle

        self.players: List[Player] = [
            Player(name=("YOU" if i == 0 else f"BOT{i}"), stack=starting_stack, seat=i, is_human=(i == 0))
            for i in range(n_players)
        ]

        self.button = 0
        self.hand_id = 1

        # Step-based state
        self._hand_gen: Optional[Generator[Dict[str, Any], None, None]] = None
        self._pending_ctx: Optional[Dict[str, Any]] = None
        self._pending_hs: Optional[HandState] = None
        self._pending_seat: Optional[int] = None
        self._last_error: str = ""

    # -----------------------------
    # Seat helpers
    # -----------------------------
    def _next(self, seat: int) -> int:
        return (seat + 1) % len(self.players)

    def _order_from(self, start: int) -> List[int]:
        out, s = [], start
        for _ in range(len(self.players)):
            out.append(s)
            s = self._next(s)
        return out

    def _seat_to_pos(self, hs: HandState, seat: int) -> str:
        n = len(self.players)
        sb = self._next(hs.button)
        bb = self._next(sb)
        if seat == hs.button:
            return "BTN"
        if seat == sb:
            return "SB"
        if seat == bb:
            return "BB"

        dist = (hs.button - seat) % n
        if n <= 6:
            if dist == 1:
                return "CO"
            if dist == 2:
                return "MP"
            return "UTG"
        else:
            if dist <= 1:
                return "CO"
            if dist <= 3:
                return "MP"
            return "UTG"

    # -----------------------------
    # Logging
    # -----------------------------
    def _log(self, hs: HandState, event: str, **kw):
        hs.action_log.append({"hand_id": hs.hand_id, "street": hs.street, "event": event, **kw})

    def _format_action_stream(self, hs: HandState) -> str:
        """
        Street-by-street action stream (live).
        """
        by_street: Dict[str, List[str]] = {"PREFLOP": [], "FLOP": [], "TURN": [], "RIVER": []}
        for e in hs.action_log:
            st = e.get("street", "PREFLOP")
            ev = e["event"]

            line = None
            if ev == "POST":
                line = f"{e['player']} posts {e['label']} ${e['amount']}"
            elif ev in ("FOLD", "CHECK"):
                line = f"{e['player']} {ev.lower()}"
            elif ev == "CALL":
                line = f"{e['player']} calls ${e['amount']}"
            elif ev == "BET":
                line = f"{e['player']} bets to ${e['to']}"
            elif ev == "RAISE":
                line = f"{e['player']} raises to ${e['to']}"
            elif ev == "ALLIN":
                line = f"{e['player']} all-in to ${e['to']}"
            elif ev == "BOARD":
                line = f"Board: {' '.join(e['cards'])}"
            elif ev == "RAKE":
                line = f"Rake: ${e['amount']}"
            elif ev in ("WIN_NO_SHOWDOWN", "SHOWDOWN"):
                # end-of-hand summary shown elsewhere
                line = None

            if line:
                by_street.setdefault(st, []).append(line)

        out = []
        for st in ["PREFLOP", "FLOP", "TURN", "RIVER"]:
            if by_street.get(st):
                out.append(f"=== {st} ===")
                out.extend(by_street[st])
                out.append("")
        return "\n".join(out).strip()

    # -----------------------------
    # Hand setup
    # -----------------------------
    def _reset_for_hand(self):
        for p in self.players:
            p.in_hand = p.stack > 0
            p.folded = False
            p.all_in = False
            p.committed = 0
            p.hole = []
            p.role = "UNKNOWN"
            p.bucket_dist = {b: 0.25 for b in BUCKETS}

    def _post(self, hs: HandState, seat: int, amt: int, label: str):
        p = self.players[seat]
        x = min(amt, p.stack)
        p.stack -= x
        p.committed += x
        if p.stack == 0:
            p.all_in = True
        self._log(hs, "POST", seat=seat, player=p.name, amount=x, label=label)

    # -----------------------------
    # Pot (LIVE vs LOCKED)
    # -----------------------------
    def _pot_total_locked(self, hs: HandState) -> int:
        return sum(p.amount for p in hs.pots)

    def _pot_total_live(self, hs: HandState) -> int:
        # LIVE pot = locked from prior streets + current-street commitments
        return self._pot_total_locked(hs) + sum(p.committed for p in self.players)

    def _build_side_pots_and_lock_street(self, hs: HandState):
        contributors = [p for p in self.players if p.committed > 0]
        if not contributors:
            return

        eligible_players = [p for p in self.players if p.in_hand and not p.folded]

        levels = sorted(set(p.committed for p in contributors))
        prev = 0
        for lvl in levels:
            pot_amt = 0
            for p in contributors:
                pot_amt += max(0, min(p.committed, lvl) - prev)

            elig = [p.seat for p in eligible_players if p.committed >= lvl]
            hs.pots.append(Pot(amount=pot_amt, eligible=elig))
            prev = lvl

        for p in self.players:
            p.committed = 0

    def _apply_rake_postflop_only(self, hs: HandState) -> int:
        if not hs.took_flop:
            return 0
        pot = self._pot_total_locked(hs)
        rake = int(pot * hs.rake_pct)
        rake = min(rake, hs.rake_cap)
        if hs.pots and rake > 0:
            hs.pots[0].amount = max(0, hs.pots[0].amount - rake)
        self._log(hs, "RAKE", amount=rake)
        return rake

    # -----------------------------
    # Actions
    # -----------------------------
    def _legal_actions(self, hs: HandState, p: Player) -> List[str]:
        if p.folded or not p.in_hand or p.all_in:
            return []
        to_call = hs.current_bet - p.committed
        if to_call <= 0:
            return ["CHECK", "BET", "ALLIN"]
        return ["FOLD", "CALL", "RAISE", "ALLIN"]

    def _size_class(self, bet_to_total: int, pot_before: int) -> str:
        if pot_before <= 0:
            return "SMALL"
        ratio = bet_to_total / pot_before
        if ratio <= 0.45:
            return "SMALL"
        if ratio <= 0.95:
            return "MED"
        return "OVER"

    def _act(self, hs: HandState, seat: int, action: str, amount: int = 0):
        p = self.players[seat]
        to_call = max(0, hs.current_bet - p.committed)

        def maybe_update_bucket(action_name: str, size_class: str):
            if (not p.is_human) and hs.street != "PREFLOP" and len(hs.board) >= 3:
                tex = board_texture(hs.board)
                p.bucket_dist = update_bucket_on_action(p.bucket_dist, action_name, size_class, tex)

        if action == "FOLD":
            p.folded = True
            p.in_hand = False
            self._log(hs, "FOLD", seat=seat, player=p.name)
            return

        if action == "CHECK":
            if to_call != 0:
                raise ValueError("Illegal CHECK")
            self._log(hs, "CHECK", seat=seat, player=p.name)
            maybe_update_bucket("CHECK", "SMALL")
            return

        if action == "CALL":
            pay = min(to_call, p.stack)
            p.stack -= pay
            p.committed += pay
            if p.stack == 0:
                p.all_in = True
            self._log(hs, "CALL", seat=seat, player=p.name, amount=pay)
            maybe_update_bucket("CALL", "SMALL")
            return

        pot_before = self._pot_total_live(hs)

        if action == "BET":
            if to_call != 0:
                raise ValueError("Illegal BET facing a bet")
            bet_to = amount
            pay = min(bet_to - p.committed, p.stack)
            p.stack -= pay
            p.committed += pay
            hs.current_bet = p.committed
            hs.last_raiser = seat
            self._log(hs, "BET", seat=seat, player=p.name, to=hs.current_bet, paid=pay)
            maybe_update_bucket("BET", self._size_class(hs.current_bet, pot_before))
            if hs.street == "PREFLOP" and hs.preflop_open_seat is None and hs.current_bet > self.bb:
                hs.preflop_open_seat = seat
            return

        if action == "RAISE":
            if to_call <= 0:
                raise ValueError("Illegal RAISE")
            raise_to = max(amount, hs.current_bet + 1)
            pay = min(raise_to - p.committed, p.stack)
            p.stack -= pay
            p.committed += pay
            if p.committed > hs.current_bet:
                hs.current_bet = p.committed
                hs.last_raiser = seat
            self._log(hs, "RAISE", seat=seat, player=p.name, to=hs.current_bet, paid=pay)
            maybe_update_bucket("RAISE", self._size_class(hs.current_bet, pot_before))
            if hs.street == "PREFLOP" and hs.preflop_open_seat is None and hs.current_bet > self.bb:
                hs.preflop_open_seat = seat
            return

        if action == "ALLIN":
            pay = p.stack
            p.stack = 0
            p.committed += pay
            p.all_in = True
            if p.committed > hs.current_bet:
                hs.current_bet = p.committed
                hs.last_raiser = seat
            self._log(hs, "ALLIN", seat=seat, player=p.name, to=p.committed, paid=pay)
            maybe_update_bucket("ALLIN", self._size_class(hs.current_bet, pot_before))
            if hs.street == "PREFLOP" and hs.preflop_open_seat is None and hs.current_bet > self.bb:
                hs.preflop_open_seat = seat
            return

        raise ValueError("Unknown action")

    # -----------------------------
    # Decisions (for UI)
    # -----------------------------
    def _decision_context(self, hs: HandState, seat: int) -> Dict[str, Any]:
        p = self.players[seat]
        legal = self._legal_actions(hs, p)
        to_call = max(0, hs.current_bet - p.committed)
        pot_live = self._pot_total_live(hs)

        seats = []
        for pl in self.players:
            seats.append({
                "seat": pl.seat,
                "name": pl.name,
                "stack": pl.stack,
                "committed": pl.committed,
                "folded": pl.folded,
                "all_in": pl.all_in,
                "in_hand": pl.in_hand,
                "is_human": pl.is_human,
            })

        return {
            "type": "HUMAN_DECISION",
            "hand_id": hs.hand_id,
            "street": hs.street,
            "board_cards": [str(c) for c in hs.board],
            "board": " ".join(str(c) for c in hs.board),
            "pot": pot_live,
            "current_bet": hs.current_bet,
            "to_call": to_call,
            "legal_actions": legal,
            "you_stack": self.players[0].stack,
            "you_committed": self.players[0].committed,
            "you_hole_cards": [str(c) for c in self.players[0].hole],
            "you_hole": " ".join(str(c) for c in self.players[0].hole),
            "seats": seats,
            "action_stream": self._format_action_stream(hs),
            "last_error": self._last_error,
        }

    def _action_code_to_engine_action(self, hs: HandState, seat: int, action_code: str) -> Tuple[str, int]:
        p = self.players[seat]
        to_call = max(0, hs.current_bet - p.committed)
        pot = self._pot_total_live(hs)

        def frac_for(code: str) -> float:
            if hs.street == "FLOP":
                mp = {"S": 0.33, "M": 0.66, "O": 1.25}
            else:
                mp = {"S": 0.50, "M": 1.00, "O": 1.50}
            return mp[code]

        if action_code == "CHECK":
            return ("CHECK", 0)
        if action_code == "CALL":
            return ("CALL", 0)
        if action_code == "FOLD":
            return ("FOLD", 0)
        if action_code == "ALLIN":
            return ("ALLIN", 0)

        if action_code.startswith("BET_"):
            if to_call > 0:
                raise ValueError("Cannot BET when facing a bet (use CALL/RAISE).")
            s = action_code.split("_", 1)[1]  # S/M/O
            amt = max(1, int(pot * frac_for(s)))
            bet_to = p.committed + min(amt, p.stack)
            return ("BET", bet_to)

        if action_code.startswith("RAISE_"):
            if to_call <= 0:
                raise ValueError("Cannot RAISE when not facing a bet (use BET).")
            s = action_code.split("_", 1)[1]
            amt = max(1, int(pot * frac_for(s)))
            raise_to = hs.current_bet + min(amt, p.stack)
            raise_to = max(raise_to, hs.current_bet + 1)
            return ("RAISE", raise_to)

        raise ValueError(f"Unknown action_code: {action_code}")

    # -----------------------------
    # Betting generator (FIXED check-around termination)
    # -----------------------------
    def _betting_round_gen(self, hs: HandState, first_to_act: int) -> Generator[Dict[str, Any], None, None]:
        order = self._order_from(first_to_act)

        def active_nonallin():
            return [pl for pl in self.players if pl.in_hand and not pl.folded and not pl.all_in]

        def everyone_matched() -> bool:
            for q in active_nonallin():
                if q.committed != hs.current_bet:
                    return False
            return True

        # Track who has acted since last aggression. When everyone has acted AND everyone matched, round ends.
        acted_since_aggr: set[int] = set()

        while True:
            if len([pl for pl in self.players if pl.in_hand and not pl.folded]) <= 1:
                return
            if len(active_nonallin()) <= 1:
                return

            progressed = False
            for seat in order:
                p = self.players[seat]
                if p.folded or not p.in_hand or p.all_in:
                    continue

                # If this seat already acted since last aggression and nothing changed, we can skip them
                # but we still need a robust end condition (below).
                if seat in acted_since_aggr and everyone_matched():
                    # do nothing; allow loop to hit end condition after the for-loop
                    continue

                if p.is_human:
                    self._pending_ctx = self._decision_context(hs, seat)
                    self._pending_hs = hs
                    self._pending_seat = seat
                    yield self._pending_ctx
                    progressed = True
                    # After UI action is applied, execution resumes here and we continue to next seats
                    acted_since_aggr.add(seat)
                    continue

                legal = self._legal_actions(hs, p)
                to_call = max(0, hs.current_bet - p.committed)
                pot_live = self._pot_total_live(hs)

                pos = self._seat_to_pos(hs, seat)
                facing_open = hs.preflop_open_seat is not None and hs.preflop_open_seat != seat
                opener_pos = self._seat_to_pos(hs, hs.preflop_open_seat) if hs.preflop_open_seat is not None else None

                you = self.players[0]
                human_fold_to_cbet = (you.stats["fold_to_cbet"] / max(1.0, you.stats["opps_fold_to_cbet"])) \
                    if you.stats["opps_fold_to_cbet"] > 0 else 0.5

                action, amt = bots.decide_action(
                    self.rng,
                    street=hs.street,
                    hole=p.hole,
                    legal_actions=legal,
                    to_call=to_call,
                    pot=pot_live,
                    current_bet=hs.current_bet,
                    committed=p.committed,
                    role=p.role,
                    bucket_dist=p.bucket_dist,
                    board=hs.board,
                    pos=pos,
                    facing_open=facing_open,
                    opener_pos=opener_pos,
                    preflop_current=hs.current_bet,
                    human_fold_to_cbet=human_fold_to_cbet,
                )

                if action not in self._legal_actions(hs, p):
                    if "CALL" in self._legal_actions(hs, p):
                        action, amt = ("CALL", 0)
                    else:
                        action, amt = ("CHECK", 0)

                # Apply action
                prev_bet = hs.current_bet
                self._act(hs, seat, action, amt)
                progressed = True

                # Update acted tracking
                acted_since_aggr.add(seat)

                # If aggression increased current bet, reset acted set (new “round”)
                if hs.current_bet > prev_bet:
                    acted_since_aggr = {seat}

            # End-of-round condition:
            # Everyone who can act has acted since last aggression AND everyone matched the current bet.
            active_seats = {pl.seat for pl in active_nonallin()}
            if active_seats.issubset(acted_since_aggr) and everyone_matched():
                return

            if not progressed:
                return

    # -----------------------------
    # Dealing
    # -----------------------------
    def _deal_board(self, deck: Deck, n: int) -> List[Card]:
        _ = deck.deal()  # burn
        return [deck.deal() for _ in range(n)]

    def _award_single_winner_if_any(self, hs: HandState) -> Optional[Dict[str, Any]]:
        remaining = [p for p in self.players if p.in_hand and not p.folded]
        if len(remaining) != 1:
            return None
        winner = remaining[0]

        self._build_side_pots_and_lock_street(hs)

        if hs.street != "PREFLOP":
            self._apply_rake_postflop_only(hs)

        amt = self._pot_total_locked(hs)
        winner.stack += amt
        self._log(hs, "WIN_NO_SHOWDOWN", seat=winner.seat, player=winner.name, amount=amt)
        return self._finalize(hs)

    # -----------------------------
    # Hand generator
    # -----------------------------
    def _hand_generator(self) -> Generator[Dict[str, Any], None, None]:
        self._reset_for_hand()
        hs = HandState(
            hand_id=self.hand_id,
            button=self.button,
            sb=self.sb,
            bb=self.bb,
            rake_pct=self.rake_pct,
            rake_cap=self.rake_cap,
        )
        self._log(hs, "HAND_START", button=self.button, seed=self.seed)
        deck = Deck(self.rng)

        # deal hole
        for _ in range(2):
            for seat in self._order_from(self._next(self.button)):
                p = self.players[seat]
                if p.stack > 0:
                    p.hole.append(deck.deal())

        # blinds
        sb_seat = self._next(self.button)
        bb_seat = self._next(sb_seat)
        self._post(hs, sb_seat, self.sb, "SB")
        self._post(hs, bb_seat, self.bb, "BB")

        # preflop init
        hs.street = "PREFLOP"
        hs.current_bet = self.bb
        hs.last_raiser = bb_seat
        hs.preflop_open_seat = None
        last_for_action = bb_seat

        # straddle: single max
        if self.straddles_enabled:
            allowed = [p.seat for p in self.players if p.seat not in (sb_seat, bb_seat) and p.stack > 0]
            if allowed and self.rng.random() < 0.45:
                s1 = self.rng.choice(allowed)
                self._post(hs, s1, self.straddle_amount, "STRADDLE")
                hs.current_bet = max(hs.current_bet, self.players[s1].committed)
                hs.last_raiser = s1
                last_for_action = s1
                self._log(hs, "STRADDLE_ON", seat=s1, amount=self.straddle_amount)

        # PREFLOP betting
        first_to_act = self._next(last_for_action)
        yield from self._betting_round_gen(hs, first_to_act)

        # lock preflop money
        self._build_side_pots_and_lock_street(hs)

        # roles after preflop
        if hs.last_raiser is not None:
            self.players[hs.last_raiser].role = "PFR"
        for p in self.players:
            if p.in_hand and not p.folded and p.role != "PFR":
                p.role = "CALLER"

        maybe = self._award_single_winner_if_any(hs)
        if maybe:
            yield {"type": "HAND_END", **maybe}
            return

        # FLOP/TURN/RIVER
        for street, n in [("FLOP", 3), ("TURN", 1), ("RIVER", 1)]:
            hs.street = street
            if street == "FLOP":
                hs.took_flop = True

            hs.board += self._deal_board(deck, n)
            self._log(hs, "BOARD", cards=[str(c) for c in hs.board])

            if street == "FLOP":
                tex = board_texture(hs.board)
                for p in self.players:
                    if p.in_hand and not p.folded and not p.is_human:
                        p.bucket_dist = initial_bucket_dist(p.role, tex)

            hs.current_bet = 0
            hs.last_raiser = None

            start = self._next(hs.button)
            for _ in range(len(self.players)):
                q = self.players[start]
                if q.in_hand and not q.folded and not q.all_in:
                    break
                start = self._next(start)

            yield from self._betting_round_gen(hs, start)

            self._build_side_pots_and_lock_street(hs)

            maybe = self._award_single_winner_if_any(hs)
            if maybe:
                yield {"type": "HAND_END", **maybe}
                return

        # SHOWDOWN
        self._apply_rake_postflop_only(hs)
        alive = [p for p in self.players if p.in_hand and not p.folded]
        strengths = {p.seat: eval_7(p.hole + hs.board) for p in alive}

        best = None
        winners: List[Player] = []
        for p in alive:
            h = strengths[p.seat]
            if best is None or h > best:
                best = h
                winners = [p]
            elif h == best:
                winners.append(p)

        pot = self._pot_total_locked(hs)
        share = pot // len(winners)
        rem = pot - share * len(winners)
        for i, w in enumerate(winners):
            w.stack += share + (1 if i < rem else 0)

        self._log(
            hs,
            "SHOWDOWN",
            board=[str(c) for c in hs.board],
            hands={self.players[s].name: [str(c) for c in self.players[s].hole] for s in strengths.keys()},
            winners=[w.name for w in winners],
            pot=pot,
        )

        self.players[0].stats["hands"] += 1.0
        final = self._finalize(hs)
        yield {"type": "HAND_END", **final}
        return

    # -----------------------------
    # Public step API
    # -----------------------------
    def start_new_hand(self):
        self._hand_gen = self._hand_generator()
        self._pending_ctx = None
        self._pending_hs = None
        self._pending_seat = None
        self._last_error = ""

    def run_until_human_or_hand_end(self) -> Dict[str, Any]:
        if self._hand_gen is None:
            self.start_new_hand()

        while True:
            try:
                msg = next(self._hand_gen)
            except StopIteration:
                self._hand_gen = None
                return {"type": "HAND_END", "hand_history_text": "", "hand_id": None, "last_error": "Hand ended unexpectedly."}

            if msg.get("type") == "HUMAN_DECISION":
                return msg
            if msg.get("type") == "HAND_END":
                self._hand_gen = None
                return msg

    def apply_human_action(self, action_code: str) -> Dict[str, Any]:
        if self._pending_ctx is None or self._pending_hs is None or self._pending_seat is None:
            self._last_error = "No pending human decision. Click Next/Continue first."
            return self.run_until_human_or_hand_end()

        hs = self._pending_hs
        seat = self._pending_seat
        p = self.players[seat]

        try:
            legal = self._legal_actions(hs, p)
            action, amt = self._action_code_to_engine_action(hs, seat, action_code)
            if action not in legal:
                raise ValueError(f"Illegal action now. Legal: {legal}")

            self._act(hs, seat, action, amt)
            self._last_error = ""
        except Exception as e:
            self._last_error = str(e)
            self._pending_ctx = self._decision_context(hs, seat)
            self._pending_ctx["last_error"] = self._last_error
            return self._pending_ctx

        self._pending_ctx = None
        self._pending_hs = None
        self._pending_seat = None
        return self.run_until_human_or_hand_end()

    # -----------------------------
    # Hand history formatting
    # -----------------------------
    def _finalize(self, hs: HandState) -> Dict[str, Any]:
        lines: List[str] = []
        lines.append(f"{HAND_SEPARATOR}Hand #{hs.hand_id} - NLHE ${hs.sb}/${hs.bb} (seed={self.seed})")

        lines.append(f"Button: Seat {hs.button} ({self.players[hs.button].name})")
        if hs.board:
            lines.append(f"Board: {' '.join(str(c) for c in hs.board)}")

        for e in hs.action_log:
            ev = e["event"]
            if ev == "POST":
                lines.append(f"{e['player']} posts {e['label']} ${e['amount']}")
            elif ev in ("FOLD", "CHECK"):
                lines.append(f"{e['player']} {ev.lower()}")
            elif ev == "CALL":
                lines.append(f"{e['player']} calls ${e['amount']}")
            elif ev == "BET":
                lines.append(f"{e['player']} bets to ${e['to']} (paid ${e['paid']})")
            elif ev == "RAISE":
                lines.append(f"{e['player']} raises to ${e['to']} (paid ${e['paid']})")
            elif ev == "ALLIN":
                lines.append(f"{e['player']} all-in to ${e['to']} (paid ${e['paid']})")
            elif ev == "BOARD":
                lines.append(f"Board now: {' '.join(e['cards'])}")
            elif ev == "RAKE":
                lines.append(f"Rake: ${e['amount']}")
            elif ev == "WIN_NO_SHOWDOWN":
                lines.append(f"{e['player']} wins ${e['amount']} (no showdown)")
            elif ev == "SHOWDOWN":
                lines.append(f"Showdown pot: ${e['pot']} | Winners: {', '.join(e['winners'])}")
                for pl, cards in e["hands"].items():
                    lines.append(f"{pl} shows {' '.join(cards)}")

        self.hand_id += 1
        self.button = self._next(self.button)

        return {
            "hand_id": hs.hand_id,
            "button": hs.button,
            "board": [str(c) for c in hs.board],
            "action_log": hs.action_log,
            "hand_history_text": "\n".join(lines),
        }


def ensure_logs_dir(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
