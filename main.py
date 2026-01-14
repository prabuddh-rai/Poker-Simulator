from __future__ import annotations

from pathlib import Path
from nlhe_sim.engine import NLHESim, ensure_logs_dir

LOG_DIR = Path("logs")
HANDS_FILE = LOG_DIR / "hands.txt"


def main():
    print("NLHE $1/$3 Cash Simulator (Terminal) — You are Seat 0 (YOU)")
    print("Inputs:")
    print("  If to_call = 0:  c | b s | b m | b o | a")
    print("  If to_call > 0:  f | k | r s | r m | r o | a")
    print()

    ensure_logs_dir(LOG_DIR)

    sim = NLHESim(
        n_players=6,
        starting_stack=300,
        rng_seed=42,
        straddles_enabled=True,
        straddle_amount=6,
        allow_restraddle=False,   # NO re-straddle
        rake_pct=0.05,
        rake_cap=5,
    )

    hand_histories = []

    while True:
        if sim.players[0].stack <= 0:
            print("\nYou are out of chips. Session over.")
            break

        res = sim.play_hand()
        print("\n" + res["hand_history_text"])
        hand_histories.append(res["hand_history_text"])

        cmd = input("\n[n]=next hand, [s]=save logs/hands.txt, [q]=quit: ").strip().lower()
        if cmd == "q":
            break
        if cmd == "s":
            HANDS_FILE.write_text("\n".join(hand_histories), encoding="utf-8")
            print(f"Saved: {HANDS_FILE.resolve()}")

    HANDS_FILE.write_text("\n".join(hand_histories), encoding="utf-8")
    print(f"Saved (on exit): {HANDS_FILE.resolve()}")


if __name__ == "__main__":
    main()
