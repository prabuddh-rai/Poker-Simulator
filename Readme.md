# NLHE $1/$3 Poker Simulator (Gradio)

A local, UI-driven No-Limit Texas Hold’em simulator for practicing **$1/$3 cash games** with:
- **5% rake (postflop only), capped at $5**
- **Straddles enabled** (no re-straddles)
- Bots with basic postflop range tracking + sizing logic
- A **Gradio desktop-friendly UI** (runs locally in your browser)
- Automatic **hand history logging** per session

> ⚠️ This is a training simulator, not a GTO solver. Bots are designed to be “strong enough to practice” and are intended to be improved by contributors.

---

## Features

- Play hands interactively via a UI (no terminal `input()`).
- Buttons only appear when actions are legal (Fold/Call/Raise vs Check/Bet).
- Displays:
  - Board cards + your hole cards (graphical)
  - Pot/to-call/current bet
  - Street-by-street action stream
- Logs every completed hand automatically.

---

## Quickstart (Windows / Mac / Linux)

### 1) Clone the repo
```bash
git clone <GITHUB_REPO_URL>
cd Poker_Simulator 

### License
This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
