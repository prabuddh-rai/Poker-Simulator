from __future__ import annotations

import gradio as gr
from pathlib import Path
from datetime import datetime

from nlhe_sim.engine import NLHESim, ensure_logs_dir

LOG_DIR = Path("logs")
ensure_logs_dir(LOG_DIR)

UPLOADED_HH_FILE = LOG_DIR / "uploaded_hands.txt"


def session_path():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return LOG_DIR / f"session_{ts}.txt"


# ---------- Card rendering (HTML) ----------
SUIT_MAP = {"s": "♠", "h": "♥", "d": "♦", "c": "♣"}


def _parse_card(card_str: str):
    if not card_str or len(card_str) < 2:
        return None, None
    rank = card_str[:-1]
    suit = card_str[-1].lower()
    return rank, suit


def card_html(card_str: str) -> str:
    rank, suit = _parse_card(card_str)
    if rank is None:
        return ""
    sym = SUIT_MAP.get(suit, "?")
    is_red = suit in ("h", "d")
    color = "#c0392b" if is_red else "#111"
    return f"""
    <span style="
      display:inline-block;
      border:1px solid #ddd;
      border-radius:10px;
      padding:8px 10px;
      margin-right:8px;
      font-family: ui-sans-serif, system-ui, Segoe UI;
      font-weight:700;
      font-size:18px;
      background:#fff;
      color:{color};
      box-shadow: 0 1px 2px rgba(0,0,0,0.06);
      min-width:44px;
      text-align:center;
    ">{rank}{sym}</span>
    """


def cards_row_html(title: str, cards: list[str]) -> str:
    if not cards:
        return f"<div><b>{title}</b>: (none)</div>"
    row = "".join(card_html(c) for c in cards)
    return f"<div style='margin:6px 0;'><b>{title}</b>: {row}</div>"


# ---------- Simulator ----------
def make_sim(seed: int, n_players: int, starting_stack: int):
    return NLHESim(
        n_players=n_players,
        starting_stack=starting_stack,
        rng_seed=int(seed),
        straddles_enabled=True,
        straddle_amount=6,
        allow_restraddle=False,
        rake_pct=0.05,
        rake_cap=5,
    )


# ---------- Button updates (use gr.update for compatibility) ----------
def button_updates(ctx: dict | None):
    """
    Return update objects for each button, showing only actions that are legal.
    Uses gr.update(...) which works across Gradio versions.
    """
    def upd(vis=False, inter=False):
        return gr.update(visible=vis, interactive=inter)

    defaults = {
        "check": upd(),
        "call": upd(),
        "fold": upd(),
        "allin": upd(),
        "b_s": upd(),
        "b_m": upd(),
        "b_o": upd(),
        "r_s": upd(),
        "r_m": upd(),
        "r_o": upd(),
    }

    if not ctx or ctx.get("type") != "HUMAN_DECISION":
        return (
            defaults["check"], defaults["call"], defaults["fold"], defaults["allin"],
            defaults["b_s"], defaults["b_m"], defaults["b_o"],
            defaults["r_s"], defaults["r_m"], defaults["r_o"],
        )

    legal = set(ctx.get("legal_actions", []))
    to_call = int(ctx.get("to_call", 0))

    defaults["allin"] = upd(vis=("ALLIN" in legal), inter=("ALLIN" in legal))

    if to_call <= 0:
        defaults["check"] = upd(vis=("CHECK" in legal), inter=("CHECK" in legal))
        show_bet = ("BET" in legal)
        defaults["b_s"] = upd(vis=show_bet, inter=show_bet)
        defaults["b_m"] = upd(vis=show_bet, inter=show_bet)
        defaults["b_o"] = upd(vis=show_bet, inter=show_bet)
    else:
        defaults["fold"] = upd(vis=("FOLD" in legal), inter=("FOLD" in legal))
        defaults["call"] = upd(vis=("CALL" in legal), inter=("CALL" in legal))
        show_raise = ("RAISE" in legal)
        defaults["r_s"] = upd(vis=show_raise, inter=show_raise)
        defaults["r_m"] = upd(vis=show_raise, inter=show_raise)
        defaults["r_o"] = upd(vis=show_raise, inter=show_raise)

    return (
        defaults["check"], defaults["call"], defaults["fold"], defaults["allin"],
        defaults["b_s"], defaults["b_m"], defaults["b_o"],
        defaults["r_s"], defaults["r_m"], defaults["r_o"],
    )


# ---------- UI rendering ----------
def render_summary(sim: NLHESim, ctx: dict | None) -> str:
    you = sim.players[0]
    if not ctx or ctx.get("type") != "HUMAN_DECISION":
        return f"Click Next/Continue to begin.\nYOU stack: ${you.stack}"

    lines = []
    lines.append(f"Hand #{ctx['hand_id']} | Street: {ctx['street']}")
    lines.append(f"Pot (live): ${ctx.get('pot', 0)} | To call: ${ctx.get('to_call', 0)} | Current bet: ${ctx.get('current_bet', 0)}")
    if ctx.get("last_error"):
        lines.append(f"ERROR: {ctx['last_error']}")
    lines.append("")
    lines.append("Seats:")
    for p in sim.players:
        flags = []
        if p.seat == sim.button:
            flags.append("BTN")
        if p.folded:
            flags.append("FOLDED")
        if p.all_in:
            flags.append("ALLIN")
        lines.append(f"  Seat {p.seat}: {p.name} | stack=${p.stack} | committed=${p.committed}" + (f" [{', '.join(flags)}]" if flags else ""))

    return "\n".join(lines)


def render_cards_html(ctx: dict | None) -> str:
    if not ctx or ctx.get("type") != "HUMAN_DECISION":
        return "<div style='color:#666;'>No active decision yet. Click Next/Continue.</div>"
    board = ctx.get("board_cards", [])
    hole = ctx.get("you_hole_cards", [])
    html = "<div style='padding:6px 2px;'>"
    html += cards_row_html("Board", board)
    html += cards_row_html("Your hand", hole)
    html += "</div>"
    return html


# ---------- Session / flow ----------
def start_session(seed, n_players, starting_stack):
    sim = make_sim(seed, n_players, starting_stack)
    sim.start_new_hand()
    hh_list = []
    sess_file = str(session_path())
    status = f"Session started. Auto-saving to: {sess_file}"

    ctx = None
    summary = render_summary(sim, ctx)
    cards = render_cards_html(ctx)
    action_stream = ""
    last_hh = ""

    btns = button_updates(ctx)
    return (sim, hh_list, sess_file, status, summary, cards, action_stream, last_hh, *btns)


def next_or_continue(sim: NLHESim, hh_list, sess_file):
    if sim is None:
        ctx = None
        btns = button_updates(ctx)
        return (None, hh_list, sess_file, "Click Start Session first.", "", "", "", "", *btns)

    msg = sim.run_until_human_or_hand_end()

    if msg["type"] == "HUMAN_DECISION":
        status = f"Your turn. Legal: {', '.join(msg['legal_actions'])}"
        summary = render_summary(sim, msg)
        cards = render_cards_html(msg)
        action_stream = msg.get("action_stream", "")
        last_hh = ""
        btns = button_updates(msg)
        return (sim, hh_list, sess_file, status, summary, cards, action_stream, last_hh, *btns)

    if msg["type"] == "HAND_END":
        hh = msg.get("hand_history_text", "")
        if hh:
            hh_list.append(hh)
            Path(sess_file).write_text("\n".join(hh_list), encoding="utf-8")

        status = f"Hand finished. Hands logged: {len(hh_list)} (auto-saved)"
        ctx = None
        summary = render_summary(sim, ctx)
        cards = render_cards_html(ctx)
        action_stream = ""
        last_hh = hh
        btns = button_updates(ctx)
        return (sim, hh_list, sess_file, status, summary, cards, action_stream, last_hh, *btns)

    ctx = None
    btns = button_updates(ctx)
    return (sim, hh_list, sess_file, "Unknown engine state.", render_summary(sim, ctx), render_cards_html(ctx), "", "", *btns)


def do_action(sim: NLHESim, hh_list, sess_file, action_code: str):
    if sim is None:
        ctx = None
        btns = button_updates(ctx)
        return (None, hh_list, sess_file, "Click Start Session first.", "", "", "", "", *btns)

    msg = sim.apply_human_action(action_code)

    if msg["type"] == "HUMAN_DECISION":
        status = f"Your turn. Legal: {', '.join(msg['legal_actions'])}"
        summary = render_summary(sim, msg)
        cards = render_cards_html(msg)
        action_stream = msg.get("action_stream", "")
        last_hh = ""
        btns = button_updates(msg)
        return (sim, hh_list, sess_file, status, summary, cards, action_stream, last_hh, *btns)

    if msg["type"] == "HAND_END":
        hh = msg.get("hand_history_text", "")
        if hh:
            hh_list.append(hh)
            Path(sess_file).write_text("\n".join(hh_list), encoding="utf-8")

        status = f"Hand finished. Hands logged: {len(hh_list)} (auto-saved)"
        ctx = None
        summary = render_summary(sim, ctx)
        cards = render_cards_html(ctx)
        action_stream = ""
        last_hh = hh
        btns = button_updates(ctx)
        return (sim, hh_list, sess_file, status, summary, cards, action_stream, last_hh, *btns)

    ctx = None
    btns = button_updates(ctx)
    return (sim, hh_list, sess_file, "Unknown engine state.", render_summary(sim, ctx), render_cards_html(ctx), "", "", *btns)


def save_session(hh_list, sess_file):
    if not sess_file:
        return "No session file yet. Click Start Session first."
    Path(sess_file).write_text("\n".join(hh_list), encoding="utf-8")
    return f"Saved {len(hh_list)} hands to {Path(sess_file).resolve()}"


def upload_hh(file_obj):
    if file_obj is None:
        return "No file selected."
    text = Path(file_obj.name).read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return "Uploaded file was empty."
    with open(UPLOADED_HH_FILE, "a", encoding="utf-8") as f:
        f.write("\n" + text + "\n")
    return f"Appended upload to {UPLOADED_HH_FILE.resolve()} (chars={len(text)})"


with gr.Blocks(title="NLHE $1/$3 Simulator") as demo:
    gr.Markdown("# NLHE $1/$3 Simulator (Gradio)\nPlay + auto-save HH + upload real HH.")

    sim_state = gr.State(None)
    hh_state = gr.State([])
    sess_file_state = gr.State("")

    with gr.Row():
        seed = gr.Number(value=42, label="Seed", precision=0)
        n_players = gr.Slider(2, 9, value=6, step=1, label="Players")
        starting_stack = gr.Slider(100, 300, value=300, step=10, label="Starting stack")

    with gr.Row():
        start_btn = gr.Button("Start Session", variant="primary")
        next_btn = gr.Button("Next / Continue", variant="secondary")

    status = gr.Textbox(label="Status", value="", interactive=False)
    summary = gr.Textbox(label="Table Summary", value="", lines=10, interactive=False)

    cards = gr.HTML(label="Cards")
    action_stream = gr.Textbox(label="Action Stream (street-by-street)", value="", lines=14, interactive=False)
    last_hh = gr.Textbox(label="Last Completed Hand History", value="", lines=14, interactive=False)

    gr.Markdown("## Actions (only legal buttons are shown/enabled)")
    with gr.Row():
        btn_check = gr.Button("Check", visible=False)
        btn_call = gr.Button("Call", visible=False)
        btn_fold = gr.Button("Fold", visible=False)
        btn_allin = gr.Button("All-in", visible=False)

    with gr.Row():
        btn_b_s = gr.Button("Bet Small", visible=False)
        btn_b_m = gr.Button("Bet Medium", visible=False)
        btn_b_o = gr.Button("Bet Over", visible=False)

    with gr.Row():
        btn_r_s = gr.Button("Raise Small", visible=False)
        btn_r_m = gr.Button("Raise Medium", visible=False)
        btn_r_o = gr.Button("Raise Over", visible=False)

    gr.Markdown("## Hand history collection")
    save_btn = gr.Button("Save Session Now")
    save_msg = gr.Textbox(label="Save message", interactive=False)

    upload = gr.File(label="Upload HH (.txt)")
    upload_msg = gr.Textbox(label="Upload message", interactive=False)

    outputs_common = [
        sim_state, hh_state, sess_file_state,
        status, summary, cards, action_stream, last_hh,
        btn_check, btn_call, btn_fold, btn_allin,
        btn_b_s, btn_b_m, btn_b_o,
        btn_r_s, btn_r_m, btn_r_o,
    ]

    start_btn.click(
        start_session,
        inputs=[seed, n_players, starting_stack],
        outputs=outputs_common,
    )

    next_btn.click(
        next_or_continue,
        inputs=[sim_state, hh_state, sess_file_state],
        outputs=outputs_common,
    )

    btn_check.click(lambda s, h, f: do_action(s, h, f, "CHECK"),
                    inputs=[sim_state, hh_state, sess_file_state],
                    outputs=outputs_common)
    btn_call.click(lambda s, h, f: do_action(s, h, f, "CALL"),
                   inputs=[sim_state, hh_state, sess_file_state],
                   outputs=outputs_common)
    btn_fold.click(lambda s, h, f: do_action(s, h, f, "FOLD"),
                   inputs=[sim_state, hh_state, sess_file_state],
                   outputs=outputs_common)
    btn_allin.click(lambda s, h, f: do_action(s, h, f, "ALLIN"),
                    inputs=[sim_state, hh_state, sess_file_state],
                    outputs=outputs_common)

    btn_b_s.click(lambda s, h, f: do_action(s, h, f, "BET_S"),
                  inputs=[sim_state, hh_state, sess_file_state],
                  outputs=outputs_common)
    btn_b_m.click(lambda s, h, f: do_action(s, h, f, "BET_M"),
                  inputs=[sim_state, hh_state, sess_file_state],
                  outputs=outputs_common)
    btn_b_o.click(lambda s, h, f: do_action(s, h, f, "BET_O"),
                  inputs=[sim_state, hh_state, sess_file_state],
                  outputs=outputs_common)

    btn_r_s.click(lambda s, h, f: do_action(s, h, f, "RAISE_S"),
                  inputs=[sim_state, hh_state, sess_file_state],
                  outputs=outputs_common)
    btn_r_m.click(lambda s, h, f: do_action(s, h, f, "RAISE_M"),
                  inputs=[sim_state, hh_state, sess_file_state],
                  outputs=outputs_common)
    btn_r_o.click(lambda s, h, f: do_action(s, h, f, "RAISE_O"),
                  inputs=[sim_state, hh_state, sess_file_state],
                  outputs=outputs_common)

    save_btn.click(save_session, inputs=[hh_state, sess_file_state], outputs=[save_msg])
    upload.change(upload_hh, inputs=[upload], outputs=[upload_msg])

demo.launch()
