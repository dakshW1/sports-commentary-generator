"""Streamlit entry point: `streamlit run app/main.py`"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so `app` imports work under streamlit
from app import core  # noqa: E402

st.set_page_config(page_title="Sports Commentary Generator", page_icon="⚽")
st.title("⚽ Sports Commentary Generator")
st.write("Game developers use this tool to adapt a language model with LoRA, generating dynamic and context-aware commentary snippets for simulated sports events like goal alerts or score changes.")


@st.cache_resource
def get_model():
    """Prefer the merged model (fastest); fall back to GPT-2 + adapter."""
    if (core.MERGED_DIR / "config.json").exists():
        model, tok = core.load_merged()
        return model, tok, "merged LoRA model"
    if core.ADAPTER_FILE.exists():
        model, tok, _ = core.load_adapted()
        return model, tok, "GPT-2 + LoRA adapter"
    return None, None, None


model, tok, kind = get_model()
if model is None:
    st.error("No trained model yet. Run `python -m app.cli train` and `python -m app.cli merge` first.")
    st.stop()
st.caption(f"Using the {kind}.")

with st.form("event"):
    c1, c2 = st.columns(2)
    etype = c1.selectbox("Event", core.EVENT_TYPES, format_func=lambda t: t.replace("_", " "))
    minute = c2.slider("Minute", 1, 90, 67)
    team = c1.text_input("Team", "Riverside Rovers")
    player = c2.text_input("Player", "Okafor")
    score = c1.text_input("Score after the event", "2-1")
    n = c2.number_input("Commentary options", 1, 5, 3)
    temperature = st.slider("Creativity (temperature)", 0.1, 1.5, 0.8, 0.1)
    submitted = st.form_submit_button("Generate commentary", type="primary")

if submitted:
    event = {"type": etype, "minute": minute, "team": team.strip(), "player": player.strip(),
             "score": score.strip()}
    st.code(core.describe(event), language=None)
    with st.spinner("Writing commentary..."):
        st.session_state["options"] = (event, core.generate(model, tok, event, n=int(n),
                                                            temperature=temperature))

if "options" in st.session_state:
    event, lines = st.session_state["options"]
    for i, line in enumerate(lines):
        rel = core.relevance(event, line)
        badges = " ".join(("✅ " if ok else "⚠️ ") + k.replace("_", " ") for k, ok in rel.items())
        c1, c2 = st.columns([5, 1])
        c1.markdown(f"**{line}**  \n<small>{badges}</small>", unsafe_allow_html=True)
        if c2.button("Save", key=f"save{i}"):
            core.save_snippet(event, line)
            st.toast("Saved")

saved = core.load_snippets()
if saved:
    with st.expander(f"Saved commentary ({len(saved)})"):
        for r in reversed(saved[-20:]):
            st.write(f"{r['event']['minute']}' {r['commentary']}")
