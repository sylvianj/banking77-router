import streamlit as st
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.predict import load_model, predict

st.set_page_config(
    page_title="Banking Intent Router",
    page_icon="🏦",
    layout="centered",
)


@st.cache_resource
def get_model():
    """Load the model once and reuse it across reruns."""
    return load_model()


st.title("Banking Intent Router")
st.write(
    "Type a customer support message. The model classifies it into one of "
    "**77 fine-grained banking intents**."
)

# Sample messages for one-click testing
with st.expander("Try a sample message", expanded=False):
    samples = [
        "I lost my card yesterday",
        "When will my new card arrive?",
        "Why was I charged twice this month?",
        "How do I change my PIN?",
        "My transfer hasn't arrived yet",
    ]
    for s in samples:
        if st.button(s, key=s):
            st.session_state.text_input = s

if "text_input" not in st.session_state:
    st.session_state.text_input = ""

text = st.text_area(
    "Customer message",
    value=st.session_state.text_input,
    height=100,
    placeholder="e.g. My card was declined at an ATM",
)

if st.button("Predict intent", type="primary") and text.strip():
    with st.spinner("Loading model... first run may take a minute"):
        tokenizer, model, label_names = get_model()

    with st.spinner("Predicting..."):
        results = predict(text, tokenizer, model, label_names, top_k=5)

    st.divider()

    top = results[0]
    st.success(f"**{top['intent']}** — {top['probability']:.1%} confidence")

    st.subheader("Top 5 predictions")
    for r in results:
        st.write(f"**{r['intent']}**")
        st.progress(float(r["probability"]), text=f"{r['probability']:.2%}")

st.divider()
st.caption(
    "Model: DistilBERT fine-tuned on Banking77 (92.8% accuracy, 77 classes). "
    "Hosted on Hugging Face Hub."
)