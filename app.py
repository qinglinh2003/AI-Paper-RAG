import os
import yaml
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.retrieval.pipeline import retrieve_then_rerank
from src.retrieval.context import build_context

st.set_page_config(page_title="AI Paper RAG Demo", page_icon="📄", layout="centered")

DEFAULT_CFG = os.getenv("RAG_CONFIG", "configs/default.yaml")

@st.cache_resource(show_spinner=False)
def load_cfg(cfg_path: str):
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)

@st.cache_resource(show_spinner=False)
def load_llm(model_name: str, dtype: str = "float16", device_map: str = "auto"):
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=getattr(torch, dtype, torch.float16),
        device_map=device_map,
    )
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    return tok, model

def llm_answer(tok, model, context: str, question: str, max_new_tokens: int = 256) -> str:
    system_text = (
        "You are a helpful research assistant.\n"
        "Answer ONLY using the provided context. When possible, copy exact wording.\n"
        "If the answer is not in the context, say 'unanswerable'."
    )
    user_text = f'Context:\n"""\n{context}\n"""\n\nQuestion: {question}\nAnswer in 1-2 sentences:'
    try:
        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        prompt = system_text + "\n\n" + user_text

    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    text = tok.decode(gen_ids, skip_special_tokens=True).strip()
    return text

st.title("📄 AI Paper RAG Demo")
st.caption("Ask a question about the indexed AI papers. The system retrieves relevant chunks and answers grounded in the text.")

cfg = load_cfg(DEFAULT_CFG)
llm_name = cfg.get("model", {}).get("llm", "Qwen/Qwen2.5-7B-Instruct")
dtype = cfg.get("model", {}).get("dtype", "float16")
device_map = cfg.get("model", {}).get("device_map", "auto")
max_ctx_docs = int(cfg.get("app", {}).get("max_context_docs", 1))
max_new_tokens = int(cfg.get("app", {}).get("max_new_tokens", 256))

with st.spinner("Loading model..."):
    tok, model = load_llm(llm_name, dtype=dtype, device_map=device_map)

question = st.text_input("Question", placeholder="e.g., What benchmark was introduced to evaluate hallucination in LVLMs?")
submitted = st.button("Submit")

output_placeholder = st.empty()

if submitted:
    q = (question or "").strip()
    if not q:
        st.warning("Please enter a question.")
    else:
        try:
            with st.spinner("Retrieving and generating..."):
                docs = retrieve_then_rerank(q, DEFAULT_CFG, topn=max_ctx_docs, mode="accuracy")
                ctx = build_context(docs, max_chars=3500, with_tags=True)
                ans = llm_answer(tok, model, ctx, q, max_new_tokens=max_new_tokens)

            output_placeholder.markdown("**Answer**")
            st.text_area(label="", value=ans, height=160)

            with st.expander("Show retrieved context"):
                st.text(ctx)

        except Exception as e:
            st.error(f"Error: {e}")
