import yaml, torch
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM

PROMPT = """You are a helpful research assistant.
Answer ONLY using the provided context. Cite sources as [Doc i].

Question: {question}

Context:
{context}

Answer:
"""

def load_cfg():
    with open("configs/default.yaml","r") as f:
        return yaml.safe_load(f)

def load_vectorstore(cfg):
    embed = HuggingFaceEmbeddings(
        model_name=cfg["embedding"]["model"],
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 16},
    )
    return FAISS.load_local(cfg["vectorstore"]["path"], embed, allow_dangerous_deserialization=True)

def retrieve_context(db, query, k=8, max_chars=1400):
    docs = db.similarity_search(query, k=k)
    ctx = []
    for i, d in enumerate(docs):
        text = d.page_content.replace("\n", " ")
        if len(text) > max_chars:
            text = text[:max_chars] + " ..."
        ctx.append(f"[Doc {i}] {text}")
    return "\n\n".join(ctx)

def load_llm(name, dtype="float16"):
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        name,
        trust_remote_code=True,
        torch_dtype=getattr(torch, dtype),
        device_map="auto",
    )
    return tok, model

def generate_answer(tok, model, prompt, max_new_tokens=256):
    ids = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False)
    text = tok.decode(out[0], skip_special_tokens=True)
    return text.split("Answer:")[-1].strip()

def answer(question: str):
    cfg = load_cfg()
    db = load_vectorstore(cfg)
    context = retrieve_context(db, question, k=cfg["retrieval"]["top_k"])
    tok, model = load_llm(cfg["model"]["llm"], dtype=cfg["model"]["dtype"])
    prompt = PROMPT.format(question=question, context=context)
    return generate_answer(tok, model, prompt, max_new_tokens=cfg["app"]["max_new_tokens"])

if __name__ == "__main__":
    print(answer("Summarize key contributions across these papers, with citations."))
