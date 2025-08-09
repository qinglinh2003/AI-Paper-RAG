import re, json, yaml, argparse
from pathlib import Path
from typing import Any, Dict, List
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import transformers
transformers.utils.logging.set_verbosity_error()

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

SMOKE_LIMIT  = None
FIVE_LEVELS  = [1.0, 0.8, 0.6, 0.3, 0.0]

def load_cfg(cfg_path: str) -> Dict[str, Any]:
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    eval_cfg = cfg.get("eval", {})
    eval_cfg.setdefault("index_dir", "data/processed/faiss_index")
    eval_cfg.setdefault("topk", 3)
    cfg["eval"] = eval_cfg
    acc_cfg = cfg.get("eval_accuracy", {})
    acc_cfg.setdefault("answer_model", "Qwen/Qwen2.5-7B-Instruct")
    acc_cfg.setdefault("judge_model",  "Qwen/Qwen2.5-7B-Instruct")
    acc_cfg.setdefault("use_8bit", True)
    acc_cfg.setdefault("max_answer_tokens", 96)
    acc_cfg.setdefault("max_judge_tokens", 64)
    cfg["eval_accuracy"] = acc_cfg
    if "embedding" not in cfg:
        cfg["embedding"] = {"model": "sentence-transformers/all-MiniLM-L6-v2"}
    return cfg

def build_retriever(cfg):
    embed_model = cfg["embedding"]["model"]
    index_dir   = cfg["eval"]["index_dir"]
    topk        = int(cfg["eval"]["topk"])
    embed = HuggingFaceEmbeddings(model_name=embed_model)
    vs = FAISS.load_local(index_dir, embed, allow_dangerous_deserialization=True)
    return vs.as_retriever(search_kwargs={"k": topk})

def load_model(name: str, use_8bit: bool):
    qcfg = BitsAndBytesConfig(load_in_8bit=True) if use_8bit else None
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(
        name,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=qcfg,
        torch_dtype="auto" if not use_8bit else None,
    )
    return tok, mdl

def chat_generate(tok, mdl, system_text: str, user_text: str, max_new_tokens: int) -> str:
    messages = [
        {"role": "system", "content": system_text.strip()},
        {"role": "user",   "content": user_text.strip()},
    ]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(mdl.device)
    with torch.no_grad():
        out = mdl.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    return tok.decode(gen_ids, skip_special_tokens=True).strip()

JUDGE_SYS = (
    "You are a strict but fair evaluator. Compare the model's answer with the reference answer.\n"
    "Return ONLY one of: 1.0, 0.8, 0.6, 0.3, 0.0\n"
    "Grading policy:\n"
    "- 1.0 (Correct): Semantically equivalent; all critical entities and numbers match (rounding tolerance allowed). "
    "Extra non-contradictory details are fine; paraphrase is fine.\n"
    "- 0.8 (Mostly correct): Main claim correct; may miss minor detail OR small numeric deviation that doesn't change the claim; no contradictions.\n"
    "- 0.6 (Partially correct): Captures part of the key idea but misses at least one critical element (e.g., key number/entity) OR too vague.\n"
    "- 0.3 (Weak/related): Loosely related; fails to convey the core idea; may include a potentially misleading detail but not a hard contradiction.\n"
    "- 0.0 (Incorrect/unsupported): Contradicts the reference; wrong entity/number; or says 'unanswerable' while the reference is specific.\n"
    "Do not penalize style. Be strict about entities/numbers. Output the score only."
)

JUDGE_USER_TMPL = """Question: {question}
ReferenceAnswer: {ref_answer}
ModelAnswer: {pred_answer}

Return ONLY one of: 1.0, 0.8, 0.6, 0.3, 0.0
"""

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", "-c", type=str, default="configs/default.yaml")
    args = p.parse_args()

    cfg = load_cfg(args.config)
    eval_cfg = cfg["eval"]
    acc_cfg  = cfg["eval_accuracy"]
    eval_data = eval_cfg["no_anchor"]

    eval_items = json.loads(Path(eval_data).read_text())
    if SMOKE_LIMIT is not None:
        eval_items = eval_items[:SMOKE_LIMIT]

    retriever = build_retriever(cfg)

    ans_name = acc_cfg["answer_model"]
    jdg_name = acc_cfg["judge_model"]
    tok_ans, mdl_ans = load_model(ans_name, acc_cfg["use_8bit"])
    if jdg_name == ans_name:
        tok_jdg, mdl_jdg = tok_ans, mdl_ans
    else:
        tok_jdg, mdl_jdg = load_model(jdg_name, acc_cfg["use_8bit"])

    results: List[Dict[str, Any]] = []

    for it in tqdm(eval_items, desc="Accuracy eval (5-level, LLM-only)", unit="q"):
        q   = (it.get("question") or "").strip()
        ref = (it.get("reference_answer") or "").strip()
        if not q or not ref:
            continue

        docs = retriever.get_relevant_documents(q)
        ctx  = "\n\n".join(d.page_content for d in docs[: int(eval_cfg["topk"])])
        ctx  = ctx[:3500]

        pred = chat_generate(
            tok_ans, mdl_ans,
            "Answer ONLY using the provided context. If not answerable, say 'unanswerable'. Be concise.",
            f'Context:\n"""\n{ctx}\n"""\n\nQuestion: {q}\nAnswer in 1-2 sentences:',
            max_new_tokens=int(acc_cfg["max_answer_tokens"])
        )

        out = chat_generate(
            tok_jdg, mdl_jdg,
            JUDGE_SYS,
            JUDGE_USER_TMPL.format(question=q, ref_answer=ref, pred_answer=pred),
            max_new_tokens=int(acc_cfg["max_judge_tokens"])
        )
        m = re.search(r"\b(1\.0|0\.8|0\.6|0\.3|0\.0)\b", out.strip())
        score = float(m.group(1)) if m else 0.0

        results.append({
            "question": q,
            "reference_answer": ref,
            "pred_answer": pred,
            "score": score
        })

    dist = {lv: 0 for lv in FIVE_LEVELS}
    for r in results: dist[r["score"]] = dist.get(r["score"], 0) + 1
    avg = sum(r["score"] for r in results) / len(results) if results else 0.0

    print("Score distribution:", {k: dist[k] for k in FIVE_LEVELS})
    print(f"Avg score: {avg:.4f}")

    output = {
        "score_distribution": dist,
        "avg_score": round(avg, 4),
        "results": results
    }
    report_file = eval_cfg["accuracy_file"]
    Path(report_file).parent.mkdir(parents=True, exist_ok=True)
    Path(report_file).write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"Saved: {report_file}")

if __name__ == "__main__":
    main()
