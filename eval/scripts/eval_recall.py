import json, yaml, re
from pathlib import Path
from typing import Dict, Any
from tqdm import tqdm

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

CFG_FILE    = "configs/default.yaml"
TOPK        = 10

def load_cfg():
    with open(CFG_FILE, "r") as f:
        return yaml.safe_load(f)

def norm_path(p: str) -> str:
    """Normalize file path for matching"""
    return re.sub(r"[\\/]+", "/", (p or "").strip().lower())

def load_retriever():
    cfg = load_cfg()
    embed_model = cfg["embedding"]["model"]
    index_dir = cfg["vectorstore"]["path"]
    embed = HuggingFaceEmbeddings(model_name=embed_model)
    vs = FAISS.load_local(index_dir, embed, allow_dangerous_deserialization=True)
    return vs.as_retriever(search_kwargs={"k": TOPK})

def evaluate():
    cfg = load_cfg()
    eval_file = cfg["eval"]["no_anchor"]
    eval_items = json.loads(Path(eval_file).read_text())
    retriever = load_retriever()

    total = 0
    hits_at = {1: 0, 3: 0, 5: 0, 10: 0}
    mrr10_sum = 0.0
    failures = []

    for it in tqdm(eval_items, desc="Evaluating (doc-level recall)", unit="q"):
        q = it.get("question") or ""
        gold_doc = norm_path(it.get("evidence", {}).get("source"))
        if not q or not gold_doc:
            continue

        total += 1
        docs = retriever.get_relevant_documents(q)

        rank = None
        top_docs_dbg = []

        for i, d in enumerate(docs):
            src_norm = norm_path(d.metadata.get("source") or d.metadata.get("path") or "")
            top_docs_dbg.append({
                "rank": i + 1,
                "source": src_norm,
                "snippet": (d.page_content or "")[:200]
            })
            if src_norm == gold_doc and rank is None:
                rank = i + 1

        for k in hits_at.keys():
            if rank is not None and rank <= k:
                hits_at[k] += 1
        if rank is not None and rank <= 10:
            mrr10_sum += 1.0 / rank
        else:
            if len(failures) < 50:
                failures.append({
                    "question": q,
                    "gold_doc": gold_doc,
                    "top_docs": top_docs_dbg
                })

    report = {
        "total": total,
        "recall@1": round(hits_at[1] / total, 4),
        "recall@3": round(hits_at[3] / total, 4),
        "recall@5": round(hits_at[5] / total, 4),
        "recall@10": round(hits_at[10] / total, 4),
        "mrr@10": round(mrr10_sum / total, 4),
        "failures_sample": failures,
        "note": "Doc-level recall using evidence.source match"
    }

    report_file = cfg["eval"]["recall_file"]

    Path(report_file).parent.mkdir(parents=True, exist_ok=True)
    Path(report_file).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved report to {report_file}")

if __name__ == "__main__":
    evaluate()
