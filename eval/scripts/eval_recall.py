import json, yaml, re, argparse
from pathlib import Path
from typing import Dict, Any
from tqdm import tqdm
from src.retrieval.pipeline import retrieve_then_rerank
import transformers
transformers.utils.logging.set_verbosity_error()

TOPK = 10

def load_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def norm_path(p: str) -> str:
    return re.sub(r"[\\/]+", "/", (p or "").strip().lower())

def evaluate(cfg_path: str):
    cfg = load_cfg(cfg_path)
    eval_file = cfg["eval"]["no_anchor"]
    eval_items = json.loads(Path(eval_file).read_text())

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
        docs = retrieve_then_rerank(q, cfg_path, topn=TOPK, mode="recall")


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
    p = argparse.ArgumentParser()
    p.add_argument("--config", "-c", type=str, default="configs/default.yaml")
    args = p.parse_args()
    evaluate(args.config)
