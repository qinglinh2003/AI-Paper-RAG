import json, random, sys, threading, itertools, re
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
import torch
import transformers
transformers.utils.logging.set_verbosity_error()  

MODEL_NAME = "Qwen/Qwen2.5-32B-Instruct"  
CHUNK_FILE = "data/processed/chunks.json"
OUTPUT_FILE = "data/evaluation_set.json"
OUTPUT_TMP  = "data/evaluation_set.tmp.json"

TARGET_QUESTIONS = 100       
MAX_CHUNKS_SCAN = None        
MIN_CHUNK_LEN   = 200       
MAX_QA_LEN      = 260      
SAVE_EVERY      = 1         

BANNED_PHRASES = [
    "given text", "given context", "the text", "this text",
    "this paper", "the paper", "this study", "the study", "context only"
]

def contains_banned(s: str) -> bool:
    s_low = (s or "").lower()
    return any(b in s_low for b in BANNED_PHRASES)

def load_partial(path):
    p = Path(path)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return []

def save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2))

class Spinner:
    def __init__(self, message="Running"):
        self.message = message
        self._stop = False
        self._thr = None
    def start(self):
        def _spin():
            for ch in itertools.cycle("|/-\\"):
                if self._stop: break
                sys.stdout.write(f"\r{self.message} {ch}")
                sys.stdout.flush()
                import time; time.sleep(0.1)
            sys.stdout.write("\r" + " " * (len(self.message) + 2) + "\r")
            sys.stdout.flush()
        self._thr = threading.Thread(target=_spin, daemon=True)
        self._thr.start()
    def stop(self):
        self._stop = True
        if self._thr:
            self._thr.join()

def load_generator():
    print("Loading model (8-bit)…")
    bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=bnb_cfg,
    )
    return tok, mdl

def chat_generate(tok, mdl, system_text, user_text, max_new_tokens):
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
    text = tok.decode(gen_ids, skip_special_tokens=True).strip()
    return text

def guess_title_from_source(src_path: str) -> str:
    name = Path(src_path or "unknown").stem
    name = re.sub(r"[_\-]+", " ", name).strip()
    return name[:120] if name else "Unknown Title"

def gen_qa(tok, mdl, chunk_text, meta, max_len, retries=2):
    source = (meta or {}).get("source", "")
    page   = (meta or {}).get("page", None)
    title  = guess_title_from_source(source)

    system_prompt = (
        "You write SELF-CONTAINED evaluation items for a RAG system. "
        "STRICT RULES: The question MUST include the explicit paper title anchor and MUST NOT use deictic phrases "
        "like 'the text', 'this paper', 'the study', 'given context'. If the output would violate any rule, "
        "you MUST regenerate and produce a compliant result."
    )

    user_template = f"""
TASK: Create EXACTLY ONE evaluation item (question + concise answer) based ONLY on the excerpt below.

HARD REQUIREMENTS (MUST follow all):
1) The question MUST be self-contained and MUST explicitly include the paper title: "{title}" (optionally include year if present).
2) DO NOT use vague references (forbidden: "the text", "this paper", "the study", "given context", etc.).
3) The answer MUST be supported by the excerpt or its immediate context in the same paper.
4) Provide evidence: source path, page index if available, and a SHORT supporting quote (<=200 chars).
5) Return VALID JSON ONLY. No markdown, no extra commentary.

Excerpt:
\"\"\"{chunk_text}\"\"\"

Output JSON ONLY:
{{
  "question": "<self-contained question that includes the exact title \\"{title}\\">",
  "answer": "<concise answer (1–2 sentences)>",
  "evidence": {{
    "source": "{source}",
    "page": {page if page is not None else "null"},
    "quote": "<short supporting quote from the excerpt>"
  }}
}}
""".strip()

    for _ in range(retries + 1):
        raw = chat_generate(tok, mdl, system_prompt, user_template, max_len)

        m = re.search(r"\{.*\}", raw, flags=re.S)
        if m:
            try:
                obj = json.loads(m.group(0))
                q = (obj.get("question") or "").strip()
                a = (obj.get("answer") or "").strip()
                ev = obj.get("evidence") or {}
                if not q or not a:
                    continue
                if contains_banned(q):
                    continue
                if title.lower() not in q.lower():
                    continue
                quote = (ev.get("quote") or "").strip()
                if not quote:
                    quote = (chunk_text or "")[:200]
                evidence = {
                    "source": ev.get("source") or source,
                    "page": ev.get("page") if ev.get("page") is not None else page,
                    "quote": quote,
                }
                return q, a, evidence, raw
            except Exception:
                pass

        continue

    return None, None, None, raw

def main():
    chunks = json.loads(Path(CHUNK_FILE).read_text())
    random.shuffle(chunks)
    chunks = chunks[:MAX_CHUNKS_SCAN]

    results = load_partial(OUTPUT_TMP)
    next_id  = len(results) + 1

    chunk_pbar = tqdm(total=len(chunks), desc="Chunks", ncols=90, leave=True)
    qa_pbar    = tqdm(total=TARGET_QUESTIONS, initial=len(results), desc="QAs", ncols=90, leave=True)

    tok, mdl = load_generator()

    try:
        for idx, ch in enumerate(chunks):
            if len(results) >= TARGET_QUESTIONS:
                break
            text = (ch.get("text") or "").strip()
            if len(text) < MIN_CHUNK_LEN:
                chunk_pbar.update(1)
                continue

            sp = Spinner(f"Generating QA on chunk #{idx}")
            sp.start()
            try:
                q, a, evidence, raw = gen_qa(tok, mdl, text, ch.get("metadata"), MAX_QA_LEN)
            finally:
                sp.stop()
            chunk_pbar.update(1)

            if not q:
                continue

            results.append({
                "id": next_id,
                "question": q,
                "reference_answer": a,
                "evidence": evidence,               
                "source_chunk_id": ch.get("id", idx)
            })
            next_id += 1
            qa_pbar.update(1)
            if len(results) % SAVE_EVERY == 0:
                save_json(OUTPUT_TMP, results)

        save_json(OUTPUT_TMP, results)
        save_json(OUTPUT_FILE, results)
        chunk_pbar.close(); qa_pbar.close()
        print(f"\n Done. Saved {len(results)} Q/A → {OUTPUT_FILE}")

    except KeyboardInterrupt:
        save_json(OUTPUT_TMP, results)
        chunk_pbar.close(); qa_pbar.close()
        print(f"\n⏸ Interrupted. Partial saved → {OUTPUT_TMP} ({len(results)} items).")

if __name__ == "__main__":
    main()
