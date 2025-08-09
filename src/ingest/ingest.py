import re
import yaml
import argparse
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_docs(raw_dir: str):
    docs = []
    for p in Path(raw_dir).rglob("*"):
        suf = p.suffix.lower()
        try:
            if suf == ".pdf":
                docs += PyPDFLoader(str(p)).load()
            elif suf == ".md":
                docs += UnstructuredMarkdownLoader(str(p)).load()
            elif suf == ".txt":
                docs += TextLoader(str(p), encoding="utf-8").load()
        except Exception as e:
            print(f"[warn] failed to load {p.name}: {e}")
    return docs

def sanitize_text(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F]", " ", s)
    s = s.replace("\u0000", " ").strip()
    s = s.encode("utf-8", "ignore").decode("utf-8", "ignore")
    s = re.sub(r"[ \t]+", " ", s)
    return s

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_dir = cfg["data"]["raw_dir"]
    vs_path = cfg["vectorstore"]["path"]

    print(f"Loading documents from {raw_dir}...")
    docs = load_docs(raw_dir)
    print(f"Loaded {len(docs)} documents.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg["chunking"]["size"],
        chunk_overlap=cfg["chunking"]["overlap"],
        separators=cfg["chunking"]["separators"],
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")

    texts, metas = [], []
    for d in chunks:
        pc = sanitize_text(getattr(d, "page_content", ""))
        if len(pc) < 10:
            continue
        texts.append(pc)
        metas.append(getattr(d, "metadata", {}))

    print(f"After cleaning, {len(texts)} chunks remain (from {len(chunks)})")

    embed = HuggingFaceEmbeddings(
        model_name=cfg["embedding"]["model"],
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
    )

    db = FAISS.from_texts(texts=texts, embedding=embed, metadatas=metas)
    Path(vs_path).parent.mkdir(parents=True, exist_ok=True)
    db.save_local(vs_path)
    print(f"FAISS index saved to {vs_path}")

if __name__ == "__main__":
    main()
