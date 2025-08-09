def build_context(docs, max_chars: int, with_tags: bool = True):
    out = []
    for i,d in enumerate(docs):
        t = (d.page_content or "").replace("\n", " ")
        if len(t) > max_chars: t = t[:max_chars] + " ..."
        out.append(f"[Doc {i}] {t}" if with_tags else t)
    return "\n\n".join(out)
