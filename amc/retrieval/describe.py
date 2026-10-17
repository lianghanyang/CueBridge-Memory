CHILD_ATTRS = ("base", "sparse", "dense", "reranker")


def describe_retriever(node, depth=0):
    if node is None:
        return None
    if depth > 5:
        return {"class": node.__class__.__name__}
    info = {}
    fn = getattr(node, "describe", None)
    if callable(fn):
        try:
            info = dict(fn() or {})
        except Exception as exc:
            info = {"describe_error": str(exc)}
    info.setdefault("class", node.__class__.__name__)
    for attr in CHILD_ATTRS:
        if attr in info:
            continue
        child = getattr(node, attr, None)
        if child is not None and not isinstance(child, (str, int, float, dict, list)):
            info[attr] = describe_retriever(child, depth + 1)
    return info
