"""视觉核验通道：当证据核验判定"不足"时，若问题是问图片内容且候选轮次含真实图片，
则取出图片让 VLM 直接观察，用观察结果补充证据。

不改变检索结果（R@gold 不受影响），只增加一条"图像观察"记录并解除弃答。
"""
from __future__ import annotations
import base64, json, logging, re, time
from typing import Any

logger = logging.getLogger("amc.visual_probe")

# 问"图片内容"才算视觉题
#   原 _IMG 不认这些 ⇒ 6/6 视觉样本被判"非视觉"⇒ VLM 探针一次都没跑。
_IMG = re.compile(r"picture|image|photo|shot|view|crop|scene|frame|screenshot|图|照片|画面|图像|截图", re.I)
_CONTENT = re.compile(
    # 允许 "in the <修饰> photo/shot/view/crop"（中间可有若干词）
    r"in (the|this|that|these|those)\s+[\w\-', ]{0,40}(picture|image|photo|shot|view|crop|scene|frame|screenshot)|"
    r"looking at the (picture|image|photo|shot|view|screenshot)|"
    r"species|breed|colou?r|same category|"
    r"which (picture|image|photo|shot|view)|depict|what is in|"
    r"\b(where|what|how many|how much|how large|which)\b.{0,60}\b(in|on|of)\b.{0,30}"
    r"(picture|image|photo|shot|view|crop|scene|frame)|"
    r"图(里|中|上)|照片(里|中|上)|画面(里|中)|"
    r"什么(颜色|种类|品种|花|植物|动物|物体)|是(什么|哪)|"
    r"images? of|photos? of|search.*(images?|photos?)", re.I)
# "是否拍过/带相机"这类是文本题，不交给视觉
_ACTION = re.compile(
    r"^\s*(did|does|has|have)\b.*(photograph|take .*photo|photo journal|"
    r"collect .*image|bring .*camera)|拍(过|了)?照|合影|相册", re.I)


def is_visual_content(q: str) -> bool:
    q = q or ""
    if not _IMG.search(q):
        return False
    if _ACTION.search(q):
        return False
    return bool(_CONTENT.search(q))


def _images_of(store, memory: dict) -> list:
    """从一个 memory dict 的 content 里取图片字节（最多 2 张）。"""
    out = []
    if not isinstance(memory, dict):
        return out
    for part in (memory.get("content") or []):
        if isinstance(part, dict) and part.get("type") == "image_ref":
            try:
                raw = store.blob_path(part["sha256"], part.get("fmt", "jpeg")).read_bytes()
            except Exception:
                continue
            out.append((raw, part.get("fmt", "jpeg")))
            if len(out) >= 2:
                break
    return out


def probe(store, memories: list, question: str, settings) -> dict | None:
    """memories: 检索到的 memory dict 列表（含 content）；返回 {"answer","request_id"} 或 None。"""
    base = (getattr(settings, "query_vision_base_url", "") or "").rstrip("/")
    model = getattr(settings, "query_vision_model", "") or ""
    key = getattr(settings, "query_vision_api_key", "") or ""
    if not (base and model and key):
        return None
    images, src = [], None
    for m in memories[:10]:
        images = _images_of(store, m)
        if images:
            src = m.get("request_id") or m.get("id")
            break
    if not images:
        return None
    # 两类问题要分开：
    #   A. 观察类（"图里是什么/什么颜色"）→ 让模型**看**，宽松提示
    #   B. 事实类（"哪一天/哪一步/多少"）→ 只允许**图中明确写出**，否则 NOT_MENTIONED
    #   官方合同："Search must not generate final answers or disguise answers as memory records"
    #   ⇒ VLM **不得作答**，只能输出"看图能看到的事实描述"，作为**证据**交给平台答案端。
    #   之前它会直接回 "(A)"/"(B)"（选择题选项字母）⇒ 等于把答案塞进搜索结果。
    if is_visual_content(question):
        _q_prompt = (f"请只描述这张图片里**能看到的内容**，不要回答问题、不要给结论。\n"
                     f"问题背景（仅用于你判断该描述什么，不要作答）：{question[:300]}\n\n"
                     "输出要求：\n"
                     "1) 用 1~3 句中文描述可见事实：有哪些物体、它们的位置（左/右/上/下/旁边）、\n"
                     "   颜色、数量、以及图里写出的文字；\n"
                     "2) **禁止**输出选项字母（A/B/C/D）、禁止写\"答案是\"\"选 X\"之类的判断；\n"
                     "3) 看不出来就写\"图片中无法辨认\"。")
    else:
        _q_prompt = (f"只依据这张图片回答问题：{question[:400]}\n\n"
                     "规则：\n"
                     "1) 只有图片里**明确写出**答案时才回答，并引用图里的原句；\n"
                     "2) 图里没有明确写、只能推测 → 只输出 NOT_MENTIONED；\n"
                     "3) 不要解释。\n\n"
                     "输出格式：\nANSWER: <答案>\nQUOTE: <原句>\n或\nNOT_MENTIONED")
    content = [{"type": "text", "text": _q_prompt}]
    for raw, fmt in images:
        f = "png" if str(fmt).lower() == "png" else "jpeg"
        content.append({"type": "image_url", "image_url":
                        {"url": f"data:image/{f};base64,{base64.b64encode(raw).decode()}"}})
    import httpx
    for attempt in range(1, 4):
        try:
            r = httpx.post(base + "/chat/completions",
                           headers={"Authorization": f"Bearer {key}"},
                           json={"model": model, "max_tokens": 200,
                                 "messages": [{"role": "user", "content": content}]},
                           timeout=120)
            r.raise_for_status()
            txt = (r.json()["choices"][0]["message"]["content"] or "").strip()
            # ⚠️ 合规过滤：VLM 若仍然给出选项字母/结论，直接丢弃（不写进 data）
            try:
                import re as _re_v
                _bad = _re_v.search(r"\([A-Da-d]\)|^\s*[A-D][\.\)]|\b(选|答案是|answer\s*(is)?)\s*[:：]?\s*\(?[A-D]\b", txt)
                if _bad:
                    logger.warning("[visual] VLM 输出含选项/结论（合规拦截）: %s", txt[:80])
                    return None
            except Exception:
                pass
            _strict = not is_visual_content(question)
            if txt and (not _strict or "NOT_MENTIONED" not in txt.upper()):
                return {"answer": txt[:300], "request_id": src}
            if txt:
                logger.info("[visual] 图里未明确写出答案 -> 不生成图像观察")
                return None
        except Exception as exc:
            _st = getattr(getattr(exc, "response", None), "status_code", None)
            if isinstance(_st, int) and 400 <= _st < 500:
                return None
            logger.warning("[visual] attempt %d failed: %s", attempt, exc)
            time.sleep(1.5 * attempt)
    return None
