import base64
import io
import re
from typing import Optional

from PIL import Image

DATA_URI_RE = re.compile(
    r"^data:image/(?P<fmt>jpeg|jpg|png|webp);base64,(?P<data>.+)$",
    re.I | re.S,
)


def sniff_format(raw: bytes) -> Optional[str]:
    try:
        with Image.open(io.BytesIO(raw)) as image:
            fmt = (image.format or "").lower()
    except Exception:
        return None
    return {
        "jpeg": "jpeg",
        "jpg": "jpeg",
        "mpo": "jpeg",
        "png": "png",
        "webp": "webp",
    }.get(fmt)


def decode_data_uri(url: str) -> bytes:
    match = DATA_URI_RE.match(url or "")
    if not match:
        raise ValueError("image_url.url must be an inline JPEG/PNG/WebP data URI")
    return base64.b64decode(match.group("data"), validate=True)


def encode_data_uri(raw: bytes, fmt: Optional[str] = None) -> str:
    fmt = fmt or sniff_format(raw)
    if fmt not in {"jpeg", "png", "webp"}:
        raise ValueError("unsupported or invalid image")
    return f"data:image/{fmt};base64,{base64.b64encode(raw).decode('ascii')}"


def fit_image(raw: bytes, limit: int) -> bytes:
    if len(raw) <= limit and sniff_format(raw):
        return raw
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    scale = 1.0
    quality = 90
    while True:
        width, height = image.size
        resized = image
        if scale < 1.0:
            resized = image.resize(
                (max(1, int(width * scale)), max(1, int(height * scale)))
            )
        buf = io.BytesIO()
        resized.save(buf, "JPEG", quality=quality)
        if buf.tell() <= limit:
            return buf.getvalue()
        scale *= 0.8
        quality = max(65, quality - 5)


# VLM 输入上限：Qwen2.5-VL 默认 max_pixels 极大（单图可达 ~16k vision token），
# 会让 decoder prompt 直接超过服务端 max-model-len → HTTP 400（实测 18105 > 16384）。
# 因此在**所有**把图片送进 VLM 的地方统一压到 ~0.8MB / 长边 <= 1024。
VLM_IMAGE_BYTES = 800 * 1024
VLM_IMAGE_LONG_SIDE = 1024


def vlm_ready_image(raw: bytes, fmt: Optional[str] = None):
    """返回 (bytes, fmt)：保证字节数与像素数都在 VLM 安全范围内。"""
    try:
        from PIL import Image
        import io as _io
        im = Image.open(_io.BytesIO(raw)).convert("RGB")
        w, h = im.size
        if max(w, h) > VLM_IMAGE_LONG_SIDE:
            sc = VLM_IMAGE_LONG_SIDE / float(max(w, h))
            im = im.resize((max(1, int(w * sc)), max(1, int(h * sc))), Image.LANCZOS)
        out = _io.BytesIO()
        im.save(out, "JPEG", quality=85)
        return out.getvalue(), "jpeg"
    except Exception:
        # 兜底：至少保证字节数不超限
        try:
            return fit_image(raw, VLM_IMAGE_BYTES), (fmt or "jpeg")
        except Exception:
            return raw, (fmt or "jpeg")
