"""任务类型路由：正则**只决定用哪个核验器**，不决定是否拒答。

优先级：冲突 > 聚合 > 视觉 > 时序比较 > 日期 > 事实
（冲突/聚合最容易被误当普通事实题，所以优先）
"""
from __future__ import annotations
import re

FACT = "FACT"
VISUAL = "VISUAL"
AGGREGATE = "AGGREGATE"
TEMPORAL_ORDER = "TEMPORAL_ORDER"
CONFLICT = "CONFLICT"
DATE = "DATE"

_PREF = re.compile(r"what do i (like|love|prefer)|my favou?rite|do i (like|love|prefer)|"
                   r"am i (into|interested)|what am i into|"
                   r"我喜欢(什么|哪)|我(最)?喜欢|偏好|爱吃什么|喜欢什么", re.I)


_EVENT = re.compile(r"when did i|where do i live|where did i move|did i move|did i get "
                    r"(married|promoted|fired)|what (job|work) do i|am i planning|did i graduate|"
                    r"我什么时候|我搬(到|去)|我现在住哪|我(在)?哪工作|我打算", re.I)


def is_event_question(q: str) -> bool:
    return bool(_EVENT.search(q or ""))


def is_preference_question(q: str) -> bool:
    return bool(_PREF.search(q or ""))


_CONFLICT = re.compile(r"conflict|contradict|inconsistent|does not match|矛盾|冲突|不一致", re.I)
_AGG = re.compile(r"how (much|many)|\btotal\b|in all|altogether|sum of|"
                  r"一共|总共|合计|总计|多少钱|多少个|几次|几张|几个", re.I)
_IMG = re.compile(r"picture|image|photo|图|照片|画面|图像|截图", re.I)
_VCON = re.compile(r"in the (picture|image|photo)|species|breed|colou?r|same category|"
                   r"which (picture|image)|depict|what is in|图(里|中|上)|照片(里|中|上)|画面(里|中)|"
                   r"什么(颜色|种类|品种|花|植物|动物|物体)|是(什么|哪)|"
                   r"images? of|photos? of|search.*(images?|photos?)|"
                   # 2026-11-01：平台视觉题常用"background / ad / screenshot / diagram"，
                   # 原规则只认 color/species/in the picture ⇒ Q1 这类被误判成 FACT，
                   r"background|backgrounds|\bad\b|advert|poster|screenshot|screen shot|"
                   r"diagram|chart|logo|心形|背景|广告|海报|截图|示意图", re.I)
_ACTION = re.compile(r"^\s*(did|does|has|have)\b.*(photograph|take .*photo|photo journal|"
                     r"collect .*image|bring .*camera)|拍(过|了)?照|合影|相册", re.I)
_ORDER = re.compile(r"^\s*(did|does|has|have|was|were|is|are)\b[^?]{0,200}?\b(before|after|first|earlier|later)\b|"
                    # `which ... earlier/later/sooner` 也是比较两个事件的先后；
                    # 原来只写了 `(came|happened) first`，导致 "Which happened earlier: X or Y"
                    # 被判成 FACT，ordering_probe 从不触发（Gem 全量里 8 道）。
                    r"which .*(came|happened|occurred)\s+(first|earlier|later|sooner)|"
                    r"先后|哪个先|早于|晚于|更早|更晚|"
                    r"先.{0,24}还是|先.{0,24}(再|后)|之前.{0,12}之后", re.I)

# 中文顺序型：**计数式**判定（不要写成一条大正则，容易把"后来…一开始"的顺序搞反）。
# 依据（2026-10-31 实测 50 条平台真实查询）：
#   · 裸加"最后/后来" → 覆盖 22/50，但大部分是"最后=最终选择/最终状态"
#     （"最后塞进行李箱的物件是什么"、"最后为什么愿意付钱"）⇒ **大量误报**
#   · 收紧为"接着/然后/随后"（明确下一步）或"≥2 个顺序词"（叙事而非单点）
#     → 覆盖 5/50，逐条人工核对均为真顺序语义
_ZH_NEXT = re.compile(r"接着|然后|随后")
_ZH_SEQ = re.compile(r"先|最初|一开始|起初|后来|最后")


def zh_sequence_query(query: str) -> bool:
    q = query or ""
    if _ZH_NEXT.search(q):
        return True
    return len(_ZH_SEQ.findall(q)) >= 2
_DATE = re.compile(r"which day|what date|on which date|when did|when was|"
                    r"哪天|哪一天|什么时候|什么日期|什么时间|几月几号|几号|几月|哪年|几年", re.I)


def route(q: str) -> str:
    q = q or ""
    if _CONFLICT.search(q):
        return CONFLICT
    if _AGG.search(q):
        return AGGREGATE
    if _IMG.search(q) and _VCON.search(q) and not _ACTION.search(q):
        return VISUAL
    if _ORDER.search(q) or zh_sequence_query(q):
        return TEMPORAL_ORDER
    if _DATE.search(q):
        return DATE
    return FACT
