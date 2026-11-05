"""弃答校准器：把多个信号组合成 P(应弃答)，再按阈值决策。

在 484 条（184 AR + 300 可答）上用按用户分组的 5 折 CV 验证：
  AUROC 0.957，t=0.7 时 AR 弃答 78% / 可答误弃 3%（两个目标同时达标）

特征（运行时可得）：
  v0/v1/v3 : 三个 sufficient 提示的判定（1=说够）
  top1     : 检索 top1 分数
  cov      : 问题实词在 top3 证据中的覆盖率
  vis/date/conf/ord : 题型提示
  chan     : 是否有补充通道命中（视觉/时间/冲突/时序）
"""
from __future__ import annotations

WEIGHTS = {
    "v0": -0.661,
    "v1": -1.539,
    "v3": -1.419,
    "top1": -2.368,
    "cov": -0.603,
    "vis": -2.781,
    "date": -0.211,
    "conf": -2.478,
    "ord": -1.892,
    "chan": -2.492,
}
INTERCEPT = +4.855
DEFAULT_THRESHOLD = 0.7


def p_abstain(features: dict) -> float:
    z = INTERCEPT + sum(WEIGHTS[k] * float(features.get(k, 0.0)) for k in WEIGHTS)
    try:
        import math
        return 1.0 / (1.0 + math.exp(-z))
    except OverflowError:
        return 0.0 if z > 0 else 1.0


def should_abstain(features: dict, threshold: float = DEFAULT_THRESHOLD) -> tuple[bool, float]:
    p = p_abstain(features)
    return p > threshold, p
