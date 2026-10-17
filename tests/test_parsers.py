"""LLM 输出解析测试：真实模型（gpt-4o-mini / qwen-flash）见过的各种形态都要能解析。"""
import pytest

from amc.retrieval.llm_rerank import _parse_ranking, _parse_selected


def test_ranking_strict_json():
    assert _parse_ranking('{"ranking": [2, 0, 1]}', 5) == [2, 0, 1]


def test_ranking_code_fence():
    """gpt-4o-mini 经常把 JSON 包在 ```json 代码块里。"""
    assert _parse_ranking('```json\n{"ranking": [1, 0]}\n```', 3) == [1, 0]


def test_ranking_order_alias_and_dedup_and_range():
    assert _parse_ranking('{"order": [0, 0, 9, 2]}', 3) == [0, 2]


def test_ranking_invalid_raises():
    with pytest.raises(Exception):
        _parse_ranking('{"ranking": []}', 3)
    with pytest.raises(Exception):
        _parse_ranking("not json at all", 3)


def test_selected_variants():
    assert _parse_selected('{"selected": [0, 2]}', 4) == [0, 2]
    assert _parse_selected('```json\n{"selected": [3]}\n```', 4) == [3]


def test_selected_tolerates_dirty_output():
    """尾随逗号 / 夹杂说明 / 截断 —— 解析器必须尽量救回来，不能整档丢弃。"""
    assert _parse_selected('{"selected": [0, 1,]}', 4) == [0, 1]
    assert _parse_selected('我选择 [2, 3]，因为……', 4) == [2, 3]
    assert set(_parse_selected('[1, 2', 4)) == {1, 2}      # 被 max_tokens 截断的数组也要救回
