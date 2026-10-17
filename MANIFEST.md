# 交付物清单

| 类别 | 路径 | 说明 |
|---|---|---|
| 运行说明 | `README.md` | 报名表"提交说明与运行指南"正文（部署/配置/API 契约/指标） |
| 配置模板 | `.env.example` | 复制为 `.env` 后填密钥；含全部开关与合规注释 |
| 依赖 | `pyproject.toml`、`requirements.txt` | `pytest` 配置已就位 |
| 容器 | `Dockerfile`、`compose.yaml`、`.dockerignore` | 一键构建/启动（GPU 预留段可删） |
| 记忆服务 | `amc/` | FastAPI 应用、存储、契约、检索与重排 |
| 运维脚本 | `scripts/` | 启动、隧道、上线前检查、公网自检 |
| 测试 | `tests/` | 契约 / 鉴权 / LLM 输出解析（20 例，离线秒级） |

## 不随仓库提交的内容

- `.env` / 任何 API 密钥（`.gitignore` 已覆盖；模板只留占位符）
- `data/`：记忆库、blob、embedding 缓存、评测报告
- 公开评测原始数据与运行结果

## 复现路径

```bash
pip install -e .            # 或 pip install -r requirements.txt
pytest -q                   # 20 例离线测试
cp .env.example .env        # 填 DASHSCOPE_API_KEY / gpt-4o-mini KEY / MEMORY_SYSTEM_KEY
bash scripts/aml_preflight.sh    # 上线前检查（key / 嵌入维度 / LLM / 视觉 / GPU / 库）
bash scripts/start_memory_api.sh # 起服务（0.0.0.0:8000）
bash scripts/aml_selfcheck.sh    # 公网自检：health / add / search / 契约 / 401
```
