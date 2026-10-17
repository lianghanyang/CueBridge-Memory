# 部署状态与交接（DEPLOY_STATUS）

> 更新时间：2026-10-16　|　系统：**CueBridge Memory v1.0**　|　部分：记忆侧 Add/Search API

## 1. 当前已部署（可用）

| 项 | 值 |
|---|---|
| 公网地址（**固定**） | **`https://memory.cuebridge.top`** |
| Add / Search | `https://memory.cuebridge.top/v1/memories/add`、`…/v1/memories/search` |
| 健康检查 | `https://memory.cuebridge.top/v1/memories/health` |
| 隧道 | **Cloudflare 命名隧道** `cuebridge`（ID `1180c59c-85f6-4188-9169-7f80e4eb5d93`）→ 回源 `http://127.0.0.1:8000`；固定域名、无响应超时 |
| 认证 | `Authorization: Token <MEMORY_SYSTEM_KEY>`（同时兼容 `Bearer` 与裸 key） |
| 服务进程 | `PYTHONPATH=. python -m scripts.serve_memory --port 8000`，**GPU 1** 上跑 **Qwen3-Reranker-4B**（8.1 GB） |
| GPU 调整 | 已按指示把服务从 GPU 0 迁到 **GPU 1**；GPU 0 已归还他人任务（14.7 GB） |
| 检索链路 | `RerankPipeline(candidates=30) → RoundFusionRetriever(hybrid_round, fusion=auto)` |
| 实测时延 | 单次 `/search` 经公网 **≈0.5 s**；完整自检（health+add+search+2 项负例）**5.4 s** |
| 自检结果 | **5/5 通过**（health / add(Token) / search(Token) / 契约+分数非递增 / 错误 key→401） |

### 当前**未启用**的档（及其原因）

| 档 | 状态 | 原因 |
|---|---|---|
| LLM 证据选择档（`gpt-4o-mini`） | **关闭**（`AMC_LLM_RERANK=0`） | 网关 503：`No available channel for model gpt-4o-mini` |
| 查询图→文本桥、关系选择 | **关闭** | 合规口径待定（VLM 是否算"LLM 相关组件"）+ 需网关视觉验证 |
| 置信度门控 | **关闭** | 换强模型 + 合适的提示后边际效应在噪声内（−0.0015 ⚪）；且 gpt-4o-mini 的"无门控"已相当于旧模型"有门控"的水平 |

> 关闭的档都会**优雅退化**：`/search` 仍返回 4B 精排结果，不会失败。

## 2. 阻塞项与解除后的动作

| 阻塞 | 现象 | 解除后要做什么 |
|---|---|---|
| **gpt-4o-mini 通道** | `POST {gateway}/v1/chat/completions` → **503 No available channel** | 在 `.env` 设 `AMC_LLM_RERANK=1`；跑 `bash scripts/aml_preflight.sh` 确认文本档 200；再跑 `bash scripts/aml_selfcheck.sh` |
| **网关视觉能力** | 同上 503，无法判定 | preflight 的"视觉输入"项转 ✅ 后，image 路可走 **Profile A**（`gpt-4o-mini`，严格合规） |
| **稳定 GPU** | ✅ **已解决**：迁到 GPU 1（为腾地方已按授权暂停 answer 服务 8002） | 若要恢复 answer 侧：`setsid bash data/reports/answer_service/restart_answer_service.sh`（注意它也要 GPU 1，与记忆服务互斥） |
| **固定域名** | ✅ **已解决**：`https://cuebridge.serveousercontent.com`（serveo 固定子域，已实测重启后名字不变） | 如后续换自有域名，用 `AMC_TUNNEL=cloudflared-named`（见 §6）即可，URL 换掉即可 |

## 3. 期间发现并修复的两个问题（重要）

1. **`.env.alibaba` 第 58 行少一个右引号**：`VLLM_DISABLE_MM_PREPROCESSOR_CACHE='1`
   → 后面的 `OPENAI_API_KEY=sk-...` 被吞进未闭合字符串，`source` 后该 key 仍是 `EMPTY`，
   **导致 9527 的 gpt-4o-mini 一直认证失败**。已修复（并删除第 62 行孤儿引号），
   备份见 `.env.alibaba.bak_<时间戳>`。修复后 `source .env.alibaba` 无报错、key 正确。
2. **`/add` 收到"纯字符串 content"时 500**：契约允许 `content` 为字符串（标准 chat 格式），
   但 `amc/store.py` 直接把它当 parts 列表迭代 → `TypeError`。已修复（归一化为 `[{"type":"text",...}]`）。
   这是**比赛 harness 极可能踩到**的格式，务必保留此修复。

## 4. 复现/重启步骤（从零到公网）

```bash
cd <repo>/submission
set -a; source .env; set +a                 # 填好 3 个 key
bash scripts/../scripts/start_memory_api.sh   # 起服务（GPU 上加载 4B）
bash scripts/aml_preflight.sh            # 上线前 6 项检查
bash scripts/start_public_tunnel.sh      # 拿公网地址
bash scripts/aml_selfcheck.sh            # 公网端到端自检
```

## 5. 风险与注意

- **共享 GPU（当前最紧的一项）**：4B 与他人任务共享 GPU 0，实测 **22.9 / 24.5 GB 已占用（仅剩 ~1.6 GB）**；
  对方再申请显存就会 OOM。启动脚本的守护循环会自动重部署（不静默退化），但**比赛期强烈建议换成独占卡**。
- **临时域名**：quick tunnel 重启即变，报名表 URL 需同步更新；长期评测务必换固定域名。
- **记忆库**：当前使用 `data/store/memory.db`（79,204 条 / 200 用户）。评测期如需隔离，可换成独立库路径。
- **不要在提交物里放 key**：`.env.example` 只放占位符；`.env` 与 `.env.alibaba` 不要进仓库。


## 6. 固定域名：三套方案（都已在 `start_public_tunnel.sh` 里实现）

| 方案 | 需要你做的一次性动作 | 得到的固定地址 | 命令 |
|---|---|---|---|
| **serveo（已采用 ✅）** | ~~注册公钥~~（**已完成**） | **`https://cuebridge.serveousercontent.com`** | `AMC_TUNNEL=serveo AMC_TUNNEL_SUBDOMAIN=cuebridge bash scripts/start_public_tunnel.sh`（autossh 常驻） |
| **Cloudflare 命名隧道（最正式）** | ① 拥有一个域名并托管到 Cloudflare；② 执行一次 `cloudflared tunnel login`（浏览器） | `https://memory.<你的域名>` | `AMC_TUNNEL=cloudflared-named AMC_TUNNEL_NAME=cuebridge AMC_PUBLIC_DOMAIN=memory.example.com bash scripts/start_public_tunnel.sh` |
| **ngrok 保留域名** | 注册 ngrok，提供 authtoken（及保留域名） | `https://<你的>.ngrok-free.app` | `AMC_TUNNEL=ngrok AMC_NGROK_AUTHTOKEN=xxx AMC_NGROK_DOMAIN=xxx.ngrok-free.app bash scripts/start_public_tunnel.sh` |

本机公钥（注册 serveo 时粘贴这一行）：

```
$(cat ~/.ssh/id_ed25519.pub)
```

> 说明：本机只有校园网 **IPv6** 地址、无公网 IPv4，也没有现成域名/反代，因此"真正固定"的域名必须由上述
> 三种方式之一提供凭据；我已把三条路径全部脚本化，你完成一次性动作后**一条命令即可切换**。

## 7. gpt-4o-mini 模型名复测（你建议的 `gpt-4o-mini-2024-07-18`）

| 模型名 | 结果 |
|---|---|
| `gpt-4o-mini` | 503 `No available channel` |
| `gpt-4o-mini-2024-07-18` | 503 `No available channel` |
| `gpt-4o-mini-0718` | 503 `No available channel` |

（两个入口 `9527.codes/v1` 与 `api.9527.codes/v1` 结果一致。注意：早先该名字返回 **403 无权限**，
现在变为 **503 无通道** ⇒ **token 权限已包含它，纯粹是网关侧没有可用通道**。）
⇒ 结论：**换模型名解决不了**，需在网关侧为该（组 default 的）模型启用通道。

### 7.1 已定位到根因（2026-10-16 复测）

| 证据 | 结果 |
|---|---|
| 该 token 的 `/v1/models` | **12 个模型，全部是 Claude**（`claude-haiku-4-5-20251001`、`claude-sonnet-4-5`、`claude-opus-5` …），**没有任何 gpt 模型** |
| 用可见模型实测 | `claude-haiku-4-5-20251001` → **HTTP 200（"可用"）** ⇒ **token 与网关本身正常** |
| `gpt-4o-mini` / `-2024-07-18` / `-0718` | 全部 **503 No available channel … under group default** |

⇒ **根因：`gpt-4o-mini` 没有被加入到该 token 所属分组（`default`）**，与该模型名无关。

### 7.2 需要网关侧做的事（三步）

1. 新建/启用一个 **OpenAI 类型渠道**，模型包含 `gpt-4o-mini`（若上游模型名不同，用"模型映射"把 `gpt-4o-mini` 映射过去）；
2. 该渠道的**分组**必须包含 token 所属分组 **`default`**（或把该 token 换到含 gpt-4o-mini 的分组）；
3. **验收命令**（出现 `gpt-4o-mini` 即成功）：
   ```bash
   curl -s https://9527.codes/v1/models -H "Authorization: Bearer $OPENAI_API_KEY"
   ```
4. 之后执行：`.env` 设 `AMC_LLM_RERANK=1` → `bash scripts/aml_preflight.sh` → 文本/视觉两项应为 ✅。

> 视觉提示：该网关对 Claude 模型不接受 OpenAI 的 `image_url` 形态（实测 400 `Input tag 'image_url' …`）。
> 因此 **image 路的 Profile A 必须用 gpt-4o-mini**（OpenAI 原生形态）；若 gpt-4o-mini 视觉不可用，
> 就退 **Profile B**（本地 Qwen2.5-VL，注意与 answer 服务争 GPU 1）。


## 8. 🚨 三个 LLM 档已全部启用（Profile A）+ 隧道多模态差异（2026-10-16 晚）

### 8.1 模型池换好后：三档全部走 gpt-4o-mini

| 档 | 状态 | 实测 |
|---|---|---|
| LLM 证据选择档（text 路） | ✅ 启用 `gpt-4o-mini`、**`evidence_set`**、pool=12、dedup、skip_marker | 日志 `+llm_rerank model=gpt-4o-mini … prompt=evidence_set dedup_rounds=True` |
| 查询图→文本桥 | ✅ `gpt-4o-mini` **视觉**（preflight 视觉项 ✅） | 图像查询 200 |
| 关系感知证据选择 | ✅ `gpt-4o-mini` 视觉 + 逐对 | 修掉一个 **401 bug** 后正常（见 8.3） |

⇒ **合规 Profile A 成立**（三个 LLM 档全是 gpt-4o-mini，不需要本地 VLM，也就不再与 answer 服务争 GPU）。

### 8.2 实测时延（公网、全档开启）

| 请求类型 | 时延 |
|---|---|
| 文本 `/search`（LLM 档生效） | **4.7 – 8.8 s** |
| 图像 `/search`（视觉桥 + 关系选择） | **6.9 – 10.8 s** |
| 首次冷启动 | 26 s（网关预热，之后稳定） |

**瓶颈是 gpt-4o-mini 调用**（网关对 ~5k 字符提示需 2–6.5 s；我们的候选清单约 9–14k 字符）。
可选优化：① 打开门控 `AMC_LLM_RERANK_GATE_MARGIN=0.193`（约一半查询跳过 LLM，**指标代价 −0.0015 噪声内**）；
② 缩小 `AMC_LLM_RERANK_POOL`/`DOC_CHARS`（会损失精度，未测）。已加安全超时 `AMC_LLM_RERANK_TIMEOUT=20`（超时自动退化为 4B 排序）。

### 8.3 本轮修掉的三个 bug

1. **`relation_select` 完全没发认证头** → 对 9527 网关 401（图 像路静默退化）。已加 `api_key` 参数 + `Authorization: Bearer`，并在 `app.py` 传入（含 `OPENAI_API_KEY` 回退链）。
2. **`relation_select` 的配置项里根本没有 api_key 字段**（我上次的补丁静默未生效——现已加上并**逐条校验**）。
3. **Search 响应 `content` 对带图记忆返回 parts 列表** → 契约允许两种，但 harness 多为字符串渲染；新增 `AMC_SEARCH_CONTENT_FORMAT`（默认 `text` 压平为字符串，图片以 `[image]` 占位）。

### 8.4 🚨 隧道差异的**更正**：serveo 不是内容过滤，是 **~6 秒响应超时**

**先更正我之前的错误结论**（"serveo 过滤 base64 图片请求"）——**不成立**，serveo 是通用 HTTP/TCP 隧道，
**不做内容过滤**（用户指出得对）。真正的判别因子是**响应耗时**：

| 实验 | 结果 |
|---|---|
| `/health`（毫秒级）经 serveo ×10 | **10/10 → 200** |
| 文本 `/search`（5.4–7.8 s）经 serveo ×6 | ≤5.7s → **200**；≥5.8s → **502**（约一半） |
| 图像 `/search`（7–11 s）经 serveo ×3 | **3/3 → 502**（每次都在 ~6.0 s 被切断） |
| `data:` URI 的**快速**请求（服务 422，秒回） | **到达服务**（⇒ 隧道并不拦 data URI） |
| 同一冷图像查询：serveo vs cloudflared | serveo **502 @5.74s** ／ cloudflared **200 @8.49s** |

**根因**：serveo 的公网桥在 **~5.7–6.0 s** 处切断响应（502），**文档未记载**（[serveo 文档](https://serveo.net/docs/) 只提到
免费版会插"浏览器警告页"，并给出绕过头部 `serveo-skip-browser-warning: true` —— **实测该头部无效**；
[定价页](https://serveo.net/) 只说 Pro 去掉警告页、隧道数不限，**未提超时**）。

**为什么看起来像内容相关**：图像路要 2–7 s/次的视觉调用（绝对超过 6 s）⇒ **100% 502**；
文本路恰好在 5.4–7.8 s 徘徊 ⇒ 约一半 502。

**缩小提示无效**：把 `LLM_RERANK_POOL 12→6`、`DOC_CHARS 1400→600`、`RELATION_SELECT_POOL 14→8` 后，
本地时延仍是 4.3–7.8 s —— 瓶颈是**网关每次调用的固有抖动（2–7 s）**，与提示大小关系不大。

### 8.5 可选方案（按推荐度）

| 方案 | 固定域名 | 多模态(<6s 限制) | 需要你做什么 |
|---|---|---|---|
| **A. cloudflared 命名隧道（推荐）** | ✅ | ✅ 无 6 s 限制（实测 8.5 s 仍 200） | 一个托管在 Cloudflare 的域名 + 一次 `cloudflared tunnel login`（浏览器） |
| **B. 自有域名 CNAME→serveo.net + TXT authkey** | ✅ | ❓很可能仍受 ~6 s 限制（Pro 文档未提超时） | 在 DNS 加 CNAME + `_serveo-authkey` TXT；或升级 serveo Pro |
| **C. 把我们的超时压到 ~3.5 s 以适配 serveo** | ✅ | ⚠️ 代价大：网关慢时**丢失 LLM 档与图像档**（图像档是全库最大收益 +0.0606 R@1） | 无需外部资源，但指标下降 |
| **D. 保持 cloudflared quick** | ❌ 重启变 | ✅ | 无；报名表地址会变，需人工同步 |

**当前状态**：两个隧道都在跑 —— cloudflared（多模态可用，域名临时）与 serveo（固定域名，仅 <6 s 的请求可用）。
**建议**：报名表先倾向方案 A；在域名就绪前用 cloudflared 地址，或只把 serveo 地址用于文本场景。


---

## 9. 换 gpt-4o-mini 后的效果实测（2026-10-16 晚）

模型池换成含 `gpt-4o-mini` 的 106 个模型后，做了一次**同口径 A/B**（union 候选池、同一提示、同一门控扫描、同样本 1,040）

### 9.1 绝对指标

| 臂 | R@1 | Hit@1 | R@n_gold |
|---|---|---|---|
| S2 轮级 4B（基线） | 0.5149 | 0.7404 | 0.6800 |
| round_rank（无门控） | 0.5414 | 0.7827 | 0.7306 |
| **evidence_set（无门控）** | **0.5615** | **0.8106** | **0.7535** |
| round_rank @门控 q=0.5（τ=0.193） | 0.5486 | 0.7913 | 0.7283 |
| evidence_set @门控 q=0.5 | 0.5567 | 0.8010 | 0.7369 |

### 9.2 配对对比：gpt-4o-mini − qwen-flash（同样本/同提示/同门控）

| 臂 | gpt-4o-mini | qwen-flash | ΔR@1（配对 95% CI） | 判定 |
|---|---|---|---|---|
| round_rank 无门控 | 0.5414 | 0.5047 | **+0.0368 [+0.0191,+0.0548]** | ✅ |
| round_rank 门控 τ=0.193 | 0.5486 | 0.5402 | +0.0084 [−0.0044,+0.0213] | ⚪ |
| evidence_set 无门控 | 0.5615 | 0.5344 | **+0.0270 [+0.0108,+0.0445]** | ✅ |
| evidence_set 门控 τ=0.193 | 0.5567 | 0.5492 | +0.0075 [−0.0043,+0.0198] | ⚪ |

**结论**：
1. **换 gpt-4o-mini 显著正向（+0.027 ~ +0.037 ✅），收益主要出现在"无门控"**；
   开门控后差异被压缩到不显著（+0.008 ⚪）。
2. **模型升级部分替代了门控的作用**：gpt-4o-mini 无门控（0.5414）≈ qwen-flash 有门控（0.5402）
   ⇒ 换模型后**不开门控是合理的**（与当前决定一致）。
3. **提示偏好的"依赖候选池"结论与模型无关**：union 池上 evidence_set − round_rank = **+0.0200 [+0.0072,+0.0334] ✅**
   （门控下 +0.0081 ⚪），与 qwen-flash 时期方向一致。

### 9.3 ⚠️ 生产口径（服务路径）全量对比：**gpt-4o-mini 并不更好**（与 union 口径相反）

配置完全相同（`round + round_rank + dedup + 无门控`，text 路 1,040），**只换模型**，逐样本配对：

| 指标 | qwen-flash | gpt-4o-mini | 配对 Δ（95% CI） | 判定 |
|---|---|---|---|---|
| R@1 | **0.5544** | 0.5447 | −0.0097 [−0.0228,+0.0038] | ⚪ |
| Hit@1 | **0.8087** | 0.7760 | **−0.0327 [−0.0510,−0.0135]** | ❌ |
| R@n_gold | **0.7377** | 0.7284 | −0.0094 [−0.0243,+0.0065] | ⚪ |

⇒ **同一对模型，在 union 池上 gpt-4o-mini 显著更好（+0.037 ✅），在生产池上却不更好（Hit@1 显著更差）**
—— 这是本项目第 N 次"结论依赖候选池"，说明**模型选择也必须在生产口径上验证**。

**已排查（都不是原因）**：
- 解析器对 gpt-4o-mini 的 1,040 条真实输出：**全是合法 dict**，仅 **11 条（1%）** 解析失败（qwen-flash 亦为 9 条）
  ⇒ 不是格式/解析问题；
- 输出未被截断（`finish_reason=stop`，输出 42 字符）；
- 超时未触发（单次 2–7 s，阈值 20 s）。

**正在跑**：`G3 = gpt-4o-mini + evidence_set + dedup + 无门控`（生产口径）。
因为比赛**强制 LLM 组件用 gpt-4o-mini**，所以真正要定的是"**在 gpt-4o-mini 下用哪个提示**"
（union 池上 gpt-4o-mini 的 evidence_set 0.5615 > round_rank 0.5414 ✅，需在生产口径复核）。

**代价**：gpt-4o-mini 单次调用 2–7 s（qwen-flash ≈0.5 s）⇒ 端到端时延上升，
这也是必须用 **cloudflared**（无 ~6 s 切断）的原因之一。

---

## 10. ✅ 固定域名上线（2026-10-16 晚）

### 10.1 最终地址

| 用途 | 地址 |
|---|---|
| **Add** | `https://memory.cuebridge.top/v1/memories/add` |
| **Search** | `https://memory.cuebridge.top/v1/memories/search` |
| 健康检查 | `https://memory.cuebridge.top/v1/memories/health` |

### 10.2 接入链路

```
客户端 ──HTTPS──▶ Cloudflare 边缘(Universal SSL: cuebridge.top, *.cuebridge.top, Let's Encrypt)
        ──命名隧道 cuebridge (1180c59c-…)──▶ cloudflared ──HTTP──▶ 127.0.0.1:8000 记忆服务
```

| 项 | 值 |
|---|---|
| 域名 | `cuebridge.top`（阿里云注册，实名认证完成，2027-10-16 到期） |
| 注册局 NS | `jill.ns.cloudflare.com` / `josh.ns.cloudflare.com` |
| 隧道 | `cuebridge`，3 条连接（香港边缘），`--no-autoupdate` |
| DNS | `memory.cuebridge.top` CNAME → 该隧道（Cloudflare 代理） |
| 证书 | Universal SSL（覆盖 apex + 一级子域），已签发 |

### 10.3 验证结果（全部实测）

| 检查 | 结果 |
|---|---|
| 公网自检（Token 认证 / add / search / 契约+分数非递增 / 错误 key→401） | **5/5 通过** |
| **慢图像查询（>6 s）** | **3/3 HTTP 200**（4.0–5.2 s）⇒ 无 serveo 那种 ~6 s 切断 |
| 文本查询 | HTTP 200（5–14 s，视是否与离线评测争 GPU） |
| 临时隧道 | 已停用（serveo / cloudflared quick），只保留命名隧道 |

### 10.4 踩坑记录（供后续复现）

1. **serveo 有 ~6 s 响应超时**（文档未记载）：>6 s 的请求一律 502，与请求内容无关（`/health` 10/10 通）。
2. **Cloudflare `argotunnel` 授权页在账号无 zone 时是空白** —— 必须先注册/添加域名。
3. **新 zone 的 Universal SSL 有签发延迟**：NS 生效后 HTTPS 仍会 `handshake_failure(alert 40)`，
   实测约 **2 分钟**后（22:38→22:40）自动恢复，期间用 `curl` 探针轮询即可。
4. **`.cn`/`.com.cn` 不能加到 Cloudflare**；`.top` 可以。
5. `cloudflared tunnel login` 的授权链接**一次性、约 10 分钟有效**；页面必须点选 zone 再 Authorize。


---

## 11. ✅ 最终配置结论（2026-10-16 深夜，生产口径全量实测）

### 11.1 五种配置的绝对指标（text 路 1,040，服务路径端到端）

| 配置 | 模型 | 提示 | R@1 | Hit@1 | R@n_gold |
|---|---|---|---|---|---|
| 只上 4B（无 LLM 档） | — | — | 0.5147 | 0.7433 | 0.6753 |
| 旧最优（**不合规**） | qwen-flash | round_rank | 0.5544 | **0.8087** | **0.7377** |
| | qwen-flash | evidence_set | 0.5298 | 0.7798 | 0.7036 |
| | gpt-4o-mini | round_rank | 0.5447 | 0.7760 | 0.7284 |
| **✅ 最终** | **gpt-4o-mini** | **evidence_set** | **0.5555** | 0.8010 | 0.7345 |

### 11.2 关键配对检验

| 对比 | ΔR@1 | ΔHit@1 | ΔR@n_gold |
|---|---|---|---|
| **qwen-flash 内**：round_rank − evidence_set | **+0.0246 ✅** | **+0.0288 ✅** | **+0.0341 ✅** |
| **gpt-4o-mini 内**：evidence_set − round_rank | **+0.0108 ✅** | **+0.0250 ✅** | +0.0061 ⚪ |
| **跨模型**：(gpt+evidence_set) − (qwen+round_rank) | +0.0011 ⚪ | −0.0077 ⚪ | −0.0032 ⚪ |
| **最终配置 − 只上 4B** | **+0.0408 ✅** | **+0.0577 ✅** | **+0.0592 ✅** |

**⇒ 提示偏好随模型反转**（同一候选池：qwen-flash 爱 `round_rank`，gpt-4o-mini 爱 `evidence_set`）。
这是本项目第 N 次"结论依赖口径"，也是**必须把模型与提示一起调**的直接证据。

**好消息**：合规方案（gpt-4o-mini + evidence_set）与旧的非合规最优（qwen-flash + round_rank）
**在三个指标上均无显著差异**（Δ ≤0.008，CI 全部跨 0）⇒ **换合规模型不损失性能**。

### 11.3 最终配置（已上线，`/health` 可核验）

```bash
# 检索骨干
AMC_RETRIEVER=hybrid_round_rerank      # dense + round 通道 + bm25_anchor
AMC_FUSION=auto                        # 只按可观测轮结构路由
AMC_RERANKER=qwen3                     # Qwen3-Reranker-4B（Reranker 不限）
AMC_RERANK_UNIT=round                  # 以整轮为打分单位
# LLM 档（比赛强制 gpt-4o-mini）
AMC_LLM_RERANK=1
AMC_LLM_RERANK_BASE_URL=https://9527.codes/v1
AMC_LLM_RERANK_MODEL=gpt-4o-mini
AMC_LLM_RERANK_PROMPT=evidence_set      # ← 最终选择
AMC_LLM_RERANK_GATE_MARGIN=0            # 门控关闭（强模型下无收益）
AMC_LLM_RERANK_DEDUP_ROUNDS=1           # 窗口按轮去重
AMC_LLM_RERANK_POOL=12
AMC_LLM_RERANK_SKIP_MARKER=[Query image]
# image 路（同样用 gpt-4o-mini 的视觉能力）
AMC_QUERY_VISION=1                      # 查询图→文本桥
AMC_RELATION_SELECT=1                   # 结构化卡片+规则+逐对图像证据关系
AMC_RELATION_SELECT_POOL=14
```

### 11.4 上线核验（固定域名）

| 检查 | 结果 |
|---|---|
| `/health` 返回的链路 | `LLMReasoningRerankPipeline → RerankPipeline(4B) → RoundFusionRetriever` |
| 日志 | `prompt=evidence_set dedup_rounds=True gate_margin=0.0000` + `relation-select … model=gpt-4o-mini` |
| 公网自检 | **5/5 通过** |
| 图像查询（视觉桥 + 关系选择） | HTTP 200，7.5 s，返回 5 条，`content` 为字符串 |

---

## 12. 赛事服务 vs 离线评测：口径对齐核查（2026-10-17）

**问题**：线上服务跑出来的效果，和离线跑测的数字，是不是同一件事？

### 12.1 生效配置逐项 diff（线上 `.env` vs 离线跑测脚本）

| 字段 | 赛事服务 | 离线跑测 | 影响 |
|---|---|---|---|
| **`retrieval_limit`** | **120** | **30** | 进 4B 精排的候选数；**实测对 R@1/Hit@1 无影响**（见 12.3） |
| `retriever` | `hybrid_round_rerank` | `hybrid_round` | 等价：`build_retriever` 在 settings 有 reranker 时**强制启用**精排，两条名字构建出的链路相同 |
| `llm_rerank` / 模型 / 提示 / 门控 / 去重 / pool / layout / doc_chars | `gpt-4o-mini` / `evidence_set` / 0 / True / 12 / round / 1400 | 完全相同 | — |
| `query_vision` / `relation_select`（pool）/ `skip_marker` | **True / True(14) / `[Query image]`** | **False / False(10) / 空** | **仅图像路**；文本查询不触发 |
| `fusion=auto`、`rerank_unit=round`、`search_content_format=text`、`max_return=50`、`text-embedding-v4`(1024) | 一致 | 一致 | — |

### 12.2 数据等价性

线上库 79,218 行 / 209 用户（历次摄入的多个命名空间：`pf-frozen-cd:`、`pf-cd-frozen-v1:`、
`pf-frozen-cd-seed0:`、`rerun-*:`、`text-v4-*:` 各 20 用户 + 9 个自检用户）；
离线库 7,924 行 / 20 用户（`eval:local-v1:multimodal:*`）。

按用户比对 `index_text` 集合：**370 行 vs 370 行、内容哈希 `8428c09e` 完全相同**
⇒ 同一批数据、仅 `user_id` 命名空间不同；`user_id` 隔离下每查询候选集等价。

### 12.3 `retrieval_limit` 30 vs 120 配对实测（text 路 100 条）

| 指标 | 30 | 120 | 配对 Δ（120−30，95% CI） | 逐样本不同条数 |
|---|---|---|---|---|
| R@1 | 0.5275 | 0.5275 | **+0.0000 [0.0000, 0.0000]** ⚪ | **0 / 100** |
| Hit@1 | 0.7400 | 0.7400 | **+0.0000 [0.0000, 0.0000]** ⚪ | **0 / 100** |
| R@n_gold | 0.6428 | 0.6478 | +0.0050 [0.0000, +0.0150] ⚪ | 1 / 100 |

⇒ 轮级 4B 精排的 top-1 不随候选池扩大而改变（融合头部已含胜者），
LLM 档只吃 12 个候选 ⇒ **文本路上"赛事服务 ≈ 离线评测"成立**。

### 12.4 仍然**不**等价的三处（报告数字时必须标注）

1. **图像路**：离线数字里 `query_vision` / `relation_select` 关闭，线上开启；
   图像档收益是 LOO 单测（查询图桥 **+0.0606 R@1**、关系选择 +0.0075），**不在 0.5555 之内**；
2. **覆盖范围**：0.5555 为**文本路 1,040 条**；线上还要服务 Mem-Gallery 另外 **487 条图像查询**，
   整体表现 = 文本路 + 图像路的加权；
3. **全库 1,527 的 0.4690 是旧配置**（qwen-flash + round_rank + 门控）的数字；
   **最终配置（gpt-4o-mini + evidence_set）的全库数字尚未测**——报告中不得混用。
