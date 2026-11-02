---
name: seekdb-hybrid-rag
description: 用 seekdb 搭建端侧混合检索知识库的实操指南：向量索引 + 全文索引建表、单条 SQL 混合检索、常见故障排查
---

# seekdb 混合检索知识库（端侧向量化 + 库内混合检索）

一个可直接落地的最小方案：**embedding 100% 在浏览器里算**（Transformers.js + ONNX Runtime Web，优先 WebGPU，自动降级 WASM），向量、原文与元数据写入 **seekdb**，查询时的「向量 + 全文」混合检索由 seekdb **一条 SQL** 完成并直接排序返回，客户端不做两路召回拼合。

```
浏览器                          本地 Node 服务                seekdb
Transformers.js                 /api/ingest   只写入不推理      kb_xxx 集合
all-MiniLM-L6-v2 INT8（384 维）──向量──▶  仅写入          ──▶  VECTOR(384) + HNSW
WebGPU ⇄ WASM 降级              /api/search   混合检索单条 SQL ◀──  FULLTEXT 索引
```

## 适用场景

- 端侧 / 离线优先的文档问答、个人知识库、内网知识检索。
- 有隐私要求、不希望把原文或查询发到云端推理服务的场景。
- 需要「语义 + 关键词」双通道召回，又不想自己维护两套召回与融合逻辑。

## 步骤一：启动 seekdb

Docker（推荐，Windows 下最省事）：

```bash
docker run -d --name seekdb -p 2881:2881 oceanbase/seekdb:latest
```

本地可执行文件方式：

```bash
seekdb.exe --base-dir <空目录> --port 2881 --nodaemon --parameter memory_limit=6G
```

连接参数通过环境变量覆盖：`SEEKDB_HOST` / `SEEKDB_PORT` / `SEEKDB_USER` / `SEEKDB_PASSWORD` / `SEEKDB_DB`。
服务启动后用 `GET /api/health` 确认数据库链路是否就绪。

## 步骤二：建表 + 向量索引 + 全文索引

表结构（由 `seekdb-js.createCollection()` 按 `schema.sql` 自动创建，也可手工建）：

| 列 | 类型 | 用途 |
| --- | --- | --- |
| doc_id / chunk_id | VARCHAR(128) / INT | 联合主键，保证增量导入幂等 |
| title / content | VARCHAR(512) / MEDIUMTEXT | 原文与展示标题 |
| meta | JSON | 来源、批次等元数据 |
| embedding | VECTOR(384) | 端侧生成的向量 |
| created_at | BIGINT | 写入时间，用于批次/会话管理 |

两类索引在写入前创建，写入只走批量 upsert，不做全量重建：

```sql
CREATE VECTOR INDEX idx_vec ON kb_default (embedding) WITH (type=HNSW, distance=cosine);
CREATE FULLTEXT INDEX idx_ft ON kb_default (content);
```

- 向量索引用 **HNSW + cosine**，支持增量插入，新写入的文档立即可查。
- 若目标版本未开启全文索引能力，自动退化为库内 `LIKE` 打分（仍是库内计算），检索响应里的 `mode` 字段会暴露实际模式，前端可据此提示。
- 集合名由界面传入，`kb_a` / `kb_b` 相互隔离，切换集合后检索结果与当前知识库严格一致。

## 步骤三：端侧向量化（离线 + INT8）

- 模型：`sentence-transformers/all-MiniLM-L6-v2`（Apache-2.0，384 维），权重可取魔搭镜像 `AI-ModelScope/all-MiniLM-L6-v2`。
- 量化：`onnxruntime` 的 `quantize_dynamic(weight_type=QuantType.QUInt8)`，对 MatMul/Gemm 权重做动态 INT8，激活在线量化。体积 86.2 MB → 21.8 MB（**3.95×**），与 FP32 的文档向量余弦均值 **0.9888**，Top-1 检索结果完全一致。
- 运行时：Transformers.js 3.0.2 + ONNX Runtime Web。先探测 `navigator.gpu`，有则 `device:'webgpu', dtype:'q8'`；初始化抛错回退 `device:'wasm'`，并在界面徽标显示当前实际后端（不静默切换）。
- 离线：`env.allowRemoteModels=false`、`env.localModelPath='/models/'`，模型从应用自身静态目录加载，不走公网。

量化命令参考：

```bash
python tools/quantize_int8.py --src web/models/all-MiniLM-L6-v2/onnx/model.onnx \
    --dst web/models/all-MiniLM-L6-v2/onnx/model_quantized.onnx \
    --tokenizer web/models/all-MiniLM-L6-v2/tokenizer.json \
    --report benchmark/quant_report.json
```

## 步骤四：一条 SQL 完成混合检索

库内同时算出 `vec_score = 1 - COSINE_DISTANCE(embedding, :q)` 与 `MATCH ... AGAINST` 的 `txt_score`，再按 `α·向量 + (1-α)·全文` 排序并 `LIMIT`，前端只拿到已排序结果：

```sql
SELECT doc_id, chunk_id, title, content,
       ROUND(1 - COSINE_DISTANCE(embedding, :q), 6) AS vec_score,
       ROUND(MATCH(content) AGAINST (:qtext), 6) AS txt_score,
       ROUND(0.7 * (1 - COSINE_DISTANCE(embedding, :q)) + 0.3 * MATCH(content) AGAINST (:qtext), 6) AS score
FROM kb_default
ORDER BY score DESC
LIMIT 8;
```

- `α`（示例 0.7）由界面滑杆控制，纯语义检索可置 1.0，纯关键词检索可置 0.0。
- 不要分两次查询再在应用层做 RRF/加权融合——那会把排序逻辑从库里搬到客户端，分页与一致性都变差。

## 步骤五：增量写入

```sql
INSERT INTO kb_default (doc_id, chunk_id, title, content, meta, embedding, created_at)
VALUES (?, ?, ?, ?, ?, ?, ?)
ON DUPLICATE KEY UPDATE content = VALUES(content), embedding = VALUES(embedding), created_at = VALUES(created_at);
```

按 `(doc_id, chunk_id)` 覆盖写，重复导入不产生重复向量；写入完成后即可检索（HNSW 增量插入），无需重建索引。

## 常见故障排查

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `seekdb.exe` 一启动就退出，退出码 `STATUS_0xC0000135` | 缺少 VC++ 运行时 DLL | 安装 Microsoft Visual C++ 运行库，或改用 `oceanbase/seekdb:latest` docker 镜像 |
| 检索响应 `mode` 显示 `like` 而非 `fulltext` | 当前版本未开启全文索引能力 | 属预期降级（仍在库内打分）；需要真全文时升级镜像/版本 |
| 服务返回 503，页面提示数据库不可用 | seekdb 未启动或连接参数错误 | 查 `/api/health`，核对 `SEEKDB_HOST/PORT/USER/PASSWORD/DB`；**不要**退化成内存检索伪装成功 |
| 界面徽标显示 `WASM（INT8）` 而非 `WebGPU（INT8）` | 浏览器/驱动未启用 WebGPU，或初始化失败已回退 | 用 `--disable-features=WebGPU` 复现降级路径，确认 import + 检索流程仍完整 |
| 模型加载 404 / 首次请求卡住 | `localModelPath` 指向的目录为空或相对路径错 | 确认 `web/models/all-MiniLM-L6-v2/` 内含 tokenizer 与 INT8 ONNX；保持 `allowRemoteModels=false` |
| 首次查询明显慢于后续 | 模型冷启动（下载/加载/预热） | 单例加载一次 pipeline，导入与查询复用同一实例 |
| 长文本导入时内存/显存尖峰 | 一次性喂入整篇 | 前端按 `BATCH=8` 分批向量化，进度按批推进 |
| 以为写进了 A 库其实写进了 B 库 | 集合名与会话状态未回显 | 每次导入回显集合名 + `total_rows` |

## 验收自查清单

1. 断开外网后仍能完成导入与检索（证明向量化在端侧）。
2. 同一段文本重复导入两次，`total_rows` 不增长（幂等）。
3. 检索结果同时含 `score / vec_score / txt_score`，三者可由 α 复算一致。
4. 关闭 seekdb 后接口明确返回 503 与启动提示，而不是静默返回空结果。
5. 切换集合名后结果不串库。

<!-- OB-S:ChjdS9b9C-5nP7eZTHwrZQ -->