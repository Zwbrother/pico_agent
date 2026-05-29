# runtime 三项改进

**改动文件**：`pico/task_state.py`（+11 / -3）、`pico/runtime.py`（+65 / -18）、`pico/tools.py`（+2 / -1）

---

## 背景

改动前，runtime 在以下三个维度存在明显短板：

| 短板 | 影响 |
|---|---|
| `retry` 类别笼统 | 空响应、JSON 格式错、XML 格式错、`<final>` 为空统统落入同一个 `retry` 桶，report 和评测无法区分 |
| trace 事件缺 `phase` 字段 | 事件序列没有阶段标注，可视化和按阶段聚合分析需要在外部重新推断 |
| delegate 调用产生孤儿 run | 子 agent 的 trace / report 与父 run 没有明确关联，多轮 delegate 后无法还原调用树 |

---

## 改动一：细化失败类别

### 1-A 新增 `STOP_REASON_BACKEND_ERROR`

**位置**：`task_state.py`

```python
# 改动前
STOP_REASON_MODEL_ERROR = "model_error"   # 已有，但从未被设置

# 改动后新增
STOP_REASON_BACKEND_ERROR = "backend_error"
```

`model_error` 原本意图覆盖"模型侧出了问题"，范围太宽。新增的 `backend_error` 专门对应 HTTP 层故障：网络连接失败、后端返回 5xx 并超过重试上限、响应体无法解析等——这些错误来自 `model_client.complete()` 抛出的 `RuntimeError`，与模型生成内容本身无关。

`TaskState` 同步新增 `stop_backend_error()` 方法，与现有 `stop_model_error()` 并列：

```python
def stop_backend_error(self, final_answer=""):
    return self.stop(STOP_REASON_BACKEND_ERROR, status=STATUS_FAILED, final_answer=final_answer)
```

### 1-B `parse()` 返回三元组，带 `parse_detail`

**位置**：`runtime.py` → `Pico.parse()`

改动前返回 `(kind, payload)`，改动后返回 `(kind, payload, parse_detail)`。

`parse_detail` 是一个 dict，精确标注本次解析的结果：

| kind | parse_detail |
|---|---|
| `tool`（JSON 格式） | `{"format": "json"}` |
| `tool`（XML 格式） | `{"format": "xml"}` |
| `final`（`<final>` 标签） | `{"format": "tagged"}` |
| `final`（纯文本） | `{"format": "plain"}` |
| `retry` | `{"retry_reason": "empty_response" \| "json_parse_error" \| "json_schema_error" \| "xml_parse_error" \| "empty_final"}` |

`retry_reason` 的五个取值覆盖了所有 parse 失败路径：

| retry_reason | 触发条件 |
|---|---|
| `empty_response` | `raw` 为空或全空白 |
| `json_parse_error` | `<tool>` 标签内的内容不是合法 JSON |
| `json_schema_error` | JSON 解析成功，但 payload 不是 dict，或缺少 `name` 字段，或 `args` 不是 dict |
| `xml_parse_error` | XML 属性格式工具调用，`parse_xml_tool()` 返回 `None` |
| `empty_final` | `<final>` 标签存在但内容为空 |

`model_parsed` trace 事件现在包含 `parse_detail`，可以直接按 `retry_reason` 聚合，不需要再反向解析 retry notice 文本。

### 1-C `model_client.complete()` 包 try/except

**位置**：`runtime.py` → `ask()` 主循环 步骤3

改动前，`model_client.complete()` 抛出的 `RuntimeError` 会穿透整个 `ask()` 直接打到 CLI 顶层，既不写 trace 也不写 report。

改动后：

```python
try:
    raw = self.model_client.complete(...)
except RuntimeError as _backend_exc:
    task_state.stop_backend_error(str(_backend_exc))
    self.record({"role": "assistant", "content": str(_backend_exc), ...})
    self.run_store.write_task_state(task_state)
    self.emit_trace(task_state, "run_finished", {
        "status": task_state.status,
        "stop_reason": task_state.stop_reason,
        "run_duration_ms": ...,
    })
    self.run_store.write_report(task_state, ...)
    return f"Stopped due to backend error: {_backend_exc}"
```

后端异常现在和其他终止条件（`step_limit`、`retry_limit`）走同一套收尾流程，run 目录下会有完整的 `task_state.json`、`trace.jsonl`、`report.json`。

---

## 改动二：补显式 phase 字段

**位置**：`runtime.py` → 常量 `TRACE_EVENT_PHASE` + `_CHECKPOINT_TRIGGER_PHASE` + `emit_trace()`

### 阶段划分

runtime 主循环天然分为五个阶段，现在通过 `phase` 字段在每条 trace 事件上显式标注：

| phase | 含义 | 对应事件 |
|---|---|---|
| `init` | 运行初始化 | `run_started`、`runtime_identity_mismatch` |
| `plan` | 感知 / 构建 prompt | `prompt_built`、`model_requested`、freshness/context 类 checkpoint |
| `decide` | 解析模型输出 | `model_parsed` |
| `act` | 执行工具 | `tool_executed`、tool 后的 checkpoint |
| `finish` | 收尾 | `run_finished`、结束时的 checkpoint |

`checkpoint_created` 事件没有固定 phase，由其 `trigger` 字段动态推导：

```python
_CHECKPOINT_TRIGGER_PHASE = {
    "tool_executed":       "act",
    "run_finished":        "finish",
    "step_limit_reached":  "finish",
    "retry_limit_reached": "finish",
    "run_stopped":         "finish",
    "freshness_mismatch":  "plan",
    "workspace_mismatch":  "plan",
    "context_reduction":   "plan",
}
```

### 实现方式

`emit_trace()` 在写入前自动注入，所有调用方无需改动：

```python
def emit_trace(self, task_state, event, payload=None):
    payload = self.redact_artifact(payload or {})
    payload["event"] = event
    payload["created_at"] = now()
    phase = TRACE_EVENT_PHASE.get(event)
    if phase is None and event == "checkpoint_created":
        trigger = payload.get("trigger", "")
        phase = _CHECKPOINT_TRIGGER_PHASE.get(trigger, "act")
    if phase:
        payload["phase"] = phase
    self.run_store.append_trace(task_state, payload)
    return payload
```

### 一条成功 run 的 trace 阶段序列

```
run_started        phase=init
prompt_built       phase=plan
model_requested    phase=plan
model_parsed       phase=decide   ← kind=tool
tool_executed      phase=act
checkpoint_created phase=act
prompt_built       phase=plan
model_requested    phase=plan
model_parsed       phase=decide   ← kind=final
run_finished       phase=finish
checkpoint_created phase=finish
```

---

## 改动三：父子 run 关系建模

### 3-A `TaskState` 新增 `parent_run_id` 字段

**位置**：`task_state.py`

```python
@dataclass
class TaskState:
    ...
    parent_run_id: str = ""   # 新增；根 run 为空串，子 run 填父 run_id
```

`create()`、`from_dict()`、`to_dict()` 均同步更新，向后兼容（旧 `task_state.json` 反序列化时缺省为 `""`）。

### 3-B `Pico.__init__` 新增 `parent_run_id` 参数

**位置**：`runtime.py`

```python
def __init__(self, ..., parent_run_id=""):
    ...
    self.parent_run_id = str(parent_run_id or "")
    self._current_run_id = ""   # ask() 启动时填充，供子 agent 回读
```

`ask()` 在创建 `TaskState` 时传入，并立即将当前 `run_id` 暴露到 `self._current_run_id`：

```python
task_state = TaskState.create(
    run_id=self.new_run_id(),
    task_id=self.new_task_id(),
    user_request=user_message,
    parent_run_id=self.parent_run_id,   # 新增
)
self._current_run_id = task_state.run_id
```

`parent_run_id` 同时写入 `run_started` trace 事件和 `report.json`。

### 3-C `tool_delegate` 透传 `parent_run_id`

**位置**：`tools.py` → `tool_delegate()`

```python
child = Pico(
    ...
    parent_run_id=getattr(agent, "_current_run_id", ""),   # 新增
)
```

`agent._current_run_id` 在 `ask()` 启动时即被设置，在 `tool_delegate` 被调用时始终有效。

### 调用树的还原方式

通过 `run_id → parent_run_id` 链，可以把任意深度的 delegate 树还原：

```
run_20260529-120000-root   parent_run_id=""
  └─ run_20260529-120001-child1   parent_run_id="run_20260529-120000-root"
       └─ run_20260529-120002-child2   parent_run_id="run_20260529-120001-child1"
```

扫描 `.pico/runs/*/report.json`，按 `parent_run_id` 归组即可重建树形结构，不依赖时间戳顺序，也不受目录扫描顺序影响。

---

## 完整改动清单

| 文件 | 改动内容 |
|---|---|
| `pico/task_state.py` | 新增 `STOP_REASON_BACKEND_ERROR`、`stop_backend_error()`、`parent_run_id` 字段 |
| `pico/runtime.py` | 新增 `TRACE_EVENT_PHASE`、`_CHECKPOINT_TRIGGER_PHASE`；更新 `emit_trace()`；`parse()` 改返回三元组；`model_client.complete()` 包 try/except；`Pico.__init__` 增 `parent_run_id` 参数；`ask()` 设 `_current_run_id`；`run_started` trace 和 `build_report` 补 `parent_run_id` |
| `pico/tools.py` | `tool_delegate` 创建子 agent 时透传 `parent_run_id` |

---

## 相关文件索引

| 文件 | 说明 |
|---|---|
| `pico/task_state.py` | 运行状态机，`stop_reason` 和 `parent_run_id` 的定义所在 |
| `pico/runtime.py` → `ask()` | 主循环，三处改动均在此落地 |
| `pico/runtime.py` → `parse()` | 模型输出解析，`parse_detail` 在此生成 |
| `pico/runtime.py` → `emit_trace()` | trace 写入入口，`phase` 在此注入 |
| `pico/tools.py` → `tool_delegate()` | delegate 工具，`parent_run_id` 在此透传 |
| `pico/run_store.py` | 负责将 `task_state`、`trace`、`report` 写入 `.pico/runs/{run_id}/` |
