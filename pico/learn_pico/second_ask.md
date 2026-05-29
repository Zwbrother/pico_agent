# pico ask() 执行流程

## 整体结构

```
agent.ask(user_message)
    │
    ▼
┌─────────────────────────────────────────┐
│ 阶段1: 初始化                            │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│ 阶段2: 主循环 (感知 → 决策 → 行动)       │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│ 阶段3: 终止处理                          │
└─────────────────────────────────────────┘
```

---

## 阶段1: 初始化

```
用户输入: user_message
    │
    ▼
1. memory.set_task_summary(user_message)
   └─► 设置任务摘要到工作记忆

2. record({"role": "user", ...})
   └─► 记录用户消息到 session history

3. TaskState.create(run_id, task_id, user_request)
   └─► 创建任务状态对象（跟踪运行元数据）

4. run_store.start_run(task_state)
   └─► 在 .pico/runs/{run_id}/ 创建目录

5. emit_trace("run_started")
   └─► 发送第一个 trace 事件
```

---

## 阶段2: 主循环

循环条件：`while tool_steps < max_steps`

### 步骤1: 感知 — 构建 Prompt

```
_build_prompt_and_metadata(user_message)
    │
    ├─► refresh_prefix()
    │   ├─► WorkspaceContext.build(root)
    │   │   └─► 扫描 Git 状态、文件列表、文档
    │   └─► build_prefix() (如需要)
    │        └─► 生成系统指令 + 工具说明
    │
    ├─► evaluate_resume_state()
    │   ├─► invalidate_stale_memory()   → 清理过期的文件摘要缓存
    │   ├─► 检查 checkpoint schema 版本
    │   ├─► 验证文件 freshness (SHA256对比)
    │   └─► 验证 runtime_identity (11字段)
    │        └─► cwd, model, approval_policy...
    │
    └─► context_manager.build(user_message)
        ├─► 计算预算分配 (total=12000 chars)
        │   ├─► prefix:          3600
        │   ├─► memory:          1600
        │   ├─► relevant_memory: 1200
        │   └─► history:         5200
        ├─► 组装 prefix + memory + history
        └─► 裁剪超限部分 (按优先级)
```

### 步骤2: 检查恢复状态并创建 Checkpoint

```
if resume_status == "partial-stale":
    └─► create_checkpoint(trigger="freshness_mismatch")

elif resume_status == "workspace-mismatch":
    └─► create_checkpoint(trigger="workspace_mismatch")

if budget_reductions:
    └─► create_checkpoint(trigger="context_reduction")
```

### 步骤3: 决策 — 调用模型

```
model_client.complete(
    prompt,
    max_new_tokens,
    prompt_cache_key,
    prompt_cache_retention
)
├─► 如果后端支持，使用 prompt_cache_key
└─► 返回 raw text (模型原始输出)
```

### 步骤4: 解析模型输出

```
parse(raw)
    ├─► 检测 <tool>...</tool>     → kind="tool"
    │   └─► 提取 JSON: {"name":"...", "args":{}}
    ├─► 检测 XML 格式             → kind="tool"
    │   └─► parse_xml_tool(raw)
    ├─► 检测 <final>...</final>   → kind="final"
    └─► 其他情况                  → kind="retry"

返回: (kind, payload)
```

### 步骤5: 行动 — 根据 kind 分支

#### 【分支A】kind == "tool"

```
run_tool(name, args)  ← 六层安全防护
    │
    ├─► 第1层: 工具存在性检查
    │   └─► self.tools.get(name)
    │
    ├─► 第2层: validate_tool(name, args)
    │   ├─► 路径逃逸检查 (锚定到 root)
    │   ├─► 参数类型校验 (str/int/dict)
    │   ├─► 必填参数检查
    │   └─► 特殊约束 (如 old_text 唯一性)
    │
    ├─► 第3层: repeated_tool_call(name, args)
    │   └─► 检查最近2次 tool 事件是否相同
    │
    ├─► 第4层: approve(name, args)
    │   ├─► read_only=True?      → 拒绝
    │   ├─► policy="auto"?       → 自动通过
    │   ├─► policy="never"?      → 拒绝
    │   └─► policy="ask"?        → input() 确认
    │
    ├─► 第5层: 执行前后快照对比
    │   ├─► before = capture_snapshot()
    │   ├─► result = tool["run"](args)
    │   └─► after  = capture_snapshot()
    │        └─► diff_workspace_snapshots()
    │
    └─► 第6层: update_memory_after_tool()
         └─► 更新 working_memory

record({"role": "tool", ...})
emit_trace("tool_executed")
create_checkpoint(trigger="tool_executed")
→ continue (进入下一轮循环)
```

#### 【分支B】kind == "retry"

```
record({"role": "assistant", payload})
→ continue (给模型重试机会)
```

#### 【分支C】kind == "final"

```
record({"role": "assistant", final})
task_state.finish_success(final)
promote_durable_memory(user, final)
    ├─► extract_durable_promotions()
    └─► memory.promote_durable()
create_checkpoint(trigger="run_finished")
emit_trace("run_finished")
build_report(task_state)
run_store.write_report()
→ return final  ✅ 退出循环
```

---

## 阶段3: 异常终止处理

```
if attempts >= max_attempts:
    └─► stop_retry_limit()
        └─► "Stopped after too many malformed responses"
else:
    └─► stop_step_limit()
        └─► "Stopped after reaching the step limit"

record({"role": "assistant", final})
promote_durable_memory()
create_checkpoint(trigger=stop_reason)
emit_trace("run_finished")
build_report()
run_store.write_report()
```

---

## run_tool() 执行流程（展开）

```
run_tool
┌──────────────────────────────────────┐
│ 1. 工具存在性检查                     │
│    tools.get(name) → None?           │
└──────────────┬───────────────────────┘
               │ exists
               ▼
┌──────────────────────────────────────┐
│ 2. 参数合法性校验                     │
│    validate_tool(name, args)         │
│    - 通用校验 (toolkit)              │
│    - delegate 深度限制               │
└──────────────┬───────────────────────┘
               │ valid
               ▼
┌──────────────────────────────────────┐
│ 3. 重复调用检测                       │
│    repeated_tool_call(name, args)    │
│    - 检查最近 2 次工具调用           │
└──────────────┬───────────────────────┘
               │ not repeated
               ▼
┌──────────────────────────────────────┐
│ 4. 审批策略检查                       │
│    approve(name, args)               │
│    - read_only / auto / ask / never  │
└──────────────┬───────────────────────┘
               │ approved
               ▼
┌──────────────────────────────────────┐
│ 5. 执行前快照 (risky 工具)            │
│    capture_workspace_snapshot()      │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│ 6. 真正执行工具                       │
│    tool["run"](args)                 │
│    clip(result)                      │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│ 7. 执行后快照 + diff                  │
│    diff_workspace_snapshots()        │
│    → affected_paths, diff_summary    │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│ 8. 更新工作记忆                       │
│    update_memory_after_tool()        │
│    record_process_note_for_tool()    │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│ 9. 返回结果 (成功/错误)               │
└──────────────────────────────────────┘
```
