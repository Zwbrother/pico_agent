# pico 启动与初始化流程

## 启动入口

```
用户启动 pico
    │
    ▼
┌─────────────────────────────────┐
│   cli.py:main(argv)             │  ← 程序入口
│                                 │
│  ① parse_args()                 │  ← 解析 CLI 参数
│  ② build_agent(args)            │  ← 装配 Agent（核心）
│  ③ build_welcome()              │  ← 打印欢迎界面
│  ④ if prompt: ask()             │  ← One-shot 模式
│     else: REPL loop             │  ← 交互模式
└─────────────────────────────────┘
```

## build_agent() 装配流程

```
build_agent(args)
    │
    ├─► WorkspaceContext.build(cwd)
    │   └─► git 命令 + 读取项目文档
    │
    ├─► load_project_env(repo_root)
    │   └─► 查找并加载 .env 文件
    │
    ├─► _configured_secret_names(args)
    │   └─► 合并敏感变量名列表
    │
    ├─► SessionStore(...)
    │   └─► 创建 session 存储目录
    │
    ├─► _build_model_client(args)
    │   └─► 创建模型客户端实例
    │
    └─► Pico.__init__(...)
        └─► 8步初始化流程（见下方）
```

## Pico.__init__() 8步初始化

```
Pico.__init__()
    │
    ├─► 1️⃣ 基础属性赋值
    │   └─► model_client, workspace, root, session_store, ...
    │
    ├─► 2️⃣ 初始化 Session 结构
    │   ├─► 生成 session ID (timestamp + UUID)
    │   └─► _ensure_session_shape()
    │       └─► history, memory, checkpoints, runtime_identity
    │
    ├─► 3️⃣ 初始化分层记忆系统
    │   └─► LayeredMemory(working + durable + task_summary)
    │
    ├─► 4️⃣ 构建工具注册表
    │   └─► build_tools()
    │       ├─► 6个基础工具 (list_files, read_file, ...)
    │       └─► delegate (如果 depth < max_depth)
    │
    ├─► 5️⃣ 构建 Prompt Prefix
    │   └─► build_prefix()
    │       ├─► 角色定义 + 规则说明
    │       ├─► 工具详细说明 + 示例
    │       ├─► 工作区快照 (git status, docs)
    │       └─► 生成 hash + fingerprint (用于缓存)
    │
    ├─► 6️⃣ 初始化上下文管理器
    │   └─► ContextManager(self)
    │       └─► 配置 prompt 预算控制策略
    │
    ├─► 7️⃣ 评估恢复状态
    │   └─► evaluate_resume_state()
    │       ├─► 清理过期文件摘要
    │       ├─► 检查 checkpoint 有效性
    │       ├─► 验证 schema 版本
    │       ├─► 检查文件新鲜度
    │       ├─► 验证运行时身份
    │       └─► 确定恢复状态 (no-checkpoint/full-valid/...)
    │
    └─► 8️⃣ 持久化 Session
        └─► session_store.save(session)
            └─► .pico/sessions/{session_id}.json
```

## evaluate_resume_state() 恢复状态评估

```
evaluate_resume_state()
    │
    ├─► 步骤1: 清理过期文件摘要
    │   └─► invalidate_stale_memory()
    │        检查 memory.file_summaries 中的缓存
    │        如果文件 SHA256 变化，清除缓存
    │
    ├─► 步骤2: 获取当前 checkpoint
    │   └─► current_checkpoint()
    │        从 session.checkpoints.items 中获取
    │
    ├─► 步骤3: 第1层验证 - Schema 版本检查
    │   └─► if checkpoint.schema_version != CHECKPOINT_SCHEMA_VERSION:
    │        status = "schema-mismatch"
    │        return  ← 直接返回，不进行后续检查
    │
    ├─► 步骤4: 第2层验证 - 文件新鲜度检查
    │   └─► for file in checkpoint.key_files:
    │        expected = file.freshness  # checkpoint 创建时的 SHA256
    │        current = file_freshness(file.path)  # 当前文件的 SHA256
    │        if expected != current:
    │          stale_paths.append(file.path)
    │
    ├─► 步骤5: 第3层验证 - 运行时身份检查
    │   └─► for key in 11个关键字段:
    │        if saved_identity[key] != current_identity[key]:
    │          mismatch_fields.append(key)
    │
    └─► 步骤6: 确定最终状态
         if stale_paths:
           status = "partial-stale"      # 文件过期但环境一致
         elif mismatch_fields:
           status = "workspace-mismatch" # 环境变化太大
         else:
           status = "full-valid"         # 完全有效
```

### 恢复状态枚举

| 状态 | 含义 |
|---|---|
| `no-checkpoint` | 无 checkpoint，全新运行 |
| `full-valid` | 完全有效，可安全恢复 |
| `partial-stale` | 部分文件已变化，需提示模型 |
| `workspace-mismatch` | 运行时环境变化太大 |
| `schema-mismatch` | checkpoint schema 版本不兼容 |
