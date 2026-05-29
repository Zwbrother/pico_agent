"""Agent 运行时核心逻辑。

Pico 就是包在模型外面的控制循环：负责组 prompt、解析模型输出、
校验并执行工具、写 trace、更新工作记忆，以及在合适的时候停下来。

## 核心职责
1. Session 管理：维护对话历史和记忆状态
2. Prompt 组装：通过 ContextManager 构建完整提示词
3. 模型交互：调用模型客户端并解析输出
4. 工具执行：校验、审批、执行工具调用
5. 状态跟踪：记录 trace、report、checkpoint
6. 安全控制：路径检查、敏感信息脱敏、重复调用检测

## Pico.__init__() 初始化流程详解
```
Pico.__init__(model_client, workspace, session_store, ...)
  │
  ├─> 1. 基础属性赋值
  │    ├─> self.model_client = model_client          # 模型客户端
  │    ├─> self.workspace = workspace                # 工作区快照
  │    ├─> self.root = Path(workspace.repo_root)     # 仓库根目录
  │    ├─> self.session_store = session_store        # Session 存储
  │    └─> ... (其他配置项)
  │
  ├─> 2. 初始化 Session 结构
  │    ├─> 生成 session ID（时间戳 + UUID）
  │    ├─> 初始化空 history 和 memory
  │    └─> _ensure_session_shape()                   # 确保完整结构
  │         ├─> history: []                          #   对话历史
  │         ├─> memory: {...}                        #   分层记忆
  │         ├─> checkpoints: {current_id, items}     #   检查点
  │         ├─> runtime_identity: {...}              #   运行时身份
  │         └─> resume_state: {...}                  #   恢复状态
  │
  ├─> 3. 初始化分层记忆系统
  │    └─> LayeredMemory(session["memory"], workspace_root)
  │         ├─> working_memory: 最近文件、笔记       #   工作记忆
  │         ├─> durable_memory: 长期沉淀的知识       #   长期记忆
  │         └─> task_summary: 当前任务摘要           #   任务摘要
  │
  ├─> 4. 构建工具注册表
  │    └─> build_tools()
  │         ├─> list_files (safe)                    #   列出文件
  │         ├─> read_file (safe)                     #   读取文件
  │         ├─> search (safe)                        #   搜索内容
  │         ├─> run_shell (risky)                    #   执行命令
  │         ├─> write_file (risky)                   #   写入文件
  │         ├─> patch_file (risky)                   #   修补文件
  │         └─> delegate (safe, 如果深度允许)        #   委托子 agent
  │
  ├─> 5. 构建 Prompt Prefix
  │    └─> build_prefix()
  │         ├─> 角色定义和规则说明
  │         ├─> 工具详细说明（名称、参数、风险等级）
  │         ├─> 工具调用示例
  │         ├─> 工作区快照（Git 状态、分支、文档）
  │         └─> 生成元数据
  │              ├─> hash: SHA256(prefix text)       #   prefix 哈希
  │              ├─> workspace_fingerprint            #   工作区指纹
  │              ├─> tool_signature                   #   工具签名
  │              └─> built_at                         #   构建时间
  │
  ├─> 6. 初始化上下文管理器
  │    └─> ContextManager(self)
  │         ├─> total_budget: 12000 chars            #   总预算
  │         ├─> section_budgets: {...}               #   各部分预算
  │         └─> reduction_order: (...)               #   裁剪顺序
  │
  ├─> 7. 评估恢复状态
  │    └─> evaluate_resume_state()
  │         ├─> invalidate_stale_memory()            #   清理过期摘要
  │         ├─> 检查 checkpoint 是否存在
  │         ├─> 验证 schema 版本匹配
  │         ├─> 检查文件新鲜度（freshness mismatch）
  │         ├─> 验证运行时身份（runtime identity）
  │         └─> 确定恢复状态
  │              ├─> no-checkpoint                    #   无 checkpoint
  │              ├─> full-valid                       #   完全有效
  │              ├─> partial-stale                    #   部分过期
  │              ├─> workspace-mismatch               #   工作区不匹配
  │              └─> schema-mismatch                  #   schema 不匹配
  │
  └─> 8. 持久化 Session
       └─> session_store.save(session)
            └─> 保存到 .pico/sessions/{session_id}.json
```
"""

import json
import os
import re
import textwrap
import uuid
import hashlib
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from . import memory as memorylib
from .context_manager import ContextManager
from .run_store import RunStore
from .task_state import TaskState
from . import tools as toolkit
from .workspace import IGNORED_PATH_NAMES, MAX_HISTORY, WorkspaceContext, clip, now

# ============================================================================
# 常量定义
# ============================================================================

# 敏感环境变量名的识别标记（用于自动检测需要脱敏的变量）
SENSITIVE_ENV_NAME_MARKERS = ("API_KEY", "TOKEN", "SECRET", "PASSWORD")

# 脱敏后的占位符
REDACTED_VALUE = "<redacted>"

# Shell 环境变量白名单（只允许这些变量传递给子进程）
DEFAULT_SHELL_ENV_ALLOWLIST = ("HOME", "LANG", "LC_ALL", "LC_CTYPE", "LOGNAME", "PATH", "PWD", "SHELL", "TERM", "TMPDIR", "TMP", "TEMP", "USER")

# 默认功能开关
DEFAULT_FEATURE_FLAGS = {
    "memory": True,              # 启用工作记忆
    "relevant_memory": True,     # 启用相关记忆检索
    "context_reduction": True,   # 启用上下文裁剪
    "prompt_cache": True,        # 启用 prompt 缓存
}

# Checkpoint 相关常量
CHECKPOINT_SCHEMA_VERSION = "phase1-v1"  # Checkpoint 结构版本
CHECKPOINT_NONE_STATUS = "no-checkpoint"  # 无 checkpoint
CHECKPOINT_FULL_VALID_STATUS = "full-valid"  # 完全有效
CHECKPOINT_PARTIAL_STALE_STATUS = "partial-stale"  # 部分过期
CHECKPOINT_WORKSPACE_MISMATCH_STATUS = "workspace-mismatch"  # 工作区不匹配
CHECKPOINT_SCHEMA_MISMATCH_STATUS = "schema-mismatch"  # Schema 不匹配

# 长期记忆意图识别模式（英文和中文）
DURABLE_MEMORY_INTENT_PATTERN = re.compile(r"(?i)\b(capture|remember|save|store|persist|note)\b")
DURABLE_MEMORY_INTENT_ZH_PATTERN = re.compile(r"(记住|保存|记录|沉淀|长期记忆|持久记忆)")

# 长期记忆行模式匹配（支持中英文）
DURABLE_MEMORY_LINE_PATTERNS = (
    ("project-conventions", re.compile(r"(?i)^Project convention:\s*(.+)$")),
    ("key-decisions", re.compile(r"(?i)^Decision:\s*(.+)$")),
    ("dependency-facts", re.compile(r"(?i)^Dependency:\s*(.+)$")),
    ("user-preferences", re.compile(r"(?i)^Preference:\s*(.+)$")),
    ("project-conventions", re.compile(r"^项目约定：\s*(.+)$")),
    ("key-decisions", re.compile(r"^决策：\s*(.+)$")),
    ("dependency-facts", re.compile(r"^依赖：\s*(.+)$")),
    ("user-preferences", re.compile(r"^偏好：\s*(.+)$")),
)

# 密钥形状文本检测模式（用于识别可能的 API key 泄露）
SECRET_SHAPED_TEXT_PATTERN = re.compile(r"(?i)(\b(api[_ -]?key|token|secret|password)\b|sk-[A-Za-z0-9_-]{6,})")

# Trace 事件的阶段映射：每个事件属于哪个执行阶段。
# 用于可视化和分析时快速过滤某一阶段的事件序列。
# checkpoint_created 的 phase 由其 trigger 字段动态推导（见 emit_trace）。
TRACE_EVENT_PHASE = {
    "run_started":                "init",
    "runtime_identity_mismatch":  "init",
    "prompt_built":               "plan",
    "model_requested":            "plan",
    "model_parsed":               "decide",
    "tool_executed":              "act",
    "run_finished":               "finish",
}

# checkpoint_created 的 trigger → phase 映射
_CHECKPOINT_TRIGGER_PHASE = {
    "tool_executed":          "act",
    "run_finished":           "finish",
    "step_limit_reached":     "finish",
    "retry_limit_reached":    "finish",
    "run_stopped":            "finish",
    "freshness_mismatch":     "plan",
    "workspace_mismatch":     "plan",
    "context_reduction":      "plan",
}


# ============================================================================
# 数据类定义
# ============================================================================

@dataclass
class PromptPrefix:
    """Prompt prefix 及其元数据。
    
    prefix 除了文本本身，还带一小份元数据，这样 runtime 才能明确判断 
    prefix 是否可以复用（基于 hash、workspace_fingerprint、tool_signature）。
    
    Attributes:
        text: prefix 的完整文本内容
        hash: text 的 SHA256 哈希值，用于判断 prefix 是否变化
        workspace_fingerprint: 工作区状态的指纹，用于判断工作区是否变化
        tool_signature: 工具注册表的签名，用于判断工具集是否变化
        built_at: 构建时间戳（ISO 格式）
    """
    text: str
    hash: str
    workspace_fingerprint: str
    tool_signature: str
    built_at: str


# ============================================================================
# Session 持久化存储
# ============================================================================

class SessionStore:
    """Session 文件的持久化存储管理器。
    
    负责将 session 状态保存到磁盘，支持：
    - 保存 session 到 JSON 文件
    - 从 JSON 文件加载 session
    - 查找最近修改的 session 文件
    
    文件路径格式：{root}/{session_id}.json
    """
    
    def __init__(self, root):
        """初始化 SessionStore。
        
        Args:
            root: Session 文件存储目录（通常是 .pico/sessions）
        """
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)  # 确保目录存在

    def path(self, session_id):
        """获取指定 session 的文件路径。
        
        Args:
            session_id: Session ID
            
        Returns:
            Path: session 文件的完整路径
        """
        return self.root / f"{session_id}.json"

    def save(self, session):
        """保存 session 到 JSON 文件。
        
        Args:
            session: session 字典对象
            
        Returns:
            Path: 保存的文件路径
        """
        path = self.path(session["id"])
        path.write_text(json.dumps(session, indent=2), encoding="utf-8")
        return path

    def load(self, session_id):
        """从 JSON 文件加载 session。
        
        Args:
            session_id: Session ID
            
        Returns:
            dict: 加载的 session 数据
        """
        return json.loads(self.path(session_id).read_text(encoding="utf-8"))

    def latest(self):
        """查找最近修改的 session 文件。
        
        Returns:
            str or None: 最近 session 的 ID，如果没有则返回 None
        """
        files = sorted(self.root.glob("*.json"), key=lambda path: path.stat().st_mtime)
        return files[-1].stem if files else None


# ============================================================================
# Pico 核心运行时类
# ============================================================================

class Pico:
    """Pico Agent 的核心运行时类。
    
    这是整个 agent 系统的控制中心，负责：
    1. 维护 session 状态（历史、记忆、checkpoint）
    2. 组装 prompt 并调用模型
    3. 解析模型输出并执行工具
    4. 记录 trace 和 report
    5. 管理安全策略（审批、脱敏、路径检查）
    
    ## 初始化流程
    详见模块级文档字符串中的流程图。
    """
    
    def __init__(
        self,
        model_client,
        workspace,
        session_store,
        session=None,
        run_store=None,
        approval_policy="ask",
        max_steps=6,
        max_new_tokens=512,
        depth=0,
        max_depth=1,
        read_only=False,
        shell_env_allowlist=None,
        secret_env_names=None,
        feature_flags=None,
        parent_run_id="",
    ):
        """初始化 Pico 运行时实例。
        
        Args:
            model_client: 模型客户端实例（OpenAI/Anthropic/Ollama）
            workspace: 工作区快照（WorkspaceContext）
            session_store: Session 存储管理器
            session: 可选的已有 session 数据（用于恢复）
            run_store: 运行记录存储（默认为 .pico/runs）
            approval_policy: 危险工具审批策略（ask/auto/never）
            max_steps: 每轮最大工具调用步数
            max_new_tokens: 模型每次输出的最大 token 数
            depth: 当前 agent 深度（用于 delegate）
            max_depth: 最大允许的深度
            read_only: 是否只读模式（禁止所有危险工具）
            shell_env_allowlist: Shell 环境变量白名单
            secret_env_names: 需要脱敏的环境变量名列表
            feature_flags: 功能开关字典
        """
        # ====================================================================
        # 步骤1: 基础属性赋值
        # ====================================================================
        self.model_client = model_client
        self.workspace = workspace
        self.root = Path(workspace.repo_root)
        self.session_store = session_store
        self.approval_policy = approval_policy
        self.max_steps = max_steps
        self.max_new_tokens = max_new_tokens
        self.depth = depth
        self.max_depth = max_depth
        self.read_only = read_only
        self.shell_env_allowlist = tuple(shell_env_allowlist or DEFAULT_SHELL_ENV_ALLOWLIST)
        self.secret_env_names = {str(name).upper() for name in (secret_env_names or ())}
        self.feature_flags = dict(DEFAULT_FEATURE_FLAGS)
        if feature_flags:
            self.feature_flags.update({str(key): bool(value) for key, value in feature_flags.items()})
        self.run_store = run_store or RunStore(Path(workspace.repo_root) / ".pico" / "runs")
        self.parent_run_id = str(parent_run_id or "")
        # _current_run_id is set at the start of ask() so tool_delegate can read it
        self._current_run_id = ""
        
        # ====================================================================
        # 步骤2: 初始化 Session 结构
        # ====================================================================
        # 如果是新 session，生成 ID 和初始状态
        self.session = session or {
            "id": datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6],
            "created_at": now(),
            "workspace_root": workspace.repo_root,
            "history": [],
            "memory": memorylib.default_memory_state(),
        }
        # 确保 session 有完整的嵌套结构（checkpoints、runtime_identity 等）
        self._ensure_session_shape()
        
        # ====================================================================
        # 步骤3: 初始化分层记忆系统
        # ====================================================================
        self.memory = memorylib.LayeredMemory(
            self.session.setdefault("memory", memorylib.default_memory_state()),
            workspace_root=self.root,
        )
        self.session["memory"] = self.memory.to_dict()
        
        # ====================================================================
        # 步骤4: 构建工具注册表
        # ====================================================================
        self.tools = self.build_tools()
        
        # ====================================================================
        # 步骤5: 构建 Prompt Prefix
        # ====================================================================
        self.prefix_state = self.build_prefix()
        self.prefix = self.prefix_state.text
        
        # ====================================================================
        # 步骤6: 初始化上下文管理器
        # ====================================================================
        self.context_manager = ContextManager(self)
        
        # ====================================================================
        # 步骤7: 评估恢复状态
        # ====================================================================
        self.resume_state = self.evaluate_resume_state()
        
        # ====================================================================
        # 步骤8: 持久化 Session
        # ====================================================================
        self.session_path = self.session_store.save(self.session)
        
        # ====================================================================
        # 运行时状态跟踪变量
        # ====================================================================
        self.current_task_state = None  # 当前任务状态
        self.current_run_dir = None  # 当前运行记录目录
        self.last_prompt_metadata = {}  # 上一轮 prompt 元数据
        self.last_completion_metadata = {}  # 上一轮模型完成元数据
        self.last_durable_promotions = []  # 上一轮提升的长期记忆
        self.last_durable_rejections = []  # 上一轮拒绝的长期记忆
        self.last_durable_superseded = []  # 上一轮被替代的长期记忆
        self._last_tool_result_metadata = {}  # 上一次工具执行结果元数据
        self._last_prefix_refresh = {
            "workspace_changed": False,
            "prefix_changed": False,
        }

    @classmethod
    def from_session(cls, model_client, workspace, session_store, session_id, **kwargs):
        """从已有 session 恢复 Pico 实例。
        
        Args:
            model_client: 模型客户端实例
            workspace: 工作区快照
            session_store: Session 存储管理器
            session_id: 要恢复的 session ID
            **kwargs: 其他传递给 __init__ 的参数
            
        Returns:
            Pico: 恢复后的 Pico 实例
        """
        return cls(
            model_client=model_client,
            workspace=workspace,
            session_store=session_store,
            session=session_store.load(session_id),
            **kwargs,
        )

    def _ensure_session_shape(self):
        """确保 session 有完整的嵌套结构。
        
        为新创建的 session 或从旧版本恢复的 session 补充缺失的字段，保证后续代码可以安全访问这些嵌套结构。
        
        为什么需要这个函数？
        - 新 session：初始化时只有基础字段（id, created_at, history, memory）
        - 旧版本 session：从磁盘加载的 JSON 可能缺少新增的字段
        - 测试场景：可能传入简化的 session 对象
        - 防御性编程：避免后续代码访问不存在的键导致 KeyError
        
        确保的四个核心结构：
        1. history: 对话历史记录列表
        2. memory: 分层记忆系统状态
        3. checkpoints: 任务检查点（支持断点续传）
        4. runtime_identity: 运行时身份标识（用于验证 checkpoint 有效性）
        5. resume_state: 恢复状态评估结果（五种状态之一）
        """
        # ====================================================================
        # 1. 确保基础结构存在
        # ====================================================================
        # history: 存储与模型的完整对话历史（user/assistant/tool 消息）
        # 用于构建 prompt 中的 "Transcript" 部分，让模型了解之前的交互
        self.session.setdefault("history", [])
        
        # memory: 分层记忆系统的完整状态字典
        # 包含 working_memory（工作记忆）、episodic_notes（临时笔记）、file_summaries（文件摘要缓存）等
        self.session.setdefault("memory", memorylib.default_memory_state())
        
        # ====================================================================
        # 2. 确保 checkpoints 结构（支持断点续传机制）
        # ====================================================================
        # checkpoints 是什么？
        # - 任务执行过程中的"快照"，记录当前进度、关键文件状态、下一步计划
        # - 当 agent 达到步数限制或遇到阻塞时，保存 checkpoint
        # - 下次启动时可以从中断处继续，而不是从头开始
        # 
        # 使用场景：
        # - 用户："帮我重构整个项目" → agent 执行 6 步后达到 max_steps
        # - 系统自动创建 checkpoint，保存当前进度
        # - 用户再次提问时，agent 从 checkpoint 恢复，继续未完成的工作
        #
        # 数据结构：
        # {
        #     "current_id": "ckpt_a1b2c3d4",  # 当前活跃的 checkpoint ID
        #     "items": {                      # 所有 checkpoint 的字典
        #         "ckpt_a1b2c3d4": {
        #             "checkpoint_id": "ckpt_a1b2c3d4",
        #             "parent_checkpoint_id": "",  # 父 checkpoint（支持链式恢复）
        #             "schema_version": "phase1-v1",  # 结构版本号（兼容性检查）
        #             "created_at": "2026-05-28T12:00:00",
        #             "current_goal": "重构用户认证模块",  # 当前目标
        #             "completed": ["读取 auth.py", "分析依赖关系"],  # 已完成步骤
        #             "excluded": [],  # 排除的文件/步骤
        #             "current_blocker": "需要确认 API 格式",  # 当前阻塞原因
        #             "next_step": "调用 run_shell 运行测试",  # 推断的下一步
        #             "key_files": [  # 关键文件及其新鲜度（SHA256）
        #                 {"path": "src/auth.py", "freshness": "abc123..."},
        #                 {"path": "tests/test_auth.py", "freshness": "def456..."}
        #             ],
        #             "freshness": {...},  # 文件新鲜度映射表
        #             "summary": "freshness_mismatch: 重构用户认证模块",  # 简要描述
        #             "runtime_identity": {...}  # 创建时的运行时身份
        #         }
        #     }
        # }
        checkpoints = self.session.setdefault("checkpoints", {})
        # 防御性检查：如果 checkpoints 不是字典类型（可能是旧版本的遗留数据），重置为空字典
        if not isinstance(checkpoints, dict):
            checkpoints = {}
            self.session["checkpoints"] = checkpoints
        # current_id: 指向当前活跃的 checkpoint，空字符串表示没有活跃 checkpoint
        checkpoints.setdefault("current_id", "")
        # items: 存储所有 checkpoint 对象的字典，key 为 checkpoint_id
        checkpoints.setdefault("items", {})
        
        # ====================================================================
        # 3. 确保 runtime_identity 结构（运行时身份标识）
        # ====================================================================
        # runtime_identity 是什么？
        # - 记录创建 checkpoint 时的"环境指纹"，包含 11 个关键字段
        # - 用于判断 checkpoint 是否可以安全恢复
        # - 如果以下任何一项发生变化，checkpoint 可能不再有效：
        #   * 工作区路径或 Git 状态变化
        #   * 模型配置变化（换了不同的 LLM）
        #   * 安全策略变化（approval_policy 从 ask 改为 auto）
        #   * 工具注册表变化（新增或删除了工具）
        #
        # 包含的字段（见 current_runtime_identity 方法）：
        # {
        #     "session_id": "...",
        #     "cwd": "/path/to/project",
        #     "model": "gpt-4",
        #     "model_client": "OpenAICompatibleModelClient",
        #     "approval_policy": "ask",
        #     "read_only": false,
        #     "max_steps": 6,
        #     "max_new_tokens": 512,
        #     "feature_flags": {...},
        #     "shell_env_allowlist": [...],
        #     "workspace_fingerprint": "...",  # 工作区状态指纹
        #     "tool_signature": "..."          # 工具注册表签名
        # }
        runtime_identity = self.session.setdefault("runtime_identity", {})
        # 防御性检查：如果不是字典类型，重置为空字典
        if not isinstance(runtime_identity, dict):
            self.session["runtime_identity"] = {}
        
        # ====================================================================
        # 4. 确保 resume_state 结构（恢复状态评估结果）
        # ====================================================================
        # resume_state 是什么？
        # - 在 __init__ 时通过 evaluate_resume_state() 计算得出
        # - 描述当前 checkpoint 的"健康状态"，决定是否可以安全恢复
        # - 返回五种状态之一：
        #   * no-checkpoint: 没有 checkpoint，全新开始
        #   * full-valid: 完全有效，可以安全恢复
        #   * partial-stale: 部分文件过期（SHA256 不匹配），需要谨慎处理
        #   * workspace-mismatch: 运行时身份不匹配（如切换了分支/模型），不能恢复
        #   * schema-mismatch: checkpoint 结构版本不兼容，不能恢复
        #
        # 数据结构：
        # {
        #     "status": "full-valid",  # 恢复状态
        #     "stale_paths": [],  # 过期的文件路径列表（freshness 不匹配）
        #     "runtime_identity_mismatch_fields": [],  # 不匹配的运行时身份字段
        #     "stale_summary_invalidations": 0  # 累计的文件摘要失效次数
        # }
        resume_state = self.session.setdefault("resume_state", {})
        # 防御性检查：如果不是字典类型，重置为空字典
        if not isinstance(resume_state, dict):
            self.session["resume_state"] = {}

    def current_runtime_identity(self):
        """获取当前运行时的身份标识。
        
        这个标识用于判断 checkpoint 是否可以安全恢复。如果以下任何一项
        发生变化，checkpoint 可能不再有效：
        - 工作区路径或状态
        - 模型配置
        - 安全策略
        - 工具注册表
        
        Returns:
            dict: 运行时身份标识字典
        """
        return {
            "session_id": self.session.get("id", ""),
            "cwd": str(self.root),
            "model": str(getattr(self.model_client, "model", "")),
            "model_client": self.model_client.__class__.__name__,
            "approval_policy": self.approval_policy,
            "read_only": bool(self.read_only),
            "max_steps": int(self.max_steps),
            "max_new_tokens": int(self.max_new_tokens),
            "feature_flags": dict(self.feature_flags),
            "shell_env_allowlist": list(self.shell_env_allowlist),
            "workspace_fingerprint": getattr(getattr(self, "prefix_state", None), "workspace_fingerprint", self.workspace.fingerprint()),
            "tool_signature": self.tool_signature(),
        }

    def checkpoint_state(self):
        """获取 checkpoint 状态。
        
        Returns:
            dict: checkpoint 状态字典
        """
        self._ensure_session_shape()
        return self.session["checkpoints"]

    def current_checkpoint(self):
        """获取当前 checkpoint。
        
        Returns:
            dict or None: 当前 checkpoint 字典，如果没有则返回 None
        """
        state = self.checkpoint_state()
        checkpoint_id = str(state.get("current_id", "")).strip()
        if not checkpoint_id:
            return None
        return state.get("items", {}).get(checkpoint_id)

    def invalidate_stale_memory(self):
        """清理过期的文件摘要。
        
        Returns:
            list: 被清理的文件路径列表
        """
        invalidated = self.memory.invalidate_stale_file_summaries()
        self.session["memory"] = self.memory.to_dict()
        return invalidated

    def evaluate_resume_state(self):
        """评估当前 session 的恢复状态。
        
        这个函数是 Pico "断点续传"机制的核心，负责判断之前保存的 checkpoint
        是否可以安全恢复。类似于游戏存档后，再次加载时需要检查：
        - 存档文件是否损坏（schema 版本检查）
        - 游戏世界是否发生了变化（文件新鲜度检查）
        - 玩家配置是否保持一致（运行时身份检查）
        
        为什么需要评估恢复状态？
        - 用户可能在两次对话之间修改了代码文件
        - 用户可能切换了 Git 分支或改变了模型配置
        - 系统升级可能导致 checkpoint 结构不兼容
        - 需要告诉用户：能否从上次中断处继续，还是需要重新开始
        
        评估流程（五层验证）：
        1. 清理过期的文件摘要缓存
        2. 检查是否有 checkpoint 存在
        3. 验证 checkpoint 结构版本兼容性
        4. 检查关键文件的 freshness（SHA256 对比）
        5. 验证 11 个运行时身份字段的一致性
        
        Returns:
            dict: 恢复状态字典，包含以下字段：
                - status: 恢复状态字符串（五种之一）
                - stale_paths: 过期的文件路径列表
                - runtime_identity_mismatch_fields: 不匹配的运行时身份字段
                - stale_summary_invalidations: 累计的文件摘要失效次数
                
        五种恢复状态：
        ┌─────────────────────┬──────────────────────────────────┐
        │ 状态                 │ 触发条件                          │
        ├─────────────────────┼──────────────────────────────────┤
        │ no-checkpoint       │ 没有 checkpoint，全新开始         │
        │ full-valid          │ 所有检查通过，可以安全恢复 ✅      │
        │ partial-stale       │ 部分文件过期，警告后恢复 ⚠️        │
        │ workspace-mismatch  │ 运行时身份不匹配，不能恢复 ❌      │
        │ schema-mismatch     │ 结构版本不兼容，不能恢复 ❌        │
        └─────────────────────┴──────────────────────────────────┘
        
        使用场景示例：
        ```python
        # 场景1：用户昨天让 agent 重构代码，达到 max_steps 后停止
        # checkpoint 保存了进度
        resume_state = agent.evaluate_resume_state()
        # → {"status": "full-valid", ...}
        # 今天用户说"继续"，agent 可以从 checkpoint 恢复
        
        # 场景2：用户在两次对话之间修改了 auth.py
        resume_state = agent.evaluate_resume_state()
        # → {"status": "partial-stale", "stale_paths": ["src/auth.py"]}
        # agent 会警告用户文件已变化，但仍可尝试恢复
        
        # 场景3：用户切换了 Git 分支
        resume_state = agent.evaluate_resume_state()
        # → {"status": "workspace-mismatch", "mismatch_fields": ["workspace_fingerprint"]}
        # agent 知道不能恢复，必须重新开始
        ```
        """
        # ====================================================================
        # 步骤1: 获取之前的恢复状态（用于累计统计）
        # ====================================================================
        previous_resume_state = dict(self.session.get("resume_state", {}) or {})
        
        # ====================================================================
        # 步骤2: 清理过期的文件摘要缓存
        # ====================================================================
        # invalidate_stale_memory() 会检查 memory.file_summaries 中的文件
        # 如果文件的 SHA256 与缓存的不一致，说明文件被外部修改了
        # 返回被清理的文件路径列表
        invalidated = self.invalidate_stale_memory()
        
        # ====================================================================
        # 步骤3: 获取当前 checkpoint
        # ====================================================================
        checkpoint = self.current_checkpoint()
        
        # 初始化默认状态：没有 checkpoint
        status = CHECKPOINT_NONE_STATUS
        # stale_paths: 记录 freshness 不匹配的文件路径
        stale_paths = list(invalidated)
        # mismatch_fields: 记录 runtime_identity 不匹配的字段名
        mismatch_fields = []
        
        # ====================================================================
        # 步骤4: 如果有 checkpoint，进行五层验证
        # ====================================================================
        if checkpoint:
            # ----------------------------------------------------------------
            # 第1层验证：检查 schema 版本兼容性
            # ----------------------------------------------------------------
            # CHECKPOINT_SCHEMA_VERSION = "phase1-v1"
            # 如果 Pico 升级导致 checkpoint 结构变化，旧版本的 checkpoint 无法恢复
            if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
                status = CHECKPOINT_SCHEMA_MISMATCH_STATUS
            else:
                # ------------------------------------------------------------
                # 第2层验证：检查关键文件的 freshness（SHA256 对比）
                # ------------------------------------------------------------
                # checkpoint 创建时记录了 key_files 中每个文件的 SHA256 哈希
                # 现在重新计算这些文件的哈希，看是否发生变化
                for item in checkpoint.get("key_files", []):
                    path = str(item.get("path", "")).strip()
                    if not path:
                        continue
                    
                    # expected: checkpoint 创建时的文件哈希
                    expected = item.get("freshness")
                    # current: 当前文件的实际哈希
                    current = memorylib.file_freshness(path, self.root)
                    
                    # 如果哈希值不同，说明文件被外部修改了
                    if expected != current and path not in stale_paths:
                        stale_paths.append(path)
                
                # ------------------------------------------------------------
                # 第3层验证：检查运行时身份一致性（11个关键字段）
                # ------------------------------------------------------------
                # saved_identity: checkpoint 创建时的环境指纹
                saved_identity = dict(
                    checkpoint.get("runtime_identity", {}) or 
                    self.session.get("runtime_identity", {}) or 
                    {}
                )
                # current_identity: 当前的环境指纹
                current_identity = self.current_runtime_identity()
                
                # 需要检查的11个关键字段
                identity_keys = (
                    "cwd",                    # 工作目录
                    "model",                  # 模型名称
                    "model_client",           # 客户端类型
                    "approval_policy",        # 审批策略
                    "read_only",              # 只读模式
                    "max_steps",              # 最大步数
                    "max_new_tokens",         # 最大 token
                    "feature_flags",          # 功能开关
                    "shell_env_allowlist",    # 环境变量白名单
                    "workspace_fingerprint",  # 工作区指纹（Git 状态）
                    "tool_signature",         # 工具注册表签名
                )
                
                # 逐个比较字段，收集不匹配的字段名
                for key in identity_keys:
                    if key not in saved_identity:
                        continue  # 跳过缺失的字段（兼容性处理）
                    if saved_identity.get(key) != current_identity.get(key):
                        mismatch_fields.append(key)
                
                # 排序以便输出稳定（便于测试和调试）
                mismatch_fields.sort()
                
                # ------------------------------------------------------------
                # 根据验证结果确定最终状态（优先级从高到低）
                # ------------------------------------------------------------
                if stale_paths:
                    # 有文件过期，但环境一致 → 部分过期，可以警告后恢复
                    status = CHECKPOINT_PARTIAL_STALE_STATUS
                elif mismatch_fields:
                    # 运行时身份不匹配 → 环境变化太大，不能恢复
                    status = CHECKPOINT_WORKSPACE_MISMATCH_STATUS
                else:
                    # 所有检查通过 → 完全有效，可以安全恢复
                    status = CHECKPOINT_FULL_VALID_STATUS
        
        # ====================================================================
        # 步骤5: 构建并保存恢复状态
        # ====================================================================
        resume_state = {
            "status": status,  # 五种状态之一
            "stale_paths": stale_paths,  # 过期的文件路径列表
            "runtime_identity_mismatch_fields": mismatch_fields,  # 不匹配的字段
            # stale_summary_invalidations: 累计的文件摘要失效次数
            # 用于追踪长期趋势，如果持续增加说明工作区频繁变化
            "stale_summary_invalidations": max(
                len(invalidated),  # 本次失效的文件数
                int(previous_resume_state.get("stale_summary_invalidations", 0))
                if status == CHECKPOINT_PARTIAL_STALE_STATUS  # 只在部分过期时累计
                else 0,
            ),
        }
        
        # 保存到 session，供后续使用（如 render_checkpoint_text、trace 记录等）
        self.session["resume_state"] = resume_state
        
        # 更新当前运行时身份（为下次评估做准备）
        self.session["runtime_identity"] = self.current_runtime_identity()
        
        return resume_state

    def render_checkpoint_text(self):
        """渲染当前 checkpoint 的文本描述。
        
        Returns:
            str: checkpoint 文本描述
        """
        checkpoint = self.current_checkpoint()
        if not checkpoint:
            return ""
        lines = [
            "Task checkpoint:",
            f"- Resume status: {self.resume_state.get('status', CHECKPOINT_NONE_STATUS)}",
            f"- Current goal: {checkpoint.get('current_goal', '-') or '-'}",
            f"- Current blocker: {checkpoint.get('current_blocker', '-') or '-'}",
            f"- Next step: {checkpoint.get('next_step', '-') or '-'}",
        ]
        key_files = [str(item.get("path", "")).strip() for item in checkpoint.get("key_files", []) if str(item.get("path", "")).strip()]
        lines.append(f"- Key files: {', '.join(key_files) or '-'}")
        if checkpoint.get("completed"):
            lines.append("- Completed: " + " | ".join(str(item) for item in checkpoint.get("completed", [])))
        if checkpoint.get("excluded"):
            lines.append("- Excluded: " + " | ".join(str(item) for item in checkpoint.get("excluded", [])))
        if self.resume_state.get("stale_paths"):
            lines.append("- Stale paths: " + ", ".join(self.resume_state["stale_paths"]))
        summary = str(checkpoint.get("summary", "")).strip()
        if summary:
            lines.append(f"- Summary: {summary}")
        return "\n".join(lines)

    @staticmethod
    def remember(bucket, item, limit):
        """将 item 添加到 bucket 中，保持 bucket 的大小不超过 limit。
        
        Args:
            bucket: 目标列表
            item: 要添加的元素
            limit: 最大元素数量
        """
        if not item:
            return
        if item in bucket:
            bucket.remove(item)
        bucket.append(item)
        del bucket[:-limit]

    def build_tools(self):
        """构建工具注册表。
        
        Returns:
            dict: 工具注册表
        """
        return toolkit.build_tool_registry(self)

    def tool_signature(self):
        """生成工具注册表的签名。
        
        Returns:
            str: 工具注册表的 SHA256 哈希值
        """
        payload = []
        for name in sorted(self.tools):
            tool = self.tools[name]
            payload.append(
                {
                    "name": name,
                    "schema": tool["schema"],
                    "risky": tool["risky"],
                    "capabilities": list(tool.get("capabilities", ())),
                    "description": tool["description"],
                }
            )
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def build_prefix(self):
        """构建 Prompt Prefix。
        
        这个函数生成 agent 的"工作手册"，包含角色定义、工具说明、调用示例和工作区快照。
        prefix 是 prompt 中最稳定的部分，通过三重指纹（text hash、workspace fingerprint、tool signature）
        实现缓存复用，避免每次请求都重新构建。
        
        Returns:
            PromptPrefix: Prompt Prefix 对象，包含文本内容和元数据
            
        Note:
            - 工具说明会动态反映当前注册的工具集
            - 工作区快照通过 git 命令实时获取最新状态
            - 返回的对象包含 SHA256 哈希值用于缓存判断
        """
        # ====================================================================
        # 步骤1: 生成工具说明文本
        # ====================================================================
        # 遍历所有已注册的工具，为每个工具生成一行可读的描述
        tool_lines = []
        for name, tool in self.tools.items():
            # 提取工具的参数 schema，格式化为 "param1: type1, param2: type2"
            fields = ", ".join(f"{key}: {value}" for key, value in tool["schema"].items())
            # 渲染 capabilities + 审批提示，让模型对每个工具的副作用面有更细的认知
            caps = tool.get("capabilities", ()) or ("read",)
            caps_label = ",".join(caps)
            risk_hint = "approval required" if tool["risky"] else "safe"
            # 组装成统一的格式：- tool_name(param1: type1, ...) [caps:... | risk_hint] description
            tool_lines.append(f"- {name}({fields}) [caps:{caps_label} | {risk_hint}] {tool['description']}")
        # 将所有工具说明用换行符连接成完整文本
        tool_text = "\n".join(tool_lines)
        
        # ====================================================================
        # 步骤2: 定义工具调用示例
        # ====================================================================
        # 提供多种工具调用格式的示例，帮助模型理解正确的语法
        examples = "\n".join(
            [
                # JSON 格式示例：简单查询类工具
                '<tool>{"name":"list_files","args":{"path":"."}}</tool>',
                '<tool>{"name":"read_file","args":{"path":"README.md","start":1,"end":80}}</tool>',
                # XML 格式示例：多行内容写入（避免 JSON 转义问题）
                '<tool name="write_file" path="binary_search.py"><content>def binary_search(nums, target):\n    return -1\n</content></tool>',
                '<tool name="patch_file" path="binary_search.py"><old_text>return -1</old_text><new_text>return mid</new_text></tool>',
                # Shell 命令执行示例（带超时控制）
                '<tool>{"name":"run_shell","args":{"command":"uv run --with pytest python -m pytest -q","timeout":20}}</tool>',
                # 最终回答格式示例
                "<final>Done.</final>",
            ]
        )
        
        # ====================================================================
        # 步骤3: 组装完整的 prefix 文本
        # ====================================================================
        # prefix 可以理解成 agent 的"工作手册"：它是谁、工具怎么调用、当前仓库是什么状态，都写在这里。
        text = textwrap.dedent(
            f"""\
            You are pico, a small local coding agent working inside a local repository.

            Rules:
            - Use tools instead of guessing about the workspace.
            - Return exactly one <tool>...</tool> or one <final>...</final>.
            - Tool calls must look like:
              <tool>{{"name":"tool_name","args":{{...}}}}</tool>
            - For write_file and patch_file with multi-line text, prefer XML style:
              <tool name="write_file" path="file.py"><content>...</content></tool>
            - Final answers must look like:
              <final>your answer</final>
            - Never invent tool results.
            - Keep answers concise and concrete.
            - If the user asks you to create or update a specific file and the path is clear, use write_file or patch_file instead of repeatedly listing files.
            - Before writing tests for existing code, read the implementation first.
            - When writing tests, match the current implementation unless the user explicitly asked you to change the code.
            - New files should be complete and runnable, including obvious imports.
            - Do not repeat the same tool call with the same arguments if it did not help. Choose a different tool or return a final answer.
            - Required tool arguments must not be empty. Do not call read_file, write_file, patch_file, run_shell, or delegate with args={{}}.

            Tools:
            {tool_text}

            Valid response examples:
            {examples}

            {self.workspace.text()}
            """
        ).strip() # 移除多余的空行
        
        # ====================================================================
        # 步骤4: 生成三重指纹元数据并返回
        # ====================================================================
        return PromptPrefix(
            text=text,                                          # 完整的 prefix 文本
            hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),  # 文本内容的 SHA256 哈希，用于判断 prefix 是否变化
            workspace_fingerprint=self.workspace.fingerprint(),     # 工作区状态指纹（Git 分支/提交/文档等），用于判断工作区是否变化
            tool_signature=self.tool_signature(),                   # 工具注册表的 SHA256 哈希，用于判断工具集是否变化
            built_at=now(),                                         # 构建时间戳（ISO 格式）
        )

    def _apply_prefix_state(self, prefix_state):
        """应用新的 Prefix State。
        
        Args:
            prefix_state: 新的 PromptPrefix 对象
        """
        self.prefix_state = prefix_state
        self.prefix = prefix_state.text

    def refresh_prefix(self, force=False):
        """刷新 Prefix。
        
        Args:
            force: 是否强制刷新
        
        Returns:
            dict: 刷新结果字典
        """
        previous_hash = getattr(getattr(self, "prefix_state", None), "hash", None)
        previous_workspace_fingerprint = getattr(getattr(self, "prefix_state", None), "workspace_fingerprint", None)

        # 工作区事实相对稳定，所以这里按整体刷新；
        # 只有这些事实真的变化了，才重建完整 prefix。
        refreshed_workspace = WorkspaceContext.build(self.root)
        refreshed_workspace_fingerprint = refreshed_workspace.fingerprint()
        workspace_changed = force or refreshed_workspace_fingerprint != previous_workspace_fingerprint
        if workspace_changed:
            self.workspace = refreshed_workspace

        prefix_state = self.build_prefix() if workspace_changed or force or previous_hash is None else self.prefix_state
        prefix_changed = force or previous_hash != prefix_state.hash
        if prefix_changed:
            self._apply_prefix_state(prefix_state)

        self._last_prefix_refresh = {
            "workspace_changed": workspace_changed,
            "prefix_changed": prefix_changed,
        }
        return dict(self._last_prefix_refresh)

    def memory_text(self):
        """获取当前记忆的文本表示。
        
        Returns:
            str: 内存文本表示
        """
        return self.memory.render_memory_text()

    def history_text(self):
        """获取当前对话历史的文本表示。
        
        Returns:
            str: 对话历史文本表示
        """
        history = self.session["history"]
        if not history:
            return "- empty"

        lines = []
        seen_reads = set()
        recent_start = max(0, len(history) - 6)
        for index, item in enumerate(history):
            recent = index >= recent_start
            if item["role"] == "tool" and item["name"] == "read_file" and not recent:
                path = str(item["args"].get("path", ""))
                if path in seen_reads:
                    continue
                seen_reads.add(path)

            if item["role"] == "tool":
                limit = 900 if recent else 180
                lines.append(f"[tool:{item['name']}] {json.dumps(item['args'], sort_keys=True)}")
                lines.append(clip(item["content"], limit))
            else:
                limit = 900 if recent else 220
                lines.append(f"[{item['role']}] {clip(item['content'], limit)}")

        return clip("\n".join(lines), MAX_HISTORY)

    def feature_enabled(self, name):
        """检查某个功能是否启用。
        
        Args:
            name: 功能名称
            
        Returns:
            bool: 是否启用
        """
        return bool(self.feature_flags.get(str(name), False))

    def prompt(self, user_message):
        """生成 prompt。
        
        Args:
            user_message: 用户输入的消息
            
        Returns:
            str: 生成的 prompt
        """
        prompt, _ = self._build_prompt_and_metadata(user_message)
        return prompt

    def record(self, item):
        """记录对话历史。
        
        Args:
            item: 对话历史项
        """
        self.session["history"].append(item)
        self.session_path = self.session_store.save(self.session)

    @staticmethod
    def looks_sensitive_env_name(name):
        """判断环境变量名是否看起来敏感。
        
        Args:
            name: 环境变量名
            
        Returns:
            bool: 是否看起来敏感
        """
        upper = str(name).upper()
        return any(upper == marker or upper.endswith(marker) or upper.endswith(f"_{marker}") for marker in SENSITIVE_ENV_NAME_MARKERS)

    def is_secret_env_name(self, name):
        """判断环境变量名是否是敏感的。
        
        Args:
            name: 环境变量名
            
        Returns:
            bool: 是否是敏感的
        """
        upper = str(name).upper()
        return upper in self.secret_env_names or self.looks_sensitive_env_name(upper)

    def configured_secret_env_items(self):
        """获取配置的敏感环境变量。
        
        Returns:
            list: 敏感环境变量列表
        """
        items = [
            (name, value)
            for name, value in os.environ.items()
            if str(name).upper() in self.secret_env_names and value
        ]
        items.sort(key=lambda item: item[0])
        return items

    def detected_secret_env_items(self):
        """检测所有敏感环境变量。
        
        Returns:
            list: 敏感环境变量列表
        """
        items = [
            (name, value)
            for name, value in os.environ.items()
            if self.is_secret_env_name(name) and value
        ]
        items.sort(key=lambda item: item[0])
        return items

    def secret_env_summary(self):
        """获取配置的敏感环境变量摘要。
        
        Returns:
            dict: 敏感环境变量摘要
        """
        names = [name for name, _ in self.configured_secret_env_items()]
        return {
            "secret_env_count": len(names),
            "secret_env_names": names,
        }

    def detected_secret_env_summary(self):
        """获取检测到的敏感环境变量摘要。
        
        Returns:
            dict: 敏感环境变量摘要
        """
        names = [name for name, _ in self.detected_secret_env_items()]
        return {
            "secret_env_count": len(names),
            "secret_env_names": names,
        }

    def redact_text(self, text):
        """脱敏文本。
        
        Args:
            text: 要脱敏的文本
            
        Returns:
            str: 脱敏后的文本
        """
        text = str(text)
        for _, value in sorted(self.detected_secret_env_items(), key=lambda item: len(item[1]), reverse=True):
            text = text.replace(value, REDACTED_VALUE)
        return text

    def redact_artifact(self, value, key=None):
        """脱敏数据结构。
        
        Args:
            value: 要脱敏的数据结构
            key: 数据结构中的键（用于判断是否是敏感变量）
            
        Returns:
            脱敏后的数据结构
        """
        if key and self.is_secret_env_name(key):
            return REDACTED_VALUE
        if isinstance(value, dict):
            return {
                str(item_key): self.redact_artifact(item_value, key=item_key)
                for item_key, item_value in value.items()
            }
        if isinstance(value, list):
            return [self.redact_artifact(item, key=key) for item in value]
        if isinstance(value, tuple):
            return [self.redact_artifact(item, key=key) for item in value]
        if isinstance(value, str):
            redacted = self.redact_text(value)
            return redacted
        return value

    def shell_env(self):
        """获取允许传递给子进程的 Shell 环境变量。
        
        Returns:
            dict: 允许传递的环境变量字典
        """
        env = {
            name: os.environ[name]
            for name in self.shell_env_allowlist
            if name in os.environ
        }
        env["PWD"] = str(self.root)
        if "PATH" not in env and os.environ.get("PATH"):
            env["PATH"] = os.environ["PATH"]
        return env

    def prompt_metadata(self, user_message, prompt):
        """获取 prompt 的元数据。
        
        Args:
            user_message: 用户输入的消息
            prompt: 生成的 prompt
            
        Returns:
            dict: prompt 的元数据
        """
        _, metadata = self._build_prompt_and_metadata(user_message)
        return metadata

    def _build_prompt_and_metadata(self, user_message):
        """构建 prompt 和元数据。
        
        Args:
            user_message: 用户输入的消息
            
        Returns:
            tuple: (prompt, metadata)
        """
        refresh = self.refresh_prefix()
        self.resume_state = self.evaluate_resume_state()
        prompt, metadata = self.context_manager.build(user_message)
        # 这里把“这轮 prompt 是怎么拼出来的”连同缓存相关状态一起记下来，
        # 后面 trace/report 才能解释清楚：为什么这一轮 prefix 变了、缓存有没有命中。
        metadata.update(
            {
                "prefix_chars": len(self.prefix),
                "workspace_chars": len(self.workspace.text()),
                "memory_chars": len(self.memory_text()),
                "history_chars": len(self.history_text()),
                "request_chars": len(user_message),
                "tool_count": len(self.tools),
                "workspace_docs": len(self.workspace.project_docs),
                "recent_commits": len(self.workspace.recent_commits),
                "prefix_hash": self.prefix_state.hash,
                "prompt_cache_key": self.prefix_state.hash,
                "workspace_fingerprint": self.prefix_state.workspace_fingerprint,
                "tool_signature": self.prefix_state.tool_signature,
                "workspace_changed": refresh["workspace_changed"],
                "prefix_changed": refresh["prefix_changed"],
                "prompt_cache_supported": bool(getattr(self.model_client, "supports_prompt_cache", False)),
                "resume_status": self.resume_state.get("status", CHECKPOINT_NONE_STATUS),
                "stale_summary_invalidations": int(self.resume_state.get("stale_summary_invalidations", 0)),
                "stale_paths": list(self.resume_state.get("stale_paths", [])),
                "runtime_identity_mismatch_fields": list(self.resume_state.get("runtime_identity_mismatch_fields", [])),
            }
        )
        metadata.update(self.detected_secret_env_summary())
        return prompt, metadata

    def emit_trace(self, task_state, event, payload=None):
        """记录 trace 事件。
        
        Args:
            task_state: 当前任务状态
            event: 事件名称
            payload: 事件负载（可选）
            
        Returns:
            dict: 记录的事件负载
        """
        payload = self.redact_artifact(payload or {})
        payload["event"] = event
        payload["created_at"] = now()
        # 注入 phase 字段，便于可视化和评测时按阶段过滤事件
        phase = TRACE_EVENT_PHASE.get(event)
        if phase is None and event == "checkpoint_created":
            trigger = payload.get("trigger", "")
            phase = _CHECKPOINT_TRIGGER_PHASE.get(trigger, "act")
        if phase:
            payload["phase"] = phase
        # trace 是运行中的逐事件时间线，适合回答“这一轮 agent 到底做了什么”。
        self.run_store.append_trace(task_state, payload)
        return payload

    def capture_workspace_snapshot(self):
        """捕获当前工作区的快照。
        
        Returns:
            dict: 工作区快照
        """
        snapshot = {}
        for path in self.root.rglob("*"):
            try:
                relative_parts = path.relative_to(self.root).parts
            except ValueError:
                continue
            if any(part in IGNORED_PATH_NAMES for part in relative_parts):
                continue
            if not path.is_file():
                continue
            try:
                snapshot[path.relative_to(self.root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
            except Exception:
                continue
        return snapshot

    @staticmethod
    def diff_workspace_snapshots(before, after):
        """比较两个工作区快照。
        
        Args:
            before: 之前的快照
            after: 之后的快照
            
        Returns:
            tuple: (变化的路径列表, 变化摘要列表)
        """
        changed_paths = []
        summaries = []
        all_paths = sorted(set(before) | set(after))
        for path in all_paths:
            if before.get(path) == after.get(path):
                continue
            changed_paths.append(path)
            if path not in before:
                summaries.append(f"created:{path}")
            elif path not in after:
                summaries.append(f"deleted:{path}")
            else:
                summaries.append(f"modified:{path}")
        return changed_paths, summaries

    def create_checkpoint(self, task_state, user_message, trigger):
        """创建 checkpoint。
        
        为什么存在：
        Checkpoint 是 Pico "断点续传"机制的核心数据结构。每次关键操作后都会创建 checkpoint，
        保存当前的执行状态、文件快照、运行时身份等信息。这样即使进程崩溃或用户中断，
        下次运行时也能从最近的 checkpoint 恢复，而不必从头开始。
        
        Checkpoint 形成链表结构：
        ```
        ckpt_001 → ckpt_002 → ckpt_003 → ... → ckpt_current
        (parent)   (parent)   (parent)         (current_id)
        ```
        
        输入 / 输出：
        - 输入：
          - task_state: 当前任务状态对象
          - user_message: 用户原始请求
          - trigger: 触发原因（tool_executed / run_finished / freshness_mismatch 等）
        - 输出：checkpoint 字典
        
        在 agent 链路里的位置：
        在以下时机自动创建：
        1. 每次工具执行成功后
        2. 运行正常结束时
        3. 运行异常终止时
        4. 检测到恢复状态异常时（freshness mismatch / workspace mismatch）
        5. 上下文预算被裁剪时
        
        Args:
            task_state: 当前任务状态
            user_message: 用户输入的消息
            trigger: 触发 checkpoint 的原因
            
        Returns:
            dict: checkpoint 数据，包含：
                - checkpoint_id: 唯一标识符（ckpt_ + UUID前8位）
                - parent_checkpoint_id: 父 checkpoint ID（形成链表）
                - schema_version: 结构版本（用于兼容性检查）
                - created_at: 创建时间
                - current_goal: 当前目标（用户请求）
                - completed: 已完成的任务列表
                - excluded: 排除的文件列表
                - current_blocker: 当前阻塞原因
                - next_step: 推断的下一步行动
                - key_files: 关键文件列表及其新鲜度
                - freshness: 文件 SHA256 哈希映射
                - summary: 简要描述
                - runtime_identity: 运行时身份指纹（11个关键字段）
        """
        # ====================================================================
        # 步骤1: 获取当前 checkpoint 状态
        # ====================================================================
        state = self.checkpoint_state()  # 获取 checkpoints 字典结构
        current = self.current_checkpoint()  # 获取当前活跃的 checkpoint
        
        # ====================================================================
        # 步骤2: 生成新的 checkpoint ID
        # ====================================================================
        # 格式：ckpt_ + UUID 的前8位十六进制字符
        # 示例：ckpt_a3f5b2c1
        checkpoint_id = "ckpt_" + uuid.uuid4().hex[:8]
        
        # ====================================================================
        # 步骤3: 收集关键文件的新鲜度信息
        # ====================================================================
        # 从 working_memory 中提取最近访问的文件列表
        # 为每个文件计算 SHA256 哈希，用于后续检测文件是否被外部修改
        key_files = []
        freshness = {}
        for path in self.memory.to_dict()["working"]["recent_files"]:
            # file_freshness() 返回文件的 SHA256 哈希值
            # 如果文件不存在，返回 None
            file_freshness = memorylib.file_freshness(path, self.root)
            freshness[path] = file_freshness
            key_files.append({"path": path, "freshness": file_freshness})
        
        # ====================================================================
        # 步骤4: 构建 checkpoint 数据结构
        # ====================================================================
        checkpoint = {
            # --- 标识信息 ---
            "checkpoint_id": checkpoint_id,  # 当前 checkpoint 的唯一ID
            
            # --- 链表结构 ---
            # 指向父 checkpoint，形成链表
            # 如果是第一个 checkpoint，parent 为空字符串
            "parent_checkpoint_id": current.get("checkpoint_id", "") if current else "",
            
            # --- 版本控制 ---
            # 用于兼容性检查，防止旧版本的 checkpoint 被错误加载
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            
            # --- 时间戳 ---
            "created_at": now(),
            
            # --- 任务进度 ---
            "current_goal": str(user_message),  # 用户的原始请求
            
            # 已完成的任务（如果有最终答案）
            "completed": [task_state.final_answer] if task_state.final_answer else [],
            
            # 排除的文件列表（预留字段，暂未使用）
            "excluded": [],
            
            # 当前阻塞原因（如果不是正常结束）
            "current_blocker": "" if str(task_state.stop_reason or "") in ("", "final_answer_returned") else str(task_state.stop_reason),
            
            # 推断的下一步行动（基于当前状态分析）
            "next_step": self.infer_next_step(task_state),
            
            # --- 文件快照 ---
            "key_files": key_files,  # 关键文件列表（含路径和新鲜度）
            "freshness": freshness,  # 文件哈希映射（快速查找）
            
            # --- 摘要信息 ---
            # 简要描述这个 checkpoint 的触发原因和上下文
            # 例如："tool_executed: 修复 auth.py 中的登录bug"
            "summary": f"{trigger}: {clip(str(user_message), 120)}",
            
            # --- 运行时身份 ---
            # 包含11个关键字段的指纹：
            # cwd, model, model_client, approval_policy, read_only,
            # max_steps, max_new_tokens, feature_flags, 
            # shell_env_allowlist, workspace_fingerprint, tool_signature
            "runtime_identity": self.current_runtime_identity(),
        }
        
        # ====================================================================
        # 步骤5: 保存 checkpoint 到 session
        # ====================================================================
        # 将新 checkpoint 添加到 items 字典中
        state["items"][checkpoint_id] = checkpoint
        
        # 更新 current_id 指向最新的 checkpoint
        state["current_id"] = checkpoint_id
        
        # 同步更新 task_state 中的 checkpoint_id
        task_state.checkpoint_id = checkpoint_id
        
        # 更新 session 中的运行时身份（供下次评估使用）
        self.session["runtime_identity"] = checkpoint["runtime_identity"]
        
        # 持久化 session 到磁盘
        # 路径：.pico/sessions/{session_id}.json
        self.session_path = self.session_store.save(self.session)
        
        return checkpoint

    def infer_next_step(self, task_state):
        """推断下一步操作。
        
        Args:
            task_state: 当前任务状态
            
        Returns:
            str: 下一步操作描述
        """
        if task_state.status == "completed":
            return "No next step recorded."
        if task_state.stop_reason == "step_limit_reached":
            return "Resume from the latest checkpoint and continue the task."
        if task_state.last_tool:
            return f"Decide the next action after {task_state.last_tool}."
        return "Continue the task from the latest checkpoint."

    def update_memory_after_tool(self, name, args, result):
        """把少量高价值工具结果沉淀到 working memory。

        为什么存在：
        并不是每个工具结果都值得长期带进下一轮 prompt。完整结果已经进了
        `history`，这里只挑少量“下一轮大概率还会用到”的事实做提纯，
        例如最近读写过哪些文件、某个文件读出来的短摘要。

        输入 / 输出：
        - 输入：工具名 `name`、参数 `args`、执行结果 `result`
        - 输出：无显式返回值，副作用是更新 `self.memory`

        在 agent 链路里的位置：
        它发生在 `run_tool()` 真正执行完工具之后、下一轮 prompt 组装之前。
        也就是说：工具结果先进入完整历史，再由这个函数择优沉淀成轻量记忆。
        """
        if not self.feature_enabled("memory"):
            return
        path = args.get("path")
        if not path:
            return

        canonical_path = self.memory.canonical_path(path)
        # 不是所有工具结果都进入工作记忆。
        # 读文件会生成摘要；写文件/patch 会让旧摘要失效，因为它们可能过期了。
        if name in {"read_file", "write_file", "patch_file"}:
            self.memory.remember_file(canonical_path)
        if name == "read_file":
            summary = memorylib.summarize_read_result(result)
            self.memory.set_file_summary(canonical_path, summary)
            self.memory.append_note(summary, tags=(canonical_path,), source=canonical_path)
        elif name in {"write_file", "patch_file"}:
            self.memory.invalidate_file_summary(canonical_path)

    def note_tool(self, name, args, result):
        """记录工具调用结果。
        
        Args:
            name: 工具名称
            args: 工具参数
            result: 工具结果
        """
        self.update_memory_after_tool(name, args, result)

    def record_process_note_for_tool(self, name, metadata):
        """记录工具调用的处理日志。
        
        Args:
            name: 工具名称
            metadata: 工具调用元数据
        """
        status = str(metadata.get("tool_status", "")).strip()
        if status not in {"partial_success", "error", "rejected"}:
            return
        affected_paths = [str(path).strip() for path in metadata.get("affected_paths", []) if str(path).strip()]
        path_text = ", ".join(affected_paths) or "workspace"
        if status == "partial_success":
            text = f"{name} partial_success on {path_text}; inspect diff before retry"
        elif status == "error":
            text = f"{name} error on {path_text}; check the failure before retry"
        else:
            text = f"{name} rejected; choose a different action before retry"
        tags = ["process", status, *affected_paths]
        self.memory.append_note(text, tags=tuple(tags), source=name, kind="process")
        self.session["memory"] = self.memory.to_dict()

    def reject_durable_reason(self, note_text):
        """判断长期记忆是否应该被拒绝。
        
        Args:
            note_text: 长期记忆内容
            
        Returns:
            str: 拒绝原因（如果有的话）
        """
        text = str(note_text or "").strip()
        lowered = text.lower()
        if not text:
            return "empty"
        if REDACTED_VALUE in text or SECRET_SHAPED_TEXT_PATTERN.search(text):
            return "secret_shaped"
        checkpoint_like_prefixes = (
            "current goal",
            "current blocker",
            "next step",
            "current phase",
            "key files",
            "freshness",
            "当前目标",
            "当前卡点",
            "下一步",
            "当前阶段",
            "关键文件",
            "已完成",
            "已排除",
        )
        if any(lowered.startswith(prefix) for prefix in checkpoint_like_prefixes):
            return "transient_task_state"
        if re.search(r"(?i)\b(stdout|stderr|traceback|exit_code)\b", text) or len(text) > 220:
            return "noisy_output"
        return ""

    def extract_durable_promotions(self, user_message, final_answer):
        """提取长期记忆的提升和拒绝。
        
        Args:
            user_message: 用户输入的消息
            final_answer: 最终答案
            
        Returns:
            tuple: (提升列表, 拒绝列表)
        """
        user_text = str(user_message or "")
        if not (DURABLE_MEMORY_INTENT_PATTERN.search(user_text) or DURABLE_MEMORY_INTENT_ZH_PATTERN.search(user_text)):
            return [], []
        promotions = []
        rejections = []
        for line in str(final_answer or "").splitlines():
            text = line.strip()
            if not text or REDACTED_VALUE in text:
                continue
            for topic, pattern in DURABLE_MEMORY_LINE_PATTERNS:
                match = pattern.match(text)
                if not match:
                    continue
                note_text = match.group(1).strip()
                if note_text:
                    reason = self.reject_durable_reason(note_text)
                    if reason:
                        rejections.append(f"{topic}:{reason}")
                        break
                    promotions.append((topic, note_text))
                break
        return promotions, rejections

    def promote_durable_memory(self, user_message, final_answer):
        """提升长期记忆。
        
        Args:
            user_message: 用户输入的消息
            final_answer: 最终答案
            
        Returns:
            tuple: (提升列表, 拒绝列表, 被替代的列表)
        """
        promotions, rejections = self.extract_durable_promotions(user_message, final_answer)
        promoted, superseded = self.memory.promote_durable(promotions)
        self.session["memory"] = self.memory.to_dict()
        self.last_durable_promotions = promoted
        self.last_durable_rejections = rejections
        self.last_durable_superseded = superseded
        return promoted, rejections, superseded

    def ask(self, user_message):
        """
        为什么存在：
        `ask()` 是整个 runtime 的总调度器。它把"用户提一个请求"扩展成一条可持续推进的控制循环：记录会话、组 prompt、调用模型、执行工具、
        写 trace/report、更新状态，直到模型给出最终答案或系统主动停下。

        输入 / 输出：
        - 输入：`user_message`，即用户这一次的任务描述
        - 输出：字符串形式的最终回答；如果中途达到步数上限或重试上限，返回的是一条停止原因说明

        在 agent 链路里的位置：
        它是 CLI 和底层工具/模型之间的核心桥梁。
        CLI 收到用户输入后基本只做一件事：调用 `agent.ask()`。而 `ask()` 内部再去驱动 `ContextManager`组 prompt、`model_client.complete()` 调模型、`run_tool()` 执行动作。
        
        执行流程概览：
        ┌─────────────────────────────────────────────────────────────┐
        │ 阶段1: 初始化准备                                            │
        │ ├─> 设置任务摘要到工作记忆                                   │
        │ ├─> 记录用户消息到历史                                       │
        │ ├─> 创建 TaskState 对象                                      │
        │ ├─> 启动运行记录 (RunStore)                                  │
        │ └─> 发送 run_started trace 事件                              │
        ├─────────────────────────────────────────────────────────────┤
        │ 阶段2: 主循环 (感知→决策→行动→记录)                          │
        │ while tool_steps < max_steps and attempts < max_attempts:   │
        │   ├─> 感知: _build_prompt_and_metadata() 构建 prompt         │
        │   │   ├─> refresh_prefix() 刷新工作区快照                    │
        │   │   ├─> evaluate_resume_state() 评估恢复状态               │
        │   │   └─> context_manager.build() 组装完整 prompt            │
        │   ├─> 检查恢复状态并创建 checkpoint（如需要）                │
        │   ├─> 决策: model_client.complete() 调用模型                 │
        │   ├─> parse() 解析模型输出                                   │
        │   │   ├─> kind="tool": 提取工具调用                          │
        │   │   ├─> kind="final": 提取最终答案                         │
        │   │   └─> kind="retry": 格式错误需要重试                     │
        │   ├─> 行动: 根据 kind 执行不同分支                           │
        │   │   ├─> tool: run_tool() 执行工具调用                      │
        │   │   │   ├─> validate_tool() 参数校验                       │
        │   │   │   ├─> repeated_tool_call() 重复检测                  │
        │   │   │   ├─> approve() 审批控制                             │
        │   │   │   ├─> 执行工具函数                                   │
        │   │   │   └─> update_memory_after_tool() 更新记忆            │
        │   │   ├─> final: 结束循环                                    │
        │   │   └─> retry: 继续下一轮                                  │
        │   └─> 记录: emit_trace(), record(), create_checkpoint()      │
        ├─────────────────────────────────────────────────────────────┤
        │ 阶段3: 异常终止处理                                          │
        │ ├─> 达到 max_attempts: 返回重试超限提示                      │
        │ └─> 达到 max_steps: 返回步数超限提示                         │
        └─────────────────────────────────────────────────────────────┘
        """
        # ====================================================================
        # 阶段1: 初始化准备
        # ====================================================================
        
        # 记录运行开始时间（用于计算总耗时）
        run_started_at = time.monotonic()
        
        # 【关键】将用户任务设置为工作记忆中的 task_summary
        # 这样后续 prompt 构建时会自动包含当前任务目标
        self.memory.set_task_summary(user_message)
        
        # 记录用户消息到 session history
        self.record({"role": "user", "content": user_message, "created_at": now()})

        # 创建新的任务状态对象
        # TaskState 跟踪整个运行的元数据：run_id, task_id, 工具调用次数等
        task_state = TaskState.create(
            run_id=self.new_run_id(),
            task_id=self.new_task_id(),
            user_request=user_message,
            parent_run_id=self.parent_run_id,
        )
        # Expose current run_id so tool_delegate can pass it as parent_run_id to children
        self._current_run_id = task_state.run_id
        
        # 设置恢复状态（从之前可能的 checkpoint 中读取）
        # 可能的值：no-checkpoint, full-valid, partial-stale, workspace-mismatch, schema-mismatch
        task_state.resume_status = self.resume_state.get("status", CHECKPOINT_NONE_STATUS)
        
        # 保存当前任务状态引用（供其他方法访问）
        self.current_task_state = task_state
        
        # 启动运行记录目录
        # 在 .pico/runs/{run_id}/ 下创建目录，用于存储 trace 和 report
        self.current_run_dir = self.run_store.start_run(task_state)
        
        # 发送第一个 trace 事件：标记运行开始
        # trace 是逐事件的时间线，用于调试和审计
        self.emit_trace(
            task_state,
            "run_started",
            {
                "task_id": task_state.task_id,
                "user_request": clip(user_message, 300),  # 截断过长内容
                "parent_run_id": task_state.parent_run_id,
            },
        )

        # ====================================================================
        # 阶段2: 主循环初始化
        # ====================================================================
        
        # tool_steps: 已执行的工具调用次数（限制每次运行的最大步数）
        tool_steps = 0
        
        # attempts: 模型调用尝试次数（防止模型反复返回无效格式）
        attempts = 0
        
        # max_attempts: 最大尝试次数
        # 策略：至少 max_steps*3，但不小于 max_steps+4
        max_attempts = max(self.max_steps * 3, self.max_steps + 4)

        # 这是 agent 的主循环，可以按"感知 -> 决策 -> 行动 -> 记录"来理解：
        # 1. 感知：重新组 prompt，把当前状态整理给模型看
        # 2. 决策：让模型返回一个工具调用，或一个最终答案
        # 3. 行动：如果是工具调用，就执行工具
        # 4. 记录：把结果写回 history / task_state / trace / memory
        # 然后进入下一轮，直到停机条件满足
        while tool_steps < self.max_steps and attempts < max_attempts:
            # ----------------------------------------------------------------
            # 循环开始：记录本次尝试
            # ----------------------------------------------------------------
            attempts += 1
            task_state.record_attempt()  # 增加 attempts 计数
            
            # 持久化任务状态到磁盘
            self.run_store.write_task_state(task_state)
            
            # =================================================================
            # 步骤1: 感知 - 构建 Prompt
            # =================================================================
            prompt_started_at = time.monotonic()
            
            # 【核心】构建本轮的完整 prompt 和元数据
            # 这个函数会：
            # 1. 刷新工作区快照（git 状态、文件列表等）
            # 2. 评估恢复状态（检查 checkpoint 是否有效）
            # 3. 通过 ContextManager 组装 prefix + memory + history + 当前请求
            prompt, prompt_metadata = self._build_prompt_and_metadata(user_message)
            
            # 记录 prompt 构建完成的 trace 事件
            self.emit_trace(
                task_state,
                "prompt_built",
                {
                    "prompt_metadata": prompt_metadata,  # 包含缓存命中率、预算使用等
                    "duration_ms": int((time.monotonic() - prompt_started_at) * 1000),
                },
            )
            
            # =================================================================
            # 步骤2: 检查恢复状态并创建 Checkpoint
            # =================================================================
            
            # 情况1: 部分文件过期（freshness mismatch）
            # 说明用户在两次对话之间修改了某些文件
            if prompt_metadata.get("resume_status") == CHECKPOINT_PARTIAL_STALE_STATUS:
                # 创建 checkpoint 标记这个不一致状态
                checkpoint = self.create_checkpoint(task_state, user_message, trigger="freshness_mismatch")
                self.run_store.write_task_state(task_state)
                self.emit_trace(
                    task_state,
                    "checkpoint_created",
                    {
                        "checkpoint_id": checkpoint["checkpoint_id"],
                        "trigger": "freshness_mismatch",
                    },
                )
            
            # 情况2: 运行时身份不匹配（workspace mismatch）
            # 说明环境发生了重大变化（如切换 Git 分支、改变模型配置等）
            elif prompt_metadata.get("resume_status") == CHECKPOINT_WORKSPACE_MISMATCH_STATUS:
                # 记录哪些字段不匹配
                self.emit_trace(
                    task_state,
                    "runtime_identity_mismatch",
                    {
                        "fields": list(prompt_metadata.get("runtime_identity_mismatch_fields", [])),
                    },
                )
                # 创建 checkpoint 标记这个严重的不一致
                checkpoint = self.create_checkpoint(task_state, user_message, trigger="workspace_mismatch")
                self.run_store.write_task_state(task_state)
                self.emit_trace(
                    task_state,
                    "checkpoint_created",
                    {
                        "checkpoint_id": checkpoint["checkpoint_id"],
                        "trigger": "workspace_mismatch",
                    },
                )
            
            # 情况3: 上下文预算被裁剪（budget reductions）
            # 说明 history/memory 太长，触发了压缩
            if prompt_metadata.get("budget_reductions"):
                # 创建 checkpoint 保存裁剪前的状态
                checkpoint = self.create_checkpoint(task_state, user_message, trigger="context_reduction")
                self.run_store.write_task_state(task_state)
                self.emit_trace(
                    task_state,
                    "checkpoint_created",
                    {
                        "checkpoint_id": checkpoint["checkpoint_id"],
                        "trigger": "context_reduction",
                    },
                )
            
            # =================================================================
            # 步骤3: 决策 - 调用模型
            # =================================================================
            self.emit_trace(
                task_state,
                "model_requested",
                {
                    "attempts": task_state.attempts,
                    "tool_steps": task_state.tool_steps,
                    "prompt_cache_key": prompt_metadata.get("prompt_cache_key"),
                },
            )
            
            # 准备 prompt 缓存参数（如果后端支持）
            prompt_cache_key = None
            prompt_cache_retention = None
            if getattr(self.model_client, "supports_prompt_cache", False):
                # 只有后端明确支持时，才把稳定前缀的 hash 作为 cache key 发出去。
                # 这样可以避免重复发送不变的 prefix，节省 token 和延迟
                prompt_cache_key = prompt_metadata.get("prompt_cache_key")
                prompt_cache_retention = "in_memory"
            
            # 【关键】调用模型客户端获取响应
            model_started_at = time.monotonic()
            try:
                raw = self.model_client.complete(
                    prompt,
                    self.max_new_tokens,
                    prompt_cache_key=prompt_cache_key,
                    prompt_cache_retention=prompt_cache_retention,
                )
            except RuntimeError as _backend_exc:
                _err_msg = str(_backend_exc)
                task_state.stop_backend_error(_err_msg)
                self.record({"role": "assistant", "content": _err_msg, "created_at": now()})
                self.run_store.write_task_state(task_state)
                self.emit_trace(
                    task_state,
                    "run_finished",
                    {
                        "status": task_state.status,
                        "stop_reason": task_state.stop_reason,
                        "run_duration_ms": int((time.monotonic() - run_started_at) * 1000),
                    },
                )
                self.run_store.write_report(task_state, self.redact_artifact(self.build_report(task_state)))
                return f"Stopped due to backend error: {_err_msg}"
            
            # 提取模型返回的元数据（usage、cache 统计等）
            completion_metadata = dict(getattr(self.model_client, "last_completion_metadata", {}) or {})
            if completion_metadata:
                # 把后端返回的 usage/cache 统计并回 prompt_metadata，
                # 方便统一写入 report 和 trace。
                prompt_metadata.update(completion_metadata)
            
            # 保存元数据供后续使用
            self.last_completion_metadata = completion_metadata
            self.last_prompt_metadata = prompt_metadata
            
            # =================================================================
            # 步骤4: 解析模型输出
            # =================================================================
            
            # 【关键】将模型的原始文本输出解析成结构化动作
            # 返回值：(kind, payload)
            # - kind="tool": payload={"name": "...", "args": {...}}
            # - kind="final": payload="最终答案文本"
            # - kind="retry": payload="错误提示信息"
            kind, payload, parse_detail = self.parse(raw)
            
            # 记录解析结果的 trace 事件
            self.emit_trace(
                task_state,
                "model_parsed",
                {
                    "kind": kind,
                    "parse_detail": parse_detail,
                    "completion_metadata": completion_metadata,
                    "duration_ms": int((time.monotonic() - model_started_at) * 1000),
                },
            )

            # =================================================================
            # 步骤5: 行动 - 根据解析结果执行不同分支
            # =================================================================
            
            # 分支1: 模型决定调用工具
            if kind == "tool":
                tool_steps += 1  # 增加工具调用计数
                
                # 提取工具名称和参数
                name = payload.get("name", "")
                args = payload.get("args", {})
                
                # 记录这次工具调用到 task_state
                task_state.record_tool(name)
                
                # 【核心】执行工具调用（带完整安全护栏）
                tool_started_at = time.monotonic()
                result = self.run_tool(name, args)
                
                # 记录工具调用结果到 session history
                self.record(
                    {
                        "role": "tool",
                        "name": name,
                        "args": args,
                        "content": result,  # 工具返回的结果（成功或错误信息）
                        "created_at": now(),
                    }
                )
                
                # 持久化任务状态
                self.run_store.write_task_state(task_state)
                
                # 记录工具执行的 trace 事件
                # `result_text` 截断给人看；`result_payload` 不截断，给评测脚本
                # 和 trace 聚合做结构化分析（exit_code、bytes_changed、match_count 等）。
                metadata = dict(self._last_tool_result_metadata or {})
                trace_payload = {
                    "name": name,
                    "args": args,
                    "result": clip(result, 500),
                    "result_chars": len(result or ""),
                    "duration_ms": int((time.monotonic() - tool_started_at) * 1000),
                    **metadata,
                }
                # 提升关键字段的可见性：放到顶层 key，trace 消费方不需要再去
                # `tool_result_payload` 里挖。
                if isinstance(metadata.get("tool_result_payload"), dict):
                    trace_payload["result_payload"] = metadata["tool_result_payload"]
                self.emit_trace(task_state, "tool_executed", trace_payload)
                
                # 【关键】每次工具执行后创建 checkpoint
                # 这样可以在下次运行时从这一步继续，而不必重头开始
                checkpoint = self.create_checkpoint(task_state, user_message, trigger="tool_executed")
                self.run_store.write_task_state(task_state)
                self.emit_trace(
                    task_state,
                    "checkpoint_created",
                    {
                        "checkpoint_id": checkpoint["checkpoint_id"],
                        "trigger": "tool_executed",
                    },
                )
                
                # 继续下一轮循环（让模型基于工具结果做下一步决策）
                continue

            # 分支2: 模型输出格式错误，需要重试
            if kind == "retry":
                # 记录重试提示到 history（让模型看到自己的错误）
                self.record({"role": "assistant", "content": payload, "created_at": now()})
                self.run_store.write_task_state(task_state)
                # 继续下一轮循环（给模型再次尝试的机会）
                continue

            # 分支3: 模型给出了最终答案
            # （既不是工具调用，也不是重试，直接视为最终回答）
            final = (payload or raw).strip()
            
            # 记录最终答案到 history
            self.record({"role": "assistant", "content": final, "created_at": now()})
            
            # 标记任务成功完成
            task_state.finish_success(final)
            
            # 【关键】提升长期记忆
            # 从本轮对话中提取有价值的知识点，沉淀到 durable memory
            self.promote_durable_memory(user_message, final)
            
            # 创建最终 checkpoint
            checkpoint = self.create_checkpoint(task_state, user_message, trigger="run_finished")
            self.run_store.write_task_state(task_state)
            self.emit_trace(
                task_state,
                "checkpoint_created",
                {
                    "checkpoint_id": checkpoint["checkpoint_id"],
                    "trigger": "run_finished",
                },
            )
            
            # 记录运行结束的 trace 事件
            self.emit_trace(
                task_state,
                "run_finished",
                {
                    "status": task_state.status,
                    "stop_reason": task_state.stop_reason,
                    "final_answer": final,
                    "run_duration_ms": int((time.monotonic() - run_started_at) * 1000),
                },
            )
            
            # 【关键】生成并保存运行报告
            # report 是对整个运行的摘要，包含指标、元数据、持久化记忆等
            self.run_store.write_report(task_state, self.redact_artifact(self.build_report(task_state)))
            
            # 返回最终答案给调用者（CLI 或其他上层组件）
            return final

        # ====================================================================
        # 阶段3: 异常终止处理（退出循环但未返回最终答案）
        # ====================================================================
        
        # 情况1: 达到最大尝试次数，但工具步数未超限
        # 说明模型反复返回无效格式，无法正常推进
        if attempts >= max_attempts and tool_steps < self.max_steps:
            final = "Stopped after too many malformed model responses without a valid tool call or final answer."
            task_state.stop_retry_limit(final)
        
        # 情况2: 达到最大工具步数
        # 说明任务太复杂，在限定步数内无法完成
        else:
            final = "Stopped after reaching the step limit without a final answer."
            task_state.stop_step_limit(final)
        
        # 记录停止原因到 history
        self.record({"role": "assistant", "content": final, "created_at": now()})
        
        # 尝试从停止状态中提取长期记忆
        self.promote_durable_memory(user_message, final)
        
        # 持久化任务状态
        self.run_store.write_task_state(task_state)
        
        # 创建 checkpoint（标记停止原因）
        checkpoint = self.create_checkpoint(task_state, user_message, trigger=task_state.stop_reason or "run_stopped")
        self.emit_trace(
            task_state,
            "checkpoint_created",
            {
                "checkpoint_id": checkpoint["checkpoint_id"],
                "trigger": task_state.stop_reason or "run_stopped",
            },
        )
        
        # 记录运行结束的 trace 事件
        self.emit_trace(
            task_state,
            "run_finished",
            {
                "status": task_state.status,
                "stop_reason": task_state.stop_reason,
                "final_answer": final,
                "run_duration_ms": int((time.monotonic() - run_started_at) * 1000),
            },
        )
        
        # 生成并保存运行报告
        self.run_store.write_report(task_state, self.redact_artifact(self.build_report(task_state)))
        
        # 返回停止原因说明
        return final

    def run_tool(self, name, args):
        """执行一次工具调用，并在执行前后套上完整护栏。

        为什么存在：
        在 agent 系统里，真正危险的不是"模型会不会想调用工具"，而是"平台有没有在执行前把边界守住"。
        这个函数就是工具层的总闸口： 所有工具调用都必须先经过它，不能让模型直接碰到底层函数。

        输入 / 输出：
        - 输入：工具名 `name`，参数字典 `args`
        - 输出：字符串结果。无论是成功结果还是错误信息，都会统一返回文本，
          这样模型下一轮都能继续消费这份反馈。

        在 agent 链路里的位置：
        它位于 `ask()` 的"模型决定要调用工具"之后，是控制循环里真正把模型意图落到外部世界的一步。因此这里串起了几乎所有安全与可控设计：
        工具是否存在、参数是否合法、是否重复、是否需要审批、执行结果是否裁剪、是否需要回写记忆。
        
        安全防护流水线（六层防护）：
        ```
        1. 工具存在性检查 → 防止调用未注册的工具
        2. validate_tool() → 参数校验（路径锚定、类型检查、必填参数）
        3. repeated_tool_call() → 重复调用检测（阻止最近2次相同调用）
        4. approve() → 审批策略控制（auto/ask/never + read_only）
        5. 执行前后快照对比 → 追踪工作区变化
        6. update_memory_after_tool() → 更新工作记忆
        ```
        
        Args:
            name: 工具名称
            args: 参数字典
            
        Returns:
            str: 工具执行结果（成功输出或错误信息）
        """
        # ====================================================================
        # 工具执行流水线：带完整安全护栏
        # 工具是否存在 -> 参数是否合法 -> 是否重复调用 -> 是否通过审批 -> 
        # 真正执行 -> 更新记忆
        # ====================================================================
        
        # --------------------------------------------------------------------
        # 第1层防护：检查工具是否存在
        # --------------------------------------------------------------------
        tool = self.tools.get(name)
        if tool is None:
            # 记录拒绝元数据（用于 trace/report）
            self._last_tool_result_metadata = {
                "tool_status": "rejected",
                "tool_error_code": "unknown_tool",
                "security_event_type": "",
                "risk_level": "high",
                "read_only": False,
                "capabilities": [],
                "tool_result_payload": {},
                "tool_result_ok": False,
                "affected_paths": [],
                "workspace_changed": False,
                "diff_summary": [],
            }
            return f"error: unknown tool '{name}'"
        
        # --------------------------------------------------------------------
        # 第2层防护：参数校验
        # --------------------------------------------------------------------
        # validate_tool() 会检查：
        # - 路径是否逃逸出 workspace root
        # - 参数类型是否正确（str/int/dict）
        # - 必填参数是否存在
        # - 特殊约束（如 patch_file 的 old_text 唯一性）
        try:
            self.validate_tool(name, args)
        except Exception as exc:
            # 获取该工具的示例用法（帮助模型理解正确格式）
            example = self.tool_example(name)
            message = f"error: invalid arguments for {name}: {exc}"
            if example:
                message += f"\nexample: {example}"
            
            # 判断是否是路径逃逸攻击
            security_event_type = "path_escape" if "path escapes workspace" in str(exc) else ""
            
            self._last_tool_result_metadata = {
                "tool_status": "rejected",
                "tool_error_code": "invalid_arguments",
                "security_event_type": security_event_type,
                "risk_level": "high" if tool["risky"] else "low",
                "read_only": not tool["risky"],
                "capabilities": list(tool.get("capabilities", ())),
                "tool_result_payload": {},
                "tool_result_ok": False,
                "affected_paths": [],
                "workspace_changed": False,
                "diff_summary": [],
            }
            return message
        
        # --------------------------------------------------------------------
        # 第3层防护：重复调用检测
        # --------------------------------------------------------------------
        # 阻止 agent 在没有新信息的情况下反复发起同一调用
        # 检查最近2次 tool 事件是否完全相同
        if self.repeated_tool_call(name, args):
            self._last_tool_result_metadata = {
                "tool_status": "rejected",
                "tool_error_code": "repeated_identical_call",
                "security_event_type": "",
                "risk_level": "high" if tool["risky"] else "low",
                "read_only": not tool["risky"],
                "capabilities": list(tool.get("capabilities", ())),
                "tool_result_payload": {},
                "tool_result_ok": False,
                "affected_paths": [],
                "workspace_changed": False,
                "diff_summary": [],
            }
            return f"error: repeated identical tool call for {name}; choose a different tool or return a final answer"
        
        # --------------------------------------------------------------------
        # 第4层防护：审批策略控制
        # --------------------------------------------------------------------
        # 对于 risky 工具，需要根据 approval_policy 进行审批
        # - read_only=True: 一律拒绝
        # - policy="auto": 自动通过
        # - policy="never": 一律拒绝
        # - policy="ask": 询问用户确认
        if tool["risky"] and not self.approve(name, args, capabilities=tool.get("capabilities", ())):
            self._last_tool_result_metadata = {
                "tool_status": "rejected",
                "tool_error_code": "approval_denied",
                "security_event_type": "read_only_block" if self.read_only else "approval_denied",
                "risk_level": "high",
                "read_only": False,
                "capabilities": list(tool.get("capabilities", ())),
                "tool_result_payload": {},
                "tool_result_ok": False,
                "affected_paths": [],
                "workspace_changed": False,
                "diff_summary": [],
            }
            return f"error: approval denied for {name}"
        
        # --------------------------------------------------------------------
        # 第5步：执行前捕获工作区快照（仅 risky 工具）
        # --------------------------------------------------------------------
        # 用于后续对比，追踪哪些文件被修改了
        before_snapshot = self.capture_workspace_snapshot() if tool["risky"] else {}
        after_snapshot = before_snapshot
        
        # --------------------------------------------------------------------
        # 第6步：真正执行工具函数
        # --------------------------------------------------------------------
        try:
            # 调用工具的实际实现函数
            # 工具统一返回 ToolResult；clip() 只对要喂回模型的 text 做截断，
            # payload 不截断，方便 trace 聚合和评测脚本拿到完整字段。
            raw_result = tool["run"](args)
            tool_result = raw_result if isinstance(raw_result, toolkit.ToolResult) else toolkit.ToolResult(text="" if raw_result is None else str(raw_result))
            result = clip(tool_result.text)
            
            # 执行后再次捕获快照（仅 risky 工具）
            after_snapshot = self.capture_workspace_snapshot() if tool["risky"] else before_snapshot
            
            # 对比执行前后的快照，找出被修改的文件
            affected_paths, diff_summary = self.diff_workspace_snapshots(before_snapshot, after_snapshot)
            workspace_changed = bool(affected_paths)
            
            # 初始化状态为成功
            tool_status = "ok"
            tool_error_code = ""
            
            # 特殊处理：run_shell 需要检查退出码
            # 现在 ToolResult.payload 直接给了结构化的 exit_code / timed_out，
            # 不再依赖正则解析文本。同时保留正则做兜底（万一某个自定义工具叫
            # run_shell 但没填 payload）。
            if name == "run_shell":
                payload = tool_result.payload or {}
                if "exit_code" in payload or "timed_out" in payload:
                    exit_code = payload.get("exit_code")
                    timed_out = bool(payload.get("timed_out"))
                    failed = timed_out or (exit_code is not None and exit_code != 0)
                else:
                    match = re.search(r"exit_code:\s*(-?\d+)", result)
                    exit_code = int(match.group(1)) if match else 0
                    failed = exit_code != 0
                if failed and workspace_changed:
                    tool_status = "partial_success"
                    tool_error_code = "tool_partial_success"
                elif failed:
                    tool_status = "error"
                    tool_error_code = "tool_failed"
            
            # --------------------------------------------------------------------
            # 第7步：更新工作记忆
            # --------------------------------------------------------------------
            # 根据工具执行结果，更新 working_memory 中的 recent_files 等
            self.update_memory_after_tool(name, args, result)
            
            # 记录成功的元数据
            self._last_tool_result_metadata = {
                "tool_status": tool_status,
                "tool_error_code": tool_error_code,
                "security_event_type": "",
                "risk_level": "high" if tool["risky"] else "low",
                "read_only": not tool["risky"],
                "capabilities": list(tool.get("capabilities", ())),
                "tool_result_payload": dict(tool_result.payload or {}),
                "tool_result_ok": bool(tool_result.ok),
                "affected_paths": affected_paths,
                "workspace_changed": workspace_changed,
                "workspace_fingerprint": self.workspace.fingerprint(),
                "diff_summary": diff_summary,
            }
            
            # 记录处理笔记（用于调试和审计）
            self.record_process_note_for_tool(name, self._last_tool_result_metadata)
            
            return result
        
        # --------------------------------------------------------------------
        # 异常处理：工具执行失败
        # --------------------------------------------------------------------
        except Exception as exc:
            # 即使出错也要捕获快照，追踪是否有部分修改
            after_snapshot = self.capture_workspace_snapshot() if tool["risky"] else before_snapshot
            affected_paths, diff_summary = self.diff_workspace_snapshots(before_snapshot, after_snapshot)
            workspace_changed = bool(affected_paths)
            
            # 判断是否是路径逃逸攻击
            security_event_type = "path_escape" if "path escapes workspace" in str(exc) else ""
            
            self._last_tool_result_metadata = {
                "tool_status": "partial_success" if workspace_changed else "error",
                "tool_error_code": "tool_partial_success" if workspace_changed else "tool_failed",
                "security_event_type": security_event_type,
                "risk_level": "high" if tool["risky"] else "low",
                "read_only": not tool["risky"],
                "capabilities": list(tool.get("capabilities", ())),
                "tool_result_payload": {},
                "tool_result_ok": False,
                "affected_paths": affected_paths,
                "workspace_changed": workspace_changed,
                "workspace_fingerprint": self.workspace.fingerprint(),
                "diff_summary": diff_summary,
            }
            
            # 记录处理笔记
            self.record_process_note_for_tool(name, self._last_tool_result_metadata)
            
            # 返回错误信息（让模型知道发生了什么）
            return f"error: tool {name} failed: {exc}"

    def repeated_tool_call(self, name, args):
        """检测是否是重复的工具调用（防止 agent 陷入无效循环）。

        为什么存在：
        Agent 很常见的一种坏循环是在没有新信息的情况下反复发起同一调用（例如
        连续两次调用 read_file 读同一个文件）。这里提前挡掉最简单的这种循环。

        检测策略：查看历史中最近的 2 次工具调用，如果两次的 name 和 args 与
        本次完全相同，则判定为重复调用。

        Args:
            name: 本次工具名称
            args: 本次工具参数

        Returns:
            bool: True 表示是重复调用，应该拒绝
        """
        # 从完整 history 中过滤出所有工具调用记录
        tool_events = [item for item in self.session["history"] if item["role"] == "tool"]
        # 少于 2 次工具调用，不可能重复
        if len(tool_events) < 2:
            return False
        # 只检查最近 2 次：如果这 2 次都与本次完全相同，则判定为重复
        recent = tool_events[-2:]
        return all(item["name"] == name and item["args"] == args for item in recent)

    @staticmethod
    def new_task_id():
        return "task_" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]

    @staticmethod
    def new_run_id():
        return "run_" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]

    def build_report(self, task_state):
        """构建运行报告（最终摘要）。

        为什么存在：
        trace 记录每一步的过程（每个工具调用、每次模型请求），而 report 是对
        整次运行的最终摘要，用于人工审计和指标统计。

        和 trace 的区别：
        - trace（.pico/runs/{run_id}/trace.jsonl）：逐事件时间线，"这一轮做了什么"
        - report（.pico/runs/{run_id}/report.json）：结果摘要，"这一轮的结论和指标"

        报告包含：
        - 运行 ID、任务 ID
        - 最终状态（completed/stopped）和停止原因
        - 最终答案文本
        - 工具调用次数、总尝试次数
        - checkpoint ID 和恢复状态
        - prompt 元数据（token 数、缓存命中等）
        - 长期记忆提升/拒绝记录
        - 脱敏后的环境变量摘要

        Args:
            task_state: 当前任务状态对象

        Returns:
            dict: 运行报告字典（调用方会通过 redact_artifact 脱敏后写入磁盘）
        """
        # report 是一次运行的最终摘要；
        # 和 trace 的区别在于，trace 关注过程，report 关注结果与关键指标。
        return {
            "run_id": task_state.run_id,
            "task_id": task_state.task_id,
            "parent_run_id": task_state.parent_run_id,
            "status": task_state.status,
            "stop_reason": task_state.stop_reason,
            "final_answer": task_state.final_answer,
            "tool_steps": task_state.tool_steps,
            "attempts": task_state.attempts,
            "checkpoint_id": task_state.checkpoint_id,
            "resume_status": task_state.resume_status,
            "task_state": task_state.to_dict(),
            "prompt_metadata": self.last_prompt_metadata,
            "durable_promotions": list(self.last_durable_promotions),
            "durable_rejections": list(self.last_durable_rejections),
            "durable_superseded": list(self.last_durable_superseded),
            "redacted_env": self.detected_secret_env_summary(),
        }

    def tool_example(self, name):
        return toolkit.tool_example(name)

    def validate_tool(self, name, args):
        """把通用工具校验和 runtime 级额外约束串起来。"""
        toolkit.validate_tool(self, name, args)
        if name == "delegate":
            if self.depth >= self.max_depth:
                raise ValueError("delegate depth exceeded")

    # ------------------------------------------------------------------
    # 工具旁路入口
    # 这些方法是给「不走 registry、直接调具体工具」的旧调用方留的兼容入口
    # （包括内部代码和测试）。工具内部统一返回 `ToolResult`，这里把 `.text`
    # 抽出来返回，让旁路调用方拿到的还是字符串，与历史接口一致。
    # 如果调用方想要结构化字段，直接走 `self.run_tool(name, args)` 或
    # 用 `self.tools[name]["run"](args)`，再读 `_last_tool_result_metadata`。
    # ------------------------------------------------------------------
    @staticmethod
    def _unwrap_tool_result(value):
        if isinstance(value, toolkit.ToolResult):
            return value.text
        return value

    def tool_list_files(self, args):
        return self._unwrap_tool_result(toolkit.tool_list_files(self, args))

    def tool_read_file(self, args):
        return self._unwrap_tool_result(toolkit.tool_read_file(self, args))

    def tool_search(self, args):
        return self._unwrap_tool_result(toolkit.tool_search(self, args))

    def tool_run_shell(self, args):
        return self._unwrap_tool_result(toolkit.tool_run_shell(self, args))

    def tool_write_file(self, args):
        return self._unwrap_tool_result(toolkit.tool_write_file(self, args))

    def tool_patch_file(self, args):
        return self._unwrap_tool_result(toolkit.tool_patch_file(self, args))

    def tool_delegate(self, args):
        return self._unwrap_tool_result(toolkit.tool_delegate(self, args))

    def _resolve_capability_policy(self, capabilities):
        """把 `approval_policy` 归一化为「针对这次工具调用」的最终决策。

        `approval_policy` 接受两种形态：
        - 旧形态：字符串 `"ask" / "auto" / "never"`，所有危险能力共享同一档；
        - 新形态：dict `{"read": "auto", "write": "ask", "exec": "ask", "net": "never"}`，
          按 capability 分档。

        归一化规则：取本次涉及到的「危险」capability（write/exec/net）对应的策略，
        合并方式按严格度选最严：
            never > ask > auto

        若工具只带 read，直接返回 "auto"，永不进入交互或拒绝路径。

        Args:
            capabilities: 本次工具声明的 capability 元组

        Returns:
            str: "auto" / "ask" / "never" 中的一种
        """
        risky_caps = [c for c in capabilities if c in toolkit.RISKY_CAPABILITIES]
        if not risky_caps:
            return "auto"

        policy = self.approval_policy
        if isinstance(policy, str):
            # 旧形态：所有危险能力共享同一档
            cap_policies = [policy] * len(risky_caps)
        elif isinstance(policy, dict):
            # 新形态：按 capability 查表，未指定的默认与旧 "ask" 等价
            cap_policies = [policy.get(c, "ask") for c in risky_caps]
        else:
            cap_policies = ["ask"]

        # 合并：选最严
        if "never" in cap_policies:
            return "never"
        if "ask" in cap_policies:
            return "ask"
        return "auto"

    def approve(self, name, args, capabilities=()):
        """根据审批策略决定是否允许执行危险工具。

        为什么存在：
        risky 工具（write_file, patch_file, run_shell, delegate）会对文件系统或外部环境造成
        不可逆的修改。这个函数是"最后一道闸"，确保在执行前经过适当的授权检查。

        四种决策路径（优先级从高到低）：
        1. read_only=True → 一律拒绝，保护只读模式下的工作区
        2. policy="auto"  → 自动通过，适合 CI 或无人值守场景
        3. policy="never" → 一律拒绝，适合纯分析场景
        4. policy="ask"   → 交互式询问用户，适合交互式 CLI

        `approval_policy` 现在也可以是 dict，按 capability 分档，例如
            {"write": "ask", "exec": "ask", "net": "never"}
        语义在 `_resolve_capability_policy()` 里归一化。

        Args:
            name: 工具名称（如 write_file, run_shell）
            args: 工具参数字典
            capabilities: 工具声明的能力元组，用于按 capability 分档审批

        Returns:
            bool: True 表示允许执行，False 表示拒绝
        """
        # 只读模式：无论策略如何，一律拒绝危险操作
        if self.read_only:
            return False
        effective = self._resolve_capability_policy(capabilities)
        # 自动审批：不弹出提示，直接通过（适合 CI 场景）
        if effective == "auto":
            return True
        # 永不审批：始终拒绝（适合只做分析、不允许修改的场景）
        if effective == "never":
            return False
        # 交互式询问：在终端显示工具调用详情，等待用户输入 y/yes
        try:
            cap_hint = ",".join(capabilities) if capabilities else "?"
            answer = input(f"approve {name} [caps:{cap_hint}] {json.dumps(args, ensure_ascii=True)}? [y/N] ")
        except EOFError:
            # 非交互式环境（如管道重定向）下 input() 抛出 EOFError，默认拒绝
            return False
        return answer.strip().lower() in {"y", "yes"}

    @staticmethod
    def parse(raw):
        """把模型原始输出解析成 runtime 可执行的动作或最终答案。

        为什么存在：
        模型输出首先是自然语言文本，而 runtime 需要的是结构化决策：
        "这是工具调用"还是"这是最终答案"。如果没有这层解析，后面的工具校验、
        审批和执行链路就没法可靠工作。

        输入 / 输出：
        - 输入：模型返回的原始文本 `raw`
        - 输出：`(kind, payload)`，其中：
          - kind="tool": payload={"name": "...", "args": {...}}
          - kind="final": payload="最终答案文本"
          - kind="retry": payload="错误提示信息"

        在 agent 链路里的位置：
        它位于 `model_client.complete()` 之后、`run_tool()` 之前，是模型输出
        进入平台控制流的第一道结构化关口。
        
        解析优先级（从高到低）：
        1. JSON 格式工具调用：<tool>{"name":"...", "args":{...}}</tool>
        2. XML 格式工具调用：<tool name="..." arg1="..." />
        3. 最终答案：<final>答案文本</final>
        4. 纯文本：直接视为最终答案
        5. 空响应：返回 retry
        
        Args:
            raw: 模型返回的原始文本
            
        Returns:
            tuple: (kind, payload)
                - kind: "tool" | "final" | "retry"
                - payload: 根据 kind 不同而不同
        """
        raw = str(raw)
        
        # ====================================================================
        # 解析策略1: 检测 <tool>...</tool> 包裹的 JSON 格式
        # ====================================================================
        # 优先检查是否包含 <tool> 标签，且出现在 <final> 之前（或没有 <final>）
        # 这种格式适合简短的工具调用，如：
        # <tool>{"name": "read_file", "args": {"path": "README.md"}}</tool>
        if "<tool>" in raw and ("<final>" not in raw or raw.find("<tool>") < raw.find("<final>")):
            # 提取 <tool> 和 </tool> 之间的内容
            body = Pico.extract(raw, "tool")
            
            try:
                # 尝试解析 JSON
                payload = json.loads(body)
            except Exception:
                # JSON 解析失败，返回 retry
                return "retry", Pico.retry_notice("model returned malformed tool JSON"), {"retry_reason": "json_parse_error"}
            
            # 校验 payload 必须是字典
            if not isinstance(payload, dict):
                return "retry", Pico.retry_notice("tool payload must be a JSON object"), {"retry_reason": "json_schema_error"}
            
            # 校验必须包含 name 字段
            if not str(payload.get("name", "")).strip():
                return "retry", Pico.retry_notice("tool payload is missing a tool name"), {"retry_reason": "json_schema_error"}
            
            # 处理 args 字段（允许缺失，默认为空字典）
            args = payload.get("args", {})
            if args is None:
                payload["args"] = {}
            elif not isinstance(args, dict):
                # args 必须是字典
                return "retry", Pico.retry_notice(), {"retry_reason": "json_schema_error"}
            
            return "tool", payload, {"format": "json"}
        
        # ====================================================================
        # 解析策略2: 检测 XML 属性格式的工具调用
        # ====================================================================
        # 支持更灵活的 XML 风格，如：
        # <tool name="write_file" path="test.txt">
        #   <content>文件内容</content>
        # </tool>
        if "<tool" in raw and ("<final>" not in raw or raw.find("<tool") < raw.find("<final>")):
            # 使用专门的 XML 解析器处理复杂结构
            payload = Pico.parse_xml_tool(raw)
            if payload is not None:
                return "tool", payload, {"format": "xml"}
            # XML 解析失败，返回 retry
            return "retry", Pico.retry_notice(), {"retry_reason": "xml_parse_error"}
        
        # ====================================================================
        # 解析策略3: 检测 <final> 标签的最终答案
        # ====================================================================
        if "<final>" in raw:
            # 提取 <final> 和 </final> 之间的内容
            final = Pico.extract(raw, "final").strip()
            if final:
                return "final", final, {"format": "tagged"}
            # 空的 <final> 标签，返回 retry
            return "retry", Pico.retry_notice("model returned an empty <final> answer"), {"retry_reason": "empty_final"}
        
        # ====================================================================
        # 解析策略4: 纯文本视为最终答案
        # ====================================================================
        raw = raw.strip()
        if raw:
            return "final", raw, {"format": "plain"}
        
        # ====================================================================
        # 解析策略5: 空响应返回 retry
        # ====================================================================
        return "retry", Pico.retry_notice("model returned an empty response"), {"retry_reason": "empty_response"}

    @staticmethod
    def retry_notice(problem=None):
        prefix = "Runtime notice"
        if problem:
            prefix += f": {problem}"
        else:
            prefix += ": model returned malformed tool output"
        return (
            f"{prefix}. Reply with a valid <tool> call or a non-empty <final> answer. "
            'For multi-line files, prefer <tool name="write_file" path="file.py"><content>...</content></tool>.'
        )

    @staticmethod
    def parse_xml_tool(raw):
        match = re.search(r"<tool(?P<attrs>[^>]*)>(?P<body>.*?)</tool>", raw, re.S)
        if not match:
            return None
        attrs = Pico.parse_attrs(match.group("attrs"))
        name = str(attrs.pop("name", "")).strip()
        if not name:
            return None

        body = match.group("body")
        args = dict(attrs)
        for key in ("content", "old_text", "new_text", "command", "task", "pattern", "path"):
            if f"<{key}>" in body:
                args[key] = Pico.extract_raw(body, key)

        body_text = body.strip("\n")
        if name == "write_file" and "content" not in args and body_text:
            args["content"] = body_text
        if name == "delegate" and "task" not in args and body_text:
            args["task"] = body_text.strip()
        return {"name": name, "args": args}

    @staticmethod
    def parse_attrs(text):
        attrs = {}
        for match in re.finditer(r"""([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:"([^"]*)"|'([^']*)')""", text):
            attrs[match.group(1)] = match.group(2) if match.group(2) is not None else match.group(3)
        return attrs

    @staticmethod
    def extract(text, tag):
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"
        start = text.find(start_tag)
        if start == -1:
            return text
        start += len(start_tag)
        end = text.find(end_tag, start)
        if end == -1:
            return text[start:].strip()
        return text[start:end].strip()

    @staticmethod
    def extract_raw(text, tag):
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"
        start = text.find(start_tag)
        if start == -1:
            return text
        start += len(start_tag)
        end = text.find(end_tag, start)
        if end == -1:
            return text[start:]
        return text[start:end]

    def reset(self):
        """重置 session 状态（清除对话历史和记忆）。

        使用场景：
        - 用户想开始一个全新的对话，不希望之前的上下文影响新任务
        - 测试场景需要干净的初始状态

        注意：这个操作是不可逆的，清除后无法恢复之前的历史。
        """
        self.session["history"] = []
        self.session["memory"].clear()
        self.session["memory"].update(memorylib.default_memory_state())
        # 重新初始化 LayeredMemory 对象（因为内存状态已被清空）
        self.memory = memorylib.LayeredMemory(self.session["memory"], workspace_root=self.root)
        # 持久化到磁盘，确保重置后的状态被保存
        self.session_store.save(self.session)

    def path(self, raw_path):
        """将相对路径或绝对路径解析为安全的绝对路径。

        核心职责是路径沙箱化：所有文件类工具都被锚定在 workspace root 之下，
        防止 agent 访问或修改工作区之外的文件。

        安全防护：
        - 防止 "../../../etc/passwd" 类型的路径遍历攻击
        - 防止符号链接（symlink）指向工作区外部的文件
        - 使用 resolve() + commonpath() 双重验证

        Args:
            raw_path: 原始路径字符串（可以是相对路径或绝对路径）

        Returns:
            Path: 解析后的绝对路径

        Raises:
            ValueError: 如果解析后的路径超出 workspace root 范围
        """
        path = Path(raw_path)
        # 相对路径：基于 workspace root 解析
        path = path if path.is_absolute() else self.root / path
        # resolve() 会展开所有 ".." 和符号链接，得到真实的绝对路径
        resolved = path.resolve()
        # 所有文件类工具都被锚定在 workspace root 之下。
        # 这样既能防住 "../" 逃逸，也能防住符号链接解析后跳出仓库。
        if os.path.commonpath([str(self.root), str(resolved)]) != str(self.root):
            raise ValueError(f"path escapes workspace: {raw_path}")
        return resolved


MiniAgent = Pico
