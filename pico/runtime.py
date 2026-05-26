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
        
        为新创建的 session 或从旧版本恢复的 session 补充缺失的字段，
        保证后续代码可以安全访问这些嵌套结构。
        """
        self.session.setdefault("history", [])
        self.session.setdefault("memory", memorylib.default_memory_state())
        
        # 确保 checkpoints 结构
        checkpoints = self.session.setdefault("checkpoints", {})
        if not isinstance(checkpoints, dict):
            checkpoints = {}
            self.session["checkpoints"] = checkpoints
        checkpoints.setdefault("current_id", "")
        checkpoints.setdefault("items", {})
        
        # 确保 runtime_identity 结构
        runtime_identity = self.session.setdefault("runtime_identity", {})
        if not isinstance(runtime_identity, dict):
            self.session["runtime_identity"] = {}
        
        # 确保 resume_state 结构
        resume_state = self.session.setdefault("resume_state", {})
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
        
        Returns:
            dict: 恢复状态字典
        """
        previous_resume_state = dict(self.session.get("resume_state", {}) or {})
        invalidated = self.invalidate_stale_memory()
        checkpoint = self.current_checkpoint()
        status = CHECKPOINT_NONE_STATUS
        stale_paths = list(invalidated)
        mismatch_fields = []
        if checkpoint:
            if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
                status = CHECKPOINT_SCHEMA_MISMATCH_STATUS
            else:
                for item in checkpoint.get("key_files", []):
                    path = str(item.get("path", "")).strip()
                    if not path:
                        continue
                    expected = item.get("freshness")
                    current = memorylib.file_freshness(path, self.root)
                    if expected != current and path not in stale_paths:
                        stale_paths.append(path)
                saved_identity = dict(checkpoint.get("runtime_identity", {}) or self.session.get("runtime_identity", {}) or {})
                current_identity = self.current_runtime_identity()
                identity_keys = (
                    "cwd",
                    "model",
                    "model_client",
                    "approval_policy",
                    "read_only",
                    "max_steps",
                    "max_new_tokens",
                    "feature_flags",
                    "shell_env_allowlist",
                    "workspace_fingerprint",
                    "tool_signature",
                )
                for key in identity_keys:
                    if key not in saved_identity:
                        continue
                    if saved_identity.get(key) != current_identity.get(key):
                        mismatch_fields.append(key)
                mismatch_fields.sort()
                if stale_paths:
                    status = CHECKPOINT_PARTIAL_STALE_STATUS
                elif mismatch_fields:
                    status = CHECKPOINT_WORKSPACE_MISMATCH_STATUS
                else:
                    status = CHECKPOINT_FULL_VALID_STATUS

        resume_state = {
            "status": status,
            "stale_paths": stale_paths,
            "runtime_identity_mismatch_fields": mismatch_fields,
            "stale_summary_invalidations": max(
                len(invalidated),
                int(previous_resume_state.get("stale_summary_invalidations", 0))
                if status == CHECKPOINT_PARTIAL_STALE_STATUS
                else 0,
            ),
        }
        self.session["resume_state"] = resume_state
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
                    "description": tool["description"],
                }
            )
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def build_prefix(self):
        """构建 Prompt Prefix。
        
        Returns:
            PromptPrefix: Prompt Prefix 对象
        """
        tool_lines = []
        for name, tool in self.tools.items():
            fields = ", ".join(f"{key}: {value}" for key, value in tool["schema"].items())
            risk = "approval required" if tool["risky"] else "safe"
            tool_lines.append(f"- {name}({fields}) [{risk}] {tool['description']}")
        tool_text = "\n".join(tool_lines)
        examples = "\n".join(
            [
                '<tool>{"name":"list_files","args":{"path":"."}}</tool>',
                '<tool>{"name":"read_file","args":{"path":"README.md","start":1,"end":80}}</tool>',
                '<tool name="write_file" path="binary_search.py"><content>def binary_search(nums, target):\n    return -1\n</content></tool>',
                '<tool name="patch_file" path="binary_search.py"><old_text>return -1</old_text><new_text>return mid</new_text></tool>',
                '<tool>{"name":"run_shell","args":{"command":"uv run --with pytest python -m pytest -q","timeout":20}}</tool>',
                "<final>Done.</final>",
            ]
        )
        # prefix 可以理解成 agent 的“工作手册”：
        # 它是谁、工具怎么调用、当前仓库是什么状态，都写在这里。
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
        ).strip()
        return PromptPrefix(
            text=text,
            hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            workspace_fingerprint=self.workspace.fingerprint(),
            tool_signature=self.tool_signature(),
            built_at=now(),
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
        
        Args:
            task_state: 当前任务状态
            user_message: 用户输入的消息
            trigger: 触发 checkpoint 的原因
            
        Returns:
            dict: checkpoint 数据
        """
        state = self.checkpoint_state()
        current = self.current_checkpoint()
        checkpoint_id = "ckpt_" + uuid.uuid4().hex[:8]
        key_files = []
        freshness = {}
        for path in self.memory.to_dict()["working"]["recent_files"]:
            file_freshness = memorylib.file_freshness(path, self.root)
            freshness[path] = file_freshness
            key_files.append({"path": path, "freshness": file_freshness})
        checkpoint = {
            "checkpoint_id": checkpoint_id,
            "parent_checkpoint_id": current.get("checkpoint_id", "") if current else "",
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "created_at": now(),
            "current_goal": str(user_message),
            "completed": [task_state.final_answer] if task_state.final_answer else [],
            "excluded": [],
            "current_blocker": "" if str(task_state.stop_reason or "") in ("", "final_answer_returned") else str(task_state.stop_reason),
            "next_step": self.infer_next_step(task_state),
            "key_files": key_files,
            "freshness": freshness,
            "summary": f"{trigger}: {clip(str(user_message), 120)}",
            "runtime_identity": self.current_runtime_identity(),
        }
        state["items"][checkpoint_id] = checkpoint
        state["current_id"] = checkpoint_id
        task_state.checkpoint_id = checkpoint_id
        self.session["runtime_identity"] = checkpoint["runtime_identity"]
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
        """执行一次完整的 agent 回合，直到产出最终答案或命中停止条件。

        为什么存在：
        `ask()` 是整个 runtime 的总调度器。它把“用户提一个请求”扩展成一条
        可持续推进的控制循环：记录会话、组 prompt、调用模型、执行工具、
        写 trace/report、更新状态，直到模型给出最终答案或系统主动停下。

        输入 / 输出：
        - 输入：`user_message`，即用户这一次的任务描述
        - 输出：字符串形式的最终回答；如果中途达到步数上限或重试上限，
          返回的是一条停止原因说明

        在 agent 链路里的位置：
        它是 CLI 和底层工具/模型之间的核心桥梁。CLI 收到用户输入后基本只做
        一件事：调用 `agent.ask()`。而 `ask()` 内部再去驱动 `ContextManager`
        组 prompt、`model_client.complete()` 调模型、`run_tool()` 执行动作。
        如果新人想理解 pico 是怎么“从一句话跑成一个 agent 流程”的，
        这里就是最关键的入口。
        """
        run_started_at = time.monotonic()
        self.memory.set_task_summary(user_message)
        self.record({"role": "user", "content": user_message, "created_at": now()})

        task_state = TaskState.create(run_id=self.new_run_id(), task_id=self.new_task_id(), user_request=user_message)
        task_state.resume_status = self.resume_state.get("status", CHECKPOINT_NONE_STATUS)
        self.current_task_state = task_state
        self.current_run_dir = self.run_store.start_run(task_state)
        self.emit_trace(
            task_state,
            "run_started",
            {
                "task_id": task_state.task_id,
                "user_request": clip(user_message, 300),
            },
        )

        tool_steps = 0
        attempts = 0
        max_attempts = max(self.max_steps * 3, self.max_steps + 4)

        # 这是 agent 的主循环，可以按“感知 -> 决策 -> 行动 -> 记录”来理解：
        # 1. 感知：重新组 prompt，把当前状态整理给模型看
        # 2. 决策：让模型返回一个工具调用，或一个最终答案
        # 3. 行动：如果是工具调用，就执行工具
        # 4. 记录：把结果写回 history / task_state / trace / memory
        # 然后进入下一轮，直到停机条件满足
        while tool_steps < self.max_steps and attempts < max_attempts:
            attempts += 1
            task_state.record_attempt()
            self.run_store.write_task_state(task_state)
            prompt_started_at = time.monotonic()
            prompt, prompt_metadata = self._build_prompt_and_metadata(user_message)
            self.emit_trace(
                task_state,
                "prompt_built",
                {
                    "prompt_metadata": prompt_metadata,
                    "duration_ms": int((time.monotonic() - prompt_started_at) * 1000),
                },
            )
            if prompt_metadata.get("resume_status") == CHECKPOINT_PARTIAL_STALE_STATUS:
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
            elif prompt_metadata.get("resume_status") == CHECKPOINT_WORKSPACE_MISMATCH_STATUS:
                self.emit_trace(
                    task_state,
                    "runtime_identity_mismatch",
                    {
                        "fields": list(prompt_metadata.get("runtime_identity_mismatch_fields", [])),
                    },
                )
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
            if prompt_metadata.get("budget_reductions"):
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
            self.emit_trace(
                task_state,
                "model_requested",
                {
                    "attempts": task_state.attempts,
                    "tool_steps": task_state.tool_steps,
                    "prompt_cache_key": prompt_metadata.get("prompt_cache_key"),
                },
            )
            prompt_cache_key = None
            prompt_cache_retention = None
            if getattr(self.model_client, "supports_prompt_cache", False):
                # 只有后端明确支持时，才把稳定前缀的 hash 作为 cache key 发出去。
                prompt_cache_key = prompt_metadata.get("prompt_cache_key")
                prompt_cache_retention = "in_memory"
            model_started_at = time.monotonic()
            raw = self.model_client.complete(
                prompt,
                self.max_new_tokens,
                prompt_cache_key=prompt_cache_key,
                prompt_cache_retention=prompt_cache_retention,
            )
            completion_metadata = dict(getattr(self.model_client, "last_completion_metadata", {}) or {})
            if completion_metadata:
                # 把后端返回的 usage/cache 统计并回 prompt_metadata，
                # 方便统一写入 report 和 trace。
                prompt_metadata.update(completion_metadata)
            self.last_completion_metadata = completion_metadata
            self.last_prompt_metadata = prompt_metadata
            kind, payload = self.parse(raw)
            self.emit_trace(
                task_state,
                "model_parsed",
                {
                    "kind": kind,
                    "completion_metadata": completion_metadata,
                    "duration_ms": int((time.monotonic() - model_started_at) * 1000),
                },
            )

            if kind == "tool":
                tool_steps += 1
                name = payload.get("name", "")
                args = payload.get("args", {})
                task_state.record_tool(name)
                tool_started_at = time.monotonic()
                result = self.run_tool(name, args)
                self.record(
                    {
                        "role": "tool",
                        "name": name,
                        "args": args,
                        "content": result,
                        "created_at": now(),
                    }
                )
                self.run_store.write_task_state(task_state)
                self.emit_trace(
                    task_state,
                    "tool_executed",
                    {
                        "name": name,
                        "args": args,
                        "result": clip(result, 500),
                        "duration_ms": int((time.monotonic() - tool_started_at) * 1000),
                        **dict(self._last_tool_result_metadata or {}),
                    },
                )
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
            elif kind == "final":
                task_state.final_answer = payload
                self.record({"role": "final", "content": payload, "created_at": now()})
                self.emit_trace(
                    task_state,
                    "checkpoint_created",
                    {
                        "checkpoint_id": checkpoint["checkpoint_id"],
                        "trigger": "tool_executed",
                    },
                )
                continue

            if kind == "retry":
                self.record({"role": "assistant", "content": payload, "created_at": now()})
                self.run_store.write_task_state(task_state)
                continue

            final = (payload or raw).strip()
            self.record({"role": "assistant", "content": final, "created_at": now()})
            task_state.finish_success(final)
            self.promote_durable_memory(user_message, final)
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
            self.run_store.write_report(task_state, self.redact_artifact(self.build_report(task_state)))
            return final

        if attempts >= max_attempts and tool_steps < self.max_steps:
            final = "Stopped after too many malformed model responses without a valid tool call or final answer."
            task_state.stop_retry_limit(final)
        else:
            final = "Stopped after reaching the step limit without a final answer."
            task_state.stop_step_limit(final)
        self.record({"role": "assistant", "content": final, "created_at": now()})
        self.promote_durable_memory(user_message, final)
        self.run_store.write_task_state(task_state)
        checkpoint = self.create_checkpoint(task_state, user_message, trigger=task_state.stop_reason or "run_stopped")
        self.emit_trace(
            task_state,
            "checkpoint_created",
            {
                "checkpoint_id": checkpoint["checkpoint_id"],
                "trigger": task_state.stop_reason or "run_stopped",
            },
        )
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
        self.run_store.write_report(task_state, self.redact_artifact(self.build_report(task_state)))
        return final

    def run_tool(self, name, args):
        """执行一次工具调用，并在执行前后套上完整护栏。

        为什么存在：
        在 agent 系统里，真正危险的不是“模型会不会想调用工具”，而是
        “平台有没有在执行前把边界守住”。这个函数就是工具层的总闸口：
        所有工具调用都必须先经过它，不能让模型直接碰到底层函数。

        输入 / 输出：
        - 输入：工具名 `name`，参数字典 `args`
        - 输出：字符串结果。无论是成功结果还是错误信息，都会统一返回文本，
          这样模型下一轮都能继续消费这份反馈。

        在 agent 链路里的位置：
        它位于 `ask()` 的“模型决定要调用工具”之后，是控制循环里真正把模型
        意图落到外部世界的一步。因此这里串起了几乎所有安全与可控设计：
        工具是否存在、参数是否合法、是否重复、是否需要审批、执行结果是否裁剪、
        是否需要回写记忆。
        """
        # 工具执行不是“直接调函数”，而是一条带护栏的流水线：
        # 工具是否存在 -> 参数是否合法 -> 是否重复调用 -> 是否通过审批
        # -> 真正执行 -> 更新记忆。
        tool = self.tools.get(name)
        if tool is None:
            self._last_tool_result_metadata = {
                "tool_status": "rejected",
                "tool_error_code": "unknown_tool",
                "security_event_type": "",
                "risk_level": "high",
                "read_only": False,
                "affected_paths": [],
                "workspace_changed": False,
                "diff_summary": [],
            }
            return f"error: unknown tool '{name}'"
        try:
            self.validate_tool(name, args)
        except Exception as exc:
            example = self.tool_example(name)
            message = f"error: invalid arguments for {name}: {exc}"
            if example:
                message += f"\nexample: {example}"
            security_event_type = "path_escape" if "path escapes workspace" in str(exc) else ""
            self._last_tool_result_metadata = {
                "tool_status": "rejected",
                "tool_error_code": "invalid_arguments",
                "security_event_type": security_event_type,
                "risk_level": "high" if tool["risky"] else "low",
                "read_only": not tool["risky"],
                "affected_paths": [],
                "workspace_changed": False,
                "diff_summary": [],
            }
            return message
        if self.repeated_tool_call(name, args):
            self._last_tool_result_metadata = {
                "tool_status": "rejected",
                "tool_error_code": "repeated_identical_call",
                "security_event_type": "",
                "risk_level": "high" if tool["risky"] else "low",
                "read_only": not tool["risky"],
                "affected_paths": [],
                "workspace_changed": False,
                "diff_summary": [],
            }
            return f"error: repeated identical tool call for {name}; choose a different tool or return a final answer"
        if tool["risky"] and not self.approve(name, args):
            self._last_tool_result_metadata = {
                "tool_status": "rejected",
                "tool_error_code": "approval_denied",
                "security_event_type": "read_only_block" if self.read_only else "approval_denied",
                "risk_level": "high",
                "read_only": False,
                "affected_paths": [],
                "workspace_changed": False,
                "diff_summary": [],
            }
            return f"error: approval denied for {name}"
        before_snapshot = self.capture_workspace_snapshot() if tool["risky"] else {}
        after_snapshot = before_snapshot
        try:
            result = clip(tool["run"](args))
            after_snapshot = self.capture_workspace_snapshot() if tool["risky"] else before_snapshot
            affected_paths, diff_summary = self.diff_workspace_snapshots(before_snapshot, after_snapshot)
            workspace_changed = bool(affected_paths)
            tool_status = "ok"
            tool_error_code = ""
            if name == "run_shell":
                match = re.search(r"exit_code:\s*(-?\d+)", result)
                exit_code = int(match.group(1)) if match else 0
                if exit_code != 0 and workspace_changed:
                    tool_status = "partial_success"
                    tool_error_code = "tool_partial_success"
                elif exit_code != 0:
                    tool_status = "error"
                    tool_error_code = "tool_failed"
            self.update_memory_after_tool(name, args, result)
            self._last_tool_result_metadata = {
                "tool_status": tool_status,
                "tool_error_code": tool_error_code,
                "security_event_type": "",
                "risk_level": "high" if tool["risky"] else "low",
                "read_only": not tool["risky"],
                "affected_paths": affected_paths,
                "workspace_changed": workspace_changed,
                "workspace_fingerprint": self.workspace.fingerprint(),
                "diff_summary": diff_summary,
            }
            self.record_process_note_for_tool(name, self._last_tool_result_metadata)
            return result
        except Exception as exc:
            after_snapshot = self.capture_workspace_snapshot() if tool["risky"] else before_snapshot
            affected_paths, diff_summary = self.diff_workspace_snapshots(before_snapshot, after_snapshot)
            workspace_changed = bool(affected_paths)
            security_event_type = "path_escape" if "path escapes workspace" in str(exc) else ""
            self._last_tool_result_metadata = {
                "tool_status": "partial_success" if workspace_changed else "error",
                "tool_error_code": "tool_partial_success" if workspace_changed else "tool_failed",
                "security_event_type": security_event_type,
                "risk_level": "high" if tool["risky"] else "low",
                "read_only": not tool["risky"],
                "affected_paths": affected_paths,
                "workspace_changed": workspace_changed,
                "workspace_fingerprint": self.workspace.fingerprint(),
                "diff_summary": diff_summary,
            }
            self.record_process_note_for_tool(name, self._last_tool_result_metadata)
            return f"error: tool {name} failed: {exc}"

    def repeated_tool_call(self, name, args):
        # agent 很常见的一种坏循环，是在没有新信息的情况下反复发起同一调用。
        # 这里提前挡掉最简单的这种循环。
        tool_events = [item for item in self.session["history"] if item["role"] == "tool"]
        if len(tool_events) < 2:
            return False
        recent = tool_events[-2:]
        return all(item["name"] == name and item["args"] == args for item in recent)

    @staticmethod
    def new_task_id():
        return "task_" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]

    @staticmethod
    def new_run_id():
        return "run_" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]

    def build_report(self, task_state):
        # report 是一次运行的最终摘要；
        # 和 trace 的区别在于，trace 关注过程，report 关注结果与关键指标。
        return {
            "run_id": task_state.run_id,
            "task_id": task_state.task_id,
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

    def tool_list_files(self, args):
        return toolkit.tool_list_files(self, args)

    def tool_read_file(self, args):
        return toolkit.tool_read_file(self, args)

    def tool_search(self, args):
        return toolkit.tool_search(self, args)

    def tool_run_shell(self, args):
        return toolkit.tool_run_shell(self, args)

    def tool_write_file(self, args):
        return toolkit.tool_write_file(self, args)

    def tool_patch_file(self, args):
        return toolkit.tool_patch_file(self, args)

    def tool_delegate(self, args):
        return toolkit.tool_delegate(self, args)

    def approve(self, name, args):
        if self.read_only:
            return False
        if self.approval_policy == "auto":
            return True
        if self.approval_policy == "never":
            return False
        try:
            answer = input(f"approve {name} {json.dumps(args, ensure_ascii=True)}? [y/N] ")
        except EOFError:
            return False
        return answer.strip().lower() in {"y", "yes"}

    @staticmethod
    def parse(raw):
        """把模型原始输出解析成 runtime 可执行的动作或最终答案。

        为什么存在：
        模型输出首先是自然语言文本，而 runtime 需要的是结构化决策：
        “这是工具调用”还是“这是最终答案”。如果没有这层解析，后面的工具校验、
        审批和执行链路就没法可靠工作。

        输入 / 输出：
        - 输入：模型返回的原始文本 `raw`
        - 输出：`(kind, payload)`，其中 `kind` 可能是 `tool`、`final`、`retry`

        在 agent 链路里的位置：
        它位于 `model_client.complete()` 之后、`run_tool()` 之前，是模型输出
        进入平台控制流的第一道结构化关口。
        """
        raw = str(raw)
        # 这里支持两种工具格式：
        # 1. <tool>...</tool> 里包 JSON，适合简短调用
        # 2. XML 风格属性/子标签，适合写文件这类多行内容
        if "<tool>" in raw and ("<final>" not in raw or raw.find("<tool>") < raw.find("<final>")):
            body = Pico.extract(raw, "tool")
            try:
                payload = json.loads(body)
            except Exception:
                return "retry", Pico.retry_notice("model returned malformed tool JSON")
            if not isinstance(payload, dict):
                return "retry", Pico.retry_notice("tool payload must be a JSON object")
            if not str(payload.get("name", "")).strip():
                return "retry", Pico.retry_notice("tool payload is missing a tool name")
            args = payload.get("args", {})
            if args is None:
                payload["args"] = {}
            elif not isinstance(args, dict):
                return "retry", Pico.retry_notice()
            return "tool", payload
        if "<tool" in raw and ("<final>" not in raw or raw.find("<tool") < raw.find("<final>")):
            payload = Pico.parse_xml_tool(raw)
            if payload is not None:
                return "tool", payload
            return "retry", Pico.retry_notice()
        if "<final>" in raw:
            final = Pico.extract(raw, "final").strip()
            if final:
                return "final", final
            return "retry", Pico.retry_notice("model returned an empty <final> answer")
        raw = raw.strip()
        if raw:
            return "final", raw
        return "retry", Pico.retry_notice("model returned an empty response")

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
        self.session["history"] = []
        self.session["memory"].clear()
        self.session["memory"].update(memorylib.default_memory_state())
        self.memory = memorylib.LayeredMemory(self.session["memory"], workspace_root=self.root)
        self.session_store.save(self.session)

    def path(self, raw_path):
        path = Path(raw_path)
        path = path if path.is_absolute() else self.root / path
        resolved = path.resolve()
        # 所有文件类工具都被锚定在 workspace root 之下。
        # 这样既能防住 "../" 逃逸，也能防住符号链接解析后跳出仓库。
        if os.path.commonpath([str(self.root), str(resolved)]) != str(self.root):
            raise ValueError(f"path escapes workspace: {raw_path}")
        return resolved


MiniAgent = Pico

"""
 agent 的核心调度器，执行流程：
┌─────────────────────────────────────────┐
│ 1. 初始化                                │
│    - 记录用户消息到 history              │
│    - 创建 TaskState                     │
│    - 启动 run (trace/report)            │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│ 2. 主循环 (while tool_steps < max)      │
│                                         │
│   ┌─────────────────────────────────┐   │
│   │ A. 感知: 构建 prompt            │   │
│   │    - _build_prompt_and_metadata │   │
│   │    - emit_trace(prompt_built)   │   │
│   │    - 检查恢复状态，必要时创建    │   │
│   │      checkpoint                 │   │
│   └──────────────┬──────────────────┘   │
│                  │                       │
│                  ▼                       │
│   ┌─────────────────────────────────┐   │
│   │ B. 决策: 调用模型               │   │
│   │    - model_client.complete()    │   │
│   │    - parse(raw) → kind/payload  │   │
│   │    - emit_trace(model_parsed)   │   │
│   └──────────────┬──────────────────┘   │
│                  │                       │
│        ┌─────────┴─────────┐            │
│        │                   │            │
│   kind=tool          kind=final/retry   │
│        │                   │            │
│        ▼                   ▼            │
│   ┌──────────────┐  ┌──────────────┐   │
│   │ C. 行动:     │  │ 记录答案     │   │
│   │ run_tool()   │  │ 完成任务     │   │
│   │ 执行工具     │  │ promote_     │   │
│   │ 更新 memory  │  │ durable_     │   │
│   │ 创建         │  │ memory()     │   │
│   │ checkpoint   │  │ 创建         │   │
│   └──────────────┘  │ checkpoint   │   │
│                     └──────────────┘   │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│ 3. 终止处理                              │
│    - 达到步数上限或重试上限              │
│    - 写入最终 report                    │
│    - 返回最终答案                       │
└─────────────────────────────────────────┘


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
│    - read_only?                      │
│    - approval_policy (auto/ask/never)│
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




"""