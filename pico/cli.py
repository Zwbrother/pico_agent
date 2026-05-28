"""命令行入口。

这个模块负责把"用户怎么启动 pico"翻译成 runtime 能理解的对象：
解析参数、挑模型后端、构建工作区快照、恢复或新建 session，
最后进入 one-shot 或交互式循环。

## 核心职责
1. 解析命令行参数（argparse）
2. 装配 Agent 实例（build_agent）
3. 打印欢迎界面（build_welcome）
4. 进入交互循环或 one-shot 模式（main）

## 调用链路概览
```
main() 
  ├─> build_arg_parser().parse_args()     # 步骤1: 解析 CLI 参数
  ├─> build_agent(args)                    # 步骤2: 装配 Agent
  │    ├─> WorkspaceContext.build()        #   2.1: 扫描工作区
  │    ├─> load_project_env()              #   2.2: 加载 .env 配置
  │    ├─> _configured_secret_names()      #   2.3: 确定敏感变量名单
  │    ├─> SessionStore()                  #   2.4: 创建 session 存储
  │    ├─> _build_model_client()           #   2.5: 构建模型客户端
  │    │    └─> _effective_model()         #       确定模型名称
  │    └─> Pico.__init__()                 #   2.6: 初始化运行时
  │         ├─> LayeredMemory()            #       初始化记忆系统
  │         ├─> build_tools()              #       注册工具集
  │         ├─> build_prefix()             #       构建 prompt prefix
  │         ├─> ContextManager()           #       初始化上下文管理器
  │         └─> evaluate_resume_state()    #       评估恢复状态
  ├─> build_welcome()                      # 步骤3: 生成并打印欢迎界面
  └─> if args.prompt: agent.ask()          # 步骤4a: one-shot 模式
      else: while True: input()->ask()     # 步骤4b: REPL 交互模式
```
"""

import argparse
import os
import shutil
import sys
import textwrap

from .config import load_project_env, provider_env
from .models import AnthropicCompatibleModelClient, OllamaModelClient, OpenAICompatibleModelClient
from .runtime import Pico, SessionStore
from .workspace import WorkspaceContext, middle

# ============================================================================
# 常量定义
# ============================================================================

# 默认需要脱敏的环境变量名（包含 API_KEY、TOKEN、SECRET 等敏感信息）
DEFAULT_SECRET_ENV_NAMES = (
    "PICO_OPENAI_API_KEY",
    "OPENAI_API_KEY",
    "OPENAI_API_TOKEN",
    "PICO_ANTHROPIC_API_KEY",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "PICO_DEEPSEEK_API_KEY",
    "DEEPSEEK_API_KEY",
    "PICO_RIGHT_CODES_API_KEY",
    "RIGHT_CODES_API_KEY",
    "GITHUB_PAT",
    "GH_PAT",
)

# 欢迎界面的 ASCII art
WELCOME_ART = (
    "        /\\___/\\\\",
    "       (  o o  )",
    "       /   ^   \\\\",
    "      /|       |\\\\",
)
WELCOME_NAME = "pico"
WELCOME_SUBTITLE = "local coding agent"
WELCOME_STATUS = "calm shell, ready for work"

# REPL 模式的帮助信息
HELP_DETAILS = textwrap.dedent(
    """\
    Commands:
    /help    Show this help message.
    /memory  Show the agent's distilled working memory.
    /session Show the path to the saved session file.
    /reset   Clear the current session history and memory.
    /exit    Exit the agent.
    """
).strip()


# 各 provider 的默认配置
DEFAULT_OLLAMA_MODEL = "qwen3.5:4b"
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_OPENAI_MODEL = "gpt-5.4"
DEFAULT_OPENAI_BASE_URL = "https://www.right.codes/codex/v1"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"
DEFAULT_ANTHROPIC_BASE_URL = "https://www.right.codes/claude/v1"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-pro"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com/anthropic"

# 敏感环境变量名的环境变量配置（支持向后兼容）
LEGACY_SECRET_ENV_NAMES_VAR = "MINI_CODING_AGENT_SECRET_ENV_NAMES"
SECRET_ENV_NAMES_VAR = "PICO_SECRET_ENV_NAMES"


# ============================================================================
# 辅助函数
# ============================================================================

def _effective_model(args, provider):
    """确定最终使用的模型名称。
    
    模型选择优先级（从高到低）：
    1. 用户显式传入 --model 参数
    2. provider 对应的环境变量（如 PICO_OPENAI_MODEL）
    3. 代码里的默认值（如 DEFAULT_OPENAI_MODEL）
    
    Args:
        args: argparse 解析后的参数对象
        provider: 模型提供商名称（ollama/openai/anthropic/deepseek）
        
    Returns:
        str: 最终确定的模型名称
    """
    # 优先级1: 检查用户是否显式指定了 --model
    explicit_model = getattr(args, "model", None)
    if explicit_model:
        return explicit_model
    
    # 优先级2 & 3: 根据 provider 读取环境变量或使用默认值
    if provider == "openai":
        model = provider_env("PICO_OPENAI_MODEL", ("OPENAI_MODEL",))
        if model:
            return model
        return DEFAULT_OPENAI_MODEL
    if provider == "anthropic":
        model = provider_env("PICO_ANTHROPIC_MODEL", ("ANTHROPIC_MODEL",))
        if model:
            return model
        return DEFAULT_ANTHROPIC_MODEL
    if provider == "deepseek":
        model = provider_env("PICO_DEEPSEEK_MODEL", ("DEEPSEEK_MODEL",))
        if model:
            return model
        return DEFAULT_DEEPSEEK_MODEL
    
    # Ollama 使用默认模型
    return DEFAULT_OLLAMA_MODEL


def _configured_secret_names(args):
    """合并所有需要脱敏的环境变量名。
    
    从三个来源收集敏感变量名：
    1. 默认的敏感变量名列表（DEFAULT_SECRET_ENV_NAMES）
    2. 用户通过 --secret-env-name 参数指定的额外变量名
    3. 环境变量 PICO_SECRET_ENV_NAMES 中配置的变量名（逗号分隔）
    
    Args:
        args: argparse 解析后的参数对象
        
    Returns:
        list[str]: 排序后的敏感环境变量名列表（大写）
    """
    # 从默认列表开始
    configured_secret_names = set(DEFAULT_SECRET_ENV_NAMES)
    
    # 添加用户通过 CLI 参数指定的变量名
    configured_secret_names.update(str(name).upper() for name in args.secret_env_names)
    
    # 从环境变量中读取额外的敏感变量名（支持新旧两种环境变量名）
    extra_names = os.environ.get(SECRET_ENV_NAMES_VAR, "")
    if not extra_names.strip():
        extra_names = os.environ.get(LEGACY_SECRET_ENV_NAMES_VAR, "")
    
    # 解析逗号分隔的变量名列表
    if extra_names.strip():
        configured_secret_names.update(
            item.strip().upper()
            for item in extra_names.split(",")
            if item.strip()
        )
    
    return sorted(configured_secret_names)


def _build_model_client(args):
    """根据 CLI 参数构建模型客户端实例。
    
    这个函数是 CLI 层到模型层的适配器，负责：
    1. 确定 provider 类型
    2. 解析模型名称、base_url、api_key 等配置
    3. 创建对应的模型客户端实例
    
    注意：CLI 只负责传递配置，真正的 HTTP 协议、提示词格式、缓存支持等细节都封装在 models.py 的各个 Client 类中。
    
    Args:
        args: argparse 解析后的参数对象
        
    Returns:
        ModelClient: 模型客户端实例（Ollama/OpenAI/Anthropic 兼容）
        
    Raises:
        ValueError: 如果 provider 不支持
    """
    provider = getattr(args, "provider", "deepseek") ##先访问args.provider 属性,如果不存在，默认使用deepseek
    
    # =========================================================================
    # OpenAI 兼容的 API（包括 Right Codes Codex）
    # =========================================================================
    if provider == "openai":
        model = _effective_model(args, provider)
        # base_url 优先级：CLI 参数 > 环境变量 > 默认值
        base_url = getattr(args, "base_url", None) or provider_env("PICO_OPENAI_API_BASE", ("OPENAI_API_BASE",), DEFAULT_OPENAI_BASE_URL)
        api_key = provider_env("PICO_OPENAI_API_KEY", ("OPENAI_API_KEY",))
        return OpenAICompatibleModelClient(
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=args.temperature,
            timeout=getattr(args, "openai_timeout", getattr(args, "ollama_timeout", 300)),
        )
    
    # =========================================================================
    # Anthropic 兼容的 API（包括 Right Codes Claude）
    # =========================================================================
    if provider == "anthropic":
        model = _effective_model(args, provider)
        base_url = getattr(args, "base_url", None) or provider_env("PICO_ANTHROPIC_API_BASE", ("ANTHROPIC_API_BASE",), DEFAULT_ANTHROPIC_BASE_URL)
        # Anthropic API key 支持多个备选环境变量名（向后兼容）
        api_key = provider_env(
            "PICO_ANTHROPIC_API_KEY",
            ("ANTHROPIC_API_KEY", "PICO_RIGHT_CODES_API_KEY", "RIGHT_CODES_API_KEY", "PICO_OPENAI_API_KEY", "OPENAI_API_KEY"),
        )
        return AnthropicCompatibleModelClient(
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=args.temperature,
            timeout=getattr(args, "openai_timeout", getattr(args, "ollama_timeout", 300)),
        )
    
    # =========================================================================
    # DeepSeek API（使用 Anthropic 兼容格式）
    # =========================================================================
    if provider == "deepseek":
        model = _effective_model(args, provider)
        base_url = getattr(args, "base_url", None) or provider_env("PICO_DEEPSEEK_API_BASE", ("DEEPSEEK_API_BASE",), DEFAULT_DEEPSEEK_BASE_URL)
        api_key = provider_env("PICO_DEEPSEEK_API_KEY", ("DEEPSEEK_API_KEY",))
        return AnthropicCompatibleModelClient(
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=args.temperature,
            timeout=getattr(args, "openai_timeout", getattr(args, "ollama_timeout", 300)),
        )

    # =========================================================================
    # Ollama 本地模型
    # =========================================================================
    model = _effective_model(args, provider)
    host = getattr(args, "host", DEFAULT_OLLAMA_HOST)
    return OllamaModelClient(
        model=model,
        host=host,
        temperature=args.temperature,
        top_p=args.top_p,
        timeout=args.ollama_timeout,
    )


def build_welcome(agent, model, host):
    """生成欢迎界面的 ASCII art 文本。
    
    这个函数负责渲染一个美观的终端欢迎界面，显示：
    - ASCII art 图案
    - 工作区路径
    - 使用的模型和分支
    - 审批策略和 session ID
    
    Args:
        agent: Pico 实例，用于获取 workspace 和 session 信息
        model: 模型名称字符串
        host: 模型服务地址
        
    Returns:
        str: 格式化的欢迎界面文本
    """
    # 计算终端宽度，限制在 68-84 字符之间
    width = max(68, min(shutil.get_terminal_size((80, 20)).columns, 84))
    inner = width - 4  # 内部可用宽度（去掉边框）
    gap = 3  # 左右两列之间的间距
    left_width = (inner - gap) // 2  # 左列宽度
    right_width = inner - gap - left_width  # 右列宽度

    # ------------------------------------------------------------------------
    # 内部辅助函数：用于格式化文本
    # ------------------------------------------------------------------------
    def row(text):
        """生成单行文本，左对齐填充"""
        body = middle(text, width - 4)
        return f"| {body.ljust(width - 4)} |"

    def divider(char="-"):
        """生成分隔线"""
        return "+" + char * (width - 2) + "+"

    def center(text):
        """生成居中文本"""
        body = middle(text, inner)
        return f"| {body.center(inner)} |"

    def cell(label, value, size):
        """生成单元格文本（标签+值）"""
        body = middle(f"{label:<9} {value}", size)
        return body.ljust(size)

    def pair(left_label, left_value, right_label, right_value):
        """生成左右配对的双列文本"""
        left = cell(left_label, left_value, left_width)
        right = cell(right_label, right_value, right_width)
        return f"| {left}{' ' * gap}{right} |"

    # ------------------------------------------------------------------------
    # 组装欢迎界面的每一行
    # ------------------------------------------------------------------------
    line = divider("=")  # 顶部和底部的双线分隔符
    rows = [center(text) for text in WELCOME_ART]  # ASCII art 图案
    rows.extend(
        [
            center(WELCOME_NAME),           # "pico"
            center(WELCOME_SUBTITLE),       # "local coding agent"
            center(WELCOME_STATUS),         # "calm shell, ready for work"
            divider("-"),                   # 单线分隔符
            row(""),                        # 空行
            row("WORKSPACE  " + middle(agent.workspace.cwd, inner - 11)),  # 工作区路径
            pair("MODEL", model, "BRANCH", agent.workspace.branch),        # 模型和分支
            pair("APPROVAL", agent.approval_policy, "SESSION", agent.session["id"]),  # 审批策略和 session ID
            row(""),                        # 空行
        ]
    )
    return "\n".join([line, *rows, line])


def build_agent(args):
    """根据 CLI 参数装配出一个可运行的 Pico 实例。

    ## 为什么存在
    命令行参数只是字符串和开关，runtime 需要的是已经装配好的对象图：model client、workspace snapshot、session store、secret 配置等。
    这个函数负责把"启动参数"翻译成"agent 运行现场"。

    ## 输入 / 输出
    - 输入：`argparse` 解析后的 `args`
    - 输出：一个新的 `Pico`，或一个从旧 session 恢复出来的 `Pico`

    ## 在 agent 链路里的位置
    它是整个程序启动链路里最靠近 runtime 的装配点。`main()` 先调它，得到 agent 后，后面无论是 one-shot 还是 REPL 模式，都会落到 `ask()`。
    
    ## 初始化流程详解
    ```
    build_agent(args)
      ├─> 1. WorkspaceContext.build(args.cwd)           # 扫描工作区
      │    ├─> git rev-parse --show-toplevel            #   查找 repo root
      │    ├─> git branch --show-current                #   获取当前分支
      │    ├─> git status --short                       #   获取工作状态
      │    ├─> git log --oneline -5                     #   获取最近提交
      │    └─> 读取项目文档（README.md, AGENTS.md 等）   #   加载项目文档
      │
      ├─> 2. load_project_env(workspace.repo_root)      # 加载 .env 文件
      │    └─> 从 repo root 向上查找 .env 并加载到 os.environ
      │
      ├─> 3. _configured_secret_names(args)             # 确定敏感变量名单
      │    ├─> 合并 DEFAULT_SECRET_ENV_NAMES
      │    ├─> 添加 --secret-env-name 参数指定的变量
      │    └─> 读取 PICO_SECRET_ENV_NAMES 环境变量
      │
      ├─> 4. SessionStore(...)                          # 创建 session 存储
      │    └─> 初始化 .pico/sessions 目录
      │
      ├─> 5. _build_model_client(args)                  # 构建模型客户端
      │    ├─> _effective_model()                       #   确定模型名称
      │    ├─> provider_env()                           #   读取 API key
      │    └─> 创建对应的 Client 实例                    #   (OpenAI/Anthropic/Ollama)
      │
      └─> 6. Pico.__init__() 或 Pico.from_session()     # 初始化运行时
           ├─> 6.1 Session 初始化
           │    ├─> 生成 session ID（时间戳 + UUID）
           │    ├─> 初始化空 history 和 memory
           │    └─> _ensure_session_shape()             #   确保完整结构
           │
           ├─> 6.2 LayeredMemory 初始化
           │    └─> 创建工作记忆、长期记忆、笔记等分层结构
           │
           ├─> 6.3 build_tools()                       # 注册工具集
           │    ├─> list_files, read_file, search       #   安全工具
           │    ├─> run_shell, write_file, patch_file   #   危险工具
           │    └─> delegate（如果深度允许）             #   子 agent 工具
           │
           ├─> 6.4 build_prefix()                      # 构建 prompt prefix
           │    ├─> 角色定义和规则说明
           │    ├─> 工具详细说明和示例
           │    ├─> 工作区快照（Git 状态、文档等）
           │    └─> 生成 hash 和 fingerprint 用于缓存
           │
           ├─> 6.5 ContextManager(self)                # 初始化上下文管理器
           │    └─> 配置 prompt 预算控制策略
           │
           ├─> 6.6 evaluate_resume_state()             # 评估恢复状态
           │    ├─> 检查 checkpoint 是否存在
           │    ├─> 验证 schema 版本匹配
           │    ├─> 检查文件新鲜度（freshness）
           │    ├─> 验证运行时身份（runtime identity）
           │    └─> 清理过期的文件摘要
           │
           └─> 6.7 session_store.save(session)         # 持久化 session
    """
    # 步骤1: 采集工作区快照（Git 信息 + 项目文档）
    workspace = WorkspaceContext.build(args.cwd)
    
    # 步骤2: 加载项目级环境配置（.env 文件）
    load_project_env(workspace.repo_root)
    
    # 步骤3: 整理敏感环境变量名单（用于 trace/report 脱敏）
    configured_secret_names = _configured_secret_names(args)
    
    # 步骤4: 创建 session 持久化存储
    store = SessionStore(workspace.repo_root + "/.pico/sessions")
    
    # 步骤5: 构建模型客户端（HTTP 连接准备）
    model = _build_model_client(args)
    
    # 步骤6: 决定是恢复旧 session 还是创建新 session
    session_id = args.resume  #指定要恢复的 session ID 参数
    if session_id == "latest":
        session_id = store.latest()  # 找到最近修改的 session 文件
    
    if session_id:
        # 从已有 session 恢复（保留历史对话和记忆）
        return Pico.from_session(
            model_client=model,
            workspace=workspace,
            session_store=store,
            session_id=session_id,
            approval_policy=args.approval,
            max_steps=args.max_steps,
            max_new_tokens=args.max_new_tokens,
            secret_env_names=configured_secret_names,
        )
    
    # 创建全新的 session
    return Pico(
        model_client=model,
        workspace=workspace,
        session_store=store,
        approval_policy=args.approval,
        max_steps=args.max_steps,
        max_new_tokens=args.max_new_tokens,
        secret_env_names=configured_secret_names,
    )


def build_arg_parser():
    """构建命令行参数解析器。    定义了所有支持的 CLI 参数及其默认值、帮助信息等。
    
    Returns:
        argparse.ArgumentParser: 配置好的参数解析器,,,Python 标准库提供的参数解析器构造函数
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Minimal coding agent for Ollama, OpenAI-compatible, Anthropic-compatible, or DeepSeek models.",
    )
    
    # 位置参数：one-shot 模式的提示词
    parser.add_argument("prompt", nargs="*", help="Optional one-shot prompt.")
    
    # 工作区配置
    parser.add_argument("--cwd", default=".", help="Workspace directory.")
    
    # 模型提供商选择
    parser.add_argument("--provider", choices=("ollama", "openai", "anthropic", "deepseek"), default="deepseek", help="Model backend to use.")
    
    # 模型名称覆盖
    parser.add_argument(
        "--model",
        default=None,
        help="Model name override. Defaults to qwen3.5:4b for Ollama, PICO_OPENAI_MODEL for openai, PICO_ANTHROPIC_MODEL for anthropic, and PICO_DEEPSEEK_MODEL for deepseek when set.",
    )
    
    # Ollama 专用配置
    parser.add_argument("--host", default=DEFAULT_OLLAMA_HOST, help="Ollama server URL.")
    parser.add_argument("--ollama-timeout", type=int, default=300, help="Ollama request timeout in seconds.")
    
    # OpenAI/Anthropic/DeepSeek 专用配置
    parser.add_argument("--base-url", default=None, help="Provider API base URL for openai, anthropic, or deepseek.")
    parser.add_argument("--openai-timeout", type=int, default=300, help="OpenAI-compatible request timeout in seconds.")
    
    # Session 管理
    parser.add_argument("--resume", default=None, help="Session id to resume or 'latest'.")
    
    # 安全策略
    parser.add_argument("--approval", choices=("ask", "auto", "never"), default="ask", help="Approval policy for risky tools.")
    parser.add_argument(
        "--secret-env-name",
        dest="secret_env_names",
        action="append",
        default=[],
        help="Extra environment variable names to treat as secrets for trace/report redaction.",
    )
    
    # 运行时控制
    parser.add_argument("--max-steps", type=int, default=6, help="Maximum tool/model iterations per request.")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Maximum model output tokens per step.")
    
    # 采样参数
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature sent to Ollama.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling value sent to Ollama.")
    
    return parser


def main(argv=None):
    """程序主入口函数。
    
    这是整个 pico 应用的启动点，负责：
    1. 解析命令行参数
    2. 装配 Agent 实例
    3. 打印欢迎界面
    4. 进入 one-shot 或 REPL 交互模式
    
    ## 执行流程
    ```
    main(argv)
      │
      ├─> 步骤1: 解析 CLI 参数
      │    └─> build_arg_parser().parse_args(argv)
      │
      ├─> 步骤2: 装配 Agent（详见 build_agent 函数）
      │    └─> agent = build_agent(args)
      │         ├─> 扫描工作区（Git + 文档）
      │         ├─> 加载 .env 配置
      │         ├─> 确定敏感变量名单
      │         ├─> 创建 session store
      │         ├─> 构建模型客户端
      │         └─> 初始化 Pico 运行时
      │              ├─> 初始化 session 结构
      │              ├─> 初始化分层记忆系统
      │              ├─> 注册工具集
      │              ├─> 构建 prompt prefix
      │              ├─> 初始化上下文管理器
      │              ├─> 评估恢复状态
      │              └─> 保存 session 到磁盘
      │
      ├─> 步骤3: 打印欢迎界面
      │    └─> print(build_welcome(agent, model, host))
      │         ├─> 显示 ASCII art
      │         ├─> 显示工作区路径
      │         ├─> 显示模型和分支
      │         └─> 显示审批策略和 session ID
      │
      └─> 步骤4: 进入执行模式
           │
           ├─> 4a. One-shot 模式（如果提供了 prompt 参数）
           │    ├─> prompt = " ".join(args.prompt).strip()
           │    ├─> print(agent.ask(prompt))  # 执行一次完整的 agent 回合
           │    └─> return 0
           │
           └─> 4b. REPL 交互模式（默认）
                └─> while True:
                     ├─> user_input = input("\\npico> ").strip()
                     ├─> 处理内置命令（/help, /memory, /session, /reset, /exit）
                     └─> print(agent.ask(user_input))  # 执行 agent 回合
    ```
    
    Args:
        argv: 命令行参数列表（默认为 sys.argv[1:]）
        
    Returns:
        int: 退出码（0 表示成功，1 表示错误）
    """
    # ========================================================================
    # 步骤1: 解析命令行参数
    # ========================================================================
    args = build_arg_parser().parse_args(argv)
    
    # ========================================================================
    # 步骤2: 装配 Agent 实例（包含完整的初始化流程）
    # ========================================================================
    agent = build_agent(args)

    # 提取模型和服务地址信息用于显示
    model = getattr(agent.model_client, "model", getattr(args, "model", DEFAULT_OLLAMA_MODEL))
    host = getattr(agent.model_client, "host", getattr(agent.model_client, "base_url", getattr(args, "host", DEFAULT_OLLAMA_HOST)))
    
    # ========================================================================
    # 步骤3: 打印欢迎界面
    # ========================================================================
    print(build_welcome(agent, model=model, host=host))

    # ========================================================================
    # 步骤4: 进入执行模式
    # ========================================================================
    if args.prompt:
        # --------------------------------------------------------------------
        # 4a. One-shot 模式：只跑一次 ask，不进入 REPL 循环
        # --------------------------------------------------------------------
        prompt = " ".join(args.prompt).strip()
        if prompt:
            print()
            try:
                print(agent.ask(prompt))
            except RuntimeError as exc:
                print(str(exc), file=sys.stderr)
                return 1
        return 0

    # ------------------------------------------------------------------------
    # 4b. REPL 交互模式：每次读取一条用户输入，交给同一个 agent
    #     因此 session history 和 working memory 会跨轮延续
    # ------------------------------------------------------------------------
    while True:
        try:
            user_input = input("\npico> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("")
            return 0

        # 跳过空输入
        if not user_input:
            continue
        
        # 处理内置命令
        if user_input in {"/exit", "/quit"}:
            return 0
        if user_input == "/help":
            print(HELP_DETAILS)
            continue
        if user_input == "/memory":
            print(agent.memory_text())
            continue
        if user_input == "/session":
            print(agent.session_path)
            continue
        if user_input == "/reset":
            agent.reset()
            print("session reset")
            continue

        # 执行正常的 agent 回合
        print()
        try:
            print(agent.ask(user_input))
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
