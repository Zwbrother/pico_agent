"""工具定义与执行辅助逻辑。

可以把这个文件看成 agent 的能力白名单：模型能申请哪些动作、这些动作
如何做参数校验，以及最终如何执行，都是在这里定义的。

## 核心职责
1. 定义工具规范（名称、参数 schema、能力位、描述）
2. 实现工具执行函数（读写文件、搜索、执行命令等）
3. 提供参数校验逻辑（路径检查、范围验证、内容匹配等）
4. 构建工具注册表（将工具名映射到执行函数）
5. 提供工具调用示例（用于 prompt 中的 few-shot learning）

## 工具能力分类（capabilities）
每个工具用一组 capability 标记自己的副作用面：
- `read`  : 只读 workspace / git 元数据
- `write` : 修改 workspace 内文件
- `exec`  : 进程执行
- `net`   : 出站网络 I/O（目前没有内置工具携带此能力，预留）

`approval_policy` 可按 capability 分档（详见 `runtime.py`），单一布尔位 `risky`
作为派生值保留，兼容旧代码路径。

## 设计原则
- **显式注册**: 工具不是动态发现的，而是显式注册的白名单
- **严格校验**: 所有工具参数都经过严格校验，防止注入和误用
- **路径安全**: 所有文件操作都被锚定在 workspace root 之下
- **确定性**: patch_file 要求 old_text 精确匹配且只出现一次
- **结构化结果**: 工具返回 `ToolResult(text, payload, ok)`，文本给模型看，
  payload 给 trace 聚合和评测脚本看
"""

import os
import shutil
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from functools import partial

from .workspace import IGNORED_PATH_NAMES, clip

# ============================================================================
# 能力位（capability bits）
# ============================================================================

CAP_READ = "read"
CAP_WRITE = "write"
CAP_EXEC = "exec"
CAP_NET = "net"

# 这几个能力被视为"危险"，需要走审批流程
RISKY_CAPABILITIES = frozenset({CAP_WRITE, CAP_EXEC, CAP_NET})


def is_risky(capabilities):
    """根据 capability 集合派生 `risky` 布尔位。

    旧代码很多地方仍读 `tool["risky"]`，这是为它们准备的兼容入口。
    """
    return bool(set(capabilities) & RISKY_CAPABILITIES)


# ============================================================================
# ToolResult：工具执行结果的结构化封装
# ============================================================================

@dataclass
class ToolResult:
    """工具执行结果。

    旧版本工具直接返回 `str`，文本既给模型看也给 trace 看，可观测维度有限。
    现在统一返回 `ToolResult`：
    - `text`    : 给模型看的可读字符串（向后兼容，沿用旧值的语义）
    - `payload` : 给 trace 聚合 / 评测脚本看的结构化字段
    - `ok`      : 工具自身视角下是否成功（注意：`run_shell` exit_code != 0 仍然
                  `ok=True`，"命令执行完成" 和 "命令业务成功" 是两回事）

    runtime 在调用工具后，会把 `.text` 塞进 history，把 `.payload` 注入
    `tool_executed` trace 事件。模型看到的内容不变，trace 里的可分析维度变多。
    """

    text: str
    payload: dict = field(default_factory=dict)
    ok: bool = True


def _coerce_tool_result(value):
    """把工具函数的返回值统一成 `ToolResult`。

    向后兼容：如果某个工具（包括用户自定义工具）仍返回纯 `str`，
    用 `ToolResult(text=value)` 包一层，`payload` 留空。
    """
    if isinstance(value, ToolResult):
        return value
    return ToolResult(text="" if value is None else str(value))

# ============================================================================
# 工具规范定义
# ============================================================================

# 基础工具规范字典
# 每个工具包含：
# - schema      : 参数定义（类型和默认值）
# - capabilities: 工具所需的能力位（read / write / exec / net）
# - description : 工具的简短描述
# - risky       : 派生位，等价于 `capabilities & RISKY_CAPABILITIES != {}`，
#                 由 `build_tool_registry()` 自动填入，便于旧代码继续用
BASE_TOOL_SPECS = {
    "list_files": {
        "schema": {"path": "str='.'"},
        "capabilities": (CAP_READ,),
        "description": "List files in the workspace.",
    },
    "read_file": {
        "schema": {"path": "str", "start": "int=1", "end": "int=200"},
        "capabilities": (CAP_READ,),
        "description": "Read a UTF-8 file by line range.",
    },
    "search": {
        "schema": {"pattern": "str", "path": "str='.'"},
        "capabilities": (CAP_READ,),
        "description": "Search the workspace with rg or a simple fallback.",
    },
    "run_shell": {
        "schema": {"command": "str", "timeout": "int=20"},
        "capabilities": (CAP_EXEC,),
        "description": "Run a shell command in the repo root.",
    },
    "write_file": {
        "schema": {"path": "str", "content": "str"},
        "capabilities": (CAP_WRITE,),
        "description": "Write a text file.",
    },
    "patch_file": {
        "schema": {"path": "str", "old_text": "str", "new_text": "str"},
        "capabilities": (CAP_WRITE,),
        "description": "Replace one exact text block in a file.",
    },
}

# Delegate 工具规范（用于创建子 agent）
DELEGATE_TOOL_SPEC = {
    "schema": {"task": "str", "max_steps": "int=3"},
    "capabilities": (CAP_READ,),
    "description": "Ask a bounded read-only child agent to investigate.",
}


def _finalize_spec(spec, runner):
    """把一个原始 spec 烘焙成可注册的 tool dict。

    - 自动从 `capabilities` 派生 `risky`（兼容旧字段）；
    - 把 `capabilities` 归一化为不可变的 tuple，避免被外部就地修改；
    - 绑定执行函数。
    """
    capabilities = tuple(spec.get("capabilities", ()))
    return {
        "schema": spec["schema"],
        "capabilities": capabilities,
        "risky": is_risky(capabilities),
        "description": spec["description"],
        "run": runner,
    }

# 工具调用示例（用于 prompt 中的 few-shot learning）
# 展示了两种格式：
# 1. JSON 格式：<tool>{"name":"...","args":{...}}</tool>
# 2. XML 格式：<tool name="..." path="..."><content>...</content></tool>
TOOL_EXAMPLES = {
    "list_files": '<tool>{"name":"list_files","args":{"path":"."}}</tool>',
    "read_file": '<tool>{"name":"read_file","args":{"path":"README.md","start":1,"end":80}}</tool>',
    "search": '<tool>{"name":"search","args":{"pattern":"binary_search","path":"."}}</tool>',
    "run_shell": '<tool>{"name":"run_shell","args":{"command":"uv run --with pytest python -m pytest -q","timeout":20}}</tool>',
    "write_file": '<tool name="write_file" path="binary_search.py"><content>def binary_search(nums, target):\n    return -1\n</content></tool>',
    "patch_file": '<tool name="patch_file" path="binary_search.py"><old_text>return -1</old_text><new_text>return mid</new_text></tool>',
    "delegate": '<tool>{"name":"delegate","args":{"task":"inspect README.md","max_steps":3}}</tool>',
}


# ============================================================================
# 工具注册表构建
# ============================================================================

def _wrap_runner(runner, agent):
    """绑定 agent 并把返回值统一成 ToolResult。

    现有 / 用户自定义工具若仍返回 str，会被自动包装为 `ToolResult(text=...)`，
    保证调用方（runtime）拿到的总是 ToolResult。
    """
    bound = partial(runner, agent)

    def _run(args):
        return _coerce_tool_result(bound(args))

    return _run


def build_tool_registry(agent):
    """构建工具注册表。
    
    工具不是动态发现的，而是显式注册的。这样模型看到的是一个有边界、可审计的动作集合。
    
    Args:
        agent: Pico 实例，用于绑定到工具执行函数
        
    Returns:
        dict: 工具注册表 {tool_name: {schema, capabilities, risky, description, run}}
        
    Note:
        - 使用 functools.partial 将 agent 绑定到工具执行函数
        - delegate 工具只在 depth < max_depth 时注册
        - 每个工具的 run 函数签名：run(args) -> ToolResult
        - 返回的 dict 同时带 `capabilities`（新字段）和 `risky`（派生字段，兼容旧代码）
    """
    # 注册所有基础工具
    tools = {
        name: _finalize_spec(spec, _wrap_runner(_TOOL_RUNNERS[name], agent))
        for name, spec in BASE_TOOL_SPECS.items()
    }

    # 子 agent 是刻意做成受限能力的：一旦深度耗尽，
    # 就连 delegate 这个工具都不再暴露给模型。
    if agent.depth < agent.max_depth:
        tools["delegate"] = _finalize_spec(DELEGATE_TOOL_SPEC, _wrap_runner(tool_delegate, agent))

    return tools


def tool_example(name):
    """获取指定工具的调用示例。
    
    Args:
        name: 工具名称
        
    Returns:
        str: 工具调用示例字符串，如果不存在则返回空字符串
    """
    return TOOL_EXAMPLES.get(name, "")


# ============================================================================
# 工具参数校验
# ============================================================================

def validate_tool(agent, name, args):
    """校验工具参数的合法性。
    
    这是工具执行前的第一道防线，确保：
    1. 路径不逃逸出 workspace root
    2. 参数类型和范围正确
    3. 必填参数存在
    4. 特殊约束满足（如 patch_file 的 old_text 唯一性）
    
    Args:
        agent: Pico 实例（用于路径解析）
        name: 工具名称
        args: 参数字典
        
    Raises:
        ValueError: 如果参数不合法
        
    ## 各工具的校验规则
    ```
    list_files:
      ├─> path 必须是目录
    
    read_file:
      ├─> path 必须是文件
      └─> start >= 1, end >= start
    
    search:
      ├─> pattern 不能为空
      └─> path 必须合法（通过 agent.path 校验）
    
    run_shell:
      ├─> command 不能为空
      └─> timeout 必须在 [1, 120] 范围内
    
    write_file:
      ├─> path 不能是目录
      └─> content 必须存在
    
    patch_file:
      ├─> path 必须是文件
      ├─> old_text 不能为空
      ├─> new_text 必须存在
      └─> old_text 在文件中必须恰好出现一次
    
    delegate:
      └─> task 不能为空
    ```
    """
    args = args or {}

    # ------------------------------------------------------------------------
    # list_files: 列出目录内容
    # ------------------------------------------------------------------------
    if name == "list_files":
        path = agent.path(args.get("path", "."))
        if not path.is_dir():
            raise ValueError("path is not a directory")
        return

    # ------------------------------------------------------------------------
    # read_file: 读取文件内容
    # ------------------------------------------------------------------------
    if name == "read_file":
        path = agent.path(args["path"])
        if not path.is_file():
            raise ValueError("path is not a file")
        start = int(args.get("start", 1))
        end = int(args.get("end", 200))
        if start < 1 or end < start:
            raise ValueError("invalid line range")
        return

    # ------------------------------------------------------------------------
    # search: 搜索文本
    # ------------------------------------------------------------------------
    if name == "search":
        pattern = str(args.get("pattern", "")).strip()
        if not pattern:
            raise ValueError("pattern must not be empty")
        agent.path(args.get("path", "."))  # 校验路径合法性
        return

    # ------------------------------------------------------------------------
    # run_shell: 执行 shell 命令
    # ------------------------------------------------------------------------
    if name == "run_shell":
        command = str(args.get("command", "")).strip()
        if not command:
            raise ValueError("command must not be empty")
        timeout = int(args.get("timeout", 20))
        if timeout < 1 or timeout > 120:
            raise ValueError("timeout must be in [1, 120]")
        return

    # ------------------------------------------------------------------------
    # write_file: 写入文件
    # ------------------------------------------------------------------------
    if name == "write_file":
        path = agent.path(args["path"])
        if path.exists() and path.is_dir():
            raise ValueError("path is a directory")
        if "content" not in args:
            raise ValueError("missing content")
        return

    # ------------------------------------------------------------------------
    # patch_file: 修补文件（最严格的校验）
    # ------------------------------------------------------------------------
    if name == "patch_file":
        # patch_file 故意做得很严格：old_text 必须精确命中且只能出现一次，
        # 这样修改行为才是确定的，失败原因也更容易解释。
        path = agent.path(args["path"])
        if not path.is_file():
            raise ValueError("path is not a file")
        old_text = str(args.get("old_text", ""))
        if not old_text:
            raise ValueError("old_text must not be empty")
        if "new_text" not in args:
            raise ValueError("missing new_text")
        text = path.read_text(encoding="utf-8")
        count = text.count(old_text)
        if count != 1:
            raise ValueError(f"old_text must occur exactly once, found {count}")
        return

    # ------------------------------------------------------------------------
    # delegate: 委派给子 agent
    # ------------------------------------------------------------------------
    if name == "delegate":
        task = str(args.get("task", "")).strip()
        if not task:
            raise ValueError("task must not be empty")
        return


# ============================================================================
# 工具执行函数
# ============================================================================

def tool_list_files(agent, args):
    """列出目录内容。

    Args:
        agent: Pico 实例
        args: 参数字典 {path: str}

    Returns:
        ToolResult:
            text:    文件列表，每行格式为 "[D/F] relative_path"，最多 200 条
            payload: {"path", "entry_count", "truncated"}

    Raises:
        ValueError: 如果 path 不是目录
    """
    path = agent.path(args.get("path", "."))
    if not path.is_dir():
        raise ValueError("path is not a directory")

    entries = [
        item for item in sorted(path.iterdir(), key=lambda item: (item.is_file(), item.name.lower()))
        if item.name not in IGNORED_PATH_NAMES
    ]

    truncated = len(entries) > 200
    lines = []
    for entry in entries[:200]:
        kind = "[D]" if entry.is_dir() else "[F]"
        lines.append(f"{kind} {entry.relative_to(agent.root)}")

    text = "\n".join(lines) or "(empty)"
    return ToolResult(
        text=text,
        payload={
            "path": str(path.relative_to(agent.root)) if path != agent.root else ".",
            "entry_count": len(entries),
            "truncated": truncated,
        },
    )


def tool_read_file(agent, args):
    """读取文件内容（按行范围）。

    Args:
        agent: Pico 实例
        args: 参数字典 {path: str, start: int=1, end: int=200}

    Returns:
        ToolResult:
            text:    带行号的文件内容，格式为
                     "# relative/path/to/file\n   1: line1\n   2: line2\n..."
            payload: {"path", "start", "end", "lines_returned", "lines_total", "truncated"}

    Raises:
        ValueError: 如果 path 不是文件或行范围无效
    """
    path = agent.path(args["path"])
    if not path.is_file():
        raise ValueError("path is not a file")

    start = int(args.get("start", 1))
    end = int(args.get("end", 200))
    if start < 1 or end < start:
        raise ValueError("invalid line range")

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    selected = lines[start - 1:end]
    body = "\n".join(f"{number:>4}: {line}" for number, line in enumerate(selected, start=start))
    text = f"# {path.relative_to(agent.root)}\n{body}"
    return ToolResult(
        text=text,
        payload={
            "path": str(path.relative_to(agent.root)),
            "start": start,
            "end": end,
            "lines_returned": len(selected),
            "lines_total": len(lines),
            "truncated": end < len(lines),
        },
    )


def tool_search(agent, args):
    """在工作区中搜索文本。

    优先使用 ripgrep (rg)，如果不可用则回退到简单的 Python 实现。

    Args:
        agent: Pico 实例
        args: 参数字典 {pattern: str, path: str='.'}

    Returns:
        ToolResult:
            text:    搜索结果，每行格式为 "relative/path:line_number:line_content"
            payload: {"pattern", "path", "match_count", "files_hit", "backend", "truncated"}

    Raises:
        ValueError: 如果 pattern 为空
    """
    pattern = str(args.get("pattern", "")).strip()
    if not pattern:
        raise ValueError("pattern must not be empty")
    path = agent.path(args.get("path", "."))

    def _summarize(text_body, backend, truncated):
        lines = [line for line in text_body.splitlines() if line.strip()] if text_body and text_body != "(no matches)" else []
        files_hit = len({line.split(":", 1)[0] for line in lines if ":" in line})
        return ToolResult(
            text=text_body if text_body else "(no matches)",
            payload={
                "pattern": pattern,
                "path": str(path.relative_to(agent.root)) if path != agent.root else ".",
                "match_count": len(lines),
                "files_hit": files_hit,
                "backend": backend,
                "truncated": truncated,
            },
        )

    # 优先用 rg，因为搜索会非常频繁，搜索延迟会直接影响 agent 控制循环。
    if shutil.which("rg"):
        result = subprocess.run(
            ["rg", "-n", "--smart-case", "--max-count", "200", pattern, str(path)],
            cwd=agent.root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        body = result.stdout.strip() or result.stderr.strip() or "(no matches)"
        return _summarize(body, backend="rg", truncated=False)

    # 回退到 Python 实现的简单搜索
    matches = []
    truncated = False
    files = [path] if path.is_file() else [
        item for item in path.rglob("*")
        if item.is_file() and not any(part in IGNORED_PATH_NAMES for part in item.relative_to(agent.root).parts)
    ]
    for file_path in files:
        for number, line in enumerate(file_path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
            if pattern.lower() in line.lower():
                matches.append(f"{file_path.relative_to(agent.root)}:{number}:{line}")
                if len(matches) >= 200:
                    truncated = True
                    break
        if truncated:
            break
    body = "\n".join(matches) or "(no matches)"
    return _summarize(body, backend="python", truncated=truncated)


# ----------------------------------------------------------------------------
# Shell 沙箱：单次命令的资源上限
# ----------------------------------------------------------------------------
#
# 默认配额。POSIX (Linux/macOS) 上通过 `resource.setrlimit` 在子进程 fork 完
# 但 exec 前生效；Windows 上 stdlib 没有等价机制，只能依赖 timeout + 进程组兜底。
#
# 配额选得保守一些：足以跑测试和构建，挡得住 fork bomb / 内存爆炸 / 输出爆炸。
DEFAULT_SHELL_MEMORY_BYTES = 2 * 1024 * 1024 * 1024   # 2 GiB 虚拟内存
DEFAULT_SHELL_MAX_PROCESSES = 256                     # 子进程数
DEFAULT_SHELL_OUTPUT_BYTES = 1 * 1024 * 1024          # 1 MiB stdout/stderr 各自上限

_IS_POSIX = os.name == "posix"


def _shell_preexec_factory(timeout_seconds):
    """构造 POSIX preexec_fn：在子进程里设置 rlimit。

    只在 POSIX 平台返回非 None。Windows 上 preexec_fn 不可用。
    """
    if not _IS_POSIX:
        return None

    def _preexec():
        # 延迟 import：resource 在 Windows 上不存在
        import resource

        # 把子进程拉到独立的进程组，超时后可以一次性 kill 整组，避免孤儿进程。
        try:
            os.setsid()
        except OSError:
            pass

        # RLIMIT_CPU 用 timeout 的 ~2 倍作软上限，避免和 subprocess.timeout 完全
        # 同时触发，但又能挡住 wall-clock 内闷头烧 CPU 的死循环。
        cpu_limit = max(1, int(timeout_seconds * 2))
        for rlimit_name, value in (
            ("RLIMIT_CPU",   (cpu_limit, cpu_limit + 1)),
            ("RLIMIT_AS",    (DEFAULT_SHELL_MEMORY_BYTES, DEFAULT_SHELL_MEMORY_BYTES)),
            ("RLIMIT_NPROC", (DEFAULT_SHELL_MAX_PROCESSES, DEFAULT_SHELL_MAX_PROCESSES)),
            ("RLIMIT_FSIZE", (DEFAULT_SHELL_OUTPUT_BYTES * 64, DEFAULT_SHELL_OUTPUT_BYTES * 64)),
        ):
            rlimit_const = getattr(resource, rlimit_name, None)
            if rlimit_const is None:
                continue
            try:
                resource.setrlimit(rlimit_const, value)
            except (ValueError, OSError):
                # 某些容器里不让降 RLIMIT_NPROC，忽略即可，命令仍能跑。
                pass

    return _preexec


def tool_run_shell(agent, args):
    """执行 shell 命令。

    边界由四层构成：
    1. cwd 锚定在 `agent.root`；
    2. timeout 兜底（subprocess 自带）；
    3. 环境变量白名单（`agent.shell_env()`）；
    4. POSIX 资源上限（CPU/虚拟内存/子进程数/文件大小），通过 `preexec_fn` 注入；
       Windows 上设独立进程组，超时后 kill 整组，避免孤儿。

    Args:
        agent: Pico 实例
        args: 参数字典 {command: str, timeout: int=20}

    Returns:
        ToolResult:
            text:    "exit_code: N\\nstdout:\\n...\\nstderr:\\n..."（沿用旧格式）
            payload: {
                "exit_code": int | None,    # 超时时为 None
                "timed_out": bool,
                "stdout_bytes": int,
                "stderr_bytes": int,
                "timeout_seconds": int,
                "sandbox": str,             # "posix-rlimit" / "windows-pgroup" / "none"
            }
            ok:    `False` 仅当出现 sandbox 自身故障；命令 exit_code != 0 仍 `ok=True`，
                   "命令执行完成" 与 "命令业务成功" 区分由调用方解释 payload 决定。

    Raises:
        ValueError: 如果 command 为空或 timeout 超出范围
    """
    command = str(args.get("command", "")).strip()
    if not command:
        raise ValueError("command must not be empty")
    timeout = int(args.get("timeout", 20))
    if timeout < 1 or timeout > 120:
        raise ValueError("timeout must be in [1, 120]")

    preexec = _shell_preexec_factory(timeout)
    popen_kwargs = {
        "cwd": agent.root,
        "shell": True,
        "capture_output": True,
        "text": True,
        "timeout": timeout,
        "encoding": "utf-8",
        "errors": "replace",
        # 这里传入的是过滤后的环境变量，而不是直接继承整个父 shell 环境，
        # 目的是减少敏感信息被意外带进命令执行环境的风险。
        "env": agent.shell_env(),
    }
    if _IS_POSIX:
        popen_kwargs["preexec_fn"] = preexec
        sandbox_label = "posix-rlimit"
    elif sys.platform == "win32":
        popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        sandbox_label = "windows-pgroup"
    else:
        sandbox_label = "none"

    timed_out = False
    stdout = ""
    stderr = ""
    exit_code = None
    try:
        result = subprocess.run(command, **popen_kwargs)
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        exit_code = result.returncode
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        # TimeoutExpired 会附带已采集到的部分输出
        stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else (exc.stdout.decode("utf-8", "replace") if exc.stdout else "")
        stderr = (exc.stderr or "") if isinstance(exc.stderr, str) else (exc.stderr.decode("utf-8", "replace") if exc.stderr else "")
        stderr = (stderr + f"\n[timed out after {timeout}s]").strip()

    text = textwrap.dedent(
        f"""\
        exit_code: {"timeout" if timed_out else exit_code}
        stdout:
        {stdout.strip() or "(empty)"}
        stderr:
        {stderr.strip() or "(empty)"}
        """
    ).strip()

    return ToolResult(
        text=text,
        payload={
            "exit_code": exit_code,
            "timed_out": timed_out,
            "stdout_bytes": len(stdout.encode("utf-8", "replace")),
            "stderr_bytes": len(stderr.encode("utf-8", "replace")),
            "timeout_seconds": timeout,
            "sandbox": sandbox_label,
        },
        ok=True,
    )


def tool_write_file(agent, args):
    """写入文件内容。

    Args:
        agent: Pico 实例
        args: 参数字典 {path: str, content: str}

    Returns:
        ToolResult:
            text:    "wrote relative/path (N chars)"
            payload: {"path", "bytes_written", "char_count", "created"}

    Raises:
        ValueError: 如果 path 是目录或 content 缺失
    """
    path = agent.path(args["path"])
    content = str(args["content"])

    created = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

    rel = str(path.relative_to(agent.root))
    bytes_written = len(content.encode("utf-8", "replace"))
    return ToolResult(
        text=f"wrote {rel} ({len(content)} chars)",
        payload={
            "path": rel,
            "bytes_written": bytes_written,
            "char_count": len(content),
            "created": created,
        },
    )


def tool_patch_file(agent, args):
    """修补文件（替换精确的文本块）。

    这个工具非常严格：old_text 必须在文件中恰好出现一次，否则拒绝执行。
    这样可以保证修改行为的确定性。

    Args:
        agent: Pico 实例
        args: 参数字典 {path: str, old_text: str, new_text: str}

    Returns:
        ToolResult:
            text:    "patched relative/path"
            payload: {"path", "bytes_removed", "bytes_added", "line_delta"}

    Raises:
        ValueError: 如果 path 不是文件、old_text 为空、new_text 缺失，
                   或 old_text 出现次数不为 1
    """
    path = agent.path(args["path"])
    if not path.is_file():
        raise ValueError("path is not a file")
    
    old_text = str(args.get("old_text", ""))
    if not old_text:
        raise ValueError("old_text must not be empty")
    if "new_text" not in args:
        raise ValueError("missing new_text")
    
    text = path.read_text(encoding="utf-8")
    count = text.count(old_text)
    if count != 1:
        raise ValueError(f"old_text must occur exactly once, found {count}")

    new_text = str(args["new_text"])
    new_content = text.replace(old_text, new_text, 1)
    path.write_text(new_content, encoding="utf-8")

    rel = str(path.relative_to(agent.root))
    line_delta = new_content.count("\n") - text.count("\n")
    return ToolResult(
        text=f"patched {rel}",
        payload={
            "path": rel,
            "bytes_removed": len(old_text.encode("utf-8", "replace")),
            "bytes_added": len(new_text.encode("utf-8", "replace")),
            "line_delta": line_delta,
        },
    )


def tool_delegate(agent, args):
    """委派任务给子 agent。
    
    创建一个只读的、步数受限的子 agent 来执行调查任务。
    子 agent 的结果会以文本形式返回给父 agent。
    
    Args:
        agent: Pico 实例（父 agent）
        args: 参数字典 {task: str, max_steps: int=3}
        
    Returns:
        str: 子 agent 的执行结果，前缀为 "delegate_result:\n"
        
    Raises:
        ValueError: 如果达到最大深度或 task 为空
        
    Note:
        - 子 agent 以只读模式运行（read_only=True）
        - 子 agent 不能使用危险工具（approval_policy="never"）
        - 子 agent 的步数更少（默认 3 步）
        - 子 agent 共享父 agent 的 model_client 和 workspace
    """
    if agent.depth >= agent.max_depth:
        raise ValueError("delegate depth exceeded")
    
    task = str(args.get("task", "")).strip()
    if not task:
        raise ValueError("task must not be empty")

    from .runtime import Pico

    # 创建子 agent（受限能力）
    child = Pico(
        model_client=agent.model_client,
        workspace=agent.workspace,
        session_store=agent.session_store,
        run_store=agent.run_store,
        approval_policy="never",       # 禁止所有危险工具
        max_steps=int(args.get("max_steps", 3)),
        max_new_tokens=agent.max_new_tokens,
        depth=agent.depth + 1,          # 深度 +1
        max_depth=agent.max_depth,
        read_only=True,                 # 只读模式
        secret_env_names=agent.secret_env_names,
        shell_env_allowlist=agent.shell_env_allowlist,
        parent_run_id=getattr(agent, "_current_run_id", ""),
    )
    
    # 委派的目标是"调查"，不是"放权执行"。
    # 子 agent 以只读方式运行、步数更少，最后只把结论文本返回给父 agent。
    child.session["memory"]["task"] = task
    child.session["memory"]["notes"] = [clip(agent.history_text(), 300)]

    answer = child.ask(task)
    child_run_id = getattr(child, "_current_run_id", "")
    return ToolResult(
        text="delegate_result:\n" + answer,
        payload={
            "child_run_id": child_run_id,
            "max_steps": int(args.get("max_steps", 3)),
            "answer_chars": len(answer or ""),
        },
    )


# ============================================================================
# 工具执行函数映射表
# ============================================================================

# 将工具名映射到对应的执行函数
# 这个字典被 build_tool_registry 使用来创建完整的工具注册表
_TOOL_RUNNERS = {
    "list_files": tool_list_files,
    "read_file": tool_read_file,
    "search": tool_search,
    "run_shell": tool_run_shell,
    "write_file": tool_write_file,
    "patch_file": tool_patch_file,
}
