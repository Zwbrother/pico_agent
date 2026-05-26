"""工具定义与执行辅助逻辑。

可以把这个文件看成 agent 的能力白名单：模型能申请哪些动作、这些动作
如何做参数校验，以及最终如何执行，都是在这里定义的。

## 核心职责
1. 定义工具规范（名称、参数 schema、风险等级、描述）
2. 实现工具执行函数（读写文件、搜索、执行命令等）
3. 提供参数校验逻辑（路径检查、范围验证、内容匹配等）
4. 构建工具注册表（将工具名映射到执行函数）
5. 提供工具调用示例（用于 prompt 中的 few-shot learning）

## 工具分类
- **安全工具**（risky=False）: list_files, read_file, search, delegate
- **危险工具**（risky=True）: run_shell, write_file, patch_file

## 设计原则
- **显式注册**: 工具不是动态发现的，而是显式注册的白名单
- **严格校验**: 所有工具参数都经过严格校验，防止注入和误用
- **路径安全**: 所有文件操作都被锚定在 workspace root 之下
- **确定性**: patch_file 要求 old_text 精确匹配且只出现一次
"""

import shutil
import subprocess
import textwrap
from functools import partial

from .workspace import IGNORED_PATH_NAMES, clip

# ============================================================================
# 工具规范定义
# ============================================================================

# 基础工具规范字典
# 每个工具包含：
# - schema: 参数定义（类型和默认值）
# - risky: 是否为危险操作（需要审批）
# - description: 工具的简短描述
BASE_TOOL_SPECS = {
    "list_files": {
        "schema": {"path": "str='.'"},
        "risky": False,
        "description": "List files in the workspace.",
    },
    "read_file": {
        "schema": {"path": "str", "start": "int=1", "end": "int=200"},
        "risky": False,
        "description": "Read a UTF-8 file by line range.",
    },
    "search": {
        "schema": {"pattern": "str", "path": "str='.'"},
        "risky": False,
        "description": "Search the workspace with rg or a simple fallback.",
    },
    "run_shell": {
        "schema": {"command": "str", "timeout": "int=20"},
        "risky": True,
        "description": "Run a shell command in the repo root.",
    },
    "write_file": {
        "schema": {"path": "str", "content": "str"},
        "risky": True,
        "description": "Write a text file.",
    },
    "patch_file": {
        "schema": {"path": "str", "old_text": "str", "new_text": "str"},
        "risky": True,
        "description": "Replace one exact text block in a file.",
    },
}

# Delegate 工具规范（用于创建子 agent）
DELEGATE_TOOL_SPEC = {
    "schema": {"task": "str", "max_steps": "int=3"},
    "risky": False,
    "description": "Ask a bounded read-only child agent to investigate.",
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

def build_tool_registry(agent):
    """构建工具注册表。
    
    工具不是动态发现的，而是显式注册的。这样模型看到的是一个有边界、
    可审计的动作集合。
    
    Args:
        agent: Pico 实例，用于绑定到工具执行函数
        
    Returns:
        dict: 工具注册表 {tool_name: {schema, risky, description, run}}
        
    Note:
        - 使用 functools.partial 将 agent 绑定到工具执行函数
        - delegate 工具只在 depth < max_depth 时注册
        - 每个工具的 run 函数签名：run(args) -> str
    """
    # 注册所有基础工具
    tools = {
        name: {**spec, "run": partial(_TOOL_RUNNERS[name], agent)}
        for name, spec in BASE_TOOL_SPECS.items()
    }
    
    # 子 agent 是刻意做成受限能力的：一旦深度耗尽，
    # 就连 delegate 这个工具都不再暴露给模型。
    if agent.depth < agent.max_depth:
        tools["delegate"] = {**DELEGATE_TOOL_SPEC, "run": partial(tool_delegate, agent)}
    
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
        str: 文件列表，每行格式为 "[D/F] relative_path"
             最多显示 200 个条目
        
    Raises:
        ValueError: 如果 path 不是目录
    """
    path = agent.path(args.get("path", "."))
    if not path.is_dir():
        raise ValueError("path is not a directory")
    
    # 排序：文件在前，目录在后；按名称字母顺序
    entries = [
        item for item in sorted(path.iterdir(), key=lambda item: (item.is_file(), item.name.lower()))
        if item.name not in IGNORED_PATH_NAMES
    ]
    
    lines = []
    for entry in entries[:200]:
        kind = "[D]" if entry.is_dir() else "[F]"
        lines.append(f"{kind} {entry.relative_to(agent.root)}")
    
    return "\n".join(lines) or "(empty)"


def tool_read_file(agent, args):
    """读取文件内容（按行范围）。
    
    Args:
        agent: Pico 实例
        args: 参数字典 {path: str, start: int=1, end: int=200}
        
    Returns:
        str: 带行号的文件内容，格式为：
             "# relative/path/to/file\n   1: line1\n   2: line2\n..."
        
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
    body = "\n".join(f"{number:>4}: {line}" for number, line in enumerate(lines[start - 1:end], start=start))
    return f"# {path.relative_to(agent.root)}\n{body}"


def tool_search(agent, args):
    """在工作区中搜索文本。
    
    优先使用 ripgrep (rg)，如果不可用则回退到简单的 Python 实现。
    
    Args:
        agent: Pico 实例
        args: 参数字典 {pattern: str, path: str='.'}
        
    Returns:
        str: 搜索结果，每行格式为 "relative/path:line_number:line_content"
             最多返回 200 条匹配
        
    Raises:
        ValueError: 如果 pattern 为空
    """
    pattern = str(args.get("pattern", "")).strip()
    if not pattern:
        raise ValueError("pattern must not be empty")
    path = agent.path(args.get("path", "."))

    # 优先用 rg，因为搜索会非常频繁，搜索延迟会直接影响 agent 控制循环。
    if shutil.which("rg"):
        result = subprocess.run(
            ["rg", "-n", "--smart-case", "--max-count", "200", pattern, str(path)],
            cwd=agent.root,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or result.stderr.strip() or "(no matches)"

    # 回退到 Python 实现的简单搜索
    matches = []
    files = [path] if path.is_file() else [
        item for item in path.rglob("*")
        if item.is_file() and not any(part in IGNORED_PATH_NAMES for part in item.relative_to(agent.root).parts)
    ]
    for file_path in files:
        for number, line in enumerate(file_path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
            if pattern.lower() in line.lower():
                matches.append(f"{file_path.relative_to(agent.root)}:{number}:{line}")
                if len(matches) >= 200:
                    return "\n".join(matches)
    return "\n".join(matches) or "(no matches)"


def tool_run_shell(agent, args):
    """执行 shell 命令。
    
    Args:
        agent: Pico 实例
        args: 参数字典 {command: str, timeout: int=20}
        
    Returns:
        str: 命令执行结果，格式为：
             "exit_code: N\nstdout:\n...\nstderr:\n..."
        
    Raises:
        ValueError: 如果 command 为空或 timeout 超出范围
        subprocess.TimeoutExpired: 如果命令执行超时
    """
    command = str(args.get("command", "")).strip()
    if not command:
        raise ValueError("command must not be empty")
    timeout = int(args.get("timeout", 20))
    if timeout < 1 or timeout > 120:
        raise ValueError("timeout must be in [1, 120]")
    
    result = subprocess.run(
        command,
        cwd=agent.root,
        shell=True,
        capture_output=True,
        text=True,
        timeout=timeout,
        # 这里传入的是过滤后的环境变量，而不是直接继承整个父 shell 环境，
        # 目的是减少敏感信息被意外带进命令执行环境的风险。
        env=agent.shell_env(),
    )
    
    return textwrap.dedent(
        f"""\
        exit_code: {result.returncode}
        stdout:
        {result.stdout.strip() or "(empty)"}
        stderr:
        {result.stderr.strip() or "(empty)"}
        """
    ).strip()


def tool_write_file(agent, args):
    """写入文件内容。
    
    Args:
        agent: Pico 实例
        args: 参数字典 {path: str, content: str}
        
    Returns:
        str: 成功消息，格式为 "wrote relative/path (N chars)"
        
    Raises:
        ValueError: 如果 path 是目录或 content 缺失
    """
    path = agent.path(args["path"])
    content = str(args["content"])
    
    # 自动创建父目录
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    
    return f"wrote {path.relative_to(agent.root)} ({len(content)} chars)"


def tool_patch_file(agent, args):
    """修补文件（替换精确的文本块）。
    
    这个工具非常严格：old_text 必须在文件中恰好出现一次，否则拒绝执行。
    这样可以保证修改行为的确定性。
    
    Args:
        agent: Pico 实例
        args: 参数字典 {path: str, old_text: str, new_text: str}
        
    Returns:
        str: 成功消息，格式为 "patched relative/path"
        
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
    
    # 只替换第一次出现（虽然理论上只有一次）
    path.write_text(text.replace(old_text, str(args["new_text"]), 1), encoding="utf-8")
    return f"patched {path.relative_to(agent.root)}"


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
    )
    
    # 委派的目标是"调查"，不是"放权执行"。
    # 子 agent 以只读方式运行、步数更少，最后只把结论文本返回给父 agent。
    child.session["memory"]["task"] = task
    child.session["memory"]["notes"] = [clip(agent.history_text(), 300)]
    
    return "delegate_result:\n" + child.ask(task)


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
