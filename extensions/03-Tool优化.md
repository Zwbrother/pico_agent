# 第三轮优化：三项改动

**改动文件**：`pico/tools.py`（+467 / -125）、`pico/runtime.py`（+169 / -49）

---

## 背景

经过两轮改动（`01-workspace-snapshot-扩展`、`02-runtime-三项改进`），pico 在「开局信息」「失败分类」「父子 run 关系」三个维度已经补齐。本轮围绕「可观测性」和「安全边界」做深，三处改动：

| 改动 | 文件 | 核心变更 |
|---|---|---|
| 一：风险模型细化 | `tools.py`、`runtime.py` | `risky: bool` → `capabilities: tuple`，审批策略支持按能力分档 |
| 二：shell 边界加固 | `tools.py` | 加 POSIX `rlimit` + Windows 进程组，`TimeoutExpired` 改为结构化处理 |
| 三：工具结果结构化 | `tools.py`、`runtime.py` | 引入 `ToolResult` dataclass，trace 写入 `result_payload` |

---

## 改动一：风险模型细化

### 改动前

`BASE_TOOL_SPECS` 里每个工具只有一个 `risky: bool`，`True` 对应「需要审批」，`False` 对应「安全」。`approval_policy` 是全局字符串，`ask / auto / never` 三档，无法区分「允许写文件但禁止跑 shell」这类需求。

### 改动后

**1-A 新增 capability 常量体系**

位置：`tools.py` 顶部

```46:60:pico/tools.py
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
```

**1-B `BASE_TOOL_SPECS` 把 `risky` 替换为 `capabilities`**

位置：`tools.py` → `BASE_TOOL_SPECS`

```108:139:pico/tools.py
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
```

`delegate` 的 `capabilities` 是 `(CAP_READ,)`——子 agent 只做调查，不写不执行。

**1-C `_finalize_spec()` 自动派生 `risky` 兼容字段**

位置：`tools.py` → `_finalize_spec()`

```149:162:pico/tools.py
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
```

`risky` 保留为派生字段，现有所有读 `tool["risky"]` 的代码不需要改。

**1-D `approval_policy` 支持按 capability 分档**

位置：`runtime.py` → `_resolve_capability_policy()`

新增的 `_resolve_capability_policy()` 把 `approval_policy` 归一化为「针对本次工具的最终决策」：

- 旧形态：字符串 `"ask"/"auto"/"never"`，等价于原来的行为；
- 新形态：dict，例如 `{"write": "auto", "exec": "never", "net": "never"}`，按能力分档；
- 合并规则：取所有危险 capability 对应的策略，**最严的赢**（`never > ask > auto`）；
- 若工具只带 `read`，永远返回 `"auto"`，绕过审批路径。

```2542:2561:pico/runtime.py
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
```

`approve()` 改为接受 `capabilities=()` 参数，通过 `_resolve_capability_policy()` 拿到 `effective` 策略后再决定放行还是拦截。交互式提示也升级为 `approve write_file [caps:write] {...}? [y/N]`，把能力位印出来。

**1-E `build_prefix()` 渲染能力位**

位置：`runtime.py` → `build_prefix()`

模型看到的工具卡片从 `[approval required]` 变成：

```922:922:pico/runtime.py
tool_lines.append(f"- {name}({fields}) [caps:{caps_label} | {risk_hint}] {tool['description']}")
```

例如：`- run_shell(command: str, timeout: int=20) [caps:exec | approval required] Run a shell command in the repo root.`

`tool_signature()` 同步把 `capabilities` 纳入指纹计算，工具能力变化会触发 prefix 重建。

### 向后兼容

- `tool["risky"]` 继续有效，值由 `_finalize_spec` 自动派生；
- `approval_policy="ask"/"auto"/"never"` 字符串行为完全不变；
- 现有测试中 `approval_policy="never"` 仍然拒绝所有危险工具，不需要修改。

---

## 改动二：shell 边界加固

### 改动前

`tool_run_shell` 执行边界：cwd 锚定 + timeout + 环境变量白名单。`TimeoutExpired` 会穿透成异常，runtime 上层捕获后返回 `error: tool run_shell failed: ...`，没有 `timed_out` 标记。

### 改动后

**2-A 资源配额常量**

位置：`tools.py`，`tool_run_shell` 之前

```545:549:pico/tools.py
DEFAULT_SHELL_MEMORY_BYTES = 2 * 1024 * 1024 * 1024   # 2 GiB 虚拟内存
DEFAULT_SHELL_MAX_PROCESSES = 256                     # 子进程数
DEFAULT_SHELL_OUTPUT_BYTES = 1 * 1024 * 1024          # 1 MiB stdout/stderr 各自上限

_IS_POSIX = os.name == "posix"
```

**2-B `_shell_preexec_factory()` 注入 rlimit**

位置：`tools.py` → `_shell_preexec_factory(timeout_seconds)`

在子进程 fork 完但 exec 前执行，做三件事：

1. `os.setsid()`：把子进程拉到独立进程组，超时后可以 `kill -TERM -pgid` 一次性干掉整棵进程树，不留孤儿；
2. 设 4 个 rlimit（CPU 时间、虚拟内存、子进程数、文件写入大小）；
3. rlimit 设置失败时静默忽略（某些容器不允许降 `RLIMIT_NPROC`），命令仍照常跑。

CPU 上限取 `timeout * 2`：比 `subprocess.timeout` 宽松，挡的是「wall-clock 内闷头烧 CPU 的死循环」，不会和超时提前争跑。

```552:588:pico/tools.py
def _shell_preexec_factory(timeout_seconds):
    if not _IS_POSIX:
        return None

    def _preexec():
        import resource
        try:
            os.setsid()
        except OSError:
            pass
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
                pass

    return _preexec
```

**2-C Windows 进程组隔离**

位置：`tools.py` → `tool_run_shell`

Windows 没有 `preexec_fn`，改用 `creationflags=CREATE_NEW_PROCESS_GROUP`。超时时 `subprocess` 会调 `TerminateProcess` 终止目标进程；子进程组内的其他进程依赖 Windows job object 来清理（当前仅做进程组隔离，不强依赖 `pywin32`）。

```642:649:pico/tools.py
if _IS_POSIX:
    popen_kwargs["preexec_fn"] = preexec
    sandbox_label = "posix-rlimit"
elif sys.platform == "win32":
    popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    sandbox_label = "windows-pgroup"
else:
    sandbox_label = "none"
```

**2-D `TimeoutExpired` 改为结构化处理**

改动前：`subprocess.TimeoutExpired` 会穿透成异常，被 `run_tool()` 的 except 块捕获，返回 `error: tool run_shell failed: ...`，没有 `timed_out` 标记。

改动后：`tool_run_shell` 自己捕获 `TimeoutExpired`，把部分输出保留，标记 `timed_out=True`，让 trace 里的 `result_payload.timed_out` 有意义：

```651:665:pico/tools.py
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
    stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else (exc.stdout.decode("utf-8", "replace") if exc.stdout else "")
    stderr = (exc.stderr or "") if isinstance(exc.stderr, str) else (exc.stderr.decode("utf-8", "replace") if exc.stderr else "")
    stderr = (stderr + f"\n[timed out after {timeout}s]").strip()
```

模型看到的文本输出 `exit_code: timeout`，trace 里的 `result_payload` 则带有 `{"timed_out": true, "exit_code": null, ...}`，可以直接做聚合，不用解析文本。

### 已落地 vs 未落地

| 项目 | 本轮状态 |
|---|---|
| POSIX rlimit（CPU/内存/进程数/文件大小） | ✅ 已落地 |
| Windows 进程组隔离 | ✅ 已落地 |
| TimeoutExpired 结构化处理 | ✅ 已落地 |
| 写路径白名单（事后 `git status` 审计） | ⬜ 未落地 |
| 容器 / bwrap 重量层沙箱 | ⬜ 未落地（设计复杂度高，留作可选） |

---

## 改动三：工具结果结构化

### 改动前

每个 `tool_*` 函数返回 `str`，runtime 原样塞进 history 和 trace 的 `result` 字段。要从 trace 里做聚合分析（非零退出占比、读文件行数分布、patch 改动量等），只能正则解文本。

### 改动后

**3-A `ToolResult` dataclass**

位置：`tools.py`

```67:84:pico/tools.py
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
```

**3-B 七个 `tool_*` 函数全部改为返回 `ToolResult`**

各工具的 `payload` 字段：

| 工具 | payload 关键字段 |
|---|---|
| `list_files` | `path`, `entry_count`, `truncated` |
| `read_file` | `path`, `start`, `end`, `lines_returned`, `lines_total`, `truncated` |
| `search` | `pattern`, `path`, `match_count`, `files_hit`, `backend`, `truncated` |
| `run_shell` | `exit_code`, `timed_out`, `stdout_bytes`, `stderr_bytes`, `timeout_seconds`, `sandbox` |
| `write_file` | `path`, `bytes_written`, `char_count`, `created` |
| `patch_file` | `path`, `bytes_removed`, `bytes_added`, `line_delta` |
| `delegate` | `child_run_id`, `max_steps`, `answer_chars` |

`text` 字段保持旧版格式不变，模型看到的完全一样。

**3-C `_wrap_runner()` 和 `_coerce_tool_result()` 做注册层兜底**

位置：`tools.py` → `_wrap_runner()`、`_coerce_tool_result()`

工具执行函数统一经 `_wrap_runner` 注册，注册时会包一层 `_coerce_tool_result`：如果某个工具（包括用户自定义工具）仍返回纯 `str`，自动包成 `ToolResult(text=..., payload={})`，调用方拿到的总是 `ToolResult`。

**3-D `tool_executed` trace 事件注入 `result_payload`**

位置：`runtime.py` → `ask()` 主循环

```2004:2020:pico/runtime.py
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
```

`result`（截断文本）和 `result_payload`（完整结构化）并存于同一条 trace 事件里，互不影响。所有 `_last_tool_result_metadata` 分支（`rejected` / `ok` / `partial_success` / `error`）都补全了 `capabilities`、`tool_result_payload`、`tool_result_ok` 三个字段，确保 trace 聚合时字段总是存在。

`run_tool()` 里的 `run_shell` 特殊处理也同步升级：现在优先读 `payload["exit_code"]` / `payload["timed_out"]` 判断成功/失败，有 payload 时不再依赖正则解文本。

**3-E 旁路入口 `tool_*` 方法保持返回 `str`**

位置：`runtime.py` → `Pico.tool_list_files()` 等七个方法

直接旁路调用（测试和内部代码）期望拿到 `str`，新增 `_unwrap_tool_result()` 把 `ToolResult.text` 抽出来返回：

```2493:2520:pico/runtime.py
@staticmethod
def _unwrap_tool_result(value):
    if isinstance(value, toolkit.ToolResult):
        return value.text
    return value

def tool_list_files(self, args):
    return self._unwrap_tool_result(toolkit.tool_list_files(self, args))

def tool_read_file(self, args):
    return self._unwrap_tool_result(toolkit.tool_read_file(self, args))
# ... 其他工具同理
```

---

## 一条 `tool_executed` trace 事件的完整字段（改动后）

```json
{
  "event": "tool_executed",
  "phase": "act",
  "name": "read_file",
  "args": {"path": "README.md", "start": 1, "end": 10},
  "result": "# README.md\n   1: ...",
  "result_chars": 342,
  "result_payload": {
    "path": "README.md",
    "start": 1,
    "end": 10,
    "lines_returned": 10,
    "lines_total": 188,
    "truncated": true
  },
  "capabilities": ["read"],
  "risk_level": "low",
  "read_only": true,
  "tool_status": "ok",
  "tool_result_ok": true,
  "tool_error_code": "",
  "workspace_changed": false,
  "duration_ms": 3,
  "created_at": "..."
}
```

`result` 给人看，`result_payload` 给脚本看，`capabilities` 记录本次副作用面，三类信息在同一条事件里，trace 聚合不需要跨事件推断。

---

## 完整改动清单

| 文件 | 改动内容 |
|---|---|
| `pico/tools.py` | 新增 `CAP_*` 常量、`RISKY_CAPABILITIES`、`is_risky()`；新增 `ToolResult` dataclass 和 `_coerce_tool_result()`；`BASE_TOOL_SPECS` / `DELEGATE_TOOL_SPEC` 从 `risky` 迁移到 `capabilities`；新增 `_finalize_spec()`、`_wrap_runner()`；新增沙箱配额常量和 `_shell_preexec_factory()`；七个 `tool_*` 函数全部改为返回 `ToolResult` |
| `pico/runtime.py` | 新增 `_resolve_capability_policy()`；`approve()` 增加 `capabilities` 参数；`build_prefix()` 渲染 `[caps:... | ...]`；`tool_signature()` 纳入 `capabilities`；`run_tool()` 里的 `ToolResult` 拆包逻辑；`tool_executed` trace 事件补 `result_payload` / `result_chars` / `capabilities`；七个旁路入口增加 `_unwrap_tool_result()` |

---

## 相关文件索引

| 文件 | 说明 |
|---|---|
| `pico/tools.py` → 顶部常量区 | capability 体系、`ToolResult`、沙箱配额 |
| `pico/tools.py` → `_finalize_spec` / `build_tool_registry` | 工具注册时烘焙 `risky` 派生字段 |
| `pico/tools.py` → `_shell_preexec_factory` | POSIX rlimit 注入逻辑 |
| `pico/tools.py` → `tool_run_shell` | shell 沙箱执行 + ToolResult payload |
| `pico/runtime.py` → `_resolve_capability_policy` | approval_policy 按 capability 分档 |
| `pico/runtime.py` → `approve` | 审批入口，消费 `_resolve_capability_policy` |
| `pico/runtime.py` → `build_prefix` | 模型看到的工具卡片渲染 |
| `pico/runtime.py` → `ask()` 主循环工具执行段 | `ToolResult` 拆包 + trace 注入 |
| `pico/runtime.py` → `_unwrap_tool_result` / `tool_*` 旁路入口 | 向后兼容层 |
