# workspace snapshot 扩展

**所属分支**：`feat/01-启动装配-workspace-扩展`  
**父提交**：`d20333f 启动装配的注释+逻辑图`  
**改动文件**：`pico/workspace.py`（+175 / -29）

---

## 背景

pico 在每次对话前会把「工作区快照」嵌进 prompt prefix，作为模型的开局信息。  
改动前，这份快照只包含：

```
Workspace:
- cwd / repo_root / branch / default_branch
- status:          ← git status --short（只告诉模型哪些文件被动了）
- recent_commits:  ← git log --oneline -5
- project_docs:    ← AGENTS.md / README.md / pyproject.toml / package.json
```

存在三个明显短板：

| 短板 | 影响 |
|---|---|
| 没有目录视图 | 模型第一轮必须调 `list_files` 探路，浪费一个来回 |
| 文档白名单太窄 | Makefile、go.mod、pytest.ini 等关键配置完全不可见 |
| 只有 `git status`，没有改动规模 | 模型不知道当前改了多少、改在哪个目录，接续工作要猜 |

---

## 改动一：文档白名单扩展

**位置**：`workspace.py` → 常量 `DOC_NAMES`

```python
# 改动前
DOC_NAMES = ("AGENTS.md", "README.md", "pyproject.toml", "package.json")

# 改动后
DOC_NAMES = (
    "AGENTS.md", "README.md",
    "pyproject.toml", "package.json",
    "Makefile", "justfile",
    "pytest.ini", "tox.ini",
    "Cargo.toml", "go.mod",
)
```

**逻辑**：`WorkspaceContext.build()` 在 repo_root 和 cwd 下扫描白名单，文件不存在时直接跳过，不会报错也不会增加任何开销。对纯 Python 项目（如本项目）影响为零；对 Rust / Go / 有 Makefile 的仓库，模型开局即可读到构建命令和测试配置，效果比继续往 README 里堆说明文字更直接。

---

## 改动二：轻量目录视图

**位置**：新增 `_dir_summary(repo_root, max_lines=35)` 函数 + 常量 `_EXT_TO_LANG`

### 生成逻辑

```
repo_root/
  ├── 扫描第一层（排除 IGNORED_PATH_NAMES 和隐藏条目）
  │    ├── 目录 → 展开第二层子项（排除同类）
  │    │         最多显示 7 个，超出附 "(+N)"
  │    └── 文件 → 直接列出
  └── 统计所有第二层文件的扩展名，生成语言分布行
       [langs: Python(25)  Markdown(2)  JSON(1)  ...]
```

`max_lines=35` 作为预算上限，超出时截断并附提示，不会撑大 prompt。

### 当前项目的实际输出

```
- dir_tree:
  assets/  screenshots/
  benchmarks/  coding_tasks.json
  docs/  ONBOARDING.md
  extensions/
  pico/  learn_pico/  __init__.py  __main__.py  cli.py  config.py  ...  (+9)
  scripts/  collect_resume_metrics.py  run_large_scale_experiments.py  run_provider_experiments.py
  tests/  fixtures/  test_context_manager.py  test_evaluator.py  ...  (+2)
  ask-flow.canvas.tsx
  pyproject.toml
  README.md
  [langs: Python(25)  Markdown(2)  JSON(1)  TypeScript(1)  TOML(1)]
```

模型拿到这段文字后，能直接看到 `pico/`（核心包）、`tests/`（测试）、`scripts/`（脚本）的全部直接子项，**第一轮不需要再调 `list_files` 探路**。

---

## 改动三：git diff 摘要

**位置**：新增 `_parse_diff_stat(stat_text)` 函数

### 生成逻辑

执行 `git diff --stat HEAD`，将多行输出浓缩为一行：

```
git diff --stat HEAD 原始输出：
  pico/runtime.py | 8 ++++----
  1 file changed, 4 insertions(+), 4 deletions(-)

_parse_diff_stat 处理后：
  1 file changed, 4 insertions(+), 4 deletions(-)  [pico/]
```

解析规则：
- 匹配 `\d+ files? changed` 的行作为汇总句
- 包含 ` | ` 的行提取文件路径，取顶层目录，去重后附在汇总句后
- 无未提交变更时返回空串，`text()` 自动省略该行，**不占 token**

### 与 `git status --short` 的互补关系

| 信息 | `status` | `diff_stat` |
|---|---|---|
| 哪些文件被改了 | ✓ | ✓（隐含） |
| 改了多少行 | ✗ | ✓ |
| 改动集中在哪个目录 | ✗ | ✓ |
| 新增/未追踪文件 | ✓ | ✗ |

两者共存，互为补充。

### 为什么选 `git diff HEAD` 而不是其他

| 命令 | 含义 |
|---|---|
| `git diff` | 仅未暂存的变更 |
| `git diff --cached` | 仅已暂存的变更 |
| `git diff HEAD` | 全部未提交变更（暂存 + 未暂存） |
| `git diff main...HEAD` | 整个分支相对主线的变更 |

选 `git diff HEAD` 原因：与 `git status --short` 口径一致，反映「从上次提交到现在所有的改动」，最直接地回答「我现在改了什么」。

---

## 对 WorkspaceContext 的影响

### 新增字段

```python
class WorkspaceContext:
    # 新增
    dir_summary: str   # 目录视图，空仓库或扫描失败时为 ""
    diff_stat:   str   # diff 摘要，干净工作树时为 ""
```

`__init__` 两个参数均有默认值 `""`，**完全向后兼容**，测试中直接构造 `WorkspaceContext(...)` 的地方不需要改动。

### build() 新增两个调用

```python
return cls(
    ...
    dir_summary = _dir_summary(repo_root),
    diff_stat   = _parse_diff_stat(git(["diff", "--stat", "HEAD"], "")),
)
```

多出一次 `git diff --stat HEAD`（同其他 git 调用，超时上限 5 秒）和一次 `os.scandir` 两层（I/O 量极小）。

### fingerprint() 纳入新字段

```python
payload = {
    ...
    "dir_summary": self.dir_summary,
    "diff_stat":   self.diff_stat,
}
```

任何文件新增/删除（影响 `dir_summary`）或代码修改（影响 `diff_stat`）都会导致 fingerprint 变化，进而触发 prefix 重建。这与现有 `status` 字段的行为一致——**工作区变化 → prefix 刷新**。

### text() 结构变化

改动前用 `textwrap.dedent(f"""...""").strip()`，因为 f-string 里嵌入的变量（`status`、`commits`、`docs`）没有与模板相同的前导空格，`textwrap.dedent` 实际上对中间行无效，造成缩进不一致。

改动后改为列表拼接，每个 section 是一个独立字符串，optional section（`diff_stat`、`dir_summary`）用 `if` 控制是否追加，消除了空行噪声：

```python
parts = [
    "Workspace:",
    f"- cwd: {self.cwd}",
    ...
    f"- status:\n{self.status}",
]
if self.diff_stat:
    parts.append(f"- diff_stat: {self.diff_stat}")
parts.append(f"- recent_commits:\n{commits}")
if self.dir_summary:
    parts.append(f"- dir_tree:\n{self.dir_summary}")
parts.append(f"- project_docs:\n{docs}")

return "\n".join(parts)
```

---

## 完整 workspace snapshot 示例（本项目）

```
Workspace:
- cwd: C:\pico
- repo_root: C:\pico
- branch: feat/01-启动装配-workspace-扩展
- default_branch: main
- status:
M pico/models.py
- diff_stat: 4 files changed, 99 insertions(+), 21 deletions(-)  [pico/]
- recent_commits:
- c560023 01启动装配扩展: workspace snapshot 补目录视图、diff摘要、文档白名单
- 532048b ask()的注释+逻辑图
- d20333f 启动装配的注释+逻辑图
- dir_tree:
  assets/  screenshots/
  benchmarks/  coding_tasks.json
  docs/  ONBOARDING.md
  extensions/  01-workspace-snapshot-扩展.md
  pico/  learn_pico/  __init__.py  __main__.py  cli.py  config.py  ...  (+9)
  scripts/  collect_resume_metrics.py  run_large_scale_experiments.py  run_provider_experiments.py
  tests/  fixtures/  test_context_manager.py  ...  (+2)
  ask-flow.canvas.tsx
  pyproject.toml
  README.md
  [langs: Python(25)  Markdown(2)  JSON(1)  TypeScript(1)  TOML(1)]
- project_docs:
- README.md
  ...（裁剪至 1200 字符）
- pyproject.toml
  ...
```

---

## 相关文件索引

| 文件 | 说明 |
|---|---|
| `pico/workspace.py` | 本次全部改动所在 |
| `pico/runtime.py` → `build_prefix()` | 调用 `self.workspace.text()`，将快照嵌入 prompt prefix |
| `pico/runtime.py` → `refresh_prefix()` | 调用 `WorkspaceContext.build()` + `fingerprint()` 判断是否刷新 |
| `docs/ONBOARDING.md` → §3 架构分层 | `workspace.py` 所属的「记忆与上下文层」说明 |
