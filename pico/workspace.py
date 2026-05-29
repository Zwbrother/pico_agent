"""工作区快照工具。

这个模块负责在 agent 按需读文件之前，先给它一份便宜的"仓库第一印象"。
这份快照刻意保持小而稳定：主要包含 Git 事实和少量白名单项目文档。

## 核心职责
1. 扫描 Git 仓库信息（分支、状态、提交历史）
2. 读取关键项目文档（README.md、AGENTS.md、Makefile 等白名单文件）
3. 生成轻量目录视图（根目录前两层 + 语言分布）
4. 生成 git diff 摘要（改动规模 + 集中目录，帮助模型快速定位工作焦点）
5. 生成工作区文本描述和指纹（用于缓存判断）

## 设计原则
- **轻量**：只收集最关键的信息，避免加载整个仓库
- **稳定**：快照内容相对稳定，适合用作 prompt prefix 的一部分
- **快速**：通过 git 命令快速获取信息，超时保护（5秒）
"""

import re
import subprocess
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

# ============================================================================
# 常量定义
# ============================================================================

# 工具输出的最大长度限制
MAX_TOOL_OUTPUT = 4000

# 历史记录的最大长度限制
MAX_HISTORY = 12000

# 预加载的项目文档白名单（这些文件最可能影响 agent 的行动方式）
# 我们不会预加载整个仓库，只会先给模型一小份"导航包"
DOC_NAMES = (
    "AGENTS.md", "README.md",
    "pyproject.toml", "package.json",
    "Makefile", "justfile",
    "pytest.ini", "tox.ini",
    "Cargo.toml", "go.mod",
)

# 需要忽略的目录名（避免扫描无关文件）
IGNORED_PATH_NAMES = {".git", ".pico", "__pycache__", ".pytest_cache", ".ruff_cache", ".venv", "venv"}


# ============================================================================
# 辅助函数
# ============================================================================

def now():
    """获取当前 UTC 时间的 ISO 格式字符串。
    
    Returns:
        str: ISO 8601 格式的时间戳（如 "2024-01-15T10:30:00+00:00"）
    """
    return datetime.now(timezone.utc).isoformat()


def clip(text, limit=MAX_TOOL_OUTPUT):
    """裁剪文本到指定长度，超出部分用省略号标记。
    
    Args:
        text: 要裁剪的文本
        limit: 最大长度限制
        
    Returns:
        str: 裁剪后的文本，如果超长则附加 truncation 提示
    """
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated {len(text) - limit} chars]"


def middle(text, limit):
    """将文本压缩到指定长度，优先保留首尾部分。
    
    用于在有限空间内显示较长的文本（如路径、commit message），
    中间用 "..." 替代。
    
    Args:
        text: 要压缩的文本
        limit: 最大长度限制
        
    Returns:
        str: 压缩后的文本（已去除换行符）
        
    Examples:
        >>> middle("hello world foo bar", 15)
        'hello...o bar'
    """
    text = str(text).replace("\n", " ")  # 去除换行符
    if len(text) <= limit:
        return text
    if limit <= 3:
        return text[:limit]
    left = (limit - 3) // 2
    right = limit - 3 - left
    return text[:left] + "..." + text[-right:]


# ============================================================================
# 目录视图辅助
# ============================================================================

# 常见扩展名 → 语言标签映射
_EXT_TO_LANG = {
    ".py": "Python", ".pyi": "Python",
    ".ts": "TypeScript", ".tsx": "TypeScript",
    ".js": "JavaScript", ".jsx": "JavaScript",
    ".rs": "Rust", ".go": "Go", ".java": "Java",
    ".c": "C", ".h": "C", ".cpp": "C++", ".hpp": "C++",
    ".toml": "TOML", ".json": "JSON", ".md": "Markdown",
    ".sh": "Shell", ".bash": "Shell",
    ".yaml": "YAML", ".yml": "YAML",
}


def _dir_summary(repo_root, max_lines=35):
    """根目录前两层目录视图，附带语言分布统计。

    只扫描两层深度，跳过 IGNORED_PATH_NAMES 和隐藏条目，保持输出小而稳定。
    语言分布基于二层文件的扩展名统计，不做完整递归扫描。

    Args:
        repo_root: 仓库根目录路径
        max_lines: 输出行数上限（超出时截断并附提示）

    Returns:
        str: 多行目录视图字符串，若扫描失败则返回空字符串
    """
    root = Path(repo_root)
    lines = []
    ext_counts: dict = {}

    def _track(p):
        if p.suffix:
            ext_counts[p.suffix] = ext_counts.get(p.suffix, 0) + 1

    try:
        # 目录优先、字母升序
        entries = sorted(root.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        for entry in entries:
            if entry.name in IGNORED_PATH_NAMES or entry.name.startswith("."):
                continue
            if entry.is_dir():
                try:
                    children = sorted(
                        [
                            c for c in entry.iterdir()
                            if c.name not in IGNORED_PATH_NAMES
                            and not c.name.startswith(".")
                        ],
                        key=lambda c: (c.is_file(), c.name.lower()),
                    )
                    for c in children:
                        if c.is_file():
                            _track(c)
                    names = [c.name + ("/" if c.is_dir() else "") for c in children]
                    preview = "  ".join(names[:7])
                    if len(names) > 7:
                        preview += f"  (+{len(names) - 7})"
                    lines.append(f"  {entry.name}/  {preview}")
                except OSError:
                    lines.append(f"  {entry.name}/")
            else:
                _track(entry)
                lines.append(f"  {entry.name}")
    except OSError:
        return ""

    # 语言分布：按文件数降序
    if ext_counts:
        lang_counts: dict = {}
        for ext, cnt in ext_counts.items():
            lang = _EXT_TO_LANG.get(ext, ext.lstrip(".").upper())
            lang_counts[lang] = lang_counts.get(lang, 0) + cnt
        lang_str = "  ".join(
            f"{lang}({cnt})"
            for lang, cnt in sorted(lang_counts.items(), key=lambda kv: -kv[1])
        )
        lines.append(f"  [langs: {lang_str}]")

    if len(lines) > max_lines:
        lines = lines[:max_lines] + [f"  ... ({len(lines) - max_lines} more)"]

    return "\n".join(lines)


def _parse_diff_stat(stat_text):
    """将 `git diff --stat HEAD` 的输出浓缩成一行摘要。

    从 stat 输出中提取：
    - 汇总行（"N files changed, X insertions(+), Y deletions(-)"）
    - 发生变动的顶层目录列表

    Args:
        stat_text: git diff --stat HEAD 的完整输出

    Returns:
        str: 单行摘要（如 "1 file changed, 4 insertions(+), 4 deletions(-)  [pico/]"），
             若无变动则返回空字符串
    """
    summary = ""
    dirs: set = set()
    for line in stat_text.splitlines():
        stripped = line.strip()
        if re.match(r"\d+ files? changed", stripped):
            summary = stripped
        elif " | " in stripped:
            fp = stripped.split(" | ")[0].strip().replace("\\", "/")
            top = fp.split("/")[0]
            if "/" in fp:
                dirs.add(top + "/")
    if not summary:
        return ""
    if dirs:
        return f"{summary}  [{', '.join(sorted(dirs))}]"
    return summary


# ============================================================================
# WorkspaceContext 类
# ============================================================================

class WorkspaceContext:
    """工作区上下文快照。
    
    这个类封装了仓库的关键信息，包括：
    - Git 状态（分支、提交、变更）
    - git diff 摘要（改动规模 + 集中目录，帮助模型快速定位工作焦点）
    - 轻量目录视图（根目录前两层 + 语言分布）
    - 项目文档（README、配置等）
    
    这些信息会被嵌入到 prompt prefix 中，让模型在第一次交互前就了解
    当前仓库的基本状况。
    
    Attributes:
        cwd: 当前工作目录
        repo_root: Git 仓库根目录
        branch: 当前分支名
        default_branch: 默认分支名（如 main/master）
        status: Git 状态摘要（简短格式）
        recent_commits: 最近 5 条提交记录
        project_docs: 项目文档字典 {相对路径: 内容摘要}
        dir_summary: 根目录前两层目录视图（含语言分布）
        diff_stat: git diff --stat HEAD 的单行摘要（含集中目录），无未提交变更时为空串
    """
    
    def __init__(self, cwd, repo_root, branch, default_branch, status, recent_commits, project_docs, dir_summary="", diff_stat=""):
        """初始化 WorkspaceContext。
        
        Args:
            cwd: 当前工作目录
            repo_root: Git 仓库根目录
            branch: 当前分支名
            default_branch: 默认分支名
            status: Git 状态摘要
            recent_commits: 最近提交列表
            project_docs: 项目文档字典
        """
        self.cwd = cwd
        self.repo_root = repo_root
        self.branch = branch
        self.default_branch = default_branch
        self.status = status
        self.recent_commits = recent_commits
        self.project_docs = project_docs
        self.dir_summary = dir_summary
        self.diff_stat = diff_stat

    @classmethod    ##作为工厂方法（Factory Method），提供多种创建对象的方式
    def build(cls, cwd, repo_root_override=None):
        """构建工作区上下文快照。
        
        这是 WorkspaceContext 的主要工厂方法，负责：
        1. 定位 Git 仓库根目录
        2. 执行 git 命令获取仓库信息
        3. 读取关键项目文档
        4. 组装成 WorkspaceContext 实例
        
        Args:
            cwd: 当前工作目录（可以是子目录）
            repo_root_override: 可选的仓库根目录覆盖（用于测试）
            
        Returns:
            WorkspaceContext: 构建好的工作区上下文
            
        Raises:
            subprocess.TimeoutExpired: 如果 git 命令超时（5秒）
            Exception: 如果 git 命令失败且无法 fallback
            
        ## 执行流程
        ```
        WorkspaceContext.build(cwd)
          │
          ├─> 1. 解析并规范化 cwd 路径
          │
          ├─> 2. 查找 Git 仓库根目录
          │    └─> git rev-parse --show-toplevel
          │         ├─> 成功: 返回 repo_root
          │         └─> 失败: fallback 到 cwd
          │
          ├─> 3. 扫描项目文档
          │    └─> 遍历 DOC_NAMES 白名单
          │         ├─> 在 repo_root 下查找
          │         ├─> 在 cwd 下查找（支持子目录启动）
          │         └─> 读取文件内容并裁剪（最多 1200 字符）
          │
          ├─> 4. 执行 Git 命令获取信息
          │    ├─> git branch --show-current          # 当前分支
          │    ├─> git symbolic-ref .../origin/HEAD   # 默认分支
          │    ├─> git status --short                 # 工作状态
          │    ├─> git log --oneline -5               # 最近提交
          │    └─> git diff --stat HEAD               # 未提交变动摘要
          │
          ├─> 5. 生成目录视图
          │    └─> _dir_summary(repo_root)            # 根目录前两层 + 语言分布
          │
          └─> 6. 组装并返回 WorkspaceContext 实例
        ```
        """
        cwd = Path(cwd).resolve()

        def git(args, fallback=""):
            """执行 git 命令并返回输出。
            
            Args:
                args: git 命令的参数列表（不含 "git"）
                fallback: 命令失败时的回退值
                
            Returns:
                str: git 命令的输出（已去除首尾空白）
                
            Note:
                - 超时时间：5 秒
                - 如果命令失败或超时，返回 fallback 值
                - 空输出会被转换为 fallback 值
            """
            try:
                result = subprocess.run(
                    ["git", *args],
                    cwd=cwd,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    check=True,
                    timeout=5,
                )
                return result.stdout.strip() or fallback
            except Exception:
                return fallback

        # --------------------------------------------------------------------
        # 步骤1: 查找 Git 仓库根目录
        # --------------------------------------------------------------------
        repo_root = (
            Path(repo_root_override).resolve()
            if repo_root_override is not None
            else Path(git(["rev-parse", "--show-toplevel"], str(cwd))).resolve()
        )
        
        # --------------------------------------------------------------------
        # 步骤2: 扫描并读取项目文档
        # --------------------------------------------------------------------
        docs = {}
        # 同时扫描 repo_root 和 cwd，这样在子目录启动时也能看到本地文档；
        # 但用相对路径做 key，避免同一份文档被重复收集。
        for base in (repo_root, cwd):
            for name in DOC_NAMES:
                path = base / name
                if not path.exists():
                    continue
                key = str(path.relative_to(repo_root))
                if key in docs:
                    continue
                docs[key] = clip(path.read_text(encoding="utf-8", errors="replace"), 1200)

        # --------------------------------------------------------------------
        # 步骤3: 执行 Git 命令并组装结果
        # --------------------------------------------------------------------
        return cls(
            cwd=str(cwd),
            repo_root=str(repo_root),
            branch=git(["branch", "--show-current"], "-") or "-",
            default_branch=(
                lambda branch: branch[len("origin/") :] if branch.startswith("origin/") else branch
            )(git(["symbolic-ref", "--short", "refs/remotes/origin/HEAD"], "origin/main") or "origin/main"),
            status=clip(git(["status", "--short"], "clean") or "clean", 1500),
            recent_commits=[line for line in git(["log", "--oneline", "-5"]).splitlines() if line],
            project_docs=docs,
            dir_summary=_dir_summary(repo_root),
            diff_stat=_parse_diff_stat(git(["diff", "--stat", "HEAD"], "")),
        )

    def text(self):
        """将工作区信息格式化为文本。
        
        这段文本会被塞进 prompt prefix，作为相对稳定的基线上下文。
        模型可以通过这些信息了解：
        - 当前在哪个分支
        - 有哪些未提交的变更（status + diff_stat）
        - 仓库的目录结构和语言组成
        - 最近的开发活动是什么
        - 项目的关键文档内容
        
        Returns:
            str: 格式化的工作区描述文本
            
        Example:
            ```
            Workspace:
            - cwd: /path/to/project
            - repo_root: /path/to/project
            - branch: feature-xyz
            - default_branch: main
            - status:
             M src/main.py
            - diff_stat: 1 file changed, 4 insertions(+), 4 deletions(-)  [src/]
            - recent_commits:
            - abc1234 Add feature X
            - dir_tree:
              src/  main.py  utils.py
              tests/  test_main.py
              [langs: Python(3)]
            - project_docs:
            - README.md
              # Project Title
              This is a sample project...
            ```
        """
        commits = "\n".join(f"- {line}" for line in self.recent_commits) or "- none"
        docs = "\n".join(f"- {path}\n{snippet}" for path, snippet in self.project_docs.items()) or "- none"

        parts = [
            "Workspace:",
            f"- cwd: {self.cwd}",
            f"- repo_root: {self.repo_root}",
            f"- branch: {self.branch}",
            f"- default_branch: {self.default_branch}",
            f"- status:\n{self.status}",
        ]
        if self.diff_stat:
            parts.append(f"- diff_stat: {self.diff_stat}")
        parts.append(f"- recent_commits:\n{commits}")
        if self.dir_summary:
            parts.append(f"- dir_tree:\n{self.dir_summary}")
        parts.append(f"- project_docs:\n{docs}")

        return "\n".join(parts)

    def fingerprint(self):
        """计算工作区状态的指纹（SHA256 哈希）。
        
        这个指纹用来判断仓库状态是否发生了足够大的变化，从而决定是否需要重建缓存中的 prompt prefix。
        
        指纹基于以下信息计算：
        - cwd 和 repo_root
        - 分支信息
        - Git 状态（status + diff_stat）
        - 最近提交
        - 项目文档内容
        - 目录视图（dir_summary）
        
        Returns:
            str: 64 字符的 SHA256 哈希值（十六进制）
            
        Note:
            - 如果任何上述信息发生变化，fingerprint 都会改变
            - 用于 prompt cache 的有效性判断
            - 与 prefix hash 配合使用，实现双层缓存策略
        """
        payload = {
            "cwd": self.cwd,
            "repo_root": self.repo_root,
            "branch": self.branch,
            "default_branch": self.default_branch,
            "status": self.status,
            "recent_commits": list(self.recent_commits),
            "project_docs": dict(self.project_docs),
            "dir_summary": self.dir_summary,
            "diff_stat": self.diff_stat,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
