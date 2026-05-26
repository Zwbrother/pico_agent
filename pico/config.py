"""Project-local configuration helpers.

这个模块负责加载和管理项目级配置，主要是 .env 文件。

## 核心职责
1. 查找项目根目录的 .env 文件（向上递归搜索）
2. 解析 .env 文件的键值对（支持注释、引号、export 前缀）
3. 将配置加载到 os.environ 中
4. 按优先级读取 provider 相关的环境变量

## 设计原则
- **自动发现**：从当前目录向上递归查找 .env 文件
- **安全解析**：严格校验变量名格式，防止注入
- **灵活覆盖**：支持 override 模式和非覆盖模式
"""

import os
import re
from pathlib import Path

# 环境变量名的合法格式模式（必须以字母或下划线开头，只包含字母、数字、下划线）
ENV_KEY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _strip_quotes(value):
    """去除字符串值两端的引号（单引号或双引号）。
    
    Args:
        value: 可能包含引号的字符串
        
    Returns:
        str: 去除引号后的字符串
        
    Examples:
        >>> _strip_quotes('"hello"')
        'hello'
        >>> _strip_quotes("'world'")
        'world'
        >>> _strip_quotes('no quotes')
        'no quotes'
    """
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _parse_env_line(line):
    """解析 .env 文件的一行，提取变量名和值。
    
    支持的格式：
    - KEY=value
    - KEY="value with spaces"
    - KEY='value with spaces'
    - export KEY=value
    - # comment (被忽略)
    - 空行 (被忽略)
    
    Args:
        line: .env 文件的一行文本
        
    Returns:
        tuple[str, str] or None: (变量名, 值) 元组，如果是注释或空行则返回 None
        
    Raises:
        ValueError: 如果行格式无效或变量名不合法
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    
    # 支持 "export KEY=value" 格式
    if line.startswith("export "):
        line = line[len("export "):].strip()
    
    if "=" not in line:
        raise ValueError(f"invalid .env line: {line}")
    
    name, value = line.split("=", 1)
    name = name.strip()
    
    # 校验变量名格式（防止注入攻击）
    if not ENV_KEY_PATTERN.match(name):
        raise ValueError(f"invalid .env variable name: {name}")
    
    return name, _strip_quotes(value)


def find_project_env(start):
    """从指定目录向上递归查找 .env 文件。
    
    搜索策略：
    1. 从 start 目录开始
    2. 逐级向上搜索父目录
    3. 直到找到第一个 .env 文件或到达文件系统根目录
    
    Args:
        start: 起始搜索路径（可以是文件或目录）
        
    Returns:
        Path or None: 找到的 .env 文件路径，如果没找到则返回 None
        
    Examples:
        >>> # 假设目录结构：/project/src/.env
        >>> find_project_env("/project/src/main.py")
        Path("/project/src/.env")
    """
    current = Path(start).resolve()
    if current.is_file():
        current = current.parent
    
    # 遍历当前目录及所有父目录
    for path in (current, *current.parents):
        env_path = path / ".env"
        if env_path.exists():
            return env_path
    return None


def load_project_env(start, override=True):
    """加载项目 .env 文件到环境变量。
    
    这是项目启动时的关键步骤，确保 API key 等敏感配置可以被正确读取。
    
    Args:
        start: 起始搜索路径（通常是 workspace root）
        override: 是否覆盖已有的环境变量
                  - True: 总是用 .env 的值覆盖 os.environ
                  - False: 只在变量不存在时设置
        
    Returns:
        dict: 加载的所有环境变量 {name: value}
        
    Note:
        - 如果找不到 .env 文件，返回空字典且不报错
        - 解析错误会抛出 ValueError
        - 加载的变量会立即生效（写入 os.environ）
        
    ## 执行流程
    ```
    load_project_env(start, override=True)
      │
      ├─> 1. find_project_env(start)              # 查找 .env 文件
      │    └─> 从 start 向上递归搜索
      │
      ├─> 2. 如果没找到 .env，返回 {}
      │
      └─> 3. 逐行解析 .env 文件
           ├─> _parse_env_line(line)              #   解析每一行
           │    ├─> 跳过注释和空行
           │    ├─> 去除 "export " 前缀
           │    ├─> 分割 KEY=VALUE
           │    ├─> 校验 KEY 格式
           │    └─> 去除 VALUE 的引号
           │
           ├─> 保存到 loaded 字典
           │
           └─> 根据 override 参数写入 os.environ
    ```
    """
    env_path = find_project_env(start)
    if env_path is None:
        return {}
    
    loaded = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_env_line(line)
        if parsed is None:
            continue
        name, value = parsed
        loaded[name] = value
        if override or name not in os.environ:
            os.environ[name] = value
    return loaded


def provider_env(name, legacy_names=(), default=""):
    """按优先级读取 provider 相关的环境变量。
    
    用于获取模型 API key、base URL、model name 等配置。
    
    Args:
        name: 首选的环境变量名（如 "PICO_OPENAI_API_KEY"）
        legacy_names: 备选的环境变量名列表（向后兼容，如 ("OPENAI_API_KEY",)）
        default: 如果所有变量都不存在时的默认值
        
    Returns:
        str: 找到的第一个非空环境变量值，或默认值
        
    Examples:
        >>> # 假设环境中设置了 OPENAI_API_KEY="sk-xxx"
        >>> provider_env("PICO_OPENAI_API_KEY", ("OPENAI_API_KEY",))
        'sk-xxx'
        
        >>> # 假设环境中没有设置任何相关变量
        >>> provider_env("NONEXISTENT", (), "default_value")
        'default_value'
    """
    for env_name in (name, *legacy_names):
        value = os.environ.get(env_name)
        if value:
            return value
    return default
