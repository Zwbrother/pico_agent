"""模型后端适配层。

runtime 只关心一件事：给我一个 prompt，我拿回一段文本。
不同 provider 在 HTTP 接口、响应结构、是否支持 prompt cache 上都有差异，
这些差异都在这里被抹平成统一的 complete() 接口。

## 核心职责
1. 统一模型调用接口（complete方法）
2. 适配不同 provider 的 HTTP API（Ollama/OpenAI/Anthropic）
3. 处理 SSE 流式响应和 JSON 响应
4. 提取 usage 和 cache 元数据
5. 实现重试机制和错误处理

## 支持的 Provider
- **Ollama**: 本地模型服务（/api/generate）
- **OpenAI-compatible**: 兼容 OpenAI Responses API（/v1/responses）
- **Anthropic-compatible**: 兼容 Anthropic Messages API（/v1/messages）

## 设计模式
- **适配器模式**: 抹平不同 provider 的 API 差异
- **策略模式**: 根据 base_url 自动判断是否支持 prompt cache
"""

import json
import time
from http.client import RemoteDisconnected
import urllib.error
import urllib.request

# OpenAI-compatible API 的 User-Agent 标识
OPENAI_COMPATIBLE_USER_AGENT = "pico/0.1"


# ============================================================================
# 测试用假客户端
# ============================================================================

class FakeModelClient:
    """用于单元测试的假模型客户端。
    
    按顺序返回预设的输出，不发起任何网络请求。
    
    Attributes:
        outputs: 预设的输出列表
        prompts: 记录所有接收到的 prompt
        supports_prompt_cache: 始终为 False
        last_completion_metadata: 空的元数据字典
    """
    
    def __init__(self, outputs):
        """初始化假客户端。
        
        Args:
            outputs: 预设的输出字符串列表
        """
        self.outputs = list(outputs)
        self.prompts = []
        self.supports_prompt_cache = False
        self.last_completion_metadata = {}

    def complete(self, prompt, max_new_tokens, **kwargs):
        """返回下一个预设输出。
        
        Args:
            prompt: 输入的 prompt（会被记录）
            max_new_tokens: 最大输出 token（被忽略）
            **kwargs: 其他参数（被忽略）
            
        Returns:
            str: 下一个预设输出
            
        Raises:
            RuntimeError: 如果输出列表已空
        """
        self.prompts.append(prompt)
        if not getattr(self, "last_completion_metadata", None):
            self.last_completion_metadata = {}
        if not self.outputs:
            raise RuntimeError("fake model ran out of outputs")
        return self.outputs.pop(0)


# ============================================================================
# Ollama 模型客户端
# ============================================================================

class OllamaModelClient:
    """Ollama 本地模型客户端。
    
    通过 HTTP POST 调用 Ollama 的 /api/generate 接口。
    
    ## 特点
    - 不支持 prompt cache
    - 使用简单的 JSON 请求/响应格式
    - 需要 ollama serve 正在运行
    
    Attributes:
        model: 模型名称（如 "qwen3.5:4b"）
        host: Ollama 服务地址（如 "http://127.0.0.1:11434"）
        temperature: 采样温度
        top_p: Top-p 采样参数
        timeout: 请求超时时间（秒）
        supports_prompt_cache: 始终为 False
        last_completion_metadata: 空的元数据字典
    """
    
    def __init__(self, model, host, temperature, top_p, timeout):
        """初始化 Ollama 客户端。
        
        Args:
            model: 模型名称
            host: Ollama 服务地址
            temperature: 采样温度（0.0-1.0）
            top_p: Top-p 采样参数（0.0-1.0）
            timeout: 请求超时时间（秒）
        """
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout
        self.supports_prompt_cache = False
        self.last_completion_metadata = {}

    def complete(self, prompt, max_new_tokens, **kwargs):
        """向 Ollama 发起一次模型调用。
        
        Ollama 当前不支持我们这里接入的 prompt cache 语义，
        所以 runtime 传下来的缓存参数会被忽略。
        
        Args:
            prompt: 完整的提示词文本
            max_new_tokens: 最大输出 token 数
            **kwargs: 其他参数（被忽略，包括 prompt_cache_key 等）
            
        Returns:
            str: 模型生成的文本
            
        Raises:
            RuntimeError: 如果 HTTP 请求失败或 Ollama 返回错误
            
        ## 执行流程
        ```
        complete(prompt, max_new_tokens)
          │
          ├─> 1. 构建请求 payload
          │    └─> {model, prompt, stream, raw, think, options}
          │
          ├─> 2. 发送 HTTP POST 到 /api/generate
          │    └─> urllib.request.urlopen()
          │
          ├─> 3. 解析 JSON 响应
          │    └─> data["response"]
          │
          └─> 4. 错误处理
               ├─> HTTPError: 提取错误信息并抛出 RuntimeError
               └─> URLError: 提供友好的连接失败提示
        ```
        """
        # Ollama 不支持 prompt cache，清空元数据
        self.last_completion_metadata = {}
        
        # 构建 Ollama API 的请求体
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,      # 非流式响应
            "raw": False,         # 不使用原始模式
            "think": False,       # 不启用思考模式
            "options": {
                "num_predict": max_new_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
            },
        }
        
        # 创建 HTTP 请求
        request = urllib.request.Request(
            self.host + "/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        
        # 发送请求并处理响应
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama request failed with HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                "Could not reach Ollama.\n"
                "Make sure `ollama serve` is running and the model is available.\n"
                f"Host: {self.host}\n"
                f"Model: {self.model}"
            ) from exc

        if data.get("error"):
            raise RuntimeError(f"Ollama error: {data['error']}")
        return data.get("response", "")


# ============================================================================
# URL 规范化辅助函数
# ============================================================================

def _normalize_versioned_base_url(base_url):
    """规范化 base URL，确保以 /v1 结尾。
    
    Args:
        base_url: 原始的 base URL
        
    Returns:
        str: 规范化后的 URL（以 /v1 结尾）
        
    Examples:
        >>> _normalize_versioned_base_url("https://api.openai.com")
        'https://api.openai.com/v1'
        >>> _normalize_versioned_base_url("https://api.example.com/v1/")
        'https://api.example.com/v1'
    """
    base = str(base_url).rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    return base


# ============================================================================
# OpenAI-compatible 响应解析辅助函数
# ============================================================================

def _extract_openai_text(data):
    """从 OpenAI-compatible 响应中提取文本。
    
    支持多种响应格式：
    1. output_text 字段（Responses API）
    2. output[].content[].text 字段
    3. choices[0].message.content 字段（Chat Completions API）
    
    Args:
        data: 解析后的 JSON 响应数据
        
    Returns:
        str: 提取的文本内容，如果找不到则返回空字符串
    """
    # 优先检查 output_text 字段（Responses API 格式）
    if data.get("output_text"):
        return data["output_text"]

    # 检查 output[].content[].text 格式
    for item in data.get("output", []):
        for content in item.get("content", []):
            if isinstance(content, dict):
                text = content.get("text")
                if text:
                    return text

    # 检查 choices[0].message.content 格式（传统 Chat API）
    choices = data.get("choices", [])
    if choices:
        message = choices[0].get("message", {})
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if text:
                        return text

    return ""


def _extract_openai_text_from_sse(body_text):
    """从 OpenAI-compatible SSE 流中提取文本。
    
    支持的事件类型：
    - response.output_text.delta: 增量文本
    - response.output_text.done: 完成文本
    - response.completed: 完整响应
    
    Args:
        body_text: SSE 格式的响应体文本
        
    Returns:
        str: 提取的完整文本
    """
    last_response = None
    deltas = []
    for line in body_text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            continue
        
        event_type = event.get("type", "")
        
        # 处理增量文本
        if event_type == "response.output_text.delta":
            delta = event.get("delta")
            if isinstance(delta, str):
                deltas.append(delta)
            continue
        
        # 处理完成文本（立即返回）
        if event_type == "response.output_text.done":
            text = event.get("text")
            if isinstance(text, str) and text:
                return text
        
        # 检查 part 字段
        part = event.get("part")
        if isinstance(part, dict):
            text = part.get("text")
            if isinstance(text, str) and text:
                return text
        
        # 检查 item 字段
        item = event.get("item")
        if isinstance(item, dict):
            text = _extract_openai_text({"output": [item]})
            if text:
                return text
        
        # 检查 response 字段
        response = event.get("response")
        if isinstance(response, dict):
            last_response = response
            text = _extract_openai_text(response)
            if text:
                return text
        
        # 直接在事件级别查找文本
        text = _extract_openai_text(event)
        if text:
            return text
    
    # 如果没有找到完成文本，返回累积的增量
    if deltas:
        return "".join(deltas)
    if isinstance(last_response, dict):
        return _extract_openai_text(last_response)
    return ""


def _extract_openai_response_from_sse(body_text):
    """从 OpenAI-compatible SSE 流中提取文本和完整响应数据。
    
    与 _extract_openai_text_from_sse 类似，但同时返回响应对象，
    用于提取 usage 和 cache 元数据。
    
    Args:
        body_text: SSE 格式的响应体文本
        
    Returns:
        tuple[str, dict]: (文本内容, 响应数据字典)
    """
    last_response = None
    deltas = []
    for line in body_text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            continue
        
        # 检查 response 字段
        response = event.get("response")
        if isinstance(response, dict):
            last_response = response
            if event.get("type") == "response.completed":
                text = _extract_openai_text(response)
                if text:
                    return text, response
        
        event_type = event.get("type", "")
        
        # 处理增量文本
        if event_type == "response.output_text.delta":
            delta = event.get("delta")
            if isinstance(delta, str):
                deltas.append(delta)
        
        # 处理完成文本
        elif event_type == "response.output_text.done":
            text = event.get("text")
            if isinstance(text, str) and text:
                return text, last_response or {}
        
        # 其他事件类型，尝试直接提取文本
        else:
            text = _extract_openai_text(event)
            if text:
                return text, event
    
    # 返回累积结果
    if deltas:
        return "".join(deltas), last_response or {}
    if isinstance(last_response, dict):
        return _extract_openai_text(last_response), last_response
    return "", {}


def _extract_usage_cache_details(data):
    """从响应数据中提取 usage 和 cache 详细信息。
    
    把不同 OpenAI-compatible 返回里的 usage 字段整理成统一结构，
    让 runtime/trace/report 不需要关心 provider 细节。
    
    Args:
        data: 解析后的 JSON 响应数据
        
    Returns:
        dict: 统一的 usage 和 cache 信息
            - input_tokens: 输入 token 数
            - output_tokens: 输出 token 数
            - total_tokens: 总 token 数（可选）
            - cached_tokens: 缓存命中的 token 数
            - cache_hit: 是否有缓存命中
    """
    usage = data.get("usage") or {}
    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens"))
    input_details = usage.get("input_tokens_details") or usage.get("prompt_tokens_details") or {}
    cached_tokens = int(input_details.get("cached_tokens") or 0)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": usage.get("total_tokens"),
        "cached_tokens": cached_tokens,
        "cache_hit": cached_tokens > 0,
    }


# ============================================================================
# OpenAI-compatible 模型客户端
# ============================================================================

class OpenAICompatibleModelClient:
    """OpenAI-compatible 模型客户端。
    
    通过 HTTP POST 调用 OpenAI Responses API（/v1/responses）。
    
    ## 支持的 Provider
    - OpenAI API (api.openai.com)
    - Right Codes Codex (www.right.codes/codex/v1)
    - 其他兼容 OpenAI Responses API 的服务
    
    ## 特点
    - 支持 prompt cache（针对特定 provider）
    - 支持 SSE 流式响应和普通 JSON 响应
    - 自动重试（最多 3 次，针对 5xx 错误）
    - 提取 usage 和 cache 元数据
    
    Attributes:
        model: 模型名称（如 "gpt-5.4"）
        base_url: API base URL（已规范化为 /v1 结尾）
        api_key: API 密钥
        temperature: 采样温度
        timeout: 请求超时时间（秒）
        supports_prompt_cache: 是否支持 prompt cache
        last_completion_metadata: 上次调用的元数据
    """
    
    def __init__(self, model, base_url, api_key, temperature, timeout):
        """初始化 OpenAI-compatible 客户端。
        
        Args:
            model: 模型名称
            base_url: API base URL
            api_key: API 密钥（可选）
            temperature: 采样温度（可选）
            timeout: 请求超时时间（秒）
        """
        self.model = model
        self.base_url = _normalize_versioned_base_url(base_url)
        self.api_key = api_key
        self.temperature = temperature
        self.timeout = timeout
        # Prompt cache 是 OpenAI Responses API 的特定功能，当前只在明确支持 prompt cache 语义的后端上启用这条链路，
        # 避免对不支持的后端传一个"看起来统一、其实没意义"的伪参数。
        self.supports_prompt_cache = any(host in self.base_url for host in ("openai.com", "right.codes"))
        self.last_completion_metadata = {}

    def complete(self, prompt, max_new_tokens, prompt_cache_key=None, prompt_cache_retention=None):
        """向 OpenAI-compatible `/responses` 接口发起一次模型调用。

        ## 为什么存在
        runtime 不应该知道 HTTP 细节、SSE 细节、usage 字段长什么样，更不应该自己去判断 prompt cache 参数要不要带。
        这个函数把这些后端细节都包起来，对上层暴露统一的 `complete()` 行为。

        ## 在 agent 链路里的位置
        它位于 `Pico.ask()` 的模型调用阶段，是稳定前缀缓存复用链路真正落到 provider API 的地方。
        
        Args:
            prompt: 完整的提示词文本
            max_new_tokens: 最大输出 token 数
            prompt_cache_key: 可选的 prompt cache key（稳定前缀的 hash）
            prompt_cache_retention: 可选的 cache 保留策略（如 "in_memory"）
            
        Returns:
            str: 模型生成的文本
            
        Raises:
            RuntimeError: 如果 HTTP 请求失败、响应解析失败或模型返回错误
            
        ## 执行流程
        ```
        complete(prompt, max_new_tokens, prompt_cache_key, prompt_cache_retention)
          │
          ├─> 1. 构建请求 payload
          │    ├─> model, input, max_output_tokens, stream
          │    └─> 如果支持 cache: prompt_cache_key, prompt_cache_retention
          │
          ├─> 2. 设置 HTTP headers
          │    ├─> Content-Type: application/json
          │    ├─> Accept: application/json
          │    ├─> User-Agent: pico/0.1
          │    └─> Authorization: Bearer {api_key}（如果有）
          │
          ├─> 3. 发送 HTTP POST 到 /responses（最多重试 3 次）
          │    └─> urllib.request.urlopen()
          │         ├─> 成功: 读取响应体
          │         ├─> 5xx 错误: 指数退避重试（0.5s, 1.0s, 1.5s）
          │         └─> 其他错误: 抛出 RuntimeError
          │
          ├─> 4. 检测响应类型
          │    ├─> SSE (text/event-stream): 调用 _extract_openai_response_from_sse()
          │    └─> JSON: 直接 json.loads()
          │
          ├─> 5. 提取文本内容
          │    ├─> SSE: 从事件流中提取
          │    └─> JSON: 从 response/output_text 中提取
          │
          └─> 6. 提取 usage/cache 元数据
               └─> _extract_usage_cache_details() -> last_completion_metadata
        ```
        """
        self.last_completion_metadata = {}
        
        # 构建 OpenAI Responses API 的请求体
        payload = {
            "model": self.model,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": prompt,
                        }
                    ],
                }
            ],
            "max_output_tokens": max_new_tokens,
            "stream": False,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        
        # runtime 传入的是"稳定前缀"的签名，而不是整段 prompt 的签名。
        # 这样缓存复用针对的是稳定段，不会因为动态 history 每轮变化而失效。
        if self.supports_prompt_cache and prompt_cache_key:
            payload["prompt_cache_key"] = prompt_cache_key
        if self.supports_prompt_cache and prompt_cache_retention:
            payload["prompt_cache_retention"] = prompt_cache_retention

        # 设置 HTTP headers
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": OPENAI_COMPATIBLE_USER_AGENT,
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # 创建 HTTP 请求
        request = urllib.request.Request(
            self.base_url + "/responses",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        
        # 发送请求（最多重试 3 次）
        attempts = 3
        for attempt in range(attempts):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body_text = response.read().decode("utf-8")
                    headers = getattr(response, "headers", {}) or {}
                    content_type = headers.get("Content-Type", "")
                break
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                # 5xx 服务器错误可以重试
                if exc.code >= 500 and attempt < attempts - 1:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise RuntimeError(f"OpenAI-compatible request failed with HTTP {exc.code}: {body}") from exc
            except (urllib.error.URLError, RemoteDisconnected) as exc:
                # 网络连接错误也可以重试
                if attempt < attempts - 1:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise RuntimeError(
                    "Could not reach the OpenAI-compatible backend.\n"
                    f"Base URL: {self.base_url}\n"
                    f"Model: {self.model}"
                ) from exc

        # --------------------------------------------------------------------
        # 解析响应：有些兼容后端返回普通 JSON，有些返回 SSE
        # 这里两种都接住，并尽量统一抽取文本和 usage/cache 元数据
        # --------------------------------------------------------------------
        if content_type.startswith("text/event-stream") or body_text.lstrip().startswith("data:"):
            # SSE 流式响应
            text, response_data = _extract_openai_response_from_sse(body_text)
            if isinstance(response_data, dict) and response_data:
                # 这些元数据会一路传回 runtime，进入 trace 和 report，
                # 用来观察 prompt cache 是否真的命中。
                self.last_completion_metadata = {
                    "prompt_cache_supported": self.supports_prompt_cache,
                    "prompt_cache_key": prompt_cache_key,
                    "prompt_cache_retention": prompt_cache_retention,
                    **_extract_usage_cache_details(response_data),
                }
            if text:
                return text
            raise RuntimeError("OpenAI-compatible error: could not extract text from event stream response")

        # JSON 响应
        try:
            data = json.loads(body_text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "OpenAI-compatible error: backend returned non-JSON content that could not be parsed"
            ) from exc
        if data.get("error"):
            raise RuntimeError(f"OpenAI-compatible error: {data['error']}")
        
        # 提取 usage/cache 元数据
        self.last_completion_metadata = {
            "prompt_cache_supported": self.supports_prompt_cache,
            "prompt_cache_key": prompt_cache_key,
            "prompt_cache_retention": prompt_cache_retention,
            **_extract_usage_cache_details(data),
        }
        return _extract_openai_text(data)


# ============================================================================
# Anthropic-compatible 响应解析辅助函数
# ============================================================================

def _extract_anthropic_text(data):
    """从 Anthropic-compatible 响应中提取文本。
    
    Args:
        data: 解析后的 JSON 响应数据
        
    Returns:
        str: 提取的文本内容，如果找不到则返回空字符串
    """
    # 优先提取 text 类型（标准响应）
    for item in data.get("content", []):
        if isinstance(item, dict) and item.get("type") == "text":
            text = item.get("text")
            if isinstance(text, str) and text:
                return text
    # 兜底：提取 thinking 类型（DeepSeek/Claude 扩展思考模式）
    for item in data.get("content", []):
        if isinstance(item, dict) and item.get("type") == "thinking":
            text = item.get("thinking")
            if isinstance(text, str) and text:
                return text
    return ""


# ============================================================================
# Anthropic-compatible 模型客户端
# ============================================================================

class AnthropicCompatibleModelClient:
    """Anthropic-compatible 模型客户端。
    
    通过 HTTP POST 调用 Anthropic Messages API（/v1/messages）。
    
    ## 支持的 Provider
    - Anthropic API (api.anthropic.com)
    - Right Codes Claude (www.right.codes/claude/v1)
    - DeepSeek API (api.deepseek.com/anthropic)
    - 其他兼容 Anthropic Messages API 的服务
    
    ## 特点
    - 不支持 prompt cache（当前实现）
    - 使用标准的 Messages API 格式
    - 自动重试（最多 3 次，针对 5xx 错误）
    
    Attributes:
        model: 模型名称（如 "claude-sonnet-4-6"）
        base_url: API base URL（已规范化为 /v1 结尾）
        api_key: API 密钥
        temperature: 采样温度
        timeout: 请求超时时间（秒）
        supports_prompt_cache: 始终为 False
        last_completion_metadata: 空的元数据字典
    """
    
    def __init__(self, model, base_url, api_key, temperature, timeout):
        """初始化 Anthropic-compatible 客户端。
        
        Args:
            model: 模型名称
            base_url: API base URL
            api_key: API 密钥
            temperature: 采样温度（可选）
            timeout: 请求超时时间（秒）
        """
        self.model = model
        self.base_url = _normalize_versioned_base_url(base_url)
        self.api_key = api_key
        self.temperature = temperature
        self.timeout = timeout
        self.supports_prompt_cache = False
        self.last_completion_metadata = {}

    def complete(self, prompt, max_new_tokens, prompt_cache_key=None, prompt_cache_retention=None):
        """向 Anthropic-compatible `/messages` 接口发起一次模型调用。
        
        为了保持统一接口，runtime 仍然会传缓存参数进来；
        这里只是显式丢弃，因为当前 Anthropic-compatible 路径没有接缓存复用。
        
        Args:
            prompt: 完整的提示词文本
            max_new_tokens: 最大输出 token 数
            prompt_cache_key: 被忽略（当前不支持）
            prompt_cache_retention: 被忽略（当前不支持）
            
        Returns:
            str: 模型生成的文本
            
        Raises:
            RuntimeError: 如果 HTTP 请求失败、响应解析失败或模型返回错误
            
        ## 执行流程
        ```
        complete(prompt, max_new_tokens)
          │
          ├─> 1. 构建请求 payload
          │    └─> {model, messages, max_tokens, stream, temperature}
          │
          ├─> 2. 设置 HTTP headers
          │    ├─> Content-Type: application/json
          │    ├─> x-api-key: {api_key}
          │    └─> anthropic-version: 2023-06-01
          │
          ├─> 3. 发送 HTTP POST 到 /messages（最多重试 3 次）
          │    └─> urllib.request.urlopen()
          │
          ├─> 4. 解析 JSON 响应
          │    └─> data["content"][0]["text"]
          │
          └─> 5. 错误处理
               ├─> HTTPError: 提取错误信息并抛出 RuntimeError
               └─> URLError: 提供友好的连接失败提示
        ```
        """
        # 显式丢弃缓存参数（当前不支持）
        del prompt_cache_key, prompt_cache_retention
        self.last_completion_metadata = {}
        
        # 构建 Anthropic Messages API 的请求体
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        }
                    ],
                }
            ],
            "max_tokens": max_new_tokens,
            "stream": False,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature

        # 设置 HTTP headers（Anthropic 特有的认证方式）
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }

        # 创建 HTTP 请求
        request = urllib.request.Request(
            self.base_url + "/messages",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        
        # 发送请求（最多重试 3 次）
        attempts = 3
        for attempt in range(attempts):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body_text = response.read().decode("utf-8")
                break
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                # 5xx 服务器错误可以重试
                if exc.code >= 500 and attempt < attempts - 1:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise RuntimeError(f"Anthropic-compatible request failed with HTTP {exc.code}: {body}") from exc
            except (urllib.error.URLError, RemoteDisconnected) as exc:
                # 网络连接错误也可以重试
                if attempt < attempts - 1:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise RuntimeError(
                    "Could not reach the Anthropic-compatible backend.\n"
                    f"Base URL: {self.base_url}\n"
                    f"Model: {self.model}"
                ) from exc

        # 解析 JSON 响应
        try:
            data = json.loads(body_text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "Anthropic-compatible error: backend returned non-JSON content that could not be parsed"
            ) from exc
        if data.get("error"):
            raise RuntimeError(f"Anthropic-compatible error: {data['error']}")
        
        # 提取文本内容
        text = _extract_anthropic_text(data)
        if text:
            return text
        raise RuntimeError("Anthropic-compatible error: could not extract text from response")
