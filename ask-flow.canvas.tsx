import React from "react";
import {
  Stack, Row, Grid, Card, CardHeader, CardBody,
  H1, H2, H3, Text, Pill, Divider, Callout, Table, Code,
  computeDAGLayout, colorPalette, useHostTheme, useCanvasState,
  Button, Spacer,
} from "cursor/canvas";

// ─── Color helpers ────────────────────────────────────────────────────────────

type NodeKind = "entry" | "phase" | "func" | "step" | "branch" | "exit" | "reject";

const KIND_COLOR: Record<NodeKind, string> = {
  entry:  colorPalette.blue,
  phase:  colorPalette.yellow,
  func:   colorPalette.purple,
  step:   colorPalette.gray,
  branch: colorPalette.orange,
  exit:   colorPalette.green,
  reject: colorPalette.pink,
};

const tint = (hex: string, a = "20") => hex.slice(0, 7) + a;

// ─── Node / Edge data ─────────────────────────────────────────────────────────

interface FN { id: string; label: string; sub?: string; kind: NodeKind }

const NODES: FN[] = [
  // ── Kick-off ──────────────────────────────────────────────────────────────
  { id:"ask",           label:"ask(user_message)",                kind:"entry" },
  { id:"set_task",      label:"memory.set_task_summary()",        sub:"写入任务摘要 → prompt 会携带当前目标",         kind:"func" },
  { id:"record_user",   label:"record({role:'user', …})",         sub:"追加到 session.history + 持久化",             kind:"func" },
  { id:"task_state",    label:"TaskState.create()",               sub:"生成 run_id / task_id，全程追踪元数据",        kind:"func" },
  { id:"start_run",     label:"run_store.start_run(task_state)",  sub:"mkdir .pico/runs/{run_id}/",                  kind:"func" },
  { id:"emit_start",    label:"emit_trace('run_started')",        sub:"写入 trace.jsonl 时间线起点",                 kind:"step" },

  // ── Main loop ─────────────────────────────────────────────────────────────
  { id:"loop",          label:"主循环",                           sub:"while tool_steps < max_steps\n   and attempts < max_attempts", kind:"phase" },
  { id:"wts",           label:"run_store.write_task_state()",     sub:"持久化当前 attempts 计数（崩溃恢复）",        kind:"step" },

  // ── Perception ────────────────────────────────────────────────────────────
  { id:"build_prompt",  label:"_build_prompt_and_metadata()",     sub:"组装本轮完整 prompt",                         kind:"func" },
  { id:"refresh",       label:"  refresh_prefix()",               sub:"WorkspaceContext.build() → 比对 fingerprint", kind:"step" },
  { id:"wp_changed",    label:"  workspace 是否变化？",            sub:"fingerprint 不同或 force=True",               kind:"branch" },
  { id:"rebuild_pfx",   label:"  build_prefix() → PromptPrefix",  sub:"重建 prefix + 3重指纹缓存",                   kind:"step" },
  { id:"eval_resume",   label:"  evaluate_resume_state()",        sub:"5层验证 checkpoint 有效性",                   kind:"step" },
  { id:"ctx_build",     label:"  context_manager.build()",        sub:"12000 chars 预算分配 + 超出裁剪",             kind:"step" },
  { id:"chk_triggers",  label:"检查 resume_status / budget",      sub:"partial-stale / workspace-mismatch\n/ budget_reductions → create_checkpoint()", kind:"step" },

  // ── Decision ─────────────────────────────────────────────────────────────
  { id:"emit_model",    label:"emit_trace('model_requested')",    sub:"记录 attempts / tool_steps / cache_key",      kind:"step" },
  { id:"model_call",    label:"model_client.complete(prompt, …)", sub:"发送 prompt + max_new_tokens (可选 cache key)", kind:"func" },
  { id:"parse",         label:"parse(raw)",                       sub:"解析模型原始输出 → (kind, payload)",           kind:"func" },
  { id:"p_json",        label:"  策略1: <tool>{JSON}</tool>",     sub:"json.loads(body) → 校验 dict + name 字段",    kind:"step" },
  { id:"p_xml",         label:"  策略2: <tool name=...>",         sub:"parse_xml_tool() 支持 content/old_text/new_text", kind:"step" },
  { id:"p_final",       label:"  策略3: <final>answer</final>",   sub:"extract() 提取，空 content 返回 retry",       kind:"step" },
  { id:"p_text",        label:"  策略4: 纯文本",                  sub:"非空裸文本 → kind='final'",                   kind:"step" },
  { id:"p_empty",       label:"  策略5: 空响应",                  sub:"→ (retry, retry_notice())",                   kind:"step" },
  { id:"branch",        label:"kind = ?",                                                                             kind:"branch" },

  // ── Action: tool ──────────────────────────────────────────────────────────
  { id:"tool_br",       label:"kind = 'tool'",                    sub:"task_state.record_tool(name)",                kind:"func" },
  { id:"rt_exist",      label:"  [1] 工具存在性",                  sub:"self.tools.get(name)",                        kind:"step" },
  { id:"rt_rej_exist",  label:"  ✗ unknown tool",                  sub:"return error: unknown tool 'x'",             kind:"reject" },
  { id:"rt_validate",   label:"  [2] validate_tool()",            sub:"路径沙箱 / 类型 / 必填 / delegate深度",       kind:"step" },
  { id:"rt_rej_val",    label:"  ✗ invalid arguments",             sub:"return error: invalid arguments + example",  kind:"reject" },
  { id:"rt_repeat",     label:"  [3] repeated_tool_call()",       sub:"history 最近2次 name+args 完全相同？",        kind:"step" },
  { id:"rt_rej_rep",    label:"  ✗ repeated identical call",       sub:"return error: repeated identical call",      kind:"reject" },
  { id:"rt_approve",    label:"  [4] approve()",                  sub:"read_only / auto / never / ask",              kind:"step" },
  { id:"rt_rej_app",    label:"  ✗ approval denied",               sub:"return error: approval denied for x",        kind:"reject" },
  { id:"rt_snap_pre",   label:"  capture_workspace_snapshot() ①", sub:"risky 工具执行前记录文件 SHA256",              kind:"step" },
  { id:"rt_exec",       label:"  [5] tool['run'](args)",          sub:"真正调用工具函数",                            kind:"func" },
  { id:"rt_snap_post",  label:"  capture_workspace_snapshot() ②", sub:"执行后对比 → affected_paths + diff_summary", kind:"step" },
  { id:"rt_memory",     label:"  [6] update_memory_after_tool()", sub:"read→摘要入记忆; write/patch→失效旧摘要",     kind:"step" },
  { id:"rt_note",       label:"  record_process_note_for_tool()", sub:"error/partial_success → 写过程笔记",          kind:"step" },
  { id:"rt_record",     label:"  record({role:'tool', …})",       sub:"结果写入 session.history + 持久化",           kind:"func" },
  { id:"rt_ckpt",       label:"  create_checkpoint(tool_executed)",sub:"保存进度快照供下次恢复",                     kind:"step" },
  { id:"emit_tool",     label:"  emit_trace('tool_executed')",    sub:"记录 name/args/result/duration_ms",           kind:"step" },

  // ── Action: retry ─────────────────────────────────────────────────────────
  { id:"retry_br",      label:"kind = 'retry'",                   sub:"record({role:'assistant', content:错误提示})", kind:"step" },

  // ── Action: final ─────────────────────────────────────────────────────────
  { id:"final_br",      label:"kind = 'final'",                   kind:"exit" },
  { id:"finish",        label:"task_state.finish_success(final)", sub:"status='completed', stop_reason=''",           kind:"func" },
  { id:"promote",       label:"promote_durable_memory()",         sub:"意图检测 → 行匹配 → 安全过滤 → durable memory", kind:"func" },
  { id:"final_ckpt",    label:"create_checkpoint(run_finished)",  kind:"func" },
  { id:"emit_finish",   label:"emit_trace('run_finished')",       sub:"记录 status / stop_reason / duration_ms",     kind:"step" },
  { id:"report",        label:"build_report() → redact → write_report()", sub:"生成脱敏运行摘要 report.json",        kind:"func" },
  { id:"ret",           label:"return final",                     kind:"exit" },

  // ── Abnormal exit ─────────────────────────────────────────────────────────
  { id:"stop",          label:"stop (step / retry limit)",        sub:"promote + checkpoint + report → return final", kind:"exit" },
];

const EDGES = [
  { from:"ask",          to:"set_task" },
  { from:"set_task",     to:"record_user" },
  { from:"record_user",  to:"task_state" },
  { from:"task_state",   to:"start_run" },
  { from:"start_run",    to:"emit_start" },
  { from:"emit_start",   to:"loop" },

  { from:"loop",         to:"wts" },
  { from:"loop",         to:"stop" },
  { from:"wts",          to:"build_prompt" },

  { from:"build_prompt", to:"refresh" },
  { from:"refresh",      to:"wp_changed" },
  { from:"wp_changed",   to:"rebuild_pfx" },
  { from:"wp_changed",   to:"eval_resume" },
  { from:"rebuild_pfx",  to:"eval_resume" },
  { from:"eval_resume",  to:"ctx_build" },
  { from:"ctx_build",    to:"chk_triggers" },
  { from:"chk_triggers", to:"emit_model" },
  { from:"emit_model",   to:"model_call" },
  { from:"model_call",   to:"parse" },

  { from:"parse",        to:"p_json" },
  { from:"parse",        to:"p_xml" },
  { from:"parse",        to:"p_final" },
  { from:"parse",        to:"p_text" },
  { from:"parse",        to:"p_empty" },
  { from:"p_json",       to:"branch" },
  { from:"p_xml",        to:"branch" },
  { from:"p_final",      to:"branch" },
  { from:"p_text",       to:"branch" },
  { from:"p_empty",      to:"branch" },

  { from:"branch",       to:"tool_br" },
  { from:"branch",       to:"retry_br" },
  { from:"branch",       to:"final_br" },

  // run_tool pipeline (happy path)
  { from:"tool_br",      to:"rt_exist" },
  { from:"rt_exist",     to:"rt_rej_exist" },
  { from:"rt_exist",     to:"rt_validate" },
  { from:"rt_validate",  to:"rt_rej_val" },
  { from:"rt_validate",  to:"rt_repeat" },
  { from:"rt_repeat",    to:"rt_rej_rep" },
  { from:"rt_repeat",    to:"rt_approve" },
  { from:"rt_approve",   to:"rt_rej_app" },
  { from:"rt_approve",   to:"rt_snap_pre" },
  { from:"rt_snap_pre",  to:"rt_exec" },
  { from:"rt_exec",      to:"rt_snap_post" },
  { from:"rt_snap_post", to:"rt_memory" },
  { from:"rt_memory",    to:"rt_note" },
  { from:"rt_note",      to:"rt_record" },
  { from:"rt_record",    to:"rt_ckpt" },
  { from:"rt_ckpt",      to:"emit_tool" },
  { from:"emit_tool",    to:"loop" },   // back-edge
  { from:"retry_br",     to:"loop" },   // back-edge

  // success path
  { from:"final_br",     to:"finish" },
  { from:"finish",       to:"promote" },
  { from:"promote",      to:"final_ckpt" },
  { from:"final_ckpt",   to:"emit_finish" },
  { from:"emit_finish",  to:"report" },
  { from:"report",       to:"ret" },
];

const LEGEND: { kind: NodeKind; label: string }[] = [
  { kind:"entry",  label:"入口" },
  { kind:"phase",  label:"循环控制" },
  { kind:"func",   label:"关键函数" },
  { kind:"step",   label:"执行步骤" },
  { kind:"branch", label:"分支判断" },
  { kind:"exit",   label:"出口/终止" },
  { kind:"reject", label:"拒绝/中止" },
];

// ─── Flow diagram ─────────────────────────────────────────────────────────────

const NW = 244, NH = 50;

function FlowDiagram() {
  const theme = useHostTheme();
  const layout = computeDAGLayout({
    nodes: NODES.map(n => ({ id: n.id })),
    edges: EDGES,
    direction: "vertical",
    nodeWidth: NW, nodeHeight: NH,
    rankGap: 22, nodeGap: 18, padding: 20,
  });
  const nodeMap = new Map(NODES.map(n => [n.id, n]));

  return (
    <div style={{ overflowX:"auto", overflowY:"auto", maxHeight:800 }}>
      <svg width={layout.width} height={layout.height} style={{ display:"block" }}>
        <defs>
          <marker id="arr"      markerWidth="8" markerHeight="7" refX="7" refY="3.5" orient="auto">
            <path d="M0,0 L0,7 L8,3.5 z" fill={theme.stroke.primary} />
          </marker>
          <marker id="arr-back" markerWidth="8" markerHeight="7" refX="7" refY="3.5" orient="auto">
            <path d="M0,0 L0,7 L8,3.5 z" fill={theme.accent.primary} />
          </marker>
        </defs>

        {layout.edges.map((e, i) => (
          <path key={i}
            d={`M${e.sourceX},${e.sourceY} C${e.sourceX},${e.sourceY+16} ${e.targetX},${e.targetY-16} ${e.targetX},${e.targetY}`}
            fill="none"
            stroke={e.isBackEdge ? theme.accent.primary : theme.stroke.secondary}
            strokeWidth={e.isBackEdge ? 2 : 1.5}
            strokeDasharray={e.isBackEdge ? "5 3" : undefined}
            markerEnd={e.isBackEdge ? "url(#arr-back)" : "url(#arr)"}
            opacity={0.6}
          />
        ))}

        {layout.nodes.map(ln => {
          const n = nodeMap.get(ln.id);
          if (!n) return null;
          const c = KIND_COLOR[n.kind];
          const hasSub = !!n.sub;
          return (
            <g key={ln.id}>
              <rect x={ln.x} y={ln.y} width={NW} height={NH} rx={5}
                fill={tint(c, "1E")} stroke={c} strokeWidth={1.5} />
              <text x={ln.x + NW/2} y={hasSub ? ln.y+16 : ln.y+NH/2+4}
                textAnchor="middle" fontSize={10.5} fontWeight={600}
                fontFamily="monospace" fill={c}>
                {n.label}
              </text>
              {hasSub && (
                <text x={ln.x + NW/2} y={ln.y+32}
                  textAnchor="middle" fontSize={9} fontFamily="sans-serif"
                  fill={theme.text.tertiary}>
                  {n.sub!.split("\n")[0]}
                </text>
              )}
            </g>
          );
        })}
      </svg>
    </div>
  );
}

// ─── Tab data ─────────────────────────────────────────────────────────────────

type TabId = "diagram" | "resume" | "parse" | "runtool" | "memory" | "checkpoint" | "session";

const TABS: { id: TabId; label: string }[] = [
  { id:"diagram",    label:"调用流程图" },
  { id:"resume",     label:"evaluate_resume_state" },
  { id:"parse",      label:"parse()" },
  { id:"runtool",    label:"run_tool() 防护" },
  { id:"memory",     label:"记忆系统" },
  { id:"checkpoint", label:"Checkpoint 结构" },
  { id:"session",    label:"Session 数据模型" },
];

// ─── Tab: evaluate_resume_state ───────────────────────────────────────────────

function ResumeTab() {
  return (
    <Stack gap={20}>
      <Stack gap={4}>
        <H2>evaluate_resume_state() — 5 层验证与 5 种状态</H2>
        <Text tone="secondary">每次 _build_prompt_and_metadata() 都会调用此函数，判断上次保存的 checkpoint 是否可以安全恢复。</Text>
      </Stack>

      <H3>验证流水线（按顺序执行，遇到失败即确定状态）</H3>
      <Table
        headers={["层", "验证项", "通过条件", "失败 → 状态"]}
        rows={[
          ["0", "清理过期文件摘要", "invalidate_stale_file_summaries() 清理 SHA256 不匹配的缓存", "— (始终执行)"],
          ["1", "checkpoint 是否存在", "session.checkpoints.current_id 非空 + items 有对应记录", "no-checkpoint"],
          ["2", "schema 版本兼容", "checkpoint.schema_version == 'phase1-v1'", "schema-mismatch"],
          ["3", "关键文件新鲜度", "key_files 中每个文件的 SHA256 与 checkpoint 记录一致", "partial-stale"],
          ["4", "运行时身份一致", "11 个字段全部匹配（见下表）", "workspace-mismatch"],
          ["全部通过", "—", "所有验证通过", "full-valid"],
        ].map(r => r.map((c, i) => i === 0
          ? <Pill size="sm" tone={c === "全部通过" ? "success" : "neutral"}>{c}</Pill>
          : <Text size="small" tone={i === 3 ? "secondary" : "primary"}>{c}</Text>
        ))}
        columnAlign={["center","left","left","left"]}
        striped
      />

      <H3>5 种 resume_status 说明</H3>
      <Grid columns={2} gap={14}>
        {[
          { status:"no-checkpoint",      tone:"neutral" as const, title:"全新开始",      desc:"没有任何 checkpoint，从零开始执行任务。新 session 的默认状态。" },
          { status:"full-valid",         tone:"success" as const, title:"可安全恢复",     desc:"所有验证通过。agent 可以直接从上次中断的地方继续，checkpoint 中的上下文完全可信。" },
          { status:"partial-stale",      tone:"warning" as const, title:"文件已变更",     desc:"某些 key_files 的 SHA256 发生了变化（用户在对话之间修改了代码）。agent 会发出警告并创建新 checkpoint，但仍可继续执行。" },
          { status:"workspace-mismatch", tone:"warning" as const, title:"运行时环境变化", desc:"11个身份字段中至少1个不匹配（如切换了 Git 分支、换了模型、改了 approval_policy）。创建新 checkpoint 记录差异，不沿用旧状态。" },
          { status:"schema-mismatch",    tone:"danger"  as const, title:"版本不兼容",     desc:"checkpoint.schema_version ≠ 'phase1-v1'。通常由 Pico 升级引起。旧 checkpoint 完全丢弃，从头开始。" },
        ].map(item => (
          <Card key={item.status}>
            <CardHeader trailing={<Pill tone={item.tone} size="sm">{item.status}</Pill>}>
              {item.title}
            </CardHeader>
            <CardBody><Text size="small" tone="secondary">{item.desc}</Text></CardBody>
          </Card>
        ))}
      </Grid>

      <H3>运行时身份 — 11 个比对字段</H3>
      <Table
        headers={["字段", "含义", "变化场景举例"]}
        rows={[
          ["cwd",                  "工作目录绝对路径",         "在不同项目目录启动 Pico"],
          ["model",                "LLM 模型名称",             "从 gpt-4o 换到 claude-3-5-sonnet"],
          ["model_client",         "客户端类名",               "从 OpenAI 换到 Anthropic 客户端"],
          ["approval_policy",      "危险工具审批策略",         "从 ask 改为 auto（CI 场景）"],
          ["read_only",            "只读模式开关",             "加了 --read-only 参数重启"],
          ["max_steps",            "最大工具步数",             "从 6 改为 12"],
          ["max_new_tokens",       "模型每次最大输出",         "从 512 改为 1024"],
          ["feature_flags",        "功能开关字典",             "关闭了 memory 功能"],
          ["shell_env_allowlist",  "Shell 环境变量白名单",     "添加了自定义环境变量"],
          ["workspace_fingerprint","工作区状态指纹",           "切换了 Git 分支"],
          ["tool_signature",       "工具注册表 SHA256",        "启用/禁用了 delegate 工具"],
        ].map(r => r.map((c, i) => i === 0
          ? <Code>{c}</Code>
          : <Text size="small" tone={i === 2 ? "secondary" : "primary"}>{c}</Text>
        ))}
        striped
      />
    </Stack>
  );
}

// ─── Tab: parse ───────────────────────────────────────────────────────────────

function ParseTab() {
  return (
    <Stack gap={20}>
      <Stack gap={4}>
        <H2>parse(raw) — 5 种解析策略</H2>
        <Text tone="secondary">模型输出是自由文本，parse() 按优先级从上到下尝试，返回第一个匹配的 (kind, payload)。</Text>
      </Stack>

      <Table
        headers={["优先级", "匹配条件", "kind", "payload", "失败处理"]}
        rows={[
          [
            "1",
            "<tool>...</tool> 且比 <final> 靠前",
            "tool",
            '{"name":"list_files","args":{"path":"."}}',
            "JSON 解析失败 / 非 dict / 无 name → (retry, retry_notice)"
          ],
          [
            "2",
            "<tool name=... > 属性格式 且比 <final> 靠前",
            "tool",
            '{"name":"write_file","args":{"path":"a.py","content":"..."}}',
            "parse_xml_tool() 返回 None → (retry, retry_notice)"
          ],
          [
            "3",
            "<final>...</final>",
            "final",
            "标签内的纯文本",
            "空内容 → (retry, 'model returned an empty <final> answer')"
          ],
          [
            "4",
            "无标签，非空纯文本",
            "final",
            "strip 后的原始文本",
            "—（不会失败）"
          ],
          [
            "5",
            "空字符串",
            "retry",
            "retry_notice('model returned an empty response')",
            "—"
          ],
        ].map(r => r.map((c, i) => i === 0
          ? <Pill size="sm">{c}</Pill>
          : i === 2
            ? <Pill size="sm" tone={c === "tool" ? "info" : c === "retry" ? "warning" : "success"}>{c}</Pill>
            : <Text size="small" tone={i === 4 ? "secondary" : "primary"}>{c}</Text>
        ))}
        columnAlign={["center","left","center","left","left"]}
        striped
      />

      <Divider />
      <H3>XML 工具调用格式详解（策略2）</H3>
      <Text tone="secondary" size="small">XML 格式专为多行内容设计，避免了 JSON 字符串里的转义问题。parse_xml_tool() 支持以下子标签：</Text>
      <Table
        headers={["子标签", "用途", "对应工具"]}
        rows={[
          ["<content>",  "文件内容（多行）",         "write_file"],
          ["<old_text>", "要替换的原始文本",          "patch_file"],
          ["<new_text>", "替换后的新文本",            "patch_file"],
          ["<command>",  "Shell 命令",               "run_shell"],
          ["<task>",     "委托给子 agent 的任务描述", "delegate"],
          ["<pattern>",  "搜索模式",                 "search"],
          ["<path>",     "文件路径（body 形式）",     "通用"],
        ].map(r => r.map((c, i) => i === 0
          ? <Code>{c}</Code>
          : <Text size="small" tone={i === 2 ? "secondary" : "primary"}>{c}</Text>
        ))}
        striped
      />

      <Divider />
      <H3>retry_notice 的作用</H3>
      <Callout tone="info" title="设计意图">
        <Text size="small">
          retry_notice 不是报错，而是一条精心设计的纠错提示，会被 record() 到 session.history 里。
          模型下一轮读到 history 时，能看到自己上次输出哪里不对，从而修正格式。
          最大尝试次数 = max(max_steps×3, max_steps+4)，保证有足够机会纠错。
        </Text>
      </Callout>
    </Stack>
  );
}

// ─── Tab: run_tool ────────────────────────────────────────────────────────────

function RunToolTab() {
  return (
    <Stack gap={20}>
      <Stack gap={4}>
        <H2>run_tool(name, args) — 完整防护流水线</H2>
        <Text tone="secondary">所有工具调用都必须通过这 6 层护栏，任何一层失败都会返回错误字符串而非抛异常，让模型能够继续消费错误反馈。</Text>
      </Stack>

      <Table
        headers={["层", "检查点", "通过条件", "拒绝返回值", "安全事件类型"]}
        rows={[
          ["1", "工具存在性",            "name 在 self.tools 注册表中",                              "error: unknown tool 'x'",                          "—"],
          ["2", "validate_tool()",       "路径在 workspace 内 + 类型正确 + 必填参数存在 + 深度未超限", "error: invalid arguments for x: …\\nexample: …",   "path_escape（路径逃逸时）"],
          ["3", "repeated_tool_call()",  "history 最近2次 tool 事件 name/args 不完全相同",            "error: repeated identical tool call for x",          "—"],
          ["4", "approve()",             "审批策略允许（见下方策略矩阵）",                             "error: approval denied for x",                       "read_only_block / approval_denied"],
          ["5", "tool['run'](args) 执行", "无异常，clip() 截断输出到合理长度",                        "error: tool x failed: {exc}",                        "path_escape（异常含路径逃逸时）"],
          ["6", "update_memory_after_tool()", "读/写文件后自动更新 working memory（无显式失败）",    "—（副作用，不拒绝）",                                 "—"],
        ].map(r => r.map((c, i) => i === 0
          ? <Pill size="sm" tone="info">{c}</Pill>
          : i === 3
            ? <Code style={{ fontSize: 10 }}>{c}</Code>
            : i === 4
              ? <Pill size="sm" tone={c === "—" ? "neutral" : "warning"}>{c}</Pill>
              : <Text size="small" tone={i === 2 ? "secondary" : "primary"}>{c}</Text>
        ))}
        columnAlign={["center","left","left","left","left"]}
        striped
      />

      <Divider />
      <H3>approve() 策略矩阵</H3>
      <Table
        headers={["条件（优先级从高到低）", "结果", "适用场景"]}
        rows={[
          ["read_only = True",         "拒绝（False）", "只读分析模式，无论什么策略都不能写文件"],
          ["approval_policy = 'auto'", "通过（True）",  "CI / 无人值守，全部自动批准"],
          ["approval_policy = 'never'","拒绝（False）", "纯探索，绝对禁止修改工作区"],
          ["approval_policy = 'ask'",  "交互询问",      "交互式 CLI，每次提示 [y/N]，EOFError 视为拒绝"],
        ].map(r => r.map((c, i) => i === 1
          ? <Pill size="sm" tone={c.includes("拒绝") ? "warning" : c === "交互询问" ? "info" : "success"}>{c}</Pill>
          : <Text size="small" tone={i === 2 ? "secondary" : "primary"}>{c}</Text>
        ))}
        striped
      />

      <Divider />
      <H3>工作区快照对比（第5层的详细机制）</H3>
      <Callout tone="info" title="为什么需要快照？">
        <Stack gap={6}>
          <Text size="small">执行前后各做一次 <Code>capture_workspace_snapshot()</Code>：遍历 workspace 下所有文件（忽略 .pico / node_modules 等），计算每个文件的 SHA256，返回路径→哈希映射。</Text>
          <Text size="small"><Code>diff_workspace_snapshots(before, after)</Code> 对比两个映射，生成 <Code>affected_paths</Code>（变化文件列表）和 <Code>diff_summary</Code>（created/modified/deleted 描述列表），写入 trace 事件。</Text>
          <Text size="small">对于 <Code>run_shell</Code>：额外解析 exit_code。exit_code≠0 且有文件变化 → <Code>partial_success</Code>；exit_code≠0 且无变化 → <Code>error</Code>。</Text>
        </Stack>
      </Callout>

      <Divider />
      <H3>update_memory_after_tool() — 差异化处理</H3>
      <Table
        headers={["工具", "操作", "对 working memory 的影响"]}
        rows={[
          ["read_file",               "读取文件",    "remember_file(path) 加入 recent_files\nsummarize_read_result(result) → set_file_summary()\nappend_note(摘要, tags=[path])"],
          ["write_file / patch_file", "修改文件",    "remember_file(path) 加入 recent_files\ninvalidate_file_summary(path) 清除旧摘要（防止过期信息）"],
          ["其他工具",                "非文件操作",  "args 中无 path 字段 → 跳过，不更新记忆"],
        ].map(r => r.map((c, i) => i === 0
          ? <Code>{c}</Code>
          : i === 2
            ? <Text size="small" tone="secondary">{c}</Text>
            : <Text size="small">{c}</Text>
        ))}
        striped
      />
    </Stack>
  );
}

// ─── Tab: memory ──────────────────────────────────────────────────────────────

function MemoryTab() {
  return (
    <Stack gap={20}>
      <Stack gap={4}>
        <H2>分层记忆系统（LayeredMemory）</H2>
        <Text tone="secondary">记忆系统负责在 prompt 有限的上下文预算内携带高价值信息，避免反复重读文件。</Text>
      </Stack>

      <Grid columns={3} gap={16}>
        {[
          {
            title:"working_memory（工作记忆）",
            color: colorPalette.blue,
            fields:[
              { k:"recent_files",    v:"最近读/写过的文件路径列表（≤8个），每次工具执行后更新" },
              { k:"file_summaries",  v:"文件内容摘要缓存（路径→摘要），read_file 后自动生成，write/patch 后失效" },
              { k:"episodic_notes",  v:"临时过程笔记（来自 append_note()），错误/部分成功时自动写入" },
              { k:"task_summary",    v:"当前任务目标描述，每次 ask() 开始时由 set_task_summary() 设置" },
            ]
          },
          {
            title:"durable_memory（长期记忆）",
            color: colorPalette.purple,
            fields:[
              { k:"project-conventions", v:"项目约定：以 'Project convention:' 开头的行" },
              { k:"key-decisions",       v:"关键决策：以 'Decision:' 开头的行" },
              { k:"dependency-facts",    v:"依赖事实：以 'Dependency:' 开头的行" },
              { k:"user-preferences",    v:"用户偏好：以 'Preference:' 开头的行" },
            ]
          },
          {
            title:"task_summary（任务摘要）",
            color: colorPalette.green,
            fields:[
              { k:"来源",    v:"每次 ask() 的 user_message" },
              { k:"用途",    v:"注入 prompt 让模型始终知道当前目标" },
              { k:"生命周期",v:"单轮 ask() 内有效，下次 ask() 会覆盖" },
              { k:"存储位置",v:"session.memory.task_summary" },
            ]
          },
        ].map(layer => (
          <Card key={layer.title}>
            <CardHeader trailing={<div style={{ width:10, height:10, borderRadius:"50%", background:layer.color }} />}>
              {layer.title}
            </CardHeader>
            <CardBody>
              <Stack gap={8}>
                {layer.fields.map(f => (
                  <Stack key={f.k} gap={2}>
                    <Code>{f.k}</Code>
                    <Text size="small" tone="secondary">{f.v}</Text>
                  </Stack>
                ))}
              </Stack>
            </CardBody>
          </Card>
        ))}
      </Grid>

      <Divider />
      <H3>记忆写入时机一览</H3>
      <Table
        headers={["触发时机", "写入目标", "写入内容"]}
        rows={[
          ["ask() 开始",          "task_summary",      "用户当前请求文本"],
          ["read_file 成功",       "file_summaries",    "文件内容的自动摘要（前N行 + 结构信息）"],
          ["read_file 成功",       "recent_files",      "文件路径（保持最近8个）"],
          ["write_file / patch",  "recent_files",      "文件路径"],
          ["write_file / patch",  "file_summaries",    "失效旧摘要（删除 key）"],
          ["run_tool 出错/部分成功","episodic_notes",   "过程笔记（如 'run_shell error on src/'）"],
          ["ask() 正常结束",       "durable memory",    "从 final_answer 中提取的长期知识条目"],
          ["evaluate_resume_state","file_summaries",   "清理 SHA256 不匹配的过期摘要"],
        ].map(r => r.map((c, i) => i === 1
          ? <Pill size="sm" tone="info">{c}</Pill>
          : <Text size="small" tone={i === 2 ? "secondary" : "primary"}>{c}</Text>
        ))}
        striped
      />

      <Divider />
      <H3>长期记忆提升流程（promote_durable_memory）</H3>
      <Callout tone="info" title="触发条件">
        <Text size="small">用户消息中含有 capture / remember / save / store / persist / note / 记住 / 保存 / 记录 / 沉淀 等意图词时才会扫描。</Text>
      </Callout>
      <Table
        headers={["阶段", "操作", "细节"]}
        rows={[
          ["1 意图检测", "正则匹配用户消息", "DURABLE_MEMORY_INTENT_PATTERN（英文）+ ZH_PATTERN（中文），两个都不匹配则直接返回 ([], [])"],
          ["2 行匹配",   "扫描 final_answer 每一行", "逐行对 8 个模式（英中各4类）做 pattern.match()，提取 (topic, text) 对"],
          ["3 安全过滤", "reject_durable_reason()", "拒绝：空文本 / 含 API key形状 / 含 <redacted> / checkpoint类前缀 / 含 stdout/traceback / 长度>220"],
          ["4 去重写入", "memory.promote_durable()", "同 topic 下相同内容不重复；新条目替换同 topic 的旧条目（superseded 列表记录被替换项）"],
        ].map(r => r.map((c, i) => i === 0
          ? <Pill size="sm">{c}</Pill>
          : <Text size="small" tone={i === 2 ? "secondary" : "primary"}>{c}</Text>
        ))}
        striped
      />
    </Stack>
  );
}

// ─── Tab: checkpoint ─────────────────────────────────────────────────────────

function CheckpointTab() {
  return (
    <Stack gap={20}>
      <Stack gap={4}>
        <H2>Checkpoint 结构与生命周期</H2>
        <Text tone="secondary">Checkpoint 是 Pico "断点续传"的核心数据结构，形成链表，保存每一步执行的完整上下文快照。</Text>
      </Stack>

      <H3>checkpoint 字典结构（完整字段）</H3>
      <Table
        headers={["字段", "类型", "含义", "示例值"]}
        rows={[
          ["checkpoint_id",         "str",  "唯一标识符",                   "ckpt_a3f5b2c1"],
          ["parent_checkpoint_id",  "str",  "父节点（链表结构）",            "ckpt_prev1234（首个为空串）"],
          ["schema_version",        "str",  "结构版本（兼容性检查）",        "phase1-v1"],
          ["created_at",            "str",  "创建时间 ISO 格式",             "2026-05-28T16:30:00"],
          ["current_goal",          "str",  "用户原始请求",                  "帮我重构认证模块"],
          ["completed",             "list", "已完成的任务项",               '["读取 auth.py","分析依赖"]'],
          ["excluded",              "list", "排除的文件/步骤（预留字段）",   "[]"],
          ["current_blocker",       "str",  "当前阻塞原因",                  "step_limit_reached / ''"],
          ["next_step",             "str",  "推断的下一步行动",              "Continue from latest checkpoint"],
          ["key_files",             "list", "关键文件列表（含新鲜度）",      '[{"path":"src/auth.py","freshness":"abc123"}]'],
          ["freshness",             "dict", "文件路径 → SHA256 快速查找表",  '{"src/auth.py":"abc123…"}'],
          ["summary",               "str",  "简要描述（trigger: 截断请求）", "tool_executed: 帮我重构认证…"],
          ["runtime_identity",      "dict", "11 字段运行时身份指纹",         '{"model":"gpt-4o","cwd":"/pico",…}'],
        ].map(r => r.map((c, i) => i === 0
          ? <Code>{c}</Code>
          : i === 1
            ? <Pill size="sm" tone="neutral">{c}</Pill>
            : <Text size="small" tone={i === 3 ? "secondary" : "primary"}>{c}</Text>
        ))}
        striped
      />

      <Divider />
      <H3>create_checkpoint() 触发时机</H3>
      <Table
        headers={["触发点", "trigger 值", "触发条件", "目的"]}
        rows={[
          ["主循环 - resume_status",    "freshness_mismatch",   "resume_status == 'partial-stale'",    "标记文件已变化的时间点"],
          ["主循环 - resume_status",    "workspace_mismatch",   "resume_status == 'workspace-mismatch'","标记环境不匹配的时间点"],
          ["主循环 - context_manager",  "context_reduction",    "prompt_metadata.budget_reductions 非空","压缩前保存完整状态"],
          ["每次工具执行后",             "tool_executed",        "每次 run_tool() 成功或失败后",         "保存工具执行进度"],
          ["正常结束",                   "run_finished",         "模型给出最终答案",                     "保存完成状态"],
          ["步数/重试超限",              "step_limit_reached 等","达到终止条件",                         "保存中断点供下次继续"],
        ].map(r => r.map((c, i) => i === 1
          ? <Pill size="sm" tone="info">{c}</Pill>
          : <Text size="small" tone={i === 2 ? "secondary" : "primary"}>{c}</Text>
        ))}
        striped
      />

      <Callout tone="info" title="链表结构">
        <Text size="small">
          每个 checkpoint 通过 <Code>parent_checkpoint_id</Code> 指向前一个，形成完整的执行链：
          ckpt_A → ckpt_B → ckpt_C → … → ckpt_current。
          session.checkpoints.current_id 始终指向最新的 checkpoint。
          下次 ask() 时通过 current_checkpoint() 读取并恢复上下文。
        </Text>
      </Callout>
    </Stack>
  );
}

// ─── Tab: session ─────────────────────────────────────────────────────────────

function SessionTab() {
  return (
    <Stack gap={20}>
      <Stack gap={4}>
        <H2>Session 数据模型</H2>
        <Text tone="secondary">session 是整个 Pico 的持久化状态容器，保存到 <Code>.pico/sessions/{"{session_id}"}.json</Code>。</Text>
      </Stack>

      <H3>顶层字段结构</H3>
      <Table
        headers={["字段", "类型", "内容", "写入时机"]}
        rows={[
          ["id",               "str",  "时间戳+UUID 格式 session ID",                           "SessionStore 创建时"],
          ["created_at",       "str",  "ISO 创建时间",                                          "SessionStore 创建时"],
          ["workspace_root",   "str",  "仓库根目录绝对路径",                                    "初始化时"],
          ["history",          "list", "完整对话历史（user/assistant/tool 消息）",               "每次 record() 调用后"],
          ["memory",           "dict", "分层记忆系统序列化状态",                                "工具执行/ask结束/记忆更新时"],
          ["checkpoints",      "dict", "current_id + items 字典（所有 checkpoint）",            "每次 create_checkpoint() 后"],
          ["runtime_identity", "dict", "最新运行时身份指纹（11字段）",                          "evaluate_resume_state() 后"],
          ["resume_state",     "dict", "最近一次的恢复状态评估结果",                            "evaluate_resume_state() 后"],
        ].map(r => r.map((c, i) => i === 0
          ? <Code>{c}</Code>
          : i === 1
            ? <Pill size="sm" tone="neutral">{c}</Pill>
            : <Text size="small" tone={i === 3 ? "secondary" : "primary"}>{c}</Text>
        ))}
        striped
      />

      <Divider />
      <H3>history 条目格式（3 种 role）</H3>
      <Table
        headers={["role", "包含字段", "写入时机", "用途"]}
        rows={[
          ["user",      "role / content / created_at",                    "ask() 开始时",           "用户原始请求，prompt 中的 [user] 段"],
          ["assistant", "role / content / created_at",                    "parse() 成功后 / retry时", "模型回复 / 错误提示，让模型看到上下文"],
          ["tool",      "role / name / args / content / created_at",      "run_tool() 完成后",       "工具调用结果，模型基于此做下一步决策"],
        ].map(r => r.map((c, i) => i === 0
          ? <Pill size="sm" tone={c==="user"?"info":c==="assistant"?"warning":"success"}>{c}</Pill>
          : <Text size="small" tone={i === 3 ? "secondary" : "primary"}>{c}</Text>
        ))}
        striped
      />

      <Divider />
      <H3>trace vs report — 两种输出文件的区别</H3>
      <Grid columns={2} gap={16}>
        <Card>
          <CardHeader trailing={<Pill size="sm">trace.jsonl</Pill>}>逐事件时间线</CardHeader>
          <CardBody>
            <Stack gap={8}>
              <Text size="small" tone="secondary">路径：<Code>.pico/runs/{"{run_id}"}/trace.jsonl</Code></Text>
              <Text size="small" tone="secondary">格式：每行一个 JSON 事件，含 event / created_at / payload</Text>
              <Text size="small">记录的事件类型：</Text>
              {["run_started","prompt_built","checkpoint_created","runtime_identity_mismatch",
                "model_requested","model_parsed","tool_executed","run_finished"].map(e => (
                <Pill key={e} size="sm" tone="neutral">{e}</Pill>
              ))}
              <Text size="small" tone="secondary">适合回答："这一轮 agent 到底做了什么，每一步花了多长时间"</Text>
            </Stack>
          </CardBody>
        </Card>
        <Card>
          <CardHeader trailing={<Pill size="sm">report.json</Pill>}>运行结果摘要</CardHeader>
          <CardBody>
            <Stack gap={8}>
              <Text size="small" tone="secondary">路径：<Code>.pico/runs/{"{run_id}"}/report.json</Code></Text>
              <Text size="small" tone="secondary">格式：单个 JSON 对象，经过 redact_artifact() 脱敏</Text>
              <Text size="small">包含的关键字段：</Text>
              {["status","stop_reason","final_answer","tool_steps","attempts",
                "checkpoint_id","prompt_metadata","durable_promotions","redacted_env"].map(e => (
                <Pill key={e} size="sm" tone="neutral">{e}</Pill>
              ))}
              <Text size="small" tone="secondary">适合回答："这次运行的结论和关键指标是什么"</Text>
            </Stack>
          </CardBody>
        </Card>
      </Grid>
    </Stack>
  );
}

// ─── Main ─────────────────────────────────────────────────────────────────────

export default function AskFlowCanvas() {
  const theme = useHostTheme();
  const [tab, setTab] = useCanvasState<TabId>("tab", "diagram");

  return (
    <Stack gap={0} style={{ background: theme.bg.editor, minHeight:"100vh" }}>

      {/* Header */}
      <Stack gap={4} style={{ padding:"24px 24px 16px" }}>
        <H1>ask(user_message) 完整调用逻辑</H1>
        <Text tone="secondary">Pico Agent 核心控制循环 · runtime.py · 感知 → 决策 → 行动 → 记录</Text>
      </Stack>

      {/* Tab bar */}
      <Row gap={6} style={{ padding:"0 24px 16px", borderBottom:`1px solid ${theme.stroke.tertiary}` }} wrap>
        {TABS.map(t => (
          <Pill key={t.id} active={tab === t.id} tone={tab === t.id ? "info" : "neutral"}
            onClick={() => setTab(t.id)}>
            {t.label}
          </Pill>
        ))}
      </Row>

      {/* Tab content */}
      <Stack gap={0} style={{ padding: 24 }}>

        {tab === "diagram" && (
          <Stack gap={20}>
            <Grid columns="3fr 2fr" gap={20} align="start">
              <Card>
                <CardHeader>调用流程图</CardHeader>
                <CardBody style={{ padding: 16 }}>
                  <Row gap={10} wrap style={{ marginBottom: 12 }}>
                    {LEGEND.map(({ kind, label }) => (
                      <Row key={kind} gap={5} align="center">
                        <div style={{ width:10, height:10, borderRadius:3,
                          background: tint(KIND_COLOR[kind], "28"),
                          border:`1.5px solid ${KIND_COLOR[kind]}` }} />
                        <Text size="small" tone="secondary">{label}</Text>
                      </Row>
                    ))}
                    <Text size="small" tone="tertiary">— 虚线 = 循环回跳</Text>
                  </Row>
                  <FlowDiagram />
                </CardBody>
              </Card>

              <Stack gap={14}>
                {[
                  { tone:"info" as const, title:"阶段 1 · 初始化准备", items:[
                    "set_task_summary() 写入工作记忆，保证每轮 prompt 携带当前目标",
                    "record(user) 写入 session.history，作为对话历史起点",
                    "TaskState.create() 创建 run_id/task_id，全程追踪",
                    "run_store.start_run() 建立 .pico/runs/{run_id}/ 目录",
                    "emit_trace('run_started') 时间线起点",
                  ]},
                  { tone:"warning" as const, title:"阶段 2 · 主循环（感知→决策→行动→记录）", items:[
                    "感知：refresh_prefix（三重指纹缓存）+ evaluate_resume_state（5层验证）+ context_manager.build（12000 chars 预算）",
                    "检查点：partial-stale / workspace-mismatch / budget_reductions 时自动创建 checkpoint",
                    "决策：model_client.complete() + parse() 解析为 5 种输出",
                    "行动(tool)：run_tool() 六层防护 + create_checkpoint('tool_executed')",
                    "行动(retry)：record 错误提示，模型下一轮纠正",
                    "行动(final)：退出循环进入阶段 3",
                  ]},
                  { tone:"success" as const, title:"阶段 3 · 正常结束", items:[
                    "finish_success(final) 标记 completed",
                    "promote_durable_memory() 沉淀长期知识",
                    "create_checkpoint('run_finished') 最终快照",
                    "emit_trace('run_finished') + write_report()",
                    "return final 返回最终答案",
                  ]},
                  { tone:"danger" as const, title:"阶段 3 · 异常终止", items:[
                    "stop_retry_limit：模型反复输出无效格式（attempts ≥ max_steps×3）",
                    "stop_step_limit：max_steps 内任务未完成",
                    "均执行：promote + checkpoint + report，返回停止原因",
                  ]},
                ].map(p => (
                  <Callout key={p.title} tone={p.tone} title={p.title}>
                    <Stack gap={3}>{p.items.map((item, i) => <Text key={i} size="small">{item}</Text>)}</Stack>
                  </Callout>
                ))}
              </Stack>
            </Grid>

            <Divider />
            <H2>停止条件</H2>
            <Grid columns={3} gap={16}>
              {[
                { t:"正常结束",  m:'parse() 返回 kind="final"',         s:"task_state.status = 'completed'" },
                { t:"步数超限",  m:"tool_steps ≥ max_steps（默认6）",    s:"stop_reason = 'step_limit_reached'" },
                { t:"重试超限",  m:"attempts ≥ max_steps×3",            s:"模型反复格式错误\nstop_reason = 'retry_limit_reached'" },
              ].map(item => (
                <Card key={item.t}>
                  <CardHeader>{item.t}</CardHeader>
                  <CardBody>
                    <Stack gap={6}>
                      <Text size="small">{item.m}</Text>
                      {item.s.split("\n").map((l, i) => <Text key={i} size="small" tone="secondary">{l}</Text>)}
                    </Stack>
                  </CardBody>
                </Card>
              ))}
            </Grid>
          </Stack>
        )}

        {tab === "resume"     && <ResumeTab />}
        {tab === "parse"      && <ParseTab />}
        {tab === "runtool"    && <RunToolTab />}
        {tab === "memory"     && <MemoryTab />}
        {tab === "checkpoint" && <CheckpointTab />}
        {tab === "session"    && <SessionTab />}

      </Stack>
    </Stack>
  );
}
