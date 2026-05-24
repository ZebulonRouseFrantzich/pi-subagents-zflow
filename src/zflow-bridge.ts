/**
 * zflow-bridge.ts — Public programmatic dispatch API for pi-zflow integration.
 *
 * Exports a factory {@link createZflowDispatchService} that returns an object
 * with `runAgent` and `runParallel` methods.  These methods delegate to the
 * same pi-subagents execution engine that the `subagent` tool uses, but
 * without requiring a live Pi ExtensionAPI/ExtensionContext.
 *
 * ## Design
 *
 * pi-subagents executes agents by spawning the `pi` CLI as a child process.
 * This is the same mechanism whether triggered from the `subagent` tool or
 * from this programmatic API.  No ExtensionContext is needed at runtime;
 * agent discovery reads from the standard filesystem paths
 * (`~/.pi/agent/agents/`, `.pi/agents/`, builtins).
 *
 * ## Usage
 *
 * ```ts
 * import { createZflowDispatchService } from "pi-subagents/zflow-bridge"
 *
 * const dispatch = createZflowDispatchService()
 *
 * const result = await dispatch.runAgent({
 *   agent: "worker",
 *   task: "Implement the approved plan",
 * })
 *
 * const parallelResult = await dispatch.runParallel({
 *   tasks: [
 *     { agent: "worker", task: "Task A" },
 *     { agent: "reviewer", task: "Review changes" },
 *   ],
 *   worktree: true,
 * })
 * ```
 *
 * @module pi-subagents/zflow-bridge
 */

import * as crypto from "node:crypto"
import * as fs from "node:fs"
import * as path from "node:path"
import { discoverAgents } from "./agents/agents.ts"
import type { AgentConfig } from "./agents/agents.ts"
import { runSync } from "./runs/foreground/execution.ts"
import {
  createWorktrees,
  diffWorktrees,
  cleanupWorktrees,
  type WorktreeSetup,
} from "./runs/shared/worktree.ts"
import type { AgentProgress, RunSyncOptions, SingleResult } from "./shared/types.ts"

// ── Shared helpers ──────────────────────────────────────────────

function findAgent(agents: AgentConfig[], name: string): AgentConfig | undefined {
  // Try exact match first, then prefix match for builtin: prefix
  return agents.find(
    (a) => a.name === name || a.name === `builtin:${name}`,
  )
}

type RunSyncUpdate = Parameters<NonNullable<RunSyncOptions["onUpdate"]>>[0]

export interface ZflowAgentProgress {
  agent: string
  status?: string
  toolCount?: number
  currentTool?: string
  currentToolArgs?: string
  recentTools?: Array<{ tool?: string; args?: string }>
  durationMs?: number
  lastActivityAt?: number
  recentOutput?: string[]
}

function mapProgress(agentName: string, progress: AgentProgress): ZflowAgentProgress {
  return {
    agent: agentName,
    status: progress.status,
    toolCount: progress.toolCount,
    currentTool: progress.currentTool,
    currentToolArgs: progress.currentToolArgs,
    recentTools: progress.recentTools?.map((tool) => ({
      tool: tool.tool,
      args: tool.args,
    })),
    durationMs: progress.durationMs,
    lastActivityAt: progress.lastActivityAt,
    recentOutput: progress.recentOutput,
  }
}

function forwardProgress(
  agentName: string,
  onUpdate: ZflowAgentInput["onUpdate"] | undefined,
): RunSyncOptions["onUpdate"] | undefined {
  if (!onUpdate) return undefined
  return (update: RunSyncUpdate) => {
    const progress = update.details?.progress?.[0]
    if (!progress) return
    onUpdate(mapProgress(agentName, progress))
  }
}

// ── Own types ───────────────────────────────────────────────────

/** Input for a single agent dispatch. */
export interface ZflowAgentInput {
  /** Agent runtime name (e.g. "worker", "builtin:scout"). */
  agent: string
  /** Task description */
  task: string
  /** Working directory override */
  cwd?: string
  /** Model override */
  model?: string
  /** Output path or false to suppress file output */
  output?: string | false
  /** Output mode */
  outputMode?: "inline" | "file-only"
  /** Output truncation limits. */
  maxOutput?: { lines?: number; bytes?: number }
  /** Live progress callback. */
  onUpdate?: (progress: ZflowAgentProgress) => void
}

/** Result of a single agent dispatch. */
export interface ZflowAgentResult {
  ok: boolean
  exitCode: number
  error?: string
  rawOutput: string
  savedOutputPath?: string
  /** Zflow-compatible alias for savedOutputPath. */
  outputPath?: string
}

/** Input for a parallel dispatch. */
export interface ZflowParallelInput {
  tasks: ZflowAgentInput[]
  concurrency?: number
  /** Create isolated git worktrees for each task */
  worktree?: boolean
  /** Working directory override for agent discovery (not per-task) */
  cwd?: string
}

/** Result from one task within a parallel dispatch. */
export interface ZflowParallelTaskResult {
  agent: string
  ok: boolean
  error?: string
  rawOutput: string
  savedOutputPath?: string
  /** Zflow-compatible alias for savedOutputPath. */
  outputPath?: string
  worktreePath?: string
  patchPath?: string
  changedFiles?: string[]
  /** Scoped verification result, when the task prompt declares one. */
  verification?: ZflowVerificationResult
}

/** Scoped verification result reported to pi-zflow. */
export interface ZflowVerificationResult {
  status: "pass" | "fail" | "skipped" | "missing"
  command?: string
  output?: string
}

/** Result of a parallel dispatch. */
export interface ZflowParallelResult {
  ok: boolean
  results: ZflowParallelTaskResult[]
}

/** Options for {@link createZflowDispatchService}. */
export interface ZflowDispatchOptions {
  /** Default working directory for agent discovery and execution. */
  cwd?: string
  /** Agent discovery scope. */
  agentScope?: "user" | "project" | "both"
  /** Override the pi executable path (default: resolves via PATH). */
  piCommand?: string
}

// ── Helpers ─────────────────────────────────────────────────────

function generateRunId(): string {
  return crypto.randomUUID().slice(0, 8)
}

/**
 * Run tasks with rolling concurrency.
 *
 * Starts up to `limit` tasks immediately.
 * When any task completes, the next queued task begins.
 * Results are returned in original task order.
 */
async function runTasksWithRollingConcurrency<T>(
  tasks: (() => Promise<T>)[],
  limit: number,
): Promise<T[]> {
  const results: T[] = new Array(tasks.length)
  let nextIndex = 0

  async function worker(): Promise<void> {
    while (nextIndex < tasks.length) {
      const idx = nextIndex++
      // If a task factory throws unexpectedly, catch and set the result to a
      // best-effort failure sentinel so the slot is freed and the pool
      // continues. Call-site task factories already catch their own errors
      // and return { ok: false, error }, so this is defence-in-depth.
      try {
        results[idx] = await tasks[idx]()
      } catch (err) {
        results[idx] = {
          ok: false,
          error: `Unexpected worker crash: ${err instanceof Error ? err.message : String(err)}`,
          rawOutput: "",
        } as unknown as T
      }
    }
  }

  const active = Math.min(limit, tasks.length)
  const workers: Promise<void>[] = []
  for (let i = 0; i < active; i++) {
    workers.push(worker())
  }
  await Promise.all(workers)
  return results
}

function mapSingleResult(result: SingleResult): ZflowAgentResult {
  return {
    ok: result.exitCode === 0 && !result.error,
    exitCode: result.exitCode,
    error: result.error,
    rawOutput: result.finalOutput ?? "",
    savedOutputPath: result.savedOutputPath,
    outputPath: result.savedOutputPath,
  }
}

function safeGetCwd(override?: string): string {
  if (override) {
    try {
      fs.accessSync(override, fs.constants.R_OK)
      return override
    } catch {
      // fall through to process.cwd
    }
  }
  return process.cwd()
}

// ── Factory ─────────────────────────────────────────────────────

/**
 * Create a zflow-compatible dispatch service backed by pi-subagents.
 *
 * The returned object's `runAgent` and `runParallel` methods use the same
 * execution engine as the `subagent` tool, spawning child `pi` processes.
 *
 * @param options - Optional configuration.
 * @returns A dispatch object.
 * @throws If pi is not available on PATH (checked on first call).
 */
export function createZflowDispatchService(
  options?: ZflowDispatchOptions,
): {
  runAgent(input: ZflowAgentInput): Promise<ZflowAgentResult>
  runParallel(input: ZflowParallelInput): Promise<ZflowParallelResult>
  readonly name: string
} {
  const defaultCwd = safeGetCwd(options?.cwd)

  // Lazy agent cache — re-discover on each call to pick up changes
  function resolveAgents(cwd: string): { agents: AgentConfig[]; error?: string } {
    try {
      const scope = options?.agentScope ?? "both"
      const result = discoverAgents(cwd, scope)
      if (result.agents.length === 0) {
        return { agents: [], error: "No agents discovered" }
      }
      return { agents: result.agents }
    } catch (err) {
      return {
        agents: [],
        error: `Agent discovery failed: ${err instanceof Error ? err.message : String(err)}`,
      }
    }
  }

  return {
    name: "pi-subagents-zflow:operational",

    async runAgent(input: ZflowAgentInput): Promise<ZflowAgentResult> {
      const cwd = safeGetCwd(input.cwd ?? defaultCwd)
      const { agents, error: discoveryError } = resolveAgents(cwd)

      if (discoveryError || agents.length === 0) {
        return {
          ok: false,
          exitCode: 1,
          error: discoveryError ?? "No agents discovered",
          rawOutput: "",
        }
      }

      const agent = findAgent(agents, input.agent)
      if (!agent) {
        const available = agents.map((a) => a.name).join(", ")
        return {
          ok: false,
          exitCode: 1,
          error: `Unknown agent "${input.agent}". Available: ${available}`,
          rawOutput: "",
        }
      }

      const options: RunSyncOptions = {
        runId: generateRunId(),
        cwd,
        modelOverride: input.model,
        outputPath: input.output === false ? undefined : (typeof input.output === "string" ? input.output : undefined),
        outputMode: input.outputMode === "file-only" ? "file-only" : undefined,
        maxOutput: input.maxOutput,
        onUpdate: forwardProgress(agent.name, input.onUpdate),
      }

      try {
        const result = await runSync(cwd, agents, agent.name, input.task, options)
        return mapSingleResult(result)
      } catch (err) {
        return {
          ok: false,
          exitCode: 1,
          error: `Dispatch error: ${err instanceof Error ? err.message : String(err)}`,
          rawOutput: "",
        }
      }
    },

    async runParallel(input: ZflowParallelInput): Promise<ZflowParallelResult> {
      // Empty tasks edge case
      if (input.tasks.length === 0) {
        return { ok: false, results: [] }
      }

      const cwd = safeGetCwd(input.cwd ?? defaultCwd)
      const { agents, error: discoveryError } = resolveAgents(cwd)

      if (discoveryError || agents.length === 0) {
        return {
          ok: false,
          results: input.tasks.map((t) => ({
            agent: t.agent,
            ok: false,
            error: discoveryError ?? "No agents discovered",
            rawOutput: "",
          })),
        }
      }

      // Validate all agents exist
      for (const task of input.tasks) {
        if (!findAgent(agents, task.agent)) {
          const available = agents.map((a) => a.name).join(", ")
          return {
            ok: false,
            results: input.tasks.map((t) => ({
              agent: t.agent,
              ok: false,
              error: t.agent === task.agent
                ? `Unknown agent "${task.agent}". Available: ${available}`
                : `Sibling task failed before start due to unknown agent "${task.agent}"`,
              rawOutput: "",
            })),
          }
        }
      }

      // ── Worktree mode ──────────────────────────────────────────
      if (input.worktree) {
        return runParallelWithWorktrees(cwd, input, agents)
      }

      // ── Plain concurrent mode ──────────────────────────────────
      return runParallelConcurrent(cwd, input, agents)
    },
  }
}

// ── Worktree parallel execution ──────────────────────────────────

async function runParallelWithWorktrees(
  cwd: string,
  input: ZflowParallelInput,
  agents: AgentConfig[],
): Promise<ZflowParallelResult> {
  const runId = generateRunId()
  const count = input.tasks.length
  const results: ZflowParallelTaskResult[] = []
  let worktreeSetup: WorktreeSetup | undefined

  try {
    worktreeSetup = createWorktrees(cwd, runId, count, {
      agents: input.tasks.map((t) => t.agent),
    })

    const concurrency = Math.min(
      input.concurrency ?? count,
      count,
    )

    // Run tasks in worktrees
    const runQueue = input.tasks.map((task, index) => async () => {
      const worktree = worktreeSetup!.worktrees[index]!
      const agentCwd = worktree.agentCwd
      const taskRunId = `${runId}-${index}`

      // Emit starting progress before the agent run so the UI transitions
      // from "queued" to "running" immediately.
      task.onUpdate?.({
        agent: task.agent,
        status: "running",
        recentOutput: ["starting worktree dispatch..."],
        lastActivityAt: Date.now(),
      })

      try {
        const options: RunSyncOptions = {
          runId: taskRunId,
          cwd: agentCwd,
          modelOverride: task.model,
          outputPath: task.output === false
            ? undefined
            : (typeof task.output === "string" ? task.output : undefined),
          outputMode: task.outputMode === "file-only" ? "file-only" : undefined,
          maxOutput: task.maxOutput,
          onUpdate: forwardProgress(task.agent, task.onUpdate),
        }

        const resolvedAgent = findAgent(agents, task.agent)!
        const result = await runSync(agentCwd, agents, resolvedAgent.name, task.task, options)

        return {
          agent: task.agent,
          ok: result.exitCode === 0 && !result.error,
          error: result.error,
          rawOutput: result.finalOutput ?? "",
          savedOutputPath: result.savedOutputPath,
          outputPath: result.savedOutputPath,
          verification: undefined,
          worktreePath: undefined as string | undefined,
          patchPath: undefined as string | undefined,
          changedFiles: undefined as string[] | undefined,
        }
      } catch (err) {
        return {
          agent: task.agent,
          ok: false,
          error: `Worktree dispatch error: ${err instanceof Error ? err.message : String(err)}`,
          rawOutput: "",
          worktreePath: undefined,
          patchPath: undefined,
          changedFiles: undefined,
        }
      }
    })

    const concurrencyLimit = Math.max(1, concurrency)

    // Run with rolling concurrency control
    const taskResults: ZflowParallelTaskResult[] = await runTasksWithRollingConcurrency(runQueue, concurrencyLimit)

    // Capture worktree diffs
    const diffsDir = path.join(cwd, ".zflow", "worktree-diffs", runId)
    try {
      fs.mkdirSync(diffsDir, { recursive: true })
    } catch {
      // Best effort
    }

    let diffs: Array<{ index: number; patchPath: string; filesChanged: number }> = []
    try {
      const worktreeDiffs = diffWorktrees(
        worktreeSetup,
        input.tasks.map((t) => t.agent),
        diffsDir,
      )
      diffs = worktreeDiffs.map((d) => ({
        index: d.index,
        patchPath: d.patchPath,
        filesChanged: d.filesChanged,
      }))
    } catch {
      // Best effort diff capture
    }

    // Merge diff info into results. Worktrees are cleaned up before returning,
    // so expose patchPath but not worktreePath; pi-zflow can apply captured
    // patches and fall back to plan-scoped files for changedFiles.
    for (const diff of diffs) {
      if (diff.index < taskResults.length) {
        taskResults[diff.index]!.patchPath = diff.patchPath
      }
    }

    const allOk = taskResults.every((r) => r.ok)
    results.push(...taskResults)

    return { ok: allOk, results }
  } finally {
    if (worktreeSetup) {
      try {
        cleanupWorktrees(worktreeSetup)
      } catch {
        // Best-effort cleanup
      }
    }
  }
}

// ── Plain concurrent parallel execution ──────────────────────────

async function runParallelConcurrent(
  cwd: string,
  input: ZflowParallelInput,
  agents: AgentConfig[],
): Promise<ZflowParallelResult> {
  const concurrency = Math.min(
    input.concurrency ?? input.tasks.length,
    input.tasks.length,
  )
  const concurrencyLimit = Math.max(1, concurrency)

  const runQueue = input.tasks.map((task, _index) => async () => {
    const taskCwd = task.cwd ?? cwd
    // Emit starting progress so the UI shows "running" immediately.
    task.onUpdate?.({
      agent: task.agent,
      status: "running",
      recentOutput: ["starting dispatch..."],
      lastActivityAt: Date.now(),
    })
    try {
      const options: RunSyncOptions = {
        runId: generateRunId(),
        cwd: taskCwd,
        modelOverride: task.model,
        outputPath: task.output === false
          ? undefined
          : (typeof task.output === "string" ? task.output : undefined),
        outputMode: task.outputMode === "file-only" ? "file-only" : undefined,
        maxOutput: task.maxOutput,
        onUpdate: forwardProgress(task.agent, task.onUpdate),
      }

      const resolvedAgent = findAgent(agents, task.agent)!
      const result = await runSync(taskCwd, agents, resolvedAgent.name, task.task, options)

      return {
        agent: task.agent,
        ok: result.exitCode === 0 && !result.error,
        error: result.error,
        rawOutput: result.finalOutput ?? "",
        savedOutputPath: result.savedOutputPath,
        outputPath: result.savedOutputPath,
      }
    } catch (err) {
      return {
        agent: task.agent,
        ok: false,
        error: `Dispatch error: ${err instanceof Error ? err.message : String(err)}`,
        rawOutput: "",
      }
    }
  })

  const taskResults = await runTasksWithRollingConcurrency(runQueue, concurrencyLimit)

  const allOk = taskResults.every((r) => r.ok)
  return { ok: allOk, results: taskResults }
}
