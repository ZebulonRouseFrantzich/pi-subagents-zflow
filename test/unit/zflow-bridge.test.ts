import assert from "node:assert/strict"
import { describe, it } from "node:test"
import { createZflowDispatchService } from "../../src/zflow-bridge.ts"

describe("createZflowDispatchService", () => {
  it("returns an object with runAgent, runParallel, and name", () => {
    const svc = createZflowDispatchService()

    assert.equal(typeof svc.runAgent, "function")
    assert.equal(typeof svc.runParallel, "function")
    assert.equal(typeof svc.name, "string")
    assert.equal(svc.name, "pi-subagents-zflow:operational")
  })

  it("returns a dispatch-wrapped service when no options are given", () => {
    const svc = createZflowDispatchService()
    // Verify the shape matches expected interface
    assert.ok("runAgent" in svc)
    assert.ok("runParallel" in svc)
    assert.ok("name" in svc)
  })
})

describe("zflow-bridge runAgent validation", () => {
  it("returns ok:false with error for an unknown agent", async () => {
    const svc = createZflowDispatchService()

    const result = await svc.runAgent({
      agent: "nonexistent-agent-xyz",
      task: "Do something",
    })

    assert.equal(result.ok, false)
    assert.equal(result.exitCode, 1)
    assert.ok(result.error, "should have an error message")
    assert.ok(
      result.error!.includes("nonexistent-agent-xyz"),
      `error should mention the agent name, got: ${result.error}`,
    )
    assert.equal(result.rawOutput, "")
  })
})

describe("zflow-bridge runParallel validation", () => {
  it("returns ok:false with per-task errors for an unknown agent", async () => {
    const svc = createZflowDispatchService()

    const result = await svc.runParallel({
      tasks: [
        { agent: "worker", task: "Do something" },
        { agent: "nonexistent-agent-xyz", task: "Do something else" },
      ],
    })

    assert.equal(result.ok, false)
    assert.ok(result.results.length === 2, "should have 2 task results")

    // First task may be unknown too (no agents discovered)
    // But at least one should have an error mentioning the unknown agent
    const hasUnknownAgentError = result.results.some(
      (r) => r.error && r.error.includes("nonexistent-agent-xyz"),
    )
    assert.ok(
      hasUnknownAgentError,
      "at least one result should mention the unknown agent",
    )
  })

  it("returns ok:false with empty tasks gracefully", async () => {
    const svc = createZflowDispatchService()

    const result = await svc.runParallel({
      tasks: [],
    })

    assert.equal(result.ok, false)
    assert.equal(result.results.length, 0)
  })
})
