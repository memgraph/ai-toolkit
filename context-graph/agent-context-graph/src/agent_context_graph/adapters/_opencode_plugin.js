import { spawn } from "node:child_process"

// No @opencode/plugin import: OpenCode resolves imports relative to this file,
// where that package is not installed, and Plugin.define is only an identity
// helper, so exporting the plugin object directly keeps this dependency-free.

// argv, spawned directly: no shell, so no login profile or ambient env is
// sourced between OpenCode and the hook (ADR 0002).
const command = __AGENT_CONTEXT_GRAPH_COMMAND__

const CAPTURED_EVENTS = new Set([
  "session.created",
  "session.deleted",
  "session.text.ended",
  "session.execution.succeeded",
  "session.execution.failed",
  "permission.asked",
])

function capture(payload) {
  return new Promise((resolve) => {
    const child = spawn(command[0], command.slice(1), { stdio: ["pipe", "ignore", "inherit"] })
    child.on("error", resolve)
    child.on("close", resolve)
    child.stdin.end(JSON.stringify(payload))
  })
}

export default {
  id: "memgraph.agent-context-graph",
  async setup(ctx) {
    const registrations = []

    registrations.push(await ctx.session.hook("prompt", async (event) => {
      await capture({
        hook_event_name: "session.prompt",
        session_id: event.sessionID,
        prompt: event.prompt.text,
        metadata: event.metadata,
      })
    }))

    registrations.push(await ctx.tool.hook("execute.before", async (event) => {
      await capture({
        hook_event_name: "tool.execute.before",
        session_id: event.sessionID,
        tool_name: event.tool,
        tool_input: event.input,
        tool_use_id: event.id,
      })
    }))

    registrations.push(await ctx.tool.hook("execute.after", async (event) => {
      await capture({
        hook_event_name: "tool.execute.after",
        session_id: event.sessionID,
        tool_name: event.tool,
        tool_input: event.input,
        tool_use_id: event.id,
        tool_result: event.status === "completed" ? event.result : undefined,
        is_error: event.status === "error",
        error: event.status === "error" ? event.error : undefined,
      })
    }))

    const controller = new AbortController()
    void (async () => {
      for await (const event of ctx.event.subscribe({ signal: controller.signal })) {
        if (!CAPTURED_EVENTS.has(event.type)) continue
        const data = event.data ?? {}
        await capture({
          hook_event_name: event.type,
          session_id: data.sessionID,
          cwd: data.location?.directory,
          model: data.model?.id,
          text: data.text,
          error: data.error,
          permission: data.action,
        })
      }
    })()

    return async () => {
      controller.abort()
      await Promise.all(registrations.map((registration) => registration.dispose()))
    }
  },
}
