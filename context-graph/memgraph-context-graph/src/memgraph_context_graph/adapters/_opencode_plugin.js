import { spawn } from "node:child_process"

// No @opencode/plugin import: OpenCode resolves imports relative to this file,
// where that package is not installed, and Plugin.define is only an identity
// helper, so exporting the plugin object directly keeps this dependency-free.

// argv, spawned directly: no shell, so no login profile or ambient env is
// sourced between OpenCode and the hook (ADR 0002).
const command = __MEMGRAPH_CONTEXT_GRAPH_COMMAND__

const CAPTURED_EVENTS = new Set([
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

// The hook's stdout, parsed: what the hook wants added to the session, or null.
function ask(payload) {
  return new Promise((resolve) => {
    const child = spawn(command[0], command.slice(1), { stdio: ["pipe", "pipe", "inherit"] })
    let out = ""
    child.stdout.on("data", (chunk) => (out += chunk))
    child.on("error", () => resolve(null))
    child.on("close", () => {
      try {
        resolve(out.trim() ? JSON.parse(out) : null)
      } catch {
        resolve(null)
      }
    })
    child.stdin.end(JSON.stringify(payload))
  })
}

export default {
  id: "memgraph.memgraph-context-graph",
  async setup(ctx) {
    const registrations = []
    // Sessions with a turn in flight. `opencode run` can exit before the bus
    // delivers session.execution.succeeded, so shutdown closes these turns.
    const openTurns = new Set()
    // Sessions already recorded as started. The bus subscription below can
    // start after `opencode run` creates its session, so session.created may
    // never arrive; the first prompt then records the start, with the
    // plugin's own directory as the session's.
    const started = new Set()
    const start = (sessionID, cwd, model) => {
      if (started.has(sessionID)) return
      started.add(sessionID)
      return capture({ hook_event_name: "session.created", session_id: sessionID, cwd, model })
    }

    // The session-start text (memory index, where the user left off), asked for once per session. The context
    // hook runs before every model request and rebuilds the system prompt each time, so it is re-added from
    // this cache on each one; compaction and title generation build their own requests and get it too.
    const sessionContext = new Map()
    const addSessionContext = async (event) => {
      if (!sessionContext.has(event.sessionID)) {
        sessionContext.set(
          event.sessionID,
          ask({ hook_event_name: "session.context", session_id: event.sessionID, cwd: ctx.location?.directory }),
        )
      }
      const response = await sessionContext.get(event.sessionID)
      if (typeof response?.system === "string" && response.system) event.system.push({ type: "text", text: response.system })
    }
    for (const name of ["context", "compaction", "generate"]) {
      registrations.push(await ctx.session.hook(name, addSessionContext))
    }

    registrations.push(await ctx.session.hook("prompt", async (event) => {
      openTurns.add(event.sessionID)
      await start(event.sessionID, ctx.location?.directory)
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
        const data = event.data ?? {}
        if (event.type === "session.created") {
          await start(data.sessionID, data.location?.directory ?? ctx.location?.directory, data.model?.id)
          continue
        }
        if (!CAPTURED_EVENTS.has(event.type)) continue
        if (event.type.startsWith("session.execution.")) openTurns.delete(data.sessionID)
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
      await Promise.all(
        [...openTurns].map((sessionID) =>
          capture({ hook_event_name: "session.execution.succeeded", session_id: sessionID }),
        ),
      )
      await Promise.all(registrations.map((registration) => registration.dispose()))
    }
  },
}
