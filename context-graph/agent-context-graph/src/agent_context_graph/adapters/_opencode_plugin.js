import { Plugin } from "@opencode/plugin"

const command = __AGENT_CONTEXT_GRAPH_COMMAND__

async function capture(payload) {
  const child = Bun.spawn(["sh", "-lc", command], {
    stdin: "pipe",
    stdout: "ignore",
    stderr: "inherit",
  })
  child.stdin.write(JSON.stringify(payload))
  child.stdin.end()
  await child.exited
}

function sessionID(value) {
  return value?.sessionID ?? value?.session_id ?? value?.session?.id ?? value?.info?.id ?? ""
}

export default Plugin.define({
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
        tool_use_id: event.callID,
      })
    }))

    registrations.push(await ctx.tool.hook("execute.after", async (event) => {
      await capture({
        hook_event_name: "tool.execute.after",
        session_id: event.sessionID,
        tool_name: event.tool,
        tool_input: event.input,
        tool_result: event.status === "completed" ? event.result : undefined,
        tool_use_id: event.callID,
        is_error: event.status === "error",
        error: event.status === "error" ? event.error : undefined,
      })
    }))

    const controller = new AbortController()
    void (async () => {
      for await (const event of ctx.event.subscribe({ signal: controller.signal })) {
        if (!["session.created", "session.deleted", "session.error", "message.updated", "permission.asked"].includes(event.type)) continue
        const properties = event.properties ?? event
        const info = properties.info ?? properties.message ?? properties
        await capture({
          hook_event_name: event.type,
          session_id: sessionID(properties),
          role: info.role,
          content: info.content ?? info.text,
          model: info.model,
          cwd: info.directory,
          error: properties.error,
          permission: properties.permission,
          event,
        })
      }
    })()

    return async () => {
      controller.abort()
      await Promise.all(registrations.map((registration) => registration.dispose()))
    }
  },
})
