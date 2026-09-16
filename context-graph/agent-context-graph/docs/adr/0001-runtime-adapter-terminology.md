# Use Runtime Adapter terminology

Agent Context Graph supports agent development SDK integrations (OpenAI Agents SDK, Claude Agent SDK) and runtime hook integrations (Codex command hooks). We use **Runtime Adapter** as canonical term: covers both integration shapes. **SDK Adapter** wrongly implies every adapter attaches to an agent development SDK. Keeps Event Protocol independent of whether events came from in-process callbacks or command-hook payloads.
