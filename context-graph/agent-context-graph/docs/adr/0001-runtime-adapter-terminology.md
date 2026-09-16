# Use Runtime Adapter terminology

Agent Context Graph supports both agent development SDK integrations, such as OpenAI Agents SDK and Claude Agent SDK, and runtime hook integrations, such as Codex command hooks. We use **Runtime Adapter** as the canonical term because it covers both integration shapes, while **SDK Adapter** incorrectly implies every adapter attaches to an agent development SDK. This keeps the Event Protocol independent of whether events came from in-process callbacks or command-hook payloads.
