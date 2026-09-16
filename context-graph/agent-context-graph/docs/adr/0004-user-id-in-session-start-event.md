# Add user_id to SessionStartEvent

The Event Protocol's `SessionStartEvent` carries no user identity. Memory Graph needs to assign ownership of every `Memory` to a user node, and Actions Graph and Skills Graph will eventually need the same. Rather than configuring user identity at `MemoryGraph` init time (which couples identity to a single component) or delegating to a caller-supplied resolver, we add an optional `user_id` field to `SessionStartEvent`. This keeps identity in the shared event stream so all graph connectors can access it without breaking existing callers that do not supply it.
