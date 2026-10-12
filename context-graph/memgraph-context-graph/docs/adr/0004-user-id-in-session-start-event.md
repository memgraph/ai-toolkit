# Add user_id to SessionStartEvent

Event Protocol's `SessionStartEvent` carried no user identity. Memory Graph needs to own every `Memory` to a user node; Actions Graph + Skills Graph eventually need same. Instead of configuring user identity at `MemoryGraph` init time (couples identity to one component) or a caller-supplied resolver: add optional `user_id` field to `SessionStartEvent`. Keeps identity in shared event stream — all graph connectors access it, no breakage for existing callers that don't supply it.
