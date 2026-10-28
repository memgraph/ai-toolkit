# MCP Memgraph Server Prompt

## TL;DR

You are working inside the `integrations/mcp-memgraph` package, a UV-managed Python project that implements a Model Context Protocol (MCP) server for connecting Memgraph graph databases with LLMs like Claude. The server exposes Memgraph operations as MCP tools, enabling AI assistants to query and interact with graph data. The primary entry point is `src/mcp_memgraph/main.py`, which supports multiple server variants via the `MCP_SERVER` environment variable. Changes usually touch:

- `src/mcp_memgraph/servers/` — Server implementations package with plugin-style registry:
  - `server.py` — Default MCP server with stable tools for Memgraph operations.
  - `experimental.py` — Experimental server for testing new features (load with `MCP_SERVER=experimental`).
  - `__init__.py` — Server registry (`AVAILABLE_SERVERS`) and dynamic loading functions (`get_server()`).
- `src/mcp_memgraph/main.py` — Server startup and routing; uses `get_server()` for dynamic loading.
- `src/mcp_memgraph/config.py` — Environment-based configuration for Memgraph connection and MCP server settings (shared across all servers).
- `tests/` — Integration tests for server tools and client connections.

The server runs in **read-only mode by default** to prevent accidental data modifications.

**Plugin-Style Architecture**: The project uses a registry system in `servers/__init__.py` that maps server names to module paths. Add new servers by creating a `servers/<name>.py` file and registering it in `AVAILABLE_SERVERS`. All servers are loaded dynamically via `importlib`, eliminating the need for explicit import statements in `main.py`. All servers share the same configuration system (`config.py`) to avoid duplication.

## Tech Stack & Tooling

- Python 3.10+ (recommended 3.13), managed with [uv](https://github.com/astral-sh/uv).
- [FastMCP](https://github.com/jlowin/fastmcp) for MCP server implementation.
- [Memgraph Toolbox](https://github.com/memgraph/memgraph-toolbox) for graph database operations.
- Memgraph as the target graph database (Bolt protocol via `neo4j` driver).
- Testing: `pytest` with async support (`pytest-asyncio`).
- Optional: Anthropic Python SDK for Claude Desktop integration testing.

## Core Concepts

- **MCP (Model Context Protocol)**: A standard protocol for connecting AI assistants to external tools and data sources. This server implements MCP to expose Memgraph as a tool provider.
- **FastMCP Server**: The `mcp` object in `server.py` is a FastMCP instance that registers tools using the `@mcp.tool()` decorator.
- **Memgraph Toolbox**: A collection of pre-built tools for common Memgraph operations (schema introspection, Cypher queries, graph algorithms). Each MCP tool wraps a corresponding toolbox tool.
- **Transport Types**: The server supports two MCP transports:
  - `stdio` (default) — Standard input/output, used by Claude Desktop and other MCP clients.
  - `streamable-http` — HTTP-based transport for web integrations.
- **Read-Only Mode**: Enabled by default (`MCP_READ_ONLY=true`). A best-effort guard: write keywords, procedures that can write, and anything Memgraph rejects in a read transaction are refused with an error. Set `MCP_READ_ONLY=false` to enable writes.
- **Multi-Server Support**: Use `MCP_SERVER` environment variable to select which server implementation to run. Options: `server` (default), `experimental`. Servers are registered in `servers/__init__.py` and loaded dynamically via `importlib`.
- **Configuration System**: Uses singleton pattern with dataclasses (`MemgraphConfig`, `MCPServerConfig`) to centralize environment variable access. Configuration is loaded once and reused across all servers, avoiding duplication.

## Architecture

```
┌─────────────────────┐
│   LLM (Claude, etc) │
└──────────┬──────────┘
           │ MCP Protocol
           │ (stdio/http)
┌──────────▼──────────┐
│   FastMCP Server    │
│   (main.py)         │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│   MCP Tools         │
│   (server.py)       │
│   - run_query       │
│   - get_schema      │
│   - get_index       │
│   - ...             │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│ Memgraph Toolbox    │
│   Tools             │
└──────────┬──────────┘
           │ Bolt Protocol
┌──────────▼──────────┐
│   Memgraph DB       │
└─────────────────────┘
```

## Available MCP Tools

The server exposes these tools (defined in `servers/server.py`):

### Core Query & Schema Tools

- `run_query(query: str)` — Execute Cypher queries (respects read-only mode).
- `get_schema()` — Retrieve graph schema (labels, relationship types, properties).
- `get_configuration()` — Fetch Memgraph configuration settings.
- `get_index()` — List all indexes.
- `get_constraint()` — List all constraints.
- `get_storage()` — Get storage usage metrics.
- `get_triggers()` — List triggers.

### Graph Algorithm Tools

- `get_betweenness_centrality()` — Calculate betweenness centrality.
- `get_page_rank()` — Calculate PageRank scores.
- `get_node_neighborhood(node_id: str, max_distance: int, limit: int)` — Find nodes within specified distance.
- `search_node_vectors(index_name: str, query_vector: List[float], limit: int)` — Perform vector similarity search.

Each tool returns `List[Dict[str, Any]]` and includes error handling that returns error dictionaries on failure.

### Experimental Tools (servers/experimental.py)

When `MCP_SERVER=experimental`, the experimental server provides these additional/alternative tools:

- `experimental_query()` — Returns hardcoded graph-like data for testing. No parameters required. Used for developing and testing new features without connecting to a real database.

## Configuration & Environment

Configuration is managed through two classes in `config.py`:

### MemgraphConfig (Connection Settings)

- `MEMGRAPH_URL` — Connection URL (default: `bolt://localhost:7687`)
- `MEMGRAPH_USER` — Username (default: `""`)
- `MEMGRAPH_PASSWORD` — Password (default: `""`)
- `MEMGRAPH_DATABASE` — Database name (default: `memgraph`)

### MCPServerConfig (Server Settings)

- `MCP_SERVER` — Server to load: `server`, `experimental` (default: `server`)
- `MCP_TRANSPORT` — Transport type: `stdio` or `streamable-http` (default: `stdio`)
- `MCP_HOST` — Bind host for HTTP transport (default: `127.0.0.1`)
- `MCP_PORT` — Bind port for HTTP transport (default: `8000`)
- `MCP_READ_ONLY` — Read-only mode: `true` or `false` (default: `true`)
- `MCP_LOG_LEVEL` — Logging level: DEBUG, INFO, WARNING, ERROR (default: `INFO`)
- `MCP_LOG_FILE` — Log file path (default: `None`, not yet fully implemented)

**Important**: Memgraph must run with `--schema-info-enabled=true` for schema introspection tools to work.

## Entry Points & Usage

### Running the Server

```bash
# Install dependencies
uv sync

# Run default server
uv run mcp-memgraph

# Run experimental server
MCP_SERVER=experimental uv run mcp-memgraph

# Or install globally and run
uv pip install -e .
mcp-memgraph

# Run specific server when installed
MCP_SERVER=experimental mcp-memgraph

# Run experimental Memgraph server
MCP_SERVER=memgraph-experimental uv run mcp-memgraph
```

### Claude Desktop Integration

The server is designed to integrate with Claude Desktop. Add to `claude_desktop_config.json`:

**Default server:**

```json
{
  "mcpServers": {
    "memgraph": {
      "command": "uv",
      "args": [
        "run",
        "--with",
        "mcp-memgraph",
        "--python",
        "3.13",
        "mcp-memgraph"
      ]
    }
  }
}
```

**Multiple servers simultaneously:**

```json
{
  "mcpServers": {
    "memgraph": {
      "command": "uv",
      "args": [
        "run",
        "--with",
        "mcp-memgraph",
        "--python",
        "3.13",
        "mcp-memgraph"
      ]
    },
    "memgraph-experimental": {
      "command": "uv",
      "args": [
        "run",
        "--with",
        "mcp-memgraph",
        "--python",
        "3.13",
        "mcp-memgraph"
      ],
      "env": {
        "MCP_SERVER": "experimental"
      }
    }
  }
}
```

**Future server examples (when implemented):**

```json
{
  "mcpServers": {
    "memgraph-experimental": {
      "command": "uv",
      "args": [
        "run",
        "--with",
        "mcp-memgraph",
        "--python",
        "3.13",
        "mcp-memgraph"
      ],
      "env": {
        "MCP_SERVER": "memgraph-experimental"
      }
    },
    "memgraph-hygm": {
      "command": "uv",
      "args": [
        "run",
        "--with",
        "mcp-memgraph",
        "--python",
        "3.13",
        "mcp-memgraph"
      ],
      "env": {
        "MCP_SERVER": "hygm"
      }
    }
  }
}
```

Location of config file:

- **macOS/Linux**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%/Claude/claude_desktop_config.json`

### Docker Support

A `Dockerfile` is provided for containerized deployments:

```bash
docker build -t mcp-memgraph .
docker run -e MEMGRAPH_URL=bolt://host.docker.internal:7687 mcp-memgraph
```

## Testing & Validation

### Running Tests

```bash
# Install test dependencies
uv sync --extra test

# Run all tests
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_server.py -v

# Run with coverage
uv run pytest tests/ -v --cov=mcp_memgraph
```

### Test Structure

- `tests/test_server.py` — Main test suite covering:
  - MCP client connection (`test_mcp_client`)
  - Query execution (`test_run_query`)
  - Read-only mode enforcement (`test_write_query_blocked_in_readonly_mode`)
  - Tool functionality for schema, index, constraint operations
  - Graph algorithm tools (neighborhood, vector search)

Tests use:

- `pytest-asyncio` for async test support
- `MCPClient` helper class for server connection
- `.env` file for test environment configuration
- Mocked or real Memgraph instances (configurable)

### Write Query Protection

When `MCP_READ_ONLY=true`, `run_cypher_query` runs queries through `read_only_query()`, a best-effort guard in layers:

- Leading comments are stripped (some Memgraph versions crash on a comment before `EXPLAIN`/`PROFILE`)
- `is_write_query()` refuses write keywords outside strings and comments (including auth commands, which Memgraph runs even in a read transaction) and calls to procedures and functions that can write although Memgraph flags them as reads, including their APOC names (`do.*`, `periodic.*`, `mg.load*`, module files, file export, model training, outbound requests)
- The rest runs in a read transaction, where Memgraph rejects writes and write procedures
- Queries that can't run in a transaction run outside one only if they are `SHOW ...`, `DATA DIRECTORY LOCK STATUS` or `PROFILE` of a read query; others (`ANALYZE GRAPH`, `FREE MEMORY`, `STORAGE MODE`, index changes, schema procedures) are blocked
- Blocked queries return an error dictionary instead of executing
- The text check is not a full Cypher parser and can be bypassed; on Enterprise, a Memgraph user with only read privileges adds a check inside Memgraph

## Development Tips

- **Adding New Tools**: Use the `@mcp.tool()` decorator in any `servers/<name>.py` file. Wrap a Memgraph Toolbox tool or implement custom logic. Follow the pattern of existing tools for error handling and logging.
- **Creating New Servers**:
  1. Create `servers/<name>.py` with `mcp` and `logger` exports
  2. Add entry to `AVAILABLE_SERVERS` in `servers/__init__.py`
  3. Server will be automatically loadable via `MCP_SERVER=<name>`
  4. No changes needed to `main.py` thanks to dynamic imports
- **Configuration Changes**: Update the appropriate config class in `config.py`. Add validation if needed. Document new env vars in this prompt and `README.md`.
- **Testing New Features**: Add tests to `tests/test_server.py`. Use the `MCPClient` class for integration tests. Mock Memgraph responses when testing error handling. Set `MCP_SERVER` in tests to target specific servers.
- **Logging**: Use the `logger` instance from your server module. Initialize it via `memgraph_toolbox.utils.logger.logger_init("server-name")`.
- **Error Handling**: Always wrap tool logic in try/except blocks. Return `[{"error": "..."}]` on failures to match the expected return type.
- **Read-Only Mode**: When adding new tools that execute Cypher, run them through `read_only_query()` if read-only enforcement is needed.
- **Transport Switching**: The `main.py` file handles transport selection. `stdio` is parameterless; `streamable-http` requires host/port.
- **Dependencies**: Use `uv` for all dependency management. Add new dependencies to `pyproject.toml` under `dependencies` or `optional-dependencies.test`.

## Common Workflows

### Adding a New Server Implementation

1. Create a new file: `src/mcp_memgraph/servers/<name>.py`.
2. Import shared configuration:

   ```python
   from mcp_memgraph.config import get_memgraph_config, get_mcp_config
   from memgraph_toolbox.utils.logger import logger_init
   from fastmcp import FastMCP

   logger = logger_init("mcp-memgraph-<name>")
   mcp = FastMCP("mcp-memgraph-<name>")
   ```

3. Add your tools with `@mcp.tool()` decorator.
4. Register in `servers/__init__.py` by adding to `AVAILABLE_SERVERS`:
   ```python
   "<name>": {
       "module": "mcp_memgraph.servers.<name>",
       "emoji": "🔬",
       "description": "Your server description",
   },
   ```
5. Test: `MCP_SERVER=<name> uv run mcp-memgraph`.
6. No changes to `main.py` required - dynamic loading handles it automatically!

### Developing a New Tool (Default Server)

1. Identify the Memgraph Toolbox tool or operation needed.
2. Add a new function in `servers/server.py` decorated with `@mcp.tool()`.
3. Implement the tool, wrapping the toolbox call with error handling.
4. Add logging for visibility.
5. Write integration tests in `tests/test_server.py`.
6. Update this prompt and `README.md` if the tool adds new user-facing capabilities.

### Developing a New Experimental Tool

1. Add a new function in `servers/experimental.py` (or your custom server) decorated with `@mcp.tool()`.
2. Implement the tool logic (can use hardcoded data, mock responses, or experimental Memgraph features).
3. Test locally with `MCP_SERVER=experimental uv run mcp-memgraph`.
4. Once stable and tested, consider migrating to `servers/server.py` for the default server.
5. All servers reuse configuration from `config.py`, so no config duplication needed.

### Testing Changes Locally

1. Start Memgraph: `docker run -p 7687:7687 memgraph/memgraph-mage --schema-info-enabled=True`
2. Set environment variables (create `.env` if needed).
3. Run the server: `uv run mcp-memgraph`
4. In another terminal, run tests: `uv run pytest tests/ -v`
5. Or test with Claude Desktop by configuring `claude_desktop_config.json` and restarting Claude.

### Debugging Issues

- Check logs for errors (server prints to stdout/stderr).
- Verify Memgraph is running and accessible: `docker ps`, `telnet localhost 7687`.
- Confirm environment variables are set correctly: `printenv | grep MEMGRAPH`.
- Test Cypher queries directly in Memgraph Lab to isolate issues.
- Use `pytest -vv` for verbose test output.
- Check `claude_desktop_config.json` syntax if Claude integration fails.

## Useful Commands

```bash
# Install dependencies
uv sync

# Install with test dependencies
uv sync --extra test

# Run the MCP server
uv run mcp-memgraph

# Run tests
uv run pytest tests/ -v

# Run specific test
uv run pytest tests/test_server.py::test_run_query -v

# Format code (if configured)
uv run black src/ tests/

# Type check (if mypy is added)
uv run mypy src/

# Build package
uv build

# Install locally in editable mode
uv pip install -e .
```
