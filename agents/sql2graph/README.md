# SQL Database to Graph Migration Agent

Intelligent database migration agent that transforms SQL databases (MySQL, PostgreSQL) into graph databases, powered by LLM analysis and LangGraph workflows.

## Overview

This package provides a sophisticated migration agent that:

- **Analyzes SQL database schemas** - Automatically discovers tables, relationships, and constraints
- **Generates optimal graph models** - Uses AI to create node and relationship structures
- **Creates indexes and constraints** - Ensures performance and data integrity
- **Handles complex relationships** - Converts foreign keys to graph relationships
- **Incremental refinement** - Review each table, adjust the model
  immediately, then enter the interactive refinement loop once all tables
  are processed
- **Comprehensive validation** - Verifies migration results and data integrity

## Installation

```bash
# Install the package
uv pip install .

# Or install in development mode
uv pip install -e .
```

## Quick Start

Run the migration agent:

```bash
uv run main
```

The agent will guide you through:

1. Environment setup and database connections
2. Graph modeling strategy selection
3. Automatic or incremental migration mode
4. Complete migration workflow with progress tracking

> **Incremental review:** The LLM now drafts the entire graph model in a single
> shot and then walks you through table-level changes detected since the last
> migration. You only need to approve (or tweak) the differences that matter.

You can also preconfigure the workflow using CLI flags or environment variables:

```bash
uv run main --mode incremental --strategy llm --meta-graph reset --log-level DEBUG
```

| Option                                 | Environment          | Description                                                   |
| -------------------------------------- | -------------------- | ------------------------------------------------------------- |
| `--mode {automatic,incremental}`       | `SQL2MG_MODE`        | Selects automatic or incremental modeling flow.               |
| `--strategy {deterministic,llm}`       | `SQL2MG_STRATEGY`    | Chooses deterministic or LLM-powered HyGM strategy.           |
| `--provider {openai,anthropic,gemini}` | `LLM_PROVIDER`       | Selects LLM provider (auto-detects if not specified).         |
| `--model MODEL_NAME`                   | `LLM_MODEL`          | Specifies LLM model name (uses provider default if not set).  |
| `--meta-graph {auto,skip,reset}`       | `SQL2MG_META_POLICY` | Controls how stored meta graph data is used (default `auto`). |
| `--log-level LEVEL`                    | `SQL2MG_LOG_LEVEL`   | Sets logging verbosity (`DEBUG`, `INFO`, etc.).               |

## Configuration

Set up your environment variables in `.env`:

```bash
# Select source database (mysql or postgresql)
SOURCE_DB_TYPE=postgresql

# PostgreSQL Database (used when SOURCE_DB_TYPE=postgresql)
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DATABASE=pagila
POSTGRES_USER=username
POSTGRES_PASSWORD=password
POSTGRES_SCHEMA=public

# MySQL Database (used when SOURCE_DB_TYPE=mysql)
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_DATABASE=sakila
MYSQL_USER=username
MYSQL_PASSWORD=password

# Memgraph Database
MEMGRAPH_URL=bolt://localhost:7687
MEMGRAPH_USERNAME=
MEMGRAPH_PASSWORD=
MEMGRAPH_DATABASE=memgraph

# LLM API Keys (for LLM-powered features - choose one or more)
OPENAI_API_KEY=your_openai_key         # For GPT models
# ANTHROPIC_API_KEY=your_anthropic_key # For Claude models
# GOOGLE_API_KEY=your_google_key       # For Gemini models

# LLM Provider Configuration (optional - auto-detects if not set)
# LLM_PROVIDER=openai                  # Options: openai, anthropic, gemini
# LLM_MODEL=gpt-4o-mini                # Specific model name

# Optional migration defaults (override CLI prompts)
SQL2MG_MODE=automatic
SQL2MG_STRATEGY=deterministic
SQL2MG_META_POLICY=auto
SQL2MG_LOG_LEVEL=INFO
```

When switching `SOURCE_DB_TYPE` remember to update the matching credential block and rerun `uv sync` so dependencies like `psycopg2-binary` are installed for PostgreSQL support.

Make sure that Memgraph is started with the `--schema-info-enabled=true`, since agent uses the schema information from Memgraph `SHOW SCHEMA INFO`.

## Multi-LLM Provider Support

The agent supports multiple LLM providers for AI-powered graph modeling:

### Supported Providers

- **OpenAI** (GPT models) - Default: `gpt-4o-mini`
- **Anthropic** (Claude models) - Default: `claude-3-5-sonnet-20241022`
- **Google** (Gemini models) - Default: `gemini-1.5-pro`

### Usage Examples

```bash
# Auto-detect provider based on API keys
uv run main --strategy llm

# Use specific provider
uv run main --strategy llm --provider anthropic

# Use specific model
uv run main --strategy llm --provider openai --model gpt-4o

# All options together
uv run main --mode incremental --strategy llm --provider gemini --model gemini-1.5-flash
```

All providers support **structured outputs** for consistent graph model generation. The system automatically validates schemas using Pydantic models.

📖 **[Full Multi-Provider Documentation](docs/MULTI_PROVIDER_SUPPORT.md)**

# Arhitecture

```
core/hygm/
├── hygm.py # Main orchestrator class
├── models/ # Data models and structures
│ ├── graph_models.py # Core graph representation
│ ├── llm_models.py # LLM-specific models
│ ├── operations.py # Interactive operations
│ └── sources.py # Source tracking
└── strategies/ # Modeling strategies
├── base.py # Abstract interface
├── deterministic.py # Rule-based modeling
└── llm.py # AI-powered modeling
```
