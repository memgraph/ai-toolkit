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

| Option                           | Environment          | Description                                                   |
| -------------------------------- | -------------------- | ------------------------------------------------------------- |
| `--mode {automatic,incremental}` | `SQL2MG_MODE`        | Selects automatic or incremental modeling flow.               |
| `--strategy {deterministic,llm}` | `SQL2MG_STRATEGY`    | Chooses deterministic or LLM-powered HyGM strategy.           |
| `--meta-graph {auto,skip,reset}` | `SQL2MG_META_POLICY` | Controls how stored meta graph data is used (default `auto`). |
| `--log-level LEVEL`              | `SQL2MG_LOG_LEVEL`   | Sets logging verbosity (`DEBUG`, `INFO`, etc.).               |

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

# OpenAI (for LLM-powered features)
OPENAI_API_KEY=your_openai_key

# Optional migration defaults (override CLI prompts)
SQL2MG_MODE=automatic
SQL2MG_STRATEGY=deterministic
SQL2MG_META_POLICY=auto
SQL2MG_LOG_LEVEL=INFO
```

When switching `SOURCE_DB_TYPE` remember to update the matching credential block and rerun `uv sync` so dependencies like `psycopg2-binary` are installed for PostgreSQL support.

Make sure that Memgraph is started with the `--schema-info-enabled=true`, since agent uses the schema information from Memgraph `SHOW SCHEMA INFO`.

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
