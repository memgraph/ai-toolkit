# SQL Database to Memgraph Migration Agent

Intelligent database migration agent that transforms SQL databases (MySQL, PostgreSQL) into graph databases using Memgraph, powered by LLM analysis and LangGraph workflows.

## Overview

This package provides a sophisticated migration agent that:

- **Analyzes SQL database schemas** - Automatically discovers tables, relationships, and constraints
- **Generates optimal graph models** - Uses AI to create node and relationship structures
- **Creates indexes and constraints** - Ensures performance and data integrity
- **Handles complex relationships** - Converts foreign keys to graph relationships
- **Incremental refinement** - Review each table, adjust the model immediately, and launch the interactive refinement loop whenever you need
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

## Configuration

Set up your environment variables in `.env`:

```bash
# Source Database (MySQL/PostgreSQL)
SOURCE_DB_HOST=localhost
SOURCE_DB_PORT=3306
SOURCE_DB_NAME=your_database
SOURCE_DB_USER=username
SOURCE_DB_PASSWORD=password
SOURCE_DB_TYPE=mysql  # or postgresql

# Memgraph Database
MEMGRAPH_HOST=localhost
MEMGRAPH_PORT=7687
MEMGRAPH_USER=
MEMGRAPH_PASSWORD=

# OpenAI (for LLM-powered features)
OPENAI_API_KEY=your_openai_key
```

Make sure that Memgraph is started with the `--schema-info-enabled=true`, since agent uses the schema information from Memgraph `SHOW SCHEMA INFO`.

# Arhitecture

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
