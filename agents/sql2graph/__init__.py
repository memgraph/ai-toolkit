"""
SQL Database to Graph Migration Agent

This package provides intelligent database migration capabilities from SQL databases
to graph databases with LLM-powered graph modeling and analysis.

## Package Structure

- `core/` - Main migration orchestration and graph modeling (HyGM)
- `database/` - Database analysis and data interface layer
- `query_generation/` - Cypher query generation and schema utilities
- `utils/` - Configuration and environment utilities
- `examples/` - Usage examples and demonstrations
- `tests/` - Tests and troubleshooting tools

## Quick Start

```python
from agents.core import SQLToMemgraphAgent, HyGM
from agents.core.hygm import ModelingMode
from agents.utils import setup_and_validate_environment

# Setup environment
mysql_config, memgraph_config = setup_and_validate_environment()

# Create migration agent
agent = SQLToMemgraphAgent(modeling_mode=ModelingMode.AUTOMATIC)

# Run migration
result = agent.migrate(source_db_config, memgraph_config)
```
"""

# Main exports
from .core import SQLToMemgraphAgent, HyGM
from .database import DatabaseAnalyzerFactory
from .query_generation import CypherGenerator, SchemaUtilities

__version__ = "0.1.0"

__all__ = [
    "SQLToMemgraphAgent",
    "HyGM",
    "DatabaseAnalyzerFactory",
    "CypherGenerator",
    "SchemaUtilities",
]
