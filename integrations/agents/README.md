# MySQL to Memgraph Migration Agent

This agent analyzes MySQL databases, generates appropriate Cypher queries, and migrates data to Memgraph using LangGraph workflow. It's specifically designed to work with the Sakila sample database but can be adapted for other MySQL databases.

## Features

- **Automatic Schema Analysis**: Connects to MySQL and analyzes table structures, relationships, and constraints
- **Intelligent Migration Planning**: Uses OpenAI GPT to generate optimal migration strategies
- **Cypher Query Generation**: Automatically generates Cypher queries for creating nodes, relationships, and constraints
- **Data Type Mapping**: Maps MySQL data types to appropriate Memgraph/Cypher types
- **Relationship Detection**: Identifies foreign key relationships and converts them to graph relationships
- **Progress Tracking**: Provides detailed progress updates and error handling
- **Verification**: Validates migration results by comparing counts and structures

## Prerequisites

1. **Python 3.10+**
2. **MySQL database** with Sakila dataset (or your own database)
3. **Memgraph** instance running and accessible
4. **OpenAI API key** for natural language processing tasks
5. **UV package manager** (already configured in the project)

## Installation

1. Navigate to the agents directory:

   ```bash
   cd integrations/agents
   ```

2. Install dependencies using UV:

   ```bash
   uv sync
   ```

3. Copy the environment configuration file:

   ```bash
   cp .env.example .env
   ```

4. Edit the `.env` file with your actual configuration:

   ```bash
   # OpenAI API Configuration
   OPENAI_API_KEY=your_actual_openai_api_key

   # MySQL Database Configuration
   MYSQL_HOST=localhost
   MYSQL_USER=root
   MYSQL_PASSWORD=your_mysql_password
   MYSQL_DATABASE=sakila
   MYSQL_PORT=3306

   # Memgraph Database Configuration
   MEMGRAPH_URL=bolt://localhost:7687
   MEMGRAPH_USER=
   MEMGRAPH_PASSWORD=
   MEMGRAPH_DATABASE=memgraph
   ```

## Setting Up Sakila Database

If you don't have the Sakila database set up:

1. Download the Sakila database from MySQL's official site
2. Import it into your MySQL instance:
   ```sql
   SOURCE sakila-schema.sql;
   SOURCE sakila-data.sql;
   ```

## Usage

### Basic Usage

Run the migration agent:

```bash
uv run python main.py
```

### Programmatic Usage

```python
from main import MySQLToMemgraphAgent

# Configure your databases
mysql_config = {
    "host": "localhost",
    "user": "root",
    "password": "",
    "database": "sakila",
    "port": 3306
}

memgraph_config = {
    "url": "bolt://localhost:7687",
    "username": "",
    "password": "",
    "database": "memgraph"
}

# Create and run the agent
agent = MySQLToMemgraphAgent()
result = agent.migrate(mysql_config, memgraph_config)

print(f"Success: {result['success']}")
print(f"Migrated {len(result['completed_tables'])} tables")
```

## How It Works

The agent follows a multi-step workflow:

1. **Schema Analysis**:

   - Connects to MySQL database
   - Extracts table schemas, foreign keys, and relationships
   - Counts rows in each table

2. **Migration Planning**:

   - Uses OpenAI GPT to analyze the database structure
   - Generates an optimal migration plan considering dependencies
   - Identifies potential issues and optimizations

3. **Query Generation**:

   - Maps MySQL data types to Cypher types
   - Generates node creation queries for each table
   - Creates relationship queries based on foreign keys
   - Generates constraint and index creation queries

4. **Query Validation**:

   - Tests connection to Memgraph
   - Validates query syntax

5. **Migration Execution**:

   - Creates constraints and indexes first
   - Migrates data table by table
   - Creates relationships between nodes
   - Handles errors gracefully

6. **Verification**:
   - Compares node and relationship counts
   - Provides detailed migration summary

## Graph Model for Sakila

The Sakila database is converted to a graph model with the following approach:

- **Tables → Node Labels**: Each table becomes a node type (e.g., `film` → `Film`)
- **Foreign Keys → Relationships**: FK relationships become directed edges
- **Primary Keys → Node IDs**: Primary keys become unique node identifiers
- **Data Types**: MySQL types are mapped to Cypher-compatible types

Example transformations:

- `film` table → `Film` nodes
- `actor` table → `Actor` nodes
- `film_actor` junction table → `ACTED_IN` relationships
- `customer` → `Customer` nodes with `PLACED` relationships to `Rental` nodes

## Customization

### Adding Custom Type Mappings

Edit `cypher_generator.py` to add custom MySQL to Cypher type mappings:

```python
self.type_mapping['your_mysql_type'] = 'CYPHER_TYPE'
```

### Custom Relationship Names

Modify the `_generate_relationship_type` method in `cypher_generator.py` to customize relationship naming logic.

### Advanced Query Generation

Override methods in `CypherGenerator` class to customize how queries are generated for your specific use case.

## Troubleshooting

### Common Issues

1. **Connection Errors**:

   - Verify MySQL and Memgraph are running
   - Check connection credentials in `.env`
   - Ensure ports are accessible

2. **Import Errors**:

   - Make sure all dependencies are installed: `uv sync`
   - Check Python path configurations

3. **Migration Failures**:

   - Check logs for specific error messages
   - Verify data integrity in source database
   - Ensure target database has sufficient permissions

4. **Memory Issues**:
   - For large databases, consider implementing batch processing
   - Monitor memory usage during migration

### Logging

The agent provides detailed logging. To increase verbosity:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

To extend the agent for other database types or add features:

1. Fork the repository
2. Create feature branches
3. Add tests for new functionality
4. Submit pull requests

## License

This project is part of the AI Toolkit and follows the same licensing terms.
