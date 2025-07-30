# MySQL to Memgraph Migration Agent

This agent analyzes MySQL databases, generates appropriate Cypher queries, and migrates data to Memgraph using LangGraph workflow. It's specifically designed to work with the Sakila sample database but can be adapted for other MySQL databases.

## Enhanced Features (New!)

### 🔗 Advanced Relationship Handling

- **Foreign Keys to Relationships**: Automatically converts foreign key columns to graph relationships and removes them from node properties
- **Join Table Detection**: Identifies many-to-many join tables and converts them to relationships with properties
- **Smart Relationship Naming**: Multiple strategies for generating meaningful relationship names

### 🎯 Configurable Relationship Naming

- **Smart Strategy** (Default): Uses intelligent patterns based on common database conventions
- **Table-Based Strategy**: Uses table names directly for relationship labels
- **LLM Strategy**: Uses OpenAI to generate contextually appropriate relationship names

### 📊 Enhanced Schema Analysis

- **Entity vs Join Table Classification**: Automatically categorizes tables as entities or join tables
- **Relationship Property Mapping**: Converts non-FK columns in join tables to relationship properties
- **Comprehensive Foreign Key Analysis**: Deep analysis of all foreign key relationships

## Core Features

- **Automatic Schema Analysis**: Connects to MySQL and analyzes table structures, relationships, and constraints
- **Intelligent Migration Planning**: Uses OpenAI GPT to generate optimal migration strategies
- **Cypher Query Generation**: Automatically generates Cypher queries for creating nodes, relationships, and constraints
- **Data Type Mapping**: Maps MySQL data types to appropriate Memgraph/Cypher types
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

### Enhanced Usage with Relationship Naming Strategies

Run the enhanced example with different relationship naming strategies:

```bash
uv run python enhanced_example.py
```

### Programmatic Usage

```python
from sql_migration_agent import MySQLToMemgraphAgent

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

# Create agent with different relationship naming strategies
strategies = ["smart", "table_based", "llm"]

for strategy in strategies:
    print(f"Using {strategy} strategy...")

    # Create agent with specific strategy
    agent = MySQLToMemgraphAgent(relationship_naming_strategy=strategy)

    # Define initial state
    initial_state = {
        "mysql_config": mysql_config,
        "memgraph_config": memgraph_config,
        "database_structure": {},
        "migration_queries": [],
        "migration_plan": "",
        "current_step": "Initializing",
        "errors": [],
        "completed_tables": [],
        "total_tables": 0
    }

    # Run migration
    result = agent.workflow.invoke(initial_state)

    print(f"Success: {len(result['errors']) == 0}")
    print(f"Migrated {len(result['completed_tables'])} tables")
    if result.get('database_structure'):
        structure = result['database_structure']
        print(f"Entity tables: {len(structure.get('entity_tables', {}))}")
        print(f"Join tables: {len(structure.get('join_tables', {}))}")
        print(f"Relationships: {len(structure.get('relationships', []))}")
```

## How It Works

The agent follows an enhanced multi-step workflow:

1. **Advanced Schema Analysis**:

   - Connects to MySQL database
   - Extracts table schemas, foreign keys, and relationships
   - **NEW**: Classifies tables as entity tables vs join tables
   - **NEW**: Detects many-to-many relationships via join tables
   - Counts rows in each table

2. **Migration Planning**:

   - Uses OpenAI GPT to analyze the database structure
   - Generates an optimal migration plan considering dependencies
   - **NEW**: Plans for both entity migration and relationship creation
   - Identifies potential issues and optimizations

3. **Enhanced Query Generation**:

   - Maps MySQL data types to Cypher types
   - **NEW**: Generates node creation queries excluding foreign key columns
   - **NEW**: Creates one-to-many relationship queries from foreign keys
   - **NEW**: Creates many-to-many relationship queries from join tables
   - **NEW**: Applies configurable relationship naming strategies
   - Generates constraint and index creation queries

4. **Query Validation**:

   - Tests connection to Memgraph
   - Validates query syntax

5. **Enhanced Migration Execution**:

   - Creates constraints and indexes first
   - **NEW**: Migrates entity tables only (excludes join tables from node creation)
   - **NEW**: Removes foreign key columns from node properties
   - **NEW**: Creates one-to-many relationships from foreign keys
   - **NEW**: Creates many-to-many relationships from join table data

6. **Verification**:
   - Validates migration by comparing node and relationship counts
   - Checks data integrity and completeness

## Relationship Naming Strategies

The agent supports three different strategies for naming relationships:

### 1. Smart Strategy (Default)

Uses intelligent patterns based on common database conventions:

```python
agent = MySQLToMemgraphAgent(relationship_naming_strategy="smart")
```

Examples:

- `customer` → `order`: `PLACED`
- `film` → `actor`: `FEATURES`
- `film_actor` join table: `ACTED_IN`
- `user` → `address`: `LOCATED_AT`

### 2. Table-Based Strategy

Uses table names directly for relationship labels:

```python
agent = MySQLToMemgraphAgent(relationship_naming_strategy="table_based")
```

Examples:

- `customer` → `order`: `HAS_ORDER`
- `film_actor` join table: `FILM_ACTOR`
- `user` → `role`: `HAS_ROLE`

### 3. LLM Strategy

Uses OpenAI to generate contextually appropriate names:

```python
agent = MySQLToMemgraphAgent(relationship_naming_strategy="llm")
```

The LLM analyzes table names and context to suggest meaningful relationship names. Falls back to smart strategy if LLM fails.

## Enhanced Database Structure Transformation

The agent performs sophisticated transformations:

### Entity Tables → Nodes

- **Tables → Node Labels**: Each entity table becomes a node type
- **Primary Keys → Node IDs**: Primary keys become unique identifiers
- **Non-FK Columns → Properties**: Regular columns become node properties
- **FK Columns → Removed**: Foreign key columns are excluded from properties

### Join Tables → Relationships

- **Junction Tables → Relationships**: Many-to-many tables become relationships
- **Additional Columns → Relationship Properties**: Non-FK columns become relationship properties
- **Table Detection**: Automatically identifies tables with mostly foreign keys

### Foreign Keys → Relationships

- **FK Constraints → Directed Edges**: Foreign keys become graph relationships
- **Configurable Names**: Relationship labels generated using selected strategy

Example transformations:

**Before (MySQL)**:

```sql
-- Entity tables
CREATE TABLE film (film_id, title, description, rating);
CREATE TABLE actor (actor_id, first_name, last_name);

-- Join table
CREATE TABLE film_actor (
    film_id INT REFERENCES film(film_id),
    actor_id INT REFERENCES actor(actor_id),
    last_update TIMESTAMP
);
```

**After (Memgraph)**:

```cypher
// Entity nodes (FK columns removed)
CREATE (f:Film {film_id: 1, title: "Movie", description: "...", rating: "PG"})
CREATE (a:Actor {actor_id: 1, first_name: "John", last_name: "Doe"})

// Relationship with properties from join table
CREATE (a)-[:ACTED_IN {last_update: "2023-01-01"}]->(f)
```

## Data Type Mappings

The agent maps MySQL types to Cypher-compatible types:

- `INT` → `Integer`
- `VARCHAR`/`CHAR` → `String`
- `TEXT` → `String`
- `DATE`/`DATETIME` → `LocalDate`/`LocalDateTime`
- `FLOAT`/`DOUBLE` → `Float`
- `DECIMAL` → `Decimal`

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
