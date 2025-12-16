# OSHA SIC Manual Scraper

A Python scraper that extracts the Standard Industrial Classification (SIC) hierarchy from the OSHA website and generates Cypher queries for importing into Memgraph. Includes **semantic search** capabilities using sentence transformer embeddings.

## SIC Hierarchy Structure

The SIC system has a 4-level hierarchical structure:

```
SIC Manual (Root)
└── Division (A-J)
    └── Major Group (2-digit code: 01-99)
        └── Industry Group (3-digit code: 011-999)
            └── Industry (4-digit code: 0111-9999)
```

### Example:

```
Division A: Agriculture, Forestry, And Fishing
└── Major Group 01: Agricultural Production Crops
    └── Industry Group 011: Cash Grains
        ├── 0111: Wheat
        ├── 0112: Rice
        ├── 0115: Corn
        └── 0116: Soybeans
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage (Full Scrape)

```bash
python main.py
```

This will:

1. Scrape all divisions, major groups, industry groups, and industries
2. Generate a JSON file with the complete hierarchy
3. Generate Cypher queries for Memgraph import

### Quick Scrape (No Industry Details)

```bash
python main.py --no-industry-details
```

Skip individual industry page scraping for faster execution.

### Custom Output Directory

```bash
python main.py -o ./my_output
```

### Batch Mode (Optimized for Large Imports)

```bash
python main.py --batch-mode
```

Generates optimized Cypher queries using UNWIND for better performance.

### Adjust Request Delay

```bash
python main.py --delay 1.0
```

Set custom delay between requests (default: 0.5 seconds).

## Output Files

After running, you'll find in the output directory:

- `sic_data.json` - Complete hierarchy in JSON format
- `sic_import.cypherl` - Cypher queries for Memgraph

## Importing into Memgraph

### Using mgconsole

```bash
mgconsole < output/sic_import.cypherl
```

### Using Memgraph Lab

1. Open Memgraph Lab
2. Connect to your Memgraph instance
3. Open the generated `.cypher` file
4. Execute the queries

### Using Python (GQLAlchemy)

```python
from gqlalchemy import Memgraph

db = Memgraph("localhost", 7687)

with open("output/sic_import.cypher", "r") as f:
    queries = f.read().split(";")
    for query in queries:
        query = query.strip()
        if query and not query.startswith("//"):
            db.execute(query)
```

## Graph Schema in Memgraph

### Node Labels

- `:SICManual` - Root node
- `:Division` - Top-level divisions (A-J)
- `:MajorGroup` - Major industry groups (2-digit codes)
- `:IndustryGroup` - Industry groups (3-digit codes)
- `:Industry` - Specific industries (4-digit codes)

### Relationships

- `(:SICManual)-[:HAS_DIVISION]->(:Division)`
- `(:Division)-[:HAS_MAJOR_GROUP]->(:MajorGroup)`
- `(:MajorGroup)-[:HAS_INDUSTRY_GROUP]->(:IndustryGroup)`
- `(:IndustryGroup)-[:HAS_INDUSTRY]->(:Industry)`

### Properties

- **Division**: `code`, `name`
- **MajorGroup**: `code`, `name`, `description`
- **IndustryGroup**: `code`, `name`
- **Industry**: `code`, `name`, `description`, `examples`

## Example Queries

### Find all industries under a division

```cypher
MATCH (d:Division {code: 'A'})-[:HAS_MAJOR_GROUP]->()-[:HAS_INDUSTRY_GROUP]->()-[:HAS_INDUSTRY]->(i:Industry)
RETURN i.code, i.name;
```

### Get the full path for an industry

```cypher
MATCH path = (root:SICManual)-[:HAS_DIVISION|HAS_MAJOR_GROUP|HAS_INDUSTRY_GROUP|HAS_INDUSTRY*]->(i:Industry {code: '0111'})
RETURN [n IN nodes(path) | coalesce(n.code, 'ROOT') + ': ' + n.name] AS hierarchy;
```

### Find all siblings of an industry

```cypher
MATCH (i:Industry {code: '0111'})<-[:HAS_INDUSTRY]-(ig:IndustryGroup)-[:HAS_INDUSTRY]->(sibling)
WHERE sibling.code <> i.code
RETURN sibling.code, sibling.name;
```

### Count industries per division

```cypher
MATCH (d:Division)-[:HAS_MAJOR_GROUP*1..3]->(:Industry)
RETURN d.code, d.name, count(*) AS industry_count
ORDER BY industry_count DESC;
```

## Data Source

Data is scraped from the OSHA SIC Manual:
https://www.osha.gov/data/sic-manual

## Semantic Search with Embeddings

The `embeddings.py` script generates sentence transformer embeddings for Industry Groups, enabling semantic search to find the closest SIC codes for any business description.

### Generate Embeddings

```bash
# From JSON data (no database connection needed)
python embeddings.py --from-json output/sic_data.json

# From Memgraph database
python embeddings.py --from-db --host localhost --port 7687
```

### Output Files

- `output/sic_embeddings.json` - Embeddings with metadata
- `output/sic_embeddings.cypherl` - Cypher to add embeddings to Memgraph

### Load Embeddings into Memgraph

```bash
mgconsole < output/sic_embeddings.cypherl
```

### Programmatic Similarity Search

```python
from embeddings import SICSimilaritySearch

# Load from JSON
searcher = SICSimilaritySearch(embeddings_json='output/sic_embeddings.json')

# Or load from Memgraph
# searcher = SICSimilaritySearch(memgraph_host='localhost', memgraph_port=7687)

# Search for matching SIC codes
results = searcher.search("software development and programming", top_k=5)
for ig, score in results:
    print(f"{ig.ig_code}: {ig.ig_name} (score: {score:.4f})")

# Or get formatted output
print(searcher.search_formatted("restaurant and food service"))
```

### Example Search Results

```
Query: "banking and financial services"

1. SIC Code: 608 (Score: 0.5945)
   Industry Group: Foreign Banking And Branches And Agencies
   Major Group: 60 - Depository Institutions
   Division: H - Finance, Insurance, And Real Estate

2. SIC Code: 602 (Score: 0.5746)
   Industry Group: Commercial Banks
   ...
```

### Interactive Mode

```bash
python embeddings.py --from-json output/sic_data.json --interactive
```

This starts an interactive prompt where you can type descriptions and get matching SIC codes.

## Notes

- The scraper includes a configurable delay between requests to be respectful to the OSHA server
- Full scraping with industry details may take 30-60 minutes depending on network speed
- Quick mode (--no-industry-details) takes about 5-10 minutes
- Embeddings use the `all-MiniLM-L6-v2` model (384 dimensions) by default
