# SIC Scraper

Scrapes the SIC (Standard Industrial Classification) hierarchy from OSHA and generates Cypher queries for Memgraph import.

## SIC Hierarchy

```
Division (A-J) → Major Group (2-digit) → Industry Group (3-digit) → Industry (4-digit)
```

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Full scrape
python main.py

# Quick scrape (skip industry details)
python main.py --no-industry-details

# With batch mode for optimized imports
python main.py --batch-mode
```

## Output

- `output/sic_data.json` - Complete hierarchy
- `output/sic_import.cypherl` - Cypher import queries

## Import to Memgraph

```bash
mgconsole < output/sic_import.cypherl
```

## Generate Embeddings

```bash
# Generate embeddings for vector search
python embeddings.py --from-json output/sic_data.json

# Load into Memgraph
mgconsole < output/sic_embeddings.cypherl
```

## Data Source

https://www.osha.gov/data/sic-manual
