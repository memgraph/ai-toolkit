"""
SIC Industry Group Embeddings Generator

This script creates embeddings for Industry Group nodes using sentence transformers.
The embeddings capture hierarchical context: Division → Major Group → Industry Group.

Usage:
    python embeddings.py --host localhost --port 7687
    python embeddings.py --from-json output/sic_data.json  # Generate without DB connection
"""

import argparse
import json
from pathlib import Path
from dataclasses import dataclass

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Please install sentence-transformers: pip install sentence-transformers")
    raise

try:
    from gqlalchemy import Memgraph
except ImportError:
    Memgraph = None
    print("Warning: gqlalchemy not installed. Database operations will be unavailable.")


# Default model - good balance of speed and quality
DEFAULT_MODEL = "all-MiniLM-L6-v2"


@dataclass
class IndustryGroupEmbedding:
    """Represents an Industry Group with its embedding context"""

    ig_code: str
    ig_name: str
    mg_code: str
    mg_name: str
    mg_description: str
    div_code: str
    div_name: str
    context_text: str
    embedding: list[float] = None

    def to_dict(self) -> dict:
        return {
            "ig_code": self.ig_code,
            "ig_name": self.ig_name,
            "mg_code": self.mg_code,
            "mg_name": self.mg_name,
            "div_code": self.div_code,
            "div_name": self.div_name,
            "context_text": self.context_text,
            "embedding": self.embedding,
        }


def build_context_text(
    div_name: str, mg_name: str, mg_description: str, ig_name: str
) -> str:
    """
    Build a rich context string for embedding.
    Combines hierarchical information into a single text for embedding.
    """
    parts = [
        f"Division: {div_name}",
        f"Major Group: {mg_name}",
        f"Industry Group: {ig_name}",
    ]

    # Add description if available (truncate if too long)
    if mg_description:
        # Clean up description
        desc = mg_description.replace("SIC Search ", "").strip()
        if len(desc) > 500:
            desc = desc[:500] + "..."
        parts.append(f"Description: {desc}")

    return " | ".join(parts)


def load_industry_groups_from_json(json_path: str) -> list[IndustryGroupEmbedding]:
    """Load industry groups from the JSON export file"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    industry_groups = []

    for division in data:
        div_code = division["code"]
        div_name = division["name"]

        for major_group in division.get("major_groups", []):
            mg_code = major_group["code"]
            mg_name = major_group["name"]
            mg_description = major_group.get("description", "")

            for ig in major_group.get("industry_groups", []):
                ig_code = ig["code"]
                ig_name = ig["name"]

                context_text = build_context_text(
                    div_name, mg_name, mg_description, ig_name
                )

                industry_groups.append(
                    IndustryGroupEmbedding(
                        ig_code=ig_code,
                        ig_name=ig_name,
                        mg_code=mg_code,
                        mg_name=mg_name,
                        mg_description=mg_description,
                        div_code=div_code,
                        div_name=div_name,
                        context_text=context_text,
                    )
                )

    return industry_groups


def load_industry_groups_from_memgraph(
    host: str, port: int
) -> list[IndustryGroupEmbedding]:
    """Load industry groups from Memgraph database"""
    if Memgraph is None:
        raise ImportError("gqlalchemy is required for database operations")

    db = Memgraph(host=host, port=port)

    query = """
    MATCH (d:Division)-[:HAS_MAJOR_GROUP]->(mg:MajorGroup)
          -[:HAS_INDUSTRY_GROUP]->(ig:IndustryGroup)
    RETURN d.code AS div_code, d.name AS div_name,
           mg.code AS mg_code, mg.name AS mg_name, 
           mg.description AS mg_description,
           ig.code AS ig_code, ig.name AS ig_name
    ORDER BY ig.code
    """

    results = list(db.execute_and_fetch(query))
    industry_groups = []

    for row in results:
        context_text = build_context_text(
            row["div_name"], row["mg_name"], row["mg_description"] or "", row["ig_name"]
        )

        industry_groups.append(
            IndustryGroupEmbedding(
                ig_code=row["ig_code"],
                ig_name=row["ig_name"],
                mg_code=row["mg_code"],
                mg_name=row["mg_name"],
                mg_description=row["mg_description"] or "",
                div_code=row["div_code"],
                div_name=row["div_name"],
                context_text=context_text,
            )
        )

    return industry_groups


def generate_embeddings(
    industry_groups: list[IndustryGroupEmbedding],
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 32,
) -> list[IndustryGroupEmbedding]:
    """Generate embeddings for all industry groups"""
    print(f"Loading sentence transformer model: {model_name}")
    model = SentenceTransformer(model_name)

    # Extract all context texts
    texts = [ig.context_text for ig in industry_groups]

    print(f"Generating embeddings for {len(texts)} industry groups...")
    embeddings = model.encode(
        texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True
    )

    # Assign embeddings back to industry groups
    for ig, emb in zip(industry_groups, embeddings):
        ig.embedding = emb.tolist()

    print(f"Generated {len(embeddings)} embeddings of dimension {len(embeddings[0])}")
    return industry_groups


def save_embeddings_to_memgraph(
    industry_groups: list[IndustryGroupEmbedding], host: str, port: int
) -> None:
    """Save embeddings to Memgraph as a property on IndustryGroup nodes"""
    if Memgraph is None:
        raise ImportError("gqlalchemy is required for database operations")

    db = Memgraph(host=host, port=port)

    print(f"Saving embeddings to Memgraph at {host}:{port}...")

    for ig in industry_groups:
        query = """
        MATCH (ig:IndustryGroup {code: $code})
        SET ig.embedding = $embedding,
            ig.context_text = $context_text
        """
        db.execute(
            query,
            {
                "code": ig.ig_code,
                "embedding": ig.embedding,
                "context_text": ig.context_text,
            },
        )

    print(f"Saved embeddings for {len(industry_groups)} industry groups")


def save_embeddings_to_json(
    industry_groups: list[IndustryGroupEmbedding], output_path: str
) -> None:
    """Save embeddings to a JSON file"""
    data = [ig.to_dict() for ig in industry_groups]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Saved embeddings to {output_path}")


def save_embeddings_to_cypherl(
    industry_groups: list[IndustryGroupEmbedding], output_path: str
) -> None:
    """Save embeddings as Cypher statements for loading into Memgraph"""
    queries = []

    for ig in industry_groups:
        # Escape single quotes in context text
        context = ig.context_text.replace("'", "\\'")
        embedding_str = json.dumps(ig.embedding)

        query = (
            f"MATCH (ig:IndustryGroup {{code: '{ig.ig_code}'}}) "
            f"SET ig.embedding = {embedding_str}, "
            f"ig.context_text = '{context}';"
        )
        queries.append(query)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(queries))

    print(f"Saved Cypher queries to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate embeddings for SIC Industry Groups"
    )

    # Input source
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--from-json", type=str, help="Load industry groups from JSON file"
    )
    input_group.add_argument(
        "--from-db",
        action="store_true",
        help="Load industry groups from Memgraph database",
    )

    # Database connection
    parser.add_argument("--host", type=str, default="localhost", help="Memgraph host")
    parser.add_argument("--port", type=int, default=7687, help="Memgraph port")

    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Sentence transformer model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for embedding generation"
    )

    # Output options
    parser.add_argument("--output-json", type=str, help="Save embeddings to JSON file")
    parser.add_argument(
        "--output-cypherl", type=str, help="Save embeddings as Cypher queries"
    )
    parser.add_argument(
        "--save-to-db", action="store_true", help="Save embeddings directly to Memgraph"
    )

    args = parser.parse_args()

    # Load industry groups
    if args.from_json:
        print(f"Loading industry groups from {args.from_json}")
        industry_groups = load_industry_groups_from_json(args.from_json)
    else:
        print(f"Loading industry groups from Memgraph at {args.host}:{args.port}")
        industry_groups = load_industry_groups_from_memgraph(args.host, args.port)

    print(f"Loaded {len(industry_groups)} industry groups")

    # Generate embeddings
    industry_groups = generate_embeddings(
        industry_groups, model_name=args.model, batch_size=args.batch_size
    )

    # Save outputs
    if args.output_json:
        save_embeddings_to_json(industry_groups, args.output_json)

    if args.output_cypherl:
        save_embeddings_to_cypherl(industry_groups, args.output_cypherl)

    if args.save_to_db:
        save_embeddings_to_memgraph(industry_groups, args.host, args.port)

    # Default output if nothing specified
    if not any([args.output_json, args.output_cypherl, args.save_to_db]):
        default_json = "output/sic_embeddings.json"
        default_cypherl = "output/sic_embeddings.cypherl"
        save_embeddings_to_json(industry_groups, default_json)
        save_embeddings_to_cypherl(industry_groups, default_cypherl)


if __name__ == "__main__":
    main()
