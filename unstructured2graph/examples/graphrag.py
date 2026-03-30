import argparse
import asyncio
import logging
import os

from memgraph_toolbox.api.memgraph import Memgraph
from openai import OpenAI

from unstructured2graph import compute_embeddings, create_vector_search_index
from loading import from_unstructured_with_prep
import prompt_templates


async def full_graphrag(args):
    #### INGESTION
    memgraph = Memgraph()
    if args.ingestion:
        await from_unstructured_with_prep()
        compute_embeddings(memgraph, "Chunk")
        create_vector_search_index(memgraph, "Chunk", "embedding")

    #### RETRIEVAL / GRAPHRAG -> The Native/One-query GraphRAG!
    prompt = "What is different under v3.7 compared to v3.6?"
    retrieved_chunks = []
    for row in memgraph.query(
        f"""
        CALL embeddings.text(['{prompt}']) YIELD embeddings, success
        CALL vector_search.search('vs_name', 5, embeddings[0]) YIELD distance, node, similarity
        MATCH (node)-[r*bfs]-(dst:Chunk)
        WITH DISTINCT dst, degree(dst) AS degree ORDER BY degree DESC
        RETURN dst LIMIT 5;
    """
    ):
        if "description" in row["dst"]:
            retrieved_chunks.append(row["dst"]["description"])
        if "text" in row["dst"]:
            retrieved_chunks.append(row["dst"]["text"])

    #### SUMMARIZATION
    if not retrieved_chunks:
        print("No chunks retrieved. Cannot generate answer.")
    else:
        context = "\n\n".join(retrieved_chunks)
        system_message = prompt_templates.system_message
        user_message = prompt_templates.user_message(context, prompt)
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set. Please set your OpenAI API key."
            )
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            temperature=0.1,
        )
        answer = completion.choices[0].message.content
        print(f"\nQuestion: {prompt}")
        print(f"\nAnswer:\n{answer}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="GraphRAG: Retrieve and answer questions using graph-based RAG"
    )
    parser.add_argument(
        "--ingestion",
        action="store_true",
        help="Run data ingestion (load documents, compute embeddings, create vector index). By default, ingestion is skipped.",
    )
    args = parser.parse_args()

    asyncio.run(full_graphrag(args))
