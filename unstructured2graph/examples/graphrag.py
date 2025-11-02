from memgraph_toolbox.api.memgraph import Memgraph

from unstructured2graph import compute_embeddings, create_vector_search_index


if __name__ == "__main__":
    #### INGESTION
    # TODO(gitbuda): Add the import here.
    memgraph = Memgraph()
    compute_embeddings(memgraph, "Chunk")
    create_vector_search_index(memgraph, "Chunk", "embedding")

    #### RETRIEVAL / GRAPHRAG
    # The Native/One-query GraphRAG!
    # TODO(gitbuda): In the current small graph, the Chunks are not connected via the entity graph.
    for row in memgraph.query(
        f"""
        CALL embeddings.text(['Hello world prompt']) YIELD embeddings, success
        CALL vector_search.search('vs_name', 10, embeddings[0]) YIELD distance, node, similarity
        MATCH (node)-[r*bfs]-(dst)
        WITH DISTINCT dst, degree(dst) AS degree ORDER BY degree DESC
        RETURN dst LIMIT 10;
    """
    ):
        if "description" in row["dst"]:
            print(row["dst"]["description"])
        if "text" in row["dst"]:
            print(row["dst"]["text"])
        print("----")

    #### SUMMARIZATION
    # TODO(gitbuda): Call LLM to generate the final answer.
