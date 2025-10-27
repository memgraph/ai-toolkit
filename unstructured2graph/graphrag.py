from memgraph_toolbox.api.memgraph import Memgraph
from memgraph import compute_embeddings, create_vector_search_index


if __name__ == "__main__":
    # TODO(gitbuda): Add the import here.

    memgraph = Memgraph()
    # TODO(gitbuda): Add options to skip the below steps.
    # create_vector_search_index(memgraph, "Chunk", "embedding")
    # compute_embeddings(memgraph, "Chunk")

    for node in memgraph.query(
        f"""
        CALL embeddings.compute_text(['Hello world prompt']) YIELD embeddings, success
        CALL vector_search.search('vs_name', 10, embeddings[0]) YIELD * RETURN *;
    """
    ):
        print(node["node"]["text"])
        print("----")
