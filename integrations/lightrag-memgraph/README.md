# 🔗 lightrag-memgraph

lightrag-memgraph is an integration that connects
[lightrag](https://github.com/HKUDS/LightRAG) and
[memgraph](https://github.com/memgraph/memgraph). The library began as a small
wrapper designed to specifically configure Memgraph within a pipeline that
processes unstructured data (various texts) and transforms it into an
ontology/entity schema graph. In other words, it enables you to extract and
enhance entities from unstructured documents, storing them in a graph for
powerful querying and analysis. Ideal for building knowledge graphs, improving
data discovery, and leveraging advanced AI techniques on top of your domain
data.

## General Notes

- Entity/relationship extraction is high-quality, but also high-cost and
relatively slow.
- The goal over time is to expose time and cost metrics (e.g., $ per your
specific document page or chunk).

## Quick start

**Prerequisites:** [Memgraph](https://memgraph.com/docs/getting-started) running
*(default `bolt://localhost:7687`), and an LLM API key (e.g. `OPENAI_API_KEY` or
*`ANTHROPIC_API_KEY`).

**Install:**

```bash
pip install lightrag-memgraph
```

**Minimal example** (async): create the wrapper, initialize with a working
*directory, insert text, then finalize.

```python
import asyncio
from lightrag_memgraph import MemgraphLightRAGWrapper

async def main():
    wrapper = MemgraphLightRAGWrapper(disable_embeddings=True)
    await wrapper.initialize(working_dir="./lightrag_storage")
    await wrapper.ainsert(input="Your document text here.", file_paths=["doc1"])
    # optional: rag = wrapper.get_lightrag(); print(await rag.get_graph_labels())
    await wrapper.afinalize()

asyncio.run(main())
```

See `example.py` in this repo for a full run with sample texts and graph output.

## Using Anthropic (Claude) as the LLM

LightRAG supports Claude via the `lightrag.llm.anthropic` module. Set your API
key and pass the LLM function and model name when initializing the wrapper. The
list of Anthropic models is available under
https://platform.claude.com/docs/en/about-claude/models.

1. **Set the API key** (required for Claude):

   ```bash
   export ANTHROPIC_API_KEY="your-anthropic-api-key"
   ```

2. **Use Anthropic in code** by passing `llm_model_func` and `llm_model_name` to `initialize()`:

   ```python
   from lightrag.llm.anthropic import anthropic_complete
   from lightrag_memgraph import MemgraphLightRAGWrapper

   wrapper = MemgraphLightRAGWrapper(disable_embeddings=True)  # or set embedding_func
   await wrapper.initialize(
       working_dir="./lightrag_storage",
       llm_model_func=anthropic_complete,
       llm_model_name="claude-3-5-sonnet-20241022",  # or claude-3-haiku-20240307, etc.
   )
   ```

   Preset functions are also available: `claude_3_opus_complete`,
   `claude_3_sonnet_complete`, `claude_3_haiku_complete` (fixed older model
   IDs). For current models, use `anthropic_complete` with the desired
   `llm_model_name`.
   
3. **Embeddings**: Anthropic does not provide embeddings. Either use
`disable_embeddings=True` (as above), or set `embedding_func` to another
provider (e.g. `openai_embed` from `lightrag.llm.openai` with `OPENAI_API_KEY`,
or Voyage AI via `lightrag.llm.anthropic.anthropic_embed` with
`VOYAGE_API_KEY`).

## Using OpenAI as the LLM

Set your API key and optionally choose a model.

1. **Set the API key**:

   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   ```

3. **Use a specific OpenAI model** by passing `llm_model_func` and optionally `llm_model_name`:

   ```python
   from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
   from lightrag_memgraph import MemgraphLightRAGWrapper

   wrapper = MemgraphLightRAGWrapper()
   await wrapper.initialize(
       working_dir="./lightrag_storage",
       llm_model_func=gpt_4o_mini_complete,
       embedding_func=openai_embed,
   )
   ```