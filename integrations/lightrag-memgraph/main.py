import os
import asyncio
from lightrag import LightRAG
from lightrag.llm.openai import gpt_4o_mini_complete, gpt_4o_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import setup_logger
from openai import AsyncOpenAI
import numpy as np
import time


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Set LightRAG working directory.
setup_logger("lightrag", level="DEBUG")
WORKING_DIR = "./rag_storage.out"
# # NOTE: Clean everything becasue we are testing.
# if os.path.exists(WORKING_DIR):
#     import shutil
#     shutil.rmtree(WORKING_DIR)
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

DUMMY_TEXTS = [
    """In the heart of the bustling city, a small bookstore stood as a
    sanctuary for dreamers and thinkers alike. Its shelves, lined with
    stories from every corner of the world, beckoned to those seeking
    adventure, solace, or simply a quiet moment away from the noise.
    The owner, an elderly gentleman with a gentle smile, greeted each
    visitor as if they were an old friend. On rainy afternoons, the
    soft patter of drops against the windows created a symphony that
    mingled with the rustle of pages. Children gathered in the reading
    nook, their imaginations ignited by tales of dragons and distant
    lands. College students found refuge among the stacks, their minds
    wandering as they prepared for exams. The bookstore was more than a
    place to buy books; it was a haven where stories came alive,
    friendships blossomed, and the magic of words wove its spell on all
    who entered.""",
    """Beneath the golden canopy of autumn leaves, a
    quiet park unfolded its charm to those who wandered its winding
    paths. Joggers traced familiar routes, their breath visible in the
    crisp morning air, while elderly couples strolled hand in hand,
    reminiscing about days gone by. Children’s laughter echoed from the
    playground, where swings soared and slides became mountains to
    conquer. A painter sat on a weathered bench, capturing the fiery
    hues of the season on her canvas, her brush dancing with
    inspiration. Nearby, a group of friends gathered for a picnic,
    sharing stories and homemade treats as squirrels darted hopefully
    around their feet. The gentle breeze carried the scent of earth and
    fallen leaves, inviting all to pause and savor the moment. In this
    tranquil oasis, time seemed to slow, offering a gentle reminder of
    nature’s beauty and the simple joys that color everyday life.""",
    """On the edge of a sleepy coastal village, a lighthouse stood
    sentinel against the relentless waves. Its beacon, steadfast and
    bright, guided fishermen safely home through fog and storm. The
    keeper, a solitary figure with weathered hands, tended the light
    with unwavering dedication, his days marked by the rhythm of tides
    and the cries of gulls. Each evening, as the sun dipped below the
    horizon, the village gathered on the shore to watch the sky ignite
    in shades of orange and violet. Children chased the surf, their
    laughter mingling with the roar of the sea. Local artisans
    displayed their crafts at the market, their wares shaped by the
    stories and traditions of generations. The lighthouse, a symbol of
    hope and resilience, reminded all who saw it that even in the
    darkest nights, a guiding light could be found, illuminating the
    path home.""",
]


# Configure the LLM client.
client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "sk-local"),
    base_url=os.getenv("OPENAI_API_BASE", "http://127.0.0.1:8000/v1"),
)


async def vllm_gpt_oss_20b_v0(prompt: str, system_prompt: str = "", **kwargs) -> str:
    resp = await client.chat.completions.create(
        model="gpt-oss-20b",  # must match --served-model-name from vLLM
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=kwargs.get("temperature", 0.0),
        # max_tokens is the output lenght
        max_tokens=min(
            kwargs.get("max_tokens", 512), 1024
        ),  # NOTE: gpt-oss-20b was configured with 16kB of context window -> min  ~15kB for the input.
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content or ""


async def vllm_gpt_oss_20b_v1(prompt: str, **kwargs) -> str:
    resp = await client.chat.completions.create(
        model="gpt-oss-20b",  # must match --served-model-name from vLLM
        messages=[{"role": "user", "content": prompt}],
        temperature=kwargs.get("temperature", 0.0),
        # max_tokens is the output lenght
        max_tokens=min(
            kwargs.get("max_tokens", 512), 2048
        ),  # NOTE: gpt-oss-20b was configured with 16kB of context window -> min  ~15kB for the input.
    )
    return resp.choices[0].message.content or ""


async def vllm_gpt_oss_20b_v2(
    prompt,
    system_prompt=None,
    history_messages=None,
    enable_cot: bool = False,
    keyword_extraction: bool = False,
    **kwargs,
) -> str:
    if history_messages is None:
        history_messages = []
    response_format = None
    if keyword_extraction:
        from lightrag.utils import GPTKeywordExtractionFormat

        response_format = GPTKeywordExtractionFormat
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    resp = await client.chat.completions.create(
        model="gpt-oss-20b",  # must match your vLLM --served-model-name
        messages=messages,
        temperature=kwargs.get("temperature", 0.0),
        max_tokens=kwargs.get("max_tokens", 1024),
        response_format=response_format,  # may be None
    )
    return resp.choices[0].message.content or ""


class DummyEmbed:
    def __init__(self, dim: int = 1):
        self.embedding_dim = dim

    async def __call__(self, texts: list[str]) -> np.ndarray:
        return np.ones((len(texts), self.embedding_dim), dtype=float)


async def initialize_rag_no_embed(working_dir: str, llm_model_func):
    rag = LightRAG(
        working_dir=working_dir,
        # llm_model_name="gpt-oss-20b", # BUG -> LightRAG has hardcoded stuff..., whatever you put here the gpt-4o-mini is going to be used
        llm_model_func=llm_model_func,
        embedding_func=DummyEmbed(dim=1),
        vector_storage="NanoVectorDBStorage",
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag


async def test_vllm_dummy():
    text = "Apple acquired Beats in 2014."
    out = await vllm_gpt_oss_20b_v1(f"Extract entities and relations from: '{text}'")
    print(out)


async def main():
    try:
        rag = await initialize_rag_no_embed(WORKING_DIR, gpt_4o_mini_complete)
        total_time = 0.0
        for idx, text in enumerate(DUMMY_TEXTS):
            start_time = time.perf_counter()
            await rag.ainsert(text)
            end_time = time.perf_counter()
            elapsed = end_time - start_time
            total_time += elapsed
            print(f"Text {idx+1} inserted in {elapsed:.4f} seconds.")
        print(
            f"Total time for inserting {len(DUMMY_TEXTS)} texts: {total_time:.4f} seconds."
        )
        if len(DUMMY_TEXTS) > 0:
            print(f"Average time per text: {total_time/len(DUMMY_TEXTS):.4f} seconds.")

        print(await rag.get_graph_labels())
        kg_data = await rag.get_knowledge_graph(node_label="City", max_depth=3)
        print("KNOWLEDGE GRAPH DATA:")
        print(kg_data)

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if rag:
            await rag.finalize_storages()


async def manual_extraction():
    system_prompt = """You are an information extraction model.
Always respond **only** with valid JSON in the format:
{
  "entities": ["Entity1", "Entity2"],
  "relations": [
    {"subject": "Entity1", "predicate": "relation", "object": "Entity2"}
  ]
}
Do not include explanations or text outside JSON.
"""
    for text in DUMMY_TEXTS:
        out = await vllm_gpt_oss_20b_v0(
            f"Extract entities and relations from: '{text}'", system_prompt
        )
        print(out)


if __name__ == "__main__":
    asyncio.run(main())
    # asyncio.run(test_vllm_dummy())
    # asyncio.run(manual_extraction())
