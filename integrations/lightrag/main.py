import csv
import os
import sys

import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, gpt_4o_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import setup_logger
import kagglehub

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from evaluations.main import TestCase, Evaluator, print_evaluation_summary

setup_logger("lightrag", level="INFO")

WORKING_DIR = "./rag_storage"
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete,
        # TODO(gitbuda): This should be somehow read from the .env file (it's ignored).
        # TODO(gitbuda): MemgraphStorage is missing under Parameters under https://github.com/HKUDS/LightRAG?tab=readme-ov-file#lightrag-init-parameters -> PR.
        graph_storage="MemgraphStorage",
    )
    # IMPORTANT: Both initialization calls are required!
    await rag.initialize_storages()  # Initialize storage backends
    await initialize_pipeline_status()  # Initialize processing pipeline
    return rag


async def naive_search(rag: LightRAG, query: str):
    return await rag.aquery(query, param=QueryParam(mode="naive"))


async def hybrid_search(rag: LightRAG, query: str):
    return await rag.aquery(query, param=QueryParam(mode="hybrid"))


async def main():
    try:
        # Initialize the dataset
        rag = await initialize_rag()
        path = os.path.join(
            kagglehub.dataset_download("dkhundley/sample-rag-knowledge-item-dataset"),
            "rag_sample_qas_from_kis.csv",
        )
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                await rag.ainsert(row["ki_text"])

        # Get answers to the test questions (naive RAG) -> BaselineRAG
        test_cases_naive = []
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                answer = await naive_search(rag, row["sample_question"])
                test_cases_naive.append(
                    {
                        "query": row["sample_question"],
                        "expected_answer": row["sample_ground_truth"],
                        "answer": answer,
                    }
                )

        # Get answers to the test questions (hybrid RAG) -> GraphRAG
        test_cases_hybrid = []
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                answer = await hybrid_search(rag, row["sample_question"])
                expected_answer = row["sample_ground_truth"]
                test_cases_hybrid.append(
                    {
                        "query": row["sample_question"],
                        "expected_answer": expected_answer,
                        "answer": answer,
                    }
                )

        # Evaluation
        evaluator = Evaluator()

        def to_testcase(tc_dict, prefix=""):
            return TestCase(
                question=tc_dict["query"],
                answer=tc_dict["answer"],
                ground_truth=tc_dict.get("expected_answer"),
                metadata={"source": prefix} if prefix else {},
            )

        test_cases_naive_objs = [
            to_testcase(tc, prefix="naive") for tc in test_cases_naive
        ]
        test_cases_hybrid_objs = [
            to_testcase(tc, prefix="hybrid") for tc in test_cases_hybrid
        ]
        test_run_naive = evaluator.evaluate(test_cases_naive_objs)
        test_run_hybrid = evaluator.evaluate(test_cases_hybrid_objs)
        print()
        print_evaluation_summary(test_run_naive, "Naive RAG Summary")
        print()
        print_evaluation_summary(test_run_hybrid, "Hybrid RAG Summary")
        print()

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if rag:
            await rag.finalize_storages()


if __name__ == "__main__":
    asyncio.run(main())
