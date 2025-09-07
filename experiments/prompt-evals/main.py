import argparse
import csv
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

from mcp_prompt_lib.prompt_lib import ask_with_tools

evaluations_path = Path(__file__).parent.parent.parent / "evaluations"
sys.path.insert(0, str(evaluations_path))
from main import (
    TestCase,
    Evaluator,
    print_evaluation_summary,
)


def run_prompt_evaluation(csv_file_path, model="ollama/llama3.2"):
    """
    Run prompt evaluation with comprehensive evaluation metrics.

    Args:
        csv_file_path: Path to the CSV file containing prompts
        model: Model to use for generating answers
    """
    # Load prompts from CSV
    prompts = {}
    with open(csv_file_path, encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            prompts[row["id"]] = row["prompt"]

    print(f"Loaded {len(prompts)} prompts from {csv_file_path}")
    print("=" * 80)
    # Create test cases and collect answers
    test_cases = []
    answers = {}

    for prompt_id, prompt in prompts.items():
        print(f"\nProcessing prompt '{prompt_id}'...")
        print(f"Prompt: {prompt}")
        print("-" * 40)

        try:
            # Get answer from the model
            answer = ask_with_tools(prompt, model=model)
            answers[prompt_id] = answer
            print(f"Answer: {answer}")

            # For now, we'll use the prompt as context since the current ask_with_tools
            # function doesn't expose tool calls and responses directly
            # In a production system, you'd want to modify ask_with_tools to return
            # both the answer and the tool context
            context = [prompt]

            # Create test case for evaluation
            test_case = TestCase(
                id=prompt_id,
                question=prompt,
                answer=answer,
                context=context,
                metadata={"model": model, "prompt_id": prompt_id},
            )
            test_cases.append(test_case)

        except Exception as e:
            print(f"Error processing prompt '{prompt_id}': {e}")
            # Create a test case with empty answer for evaluation
            test_case = TestCase(
                id=prompt_id,
                question=prompt,
                answer="",
                context=[prompt],
                metadata={"model": model, "prompt_id": prompt_id, "error": str(e)},
            )
            test_cases.append(test_case)

    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)

    # Run evaluation
    evaluator = Evaluator()
    test_run = evaluator.evaluate(test_cases)

    # Display detailed results for each test case
    for test_case in test_run.test_cases:
        print(f"\nTest Case ID: {test_case.id}")
        print(f"Question: {test_case.question}")
        print(f"Answer: {test_case.answer}")

        # Get results for this test case
        case_results = [r for r in test_run.results if r.test_case_id == test_case.id]

        if case_results:
            print(f"\nEvaluation Results ({len(case_results)} metrics):")
            for result in case_results:
                status = "✅ PASS" if result.passed else "❌ FAIL"
                print(f"  {result.metric_name.replace('_', ' ').title()}: {status}")
                print(
                    f"    Score: {result.score.name} ({result.score_numeric:.2f}/5.0)"
                )
                print(f"    Reasoning: {result.reasoning}")
                if result.metadata:
                    print(f"    Metadata: {result.metadata}")
        else:
            print("  No evaluation results available")

        print("-" * 60)

    # Print summary
    print_evaluation_summary(test_run, "Overall Evaluation Summary")

    return test_run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run prompt evaluation with comprehensive metrics."
    )
    parser.add_argument(
        "csv_file_path", help="Path to the CSV file containing prompts."
    )
    parser.add_argument(
        "--model",
        default="ollama/llama3.2",
        help="Model to use for generating answers.",
    )
    args = parser.parse_args()

    test_run = run_prompt_evaluation(args.csv_file_path, args.model)
