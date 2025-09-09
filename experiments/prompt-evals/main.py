import argparse
import csv
import logging

from mcp_prompt_lib.prompt_lib import ask_with_tools
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
)
from llm_model_resolver import DeepEvalModelResolver, LiteLLMModelResolver

logger = logging.getLogger(__name__)


def configure_logging(level=logging.INFO, format_string=None):
    """
    Configure logging for all modules in the prompt evaluation system.
    Uses proper logging hierarchy - child loggers inherit from root logger.
    Args:
        level: Logging level (default: logging.INFO)
        format_string: Custom format string for log messages
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Configure root logger - all child loggers will inherit this configuration
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=level,
            format=format_string,
            force=True,  # Override any existing configuration
        )
    # Set level for this specific logger
    logger.setLevel(level)
    # Optional: Set specific levels for noisy third-party libraries
    # logging.getLogger('deepeval').setLevel(logging.WARNING)  # Reduce noise from deepeval


def run_prompt_evaluation(
    csv_file_path, model="ollama/llama3.2:latest", skip_evaluations=False
):
    """
    Run prompt evaluation with comprehensive evaluation metrics using DeepEval.

    Args:
        csv_file_path: Path to the CSV file containing prompts
        model: Model to use for generating answers
        skip_evaluations: If True, skip the evaluation step and only generate answers
    """
    # Load prompts from CSV
    prompts = {}
    with open(csv_file_path, encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            prompts[row["id"]] = row["prompt"]

    logger.info(f"Loaded {len(prompts)} prompts from {csv_file_path}")
    logger.info("=" * 80)

    # Create test cases and collect answers
    test_cases = []
    answers = {}
    litellm_model_resolver = LiteLLMModelResolver(model)

    for prompt_id, prompt in prompts.items():
        logger.info(f"\nProcessing prompt '{prompt_id}'...")
        logger.info(f"Prompt: {prompt}")
        logger.info("-" * 40)

        try:
            # Get answer from the model
            answer = ask_with_tools(
                prompt, model=litellm_model_resolver.get_model_name()
            )
            answers[prompt_id] = answer
            logger.info(f"Answer: {answer}")

            # Handle None or empty answers - DeepEval requires non-null actual_output
            if answer is None or answer == "":
                answer = "No response generated"
                logger.warning(
                    f"No response generated for prompt '{prompt_id}', using placeholder"
                )

            # For now, we'll use the prompt as context since the current ask_with_tools
            # function doesn't expose tool calls and responses directly
            # In a production system, you'd want to modify ask_with_tools to return
            # both the answer and the tool context
            context = [prompt]

            # Create DeepEval test case for evaluation
            test_case = LLMTestCase(
                input=prompt,
                actual_output=answer,
                # TODO(gitbuda): Use expected output instead of actual output (e.g. using human input or another LLM).
                expected_output=answer,  # For now, use actual output as expected
                retrieval_context=context,
                metadata={"model": model, "prompt_id": prompt_id},
            )
            test_cases.append(test_case)

        except Exception as e:
            logger.error(f"Error processing prompt '{prompt_id}': {e}")
            # Create a test case with placeholder answer for evaluation
            test_case = LLMTestCase(
                input=prompt,
                actual_output="Error occurred during processing",
                expected_output="Error occurred during processing",  # For now, use actual output as expected
                retrieval_context=[prompt],
                metadata={"model": model, "prompt_id": prompt_id, "error": str(e)},
            )
            test_cases.append(test_case)

    if skip_evaluations:
        logger.info("\n" + "=" * 80)
        logger.info("EVALUATIONS SKIPPED")
        logger.info("=" * 80)
        logger.info(f"Generated answers for {len(test_cases)} prompts.")
        logger.info("Evaluations were skipped as requested.")
        return []

    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 80)

    # Check if we have any valid test cases
    if not test_cases:
        logger.warning("No test cases to evaluate!")
        return []

    deepeval_model_resolver = DeepEvalModelResolver(model)
    logger.info(f"Using DeepEval model: {deepeval_model_resolver.get_model_name()}")
    deepeval_model = deepeval_model_resolver.get_model()

    # Define DeepEval metrics with Ollama model
    metrics = [
        AnswerRelevancyMetric(threshold=0.5, model=deepeval_model),
        FaithfulnessMetric(threshold=0.5, model=deepeval_model),
        ContextualPrecisionMetric(threshold=0.5, model=deepeval_model),
    ]

    try:
        # Run evaluation using DeepEval
        logger.info(f"Running evaluation with Ollama model: {model}")
        results = evaluate(test_cases, metrics)
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        if "api_key" in str(e).lower() or "openai" in str(e).lower():
            logger.error("\nDeepEval is trying to use OpenAI instead of Ollama.")
            logger.error("This might be due to model configuration issues.")
        elif "ollama" in str(e).lower() or "connection" in str(e).lower():
            logger.error(f"\nOllama connection error. Please ensure:")
            logger.error(f"1. Ollama is running: ollama serve")
            logger.error(f"2. The model '{model}' is available: ollama list")
            logger.error(f"3. The model is pulled: ollama pull {model}")
            logger.error(f"4. Environment variable is set:")
            logger.error(f"   - OLLAMA_BASE_URL (default: http://localhost:11434)")
        else:
            logger.error(
                "This might be due to missing API keys or other configuration issues."
            )
            logger.error(
                "Please ensure you have the required setup for DeepEval with Ollama."
            )
        return []

    # DeepEval already provides comprehensive output, so we'll just show a simple summary
    logger.info(f"\nEvaluation completed for {len(test_cases)} test cases.")
    logger.info("Detailed results are shown above in the DeepEval output.")

    # Print a simple summary
    if results:
        logger.info(f"\nEvaluation completed successfully!")
        logger.info(
            "See the detailed metrics summary above for individual test case results."
        )
    else:
        logger.warning("No evaluation results to summarize.")

    return results


def print_evaluation_summary(results, title="Evaluation Summary"):
    """
    Print a formatted evaluation summary for DeepEval results.

    Args:
        results: List of DeepEval evaluation results
        title: Optional title for the summary
    """
    logger.info(f"\n{title}:")
    logger.info("=" * 50)

    if not results:
        logger.warning("No evaluation results available")
        return

    # Group results by metric
    metric_results = {}
    for result in results:
        if isinstance(result, tuple) and len(result) >= 2:
            # Extract metric result from tuple
            metric_result = result[1]
            metric_name = metric_result.metric_name
            if metric_name not in metric_results:
                metric_results[metric_name] = []
            metric_results[metric_name].append(metric_result)

    # Calculate summary statistics
    total_tests = len(
        set(r[0] for r in results if isinstance(r, tuple) and len(r) >= 2)
    )
    total_metrics = len(results)

    # Count passed/failed by metric
    metric_summary = {}
    for metric_name, metric_results_list in metric_results.items():
        passed = sum(1 for r in metric_results_list if r.success)
        total = len(metric_results_list)
        avg_score = (
            sum(r.score for r in metric_results_list) / total if total > 0 else 0
        )

        metric_summary[metric_name] = {
            "passed": passed,
            "total": total,
            "pass_rate": passed / total if total > 0 else 0,
            "avg_score": avg_score,
        }

    # Overall pass rate (test cases that pass ALL their metrics)
    test_case_results = {}
    for result in results:
        if isinstance(result, tuple) and len(result) >= 2:
            test_case, metric_result = result[0], result[1]
            if test_case not in test_case_results:
                test_case_results[test_case] = []
            test_case_results[test_case].append(metric_result)

    passed_test_cases = sum(
        1
        for test_case_results_list in test_case_results.values()
        if all(r.success for r in test_case_results_list)
    )
    total_test_cases = len(test_case_results)
    overall_pass_rate = (
        passed_test_cases / total_test_cases if total_test_cases > 0 else 0
    )

    logger.info(f"Total Test Cases: {total_test_cases}")
    logger.info(f"Total Metric Evaluations: {total_metrics}")
    logger.info(
        f"Overall Pass Rate: {overall_pass_rate:.2%} ({passed_test_cases}/{total_test_cases})"
    )
    logger.info("")

    logger.info("Metric Summary:")
    for metric_name, summary in metric_summary.items():
        logger.info(f"  {metric_name.replace('_', ' ').title()}:")
        logger.info(
            f"    Pass Rate: {summary['pass_rate']:.2%} ({summary['passed']}/{summary['total']})"
        )
        logger.info(f"    Average Score: {summary['avg_score']:.2f}")
        logger.info("")


if __name__ == "__main__":
    configure_logging(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Run prompt evaluation with comprehensive metrics."
    )
    parser.add_argument(
        "csv_file_path", help="Path to the CSV file containing prompts."
    )
    parser.add_argument(
        "--model",
        default="ollama/llama3.2:latest",
        help="Model to use for both generating answers and evaluating.",
    )
    parser.add_argument(
        "--skip-evaluations",
        action="store_true",
        help="Skip the evaluation step and only generate answers.",
    )
    args = parser.parse_args()

    test_run = run_prompt_evaluation(
        args.csv_file_path, args.model, args.skip_evaluations
    )
