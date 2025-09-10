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
from deepeval.evaluate.configs import DisplayConfig
from llm_model_resolver import DeepEvalModelResolver, LiteLLMModelResolver
from metrics_aggregator import aggregate_metrics

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
    logging.getLogger("deepeval").setLevel(
        logging.WARNING
    )  # Reduce noise from deepeval


def run_prompt_evaluation(args):
    """
    Run prompt evaluation with comprehensive evaluation metrics using DeepEval.

    Args:
        csv_file_path: Path to the CSV file containing prompts
        model: Model to use for generating answers
        skip_evaluations: If True, skip the evaluation step and only generate answers
    """
    # Load prompts from CSV
    prompts = {}
    with open(args.csv_file_path, encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            prompts[row["id"]] = row["prompt"]
    logger.info(f"Loaded {len(prompts)} prompts from {args.csv_file_path}")
    logger.info("=" * 80)

    if args.llm_generation:
        lll_model_generation = args.llm_generation
    else:
        llm_model_generation = args.llm
    logger.info(f"Using LLM model for generation: {llm_model_generation}")
    litellm_model_resolver = LiteLLMModelResolver(llm_model_generation)

    answers = {}
    test_cases = []
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
                metadata={"model": llm_model_generation, "prompt_id": prompt_id},
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
                metadata={
                    "model": llm_model_generation,
                    "prompt_id": prompt_id,
                    "error": str(e),
                },
            )
            test_cases.append(test_case)

    if args.skip_evaluation:
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

    if args.llm_evaluation:
        llm_model_evaluation = args.llm_evaluation
    else:
        llm_model_evaluation = args.llm
    deepeval_model_resolver = DeepEvalModelResolver(llm_model_evaluation)
    logger.info(f"Using DeepEval model: {deepeval_model_resolver.get_model_name()}")
    deepeval_model = deepeval_model_resolver.get_model()
    metrics = [
        AnswerRelevancyMetric(threshold=0.5, model=deepeval_model),
        FaithfulnessMetric(threshold=0.5, model=deepeval_model),
        ContextualPrecisionMetric(threshold=0.5, model=deepeval_model),
    ]

    try:
        # Run evaluation using DeepEval
        logger.info(f"Running evaluation with Ollama model: {llm_model_generation}")

        # Configure display to disable all output
        if args.evaluation_verbosity == "high-level":
            display_config = DisplayConfig(
                show_indicator=True,
                print_results=False,
                verbose_mode=False,
            )
        else:
            display_config = DisplayConfig(
                show_indicator=True,
                print_results=True,
                verbose_mode=True,
            )
        logger.info(f"Display config: {display_config}")
        results = evaluate(test_cases, metrics, display_config=display_config)
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        if "api_key" in str(e).lower() or "openai" in str(e).lower():
            logger.error("DeepEval is trying to use OpenAI instead of Ollama.")
            logger.error("This might be due to model configuration issues.")
        elif "ollama" in str(e).lower() or "connection" in str(e).lower():
            logger.error(f"Ollama connection error. Please ensure:")
            logger.error(f"1. Ollama is running: ollama serve")
            logger.error(
                f"2. The model '{llm_model_generation}' is available: ollama list"
            )
            logger.error(f"3. The model is pulled: ollama pull {llm_model_generation}")
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

    # Compute and print aggregated metrics for all test cases
    aggregate_metrics(results)

    # DeepEval already provides comprehensive output, so we'll just show a simple summary
    logger.info(f"Evaluation completed for {len(test_cases)} test cases.")
    logger.info("Detailed results are shown above in the DeepEval output.")

    # Print a simple summary
    if results:
        logger.info(f"Evaluation completed successfully!")
        logger.info(
            "See the detailed metrics summary above for individual test case results."
        )
    else:
        logger.warning("No evaluation results to summarize.")

    return results


if __name__ == "__main__":
    configure_logging(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Run prompt evaluation with comprehensive metrics."
    )
    parser.add_argument(
        "csv_file_path", help="Path to the CSV file containing prompts."
    )
    parser.add_argument(
        "--llm",
        default="ollama/llama3.2:latest",
        help="Model to use for both generating answers and evaluating.",
    )
    parser.add_argument(
        "--llm-generation",
        default=None,
        help="Model to use for generating answers.",
    )
    parser.add_argument(
        "--llm-evaluation",
        default=None,
        help="Model to use for both generating answers and evaluating.",
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip the evaluation step and only generate answers.",
    )
    parser.add_argument(
        "--evaluation-verbosity",
        default="high-level",
        choices=["high-level", "granular"],
        help="Verbosity level for the evaluation.",
    )
    args = parser.parse_args()
    test_run = run_prompt_evaluation(args)
