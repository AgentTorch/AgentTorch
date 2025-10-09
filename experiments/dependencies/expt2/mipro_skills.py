import os
import time

import dspy
import torch
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
import pandas as pd
import mlflow

from o_net_data_processor import create_shared_data_processor, JobMetrics

# Import our DSPy token tracking wrapper
import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from dspy_token_tracking import configure_dspy_with_token_tracking, TokenTrackingLM
from token_tracker import TokenTracker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Token tracking logging (set to WARNING to reduce verbosity)
token_logger = logging.getLogger('dspy_token_tracking')
token_logger.setLevel(logging.WARNING)
token_tracker_logger = logging.getLogger('token_tracker') 
token_tracker_logger.setLevel(logging.WARNING)

# "exp1" = Use processed binary skill values (resolved skills)
# "exp2" = Use original work activity and skills strings
DATA_TYPE = "exp1"
OPTIMIZER_MODE = "light"
EXPERIMENT_ID = f"{DATA_TYPE}_gepa_{OPTIMIZER_MODE}_100"

# Generate unique experiment ID with timestamp for this run
EXPERIMENT_START_TIME = int(time.time())
FULL_EXPERIMENT_ID = f"{EXPERIMENT_ID}_{EXPERIMENT_START_TIME}"

model_path = f"./expt2_models/dspy_models/{FULL_EXPERIMENT_ID}.json"
logs_dir = f"./expt2_models/dspy_logs/{FULL_EXPERIMENT_ID}"
token_logs = f"./expt2_models/token_logs/{FULL_EXPERIMENT_ID}"


from pathlib import Path

mlflow_tracking_dir = os.path.abspath("./expt2_models/mlflow_tracking")
os.makedirs(mlflow_tracking_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(token_logs, exist_ok=True)

# Honor env var if provided; normalize to a proper file URI on Windows
tracking_uri_env = os.environ.get("MLFLOW_TRACKING_URI")
if tracking_uri_env:
    # If no scheme present (e.g., "C:\\path"), convert to file URI
    if "://" not in tracking_uri_env:
        mlflow.set_tracking_uri(Path(tracking_uri_env).resolve().as_uri())
    else:
        mlflow.set_tracking_uri(tracking_uri_env)
else:
    # Default to a proper file URI (file:///C:/...)
    mlflow.set_tracking_uri(Path(mlflow_tracking_dir).resolve().as_uri())

# Enable autologging with all features (official DSPy approach)
mlflow.dspy.autolog(
    log_compiles=True,              # Track optimization process
    log_evals=True,                 # Track evaluation results
    log_traces_from_compile=True    # Track program traces during optimization
)

# Set experiment name
mlflow.set_experiment("DSPY optimization BLS")

# Create token tracker instance directly
token_tracker = TokenTracker.get_instance(
    session_id=FULL_EXPERIMENT_ID,
    log_dir=token_logs,
    auto_save=True
)

# Configure DSPy with Gemini and token tracking using the created tracker
# Does dspy.configure(lm=tracking_lm) internally
tracking_lm = configure_dspy_with_token_tracking(
    model='gemini/gemini-2.5-flash',
    tracker=token_tracker  # Pass the tracker directly
)
dspy.configure_cache(
    enable_disk_cache=False,
    enable_memory_cache=False,
)

logger.info(f"DSPy configured with TokenTrackingLM for session: {FULL_EXPERIMENT_ID}")

# Create shared data processor to ensure consistency with P3O experiment
# Using the same random seed for reproducible results
data_processor = create_shared_data_processor(random_seed=42)

# Load and split data once at module initialization (using same max_data_points as P3O experiment)
MAX_DATA_POINTS = 900  # Must match P3O experiment setting
logger.info("Loading and splitting data once at module initialization...")
train_queries_raw, test_queries_raw = data_processor.get_train_test_split(max_data_points=MAX_DATA_POINTS, data_type=DATA_TYPE)
logger.info(f"Stored {len(train_queries_raw)} train queries and {len(test_queries_raw)} test queries")

class JobAnalysisSignature(dspy.Signature):
    """Analyze skill information and predict knowledge requirements across multiple domains."""
    job_info: str = dspy.InputField(desc="Relevant skills and capabilities (no job title or identifying information)")
    job_metrics: JobMetrics = dspy.OutputField(desc="Predicted knowledge requirements for each domain (0-100 scale)")

class JobAnalysisModule(dspy.Module):
    """DSPy module for job knowledge prediction with optimizable prompts."""
    
    def __init__(self) -> None:
        super().__init__()
        # Use Predict so downstream converters can access .signature
        self.predictor = dspy.Predict(JobAnalysisSignature)
    
    def forward(self, job_info: str) -> dspy.Prediction:
        """Forward pass through the job analysis module."""
        return self.predictor(job_info=job_info)

def create_training_examples(max_examples: int = 20, data_type: str = "exp1") -> List[dspy.Example]:
    """Create training examples from the stored train data.
    
    Args:
        max_examples: Maximum number of examples to create
        data_type: Experiment format ("exp1" for binary skills, "exp2" for original strings)
    """
    logger.info(f"Creating training examples from stored train data for {data_type}...")
    
    # Use stored train queries and convert to DSPy format
    train_queries = train_queries_raw[:max_examples] if max_examples else train_queries_raw
    examples = data_processor.convert_p3o_to_dspy_format(train_queries, data_type=data_type)
    
    logger.info(f"Created {len(examples)} training examples from stored data for {data_type}")
    return examples

def job_metrics_metric_mipro(example: dspy.Example, pred: dspy.Prediction, trace: Optional[Any] = None) -> float:
    """Metric function for MIPROv2 and other optimizers that expect 3 parameters."""
    if not hasattr(pred, 'job_metrics') or pred.job_metrics is None:
        return 0.0
    
    try:
        # Convert both to tensors for comparison
        true_tensor = example.job_metrics.to_torch_tensor()
        pred_tensor = pred.job_metrics.to_torch_tensor()
        
        # Calculate MSE and convert to a reward (lower MSE = higher reward)
        mse = torch.mean((true_tensor - pred_tensor) ** 2)
        
        # Convert MSE to a score (higher is better)
        # Using linear scoring: score = 10000 - mse (since max MSE is 10000)
        score = 10000 - mse.item()

        score = min(10000, score)
        score = max(0, score)
        # score = - mse.item()
        return score
        
    except Exception as e:
        logger.warning(f"Error calculating metric: {e}")
        return 0.0

def job_metrics_metric_gepa(gold: dspy.Example, pred: dspy.Prediction, trace: Optional[Any] = None, pred_name: Optional[str] = None, pred_trace: Optional[Any] = None) -> float:
    """Metric function for GEPA optimizer that expects 5 parameters.
    
    GEPA metric function that accepts five arguments as required:
    (gold, pred, trace, pred_name, pred_trace)
    """
    if not hasattr(pred, 'job_metrics') or pred.job_metrics is None:
        return 0.0
    
    try:
        # Convert both to tensors for comparison
        true_tensor = gold.job_metrics.to_torch_tensor()
        pred_tensor = pred.job_metrics.to_torch_tensor()
        
        # Calculate MSE and convert to a reward (lower MSE = higher reward)
        mse = torch.mean((true_tensor - pred_tensor) ** 2)
        
        # Convert MSE to a score (higher is better)
        # Using linear scoring: score = 10000 - mse (since max MSE is 10000)
        score = 10000 - mse.item()

        score = min(10000, score)
        score = max(0, score)
        # score = - mse.item()
        return score
        
    except Exception as e:
        logger.warning(f"Error calculating metric: {e}")
        return 0.0

# RewardTracker removed - using MLflow for all tracking

# Log parsing functions removed - using MLflow exclusively for tracking

def _log_token_usage_to_mlflow(tracker: TokenTracker):
    """Log comprehensive token usage statistics to MLflow."""
    try:
        # Get detailed session stats from TokenTracker
        session_stats = tracker.get_session_stats()
        
        # Log comprehensive session metrics
        mlflow.log_metric("token_session_total_requests", session_stats.total_requests)
        mlflow.log_metric("token_session_total_tokens", session_stats.total_tokens)
        mlflow.log_metric("token_session_prompt_tokens", session_stats.total_prompt_tokens)
        mlflow.log_metric("token_session_completion_tokens", session_stats.total_completion_tokens)
        mlflow.log_metric("token_session_total_cost_usd", session_stats.total_cost_usd)
        mlflow.log_metric("token_session_avg_tokens_per_request", session_stats.average_tokens_per_request)
        mlflow.log_metric("token_session_avg_cost_per_request", session_stats.average_cost_per_request)
        
        # Log model and provider usage
        for model, count in session_stats.models_used.items():
            mlflow.log_metric(f"token_model_{model}_requests", count)
        
        for provider, count in session_stats.providers_used.items():
            mlflow.log_metric(f"token_provider_{provider}_requests", count)
        
        # Export and log token usage data as artifacts
        token_json_path = tracker.export_to_json(
            filename=f"token_usage_{tracker.session_id}.json"
        )
        token_csv_path = tracker.export_to_csv(
            filename=f"token_usage_{tracker.session_id}.csv"
        )
        
        # Log the token usage files as MLflow artifacts
        mlflow.log_artifact(token_json_path, "token_logs")
        mlflow.log_artifact(token_csv_path, "token_logs")
        
        logger.info("✅ Token usage logged to MLflow")
        
    except Exception as e:
        logger.error(f"Error logging token usage to MLflow: {e}")

def run_mipro_optimization(data_type: str = "exp1", tracking_lm = None) -> Tuple[JobAnalysisModule, List[dspy.Example]]:
    """Run MIPROv2 optimization on the job analysis task.
    
    Following the official DSPy MLflow tracking approach:
    https://dspy.ai/tutorials/optimizer_tracking/
    
    Args:
        data_type: Experiment format ("exp1" for binary skills, "exp2" for original strings)
        
    Returns:
        Tuple of optimized analyzer and training examples
    """
    logger.info(f"Starting MIPROv2 optimization for {data_type}...")
    
    # Create training examples
    trainset = create_training_examples(max_examples=MAX_DATA_POINTS, data_type=data_type)
    
    # Create the module to optimize
    job_analyzer = JobAnalysisModule()
    
    # Test the module before optimization
    logger.info("Testing module before optimization...")
    test_example = trainset[0]
    test_result = job_analyzer(job_info=test_example.job_info)
    logger.info(f"Test prediction type: {type(test_result.job_metrics)}")
    
    # Initialize MIPROv2 optimizer (MLflow autologging will track everything)
    from dspy.teleprompt import MIPROv2

    optimizer = MIPROv2(
        metric=job_metrics_metric_mipro,
        auto=OPTIMIZER_MODE,
        num_threads=2
    )
    
    logger.info("Compiling optimized program with MLflow tracking...")
    
    # Split data for training and validation
    train_size = max(1, len(trainset) // 2)
    train_examples = trainset[:train_size]
    val_examples = trainset[train_size:train_size+2] if len(trainset) > train_size else trainset[:1]
    
    # Start MLflow run with custom run name using FULL_EXPERIMENT_ID
    with mlflow.start_run(run_name=FULL_EXPERIMENT_ID) as run:
        logger.info(f"Started MLflow run with ID: {run.info.run_id}")
        logger.info(f"Run name: {FULL_EXPERIMENT_ID}")
        
        # Log experiment configuration
        mlflow.log_param("experiment_id", EXPERIMENT_ID)
        mlflow.log_param("start_time", EXPERIMENT_START_TIME)
        mlflow.log_param("full_experiment_id", FULL_EXPERIMENT_ID)
        mlflow.log_param("exp_format", data_type)
        
        # Set operation context for training phase
        if tracking_lm and hasattr(tracking_lm, 'set_operation'):
            tracking_lm.set_operation("train")
            logger.info("Token tracking operation set to: train")
        
        # The optimization process will be automatically tracked by TokenTrackingLM and MLflow
        logger.info("Starting MIPRO optimization with automatic token tracking...")
        optimized_analyzer = optimizer.compile(
            student=job_analyzer,
            trainset=train_examples,
            valset=val_examples,
            requires_permission_to_run=False
        )
        
        logger.info("Optimization completed!")
        
        # Set operation context for validation/testing phase
        if tracking_lm and hasattr(tracking_lm, 'set_operation'):
            tracking_lm.set_operation("eval")
            logger.info("Token tracking operation set to: eval (validation)")
        
        # Test the optimized program
        logger.info("Testing optimized program...")
        for i, example in enumerate(val_examples[:2]):
            original_result = job_analyzer(job_info=example.job_info)
            optimized_result = optimized_analyzer(job_info=example.job_info)
            
            original_score = job_metrics_metric_mipro(example, original_result)
            optimized_score = job_metrics_metric_mipro(example, optimized_result)
            improvement = optimized_score - original_score
            
            logger.info(f"Example {i+1}:")
            logger.info(f"  Original score: {original_score:.3f}")
            logger.info(f"  Optimized score: {optimized_score:.3f}")
            logger.info(f"  Improvement: {improvement:.3f}")
            
            # Log test results to MLflow
            mlflow.log_metric(f"test_original_score_{i+1}", original_score)
            mlflow.log_metric(f"test_optimized_score_{i+1}", optimized_score)
            mlflow.log_metric(f"test_improvement_{i+1}", improvement)
        
        # Save the optimized program and log as artifact
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        optimized_analyzer.save(model_path)
        logger.info(f"Optimized program saved to {model_path}")
        
        # Log the model as an artifact
        mlflow.log_artifact(model_path, "optimized_model")
        
        # Set operation context for evaluation phase
        if tracking_lm and hasattr(tracking_lm, 'set_operation'):
            tracking_lm.set_operation("eval")
            logger.info("Token tracking operation set to: eval")
        
        # Test on holdout data and log results to MLflow
        logger.info("Running holdout test and logging results to MLflow...")
        holdout_summary = test_optimized_program_on_holdout(
            optimized_analyzer, 
            max_test_points=100, 
            exp_id=data_type
        )
        
        if holdout_summary:
            # Log holdout test results to MLflow
            mlflow.log_metric("holdout_num_samples", holdout_summary['num_test_samples'])
            mlflow.log_metric("holdout_avg_optimized_score", holdout_summary['avg_optimized_score'])
            mlflow.log_metric("holdout_avg_baseline_score", holdout_summary['avg_baseline_score'])
            mlflow.log_metric("holdout_avg_improvement", holdout_summary['avg_improvement'])
            
            # Log some individual sample results
            for i, result in enumerate(holdout_summary['test_results'][:5]):  # First 5 samples
                mlflow.log_metric(f"holdout_sample_{i+1}_optimized", result['optimized_score'])
                mlflow.log_metric(f"holdout_sample_{i+1}_baseline", result['baseline_score'])
                mlflow.log_metric(f"holdout_sample_{i+1}_improvement", result['improvement'])
            
            logger.info("✅ Holdout test results logged to MLflow")
        
        # Log final token usage to MLflow
        _log_token_usage_to_mlflow(token_tracker)
        
        return optimized_analyzer, trainset, holdout_summary

def test_optimized_program_on_holdout(
    optimized_analyzer: JobAnalysisModule, 
    max_test_points: Optional[int] = None, 
    exp_id: str = "exp1"
) -> Optional[Dict[str, Any]]:
    """Test the optimized MIPROv2 program on held-out test data.
    
    Args:
        optimized_analyzer: The optimized model to test
        max_test_points: Maximum number of test points to evaluate
        exp_id: Experiment format ("exp1" for binary skills, "exp2" for original strings)
        
    Returns:
        Dictionary containing test summary results, or None if test fails
    """
    logger.info("=" * 60)
    logger.info(f"TESTING OPTIMIZED MIPRO MODEL ON HELD-OUT TEST DATA ({exp_id})")
    logger.info("=" * 60)
    
    # Use stored test data and convert to DSPy format
    test_queries = test_queries_raw[:max_test_points] if max_test_points else test_queries_raw
    test_examples = data_processor.convert_p3o_to_dspy_format(test_queries, data_type=exp_id)
    logger.info(f"Testing on {len(test_examples)} held-out test samples for {exp_id}")
    
    if not test_examples:
        logger.error("No test data available!")
        return None
    
    # Create baseline model for comparison
    baseline_analyzer = JobAnalysisModule()
    
    # Test both models
    total_optimized_score = 0.0
    total_baseline_score = 0.0
    test_results = []
    
    logger.info("Running test evaluation...")
    
    from tqdm import tqdm
    for i, test_example in enumerate(tqdm(test_examples, desc="Testing on holdout set")):
        try:
            # Get predictions from both models
            optimized_result = optimized_analyzer(job_info=test_example.job_info)
            baseline_result = baseline_analyzer(job_info=test_example.job_info)
            
            # Calculate scores
            optimized_score = job_metrics_metric_mipro(test_example, optimized_result)
            baseline_score = job_metrics_metric_mipro(test_example, baseline_result)
            
            total_optimized_score += optimized_score
            total_baseline_score += baseline_score
            
            test_results.append({
                'example_idx': i,
                'optimized_score': optimized_score,
                'baseline_score': baseline_score,
                'improvement': optimized_score - baseline_score,
                'job_info': test_example.job_info[:100] + "..." if len(test_example.job_info) > 100 else test_example.job_info
            })
            
            # Log progress every 10 samples
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(test_examples)} test samples")
                
        except Exception as e:
            logger.error(f"Error processing test sample {i}: {e}")
            continue
    
    # Calculate average scores
    avg_optimized_score = total_optimized_score / len(test_examples)
    avg_baseline_score = total_baseline_score / len(test_examples)
    avg_improvement = avg_optimized_score - avg_baseline_score
    
    logger.info("\n" + "-" * 40)
    logger.info("TEST RESULTS")
    logger.info("-" * 40)
    logger.info(f"Average Optimized Score: {avg_optimized_score:.3f}")
    logger.info(f"Average Baseline Score: {avg_baseline_score:.3f}")
    logger.info(f"Average Improvement: {avg_improvement:.3f}")
    
    # Show sample results
    logger.info(f"\nSample Test Results (first 3):")
    for i, result in enumerate(test_results[:3]):
        logger.info(f"Sample {i+1}:")
        logger.info(f"  Job Info: {result['job_info']}")
        logger.info(f"  Optimized Score: {result['optimized_score']:.3f}")
        logger.info(f"  Baseline Score: {result['baseline_score']:.3f}")
        logger.info(f"  Improvement: {result['improvement']:.3f}")
    
    test_summary = {
        'num_test_samples': len(test_examples),
        'avg_optimized_score': avg_optimized_score,
        'avg_baseline_score': avg_baseline_score,
        'avg_improvement': avg_improvement,
        'test_results': test_results
    }
    
    logger.info(f"\n✅ Test evaluation complete!")
    return test_summary


def load_optimized_program(model_path: str = "./saved_models/optimized_job_analyzer") -> JobAnalysisModule:
    """
    Load a previously optimized MIPRO program using DSPy's standard method.
    
    Args:
        model_path: Path to the saved model (default: "./saved_models/optimized_job_analyzer")
        
    Returns:
        Loaded optimized program
    """
    optimized_program = JobAnalysisModule()
    optimized_program.load(model_path)
    logger.info(f"Optimized program loaded from {model_path}")
    return optimized_program


if __name__ == "__main__":
    # Configuration: Choose experiment format
    
    try:
        optimized_program, examples, test_summary = run_mipro_optimization(data_type=DATA_TYPE, tracking_lm=tracking_lm)
        logger.info(f"MIPROv2 optimization completed successfully for {FULL_EXPERIMENT_ID}!")
        
        # Summary results
        logger.info("\n" + "=" * 60)
        logger.info("MIPRO OPTIMIZATION COMPLETE")
        logger.info("=" * 60)
        
        if test_summary:
            logger.info(f"Final Test Results (logged to MLflow):")
            logger.info(f"  Test Samples: {test_summary['num_test_samples']}")
            logger.info(f"  Average Optimized Score: {test_summary['avg_optimized_score']:.3f}")
            logger.info(f"  Average Baseline Score: {test_summary['avg_baseline_score']:.3f}")
            logger.info(f"  Average Improvement: {test_summary['avg_improvement']:.3f}")
        
        logger.info("\n✅ All results logged to MLflow experiment")
        logger.info("🚀 Launch MLflow UI to view complete results:")
        logger.info("   python3 launch_mlflow_ui.py")
        logger.info("📊 Token usage data available in MLflow artifacts and token logs")
        
    except Exception as e:
        logger.error(f"Error during optimization: {e}")
        import traceback
        traceback.print_exc()
        