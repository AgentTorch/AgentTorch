"""
Knowledge Prediction Prompt Optimization using P3O
==================================================

This script demonstrates how to use the P3O library to optimize prompts
for job knowledge prediction tasks using O-NET data.

The script shows:
1. How to define prompt templates with placeholders for job data
2. How to create data-aware lambda functions for dynamic prompt generation
3. How to design reward functions for job knowledge prediction tasks
4. How to train and optimize prompts using policy gradient methods
5. How to analyze and interpret the learned prompt parameters

Key Features:
- Template-based prompt generation with {{placeholder}} syntax
- Data-aware prompt customization using lambda functions
- Policy gradient optimization with reward shaping
- Comprehensive training analysis and visualization
- Job-specific knowledge prediction task support
"""
import time

NUM_EPOCHS = 50
MAX_TEST_POINTS = 100
BATCH_SIZE = 50
EXPLORATION_TYPE = "long"
DATA_TYPE = "exp2"
LEARNING_RATE = 0.005 # deault was 0.1

EXPERIMENT_ID = f"{DATA_TYPE}_p3o_b_reward10_normalize_advantages_False"


# Generate unique experiment ID with timestamp for this run
EXPERIMENT_START_TIME = int(time.time())
FULL_EXPERIMENT_ID = f"{EXPERIMENT_ID}_{EXPERIMENT_START_TIME}"

input_data_path_numeric = "./data/job_data_processed/job_data_processed.csv"  # path where numeric skill values live
input_data_slots = "./expt2_data/skill_dimensions_updated/updated_job_vectors.json"  # path where skill vectors live for all the jobs
all_discovered_slots = "/Users/ferozahmad/hackspace/prompt-opt/expt2_data/skill_dimensions_updated/all_skills.csv"   # all skills normalized in lower case
output_path = "./data/high_res_imp_slots"
output_column_prefix = "NUMERIC_knowledge_"
log_dir = "./expt2_models/logs"
token_logs = f"./expt2_models/token_logs/{FULL_EXPERIMENT_ID}"
SKIP_COLUMNS = ["onet_code", "job_title"]

import logging
import os
import sys
from typing import Dict, Any, Tuple, Callable, Optional

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch

# Add parent directory to path to import p3o and llm_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from p3o import PromptGeneratorModule, P3OTrainer, get_exploration_config
from llm_utils import call_llm_structured_output, call_llm_structured_output_gemini
from o_net_data_processor import create_shared_data_processor, JobMetrics as ProcessorJobMetrics, SkillDimensionManager as ProcessorSkillDimensionManager
from token_tracker import TokenTracker

NEW_LINE = "\n"
# Configure logging to output to stdout
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
    force=True
)

# Set instructor logging level to ERROR only to reduce verbosity
logging.getLogger('instructor').setLevel(logging.ERROR)

logger = logging.getLogger(__name__)


# SkillDimensionManager is now imported from o_net_data_processor
SkillDimensionManager = ProcessorSkillDimensionManager


def plot_reward_progression(history, output_path):
    """
    Plot the reward progression during training and save to output path.
    """
    os.makedirs(output_path, exist_ok=True)
    
    rewards = [step['reward'] for step in history]
    losses = [step['loss'] for step in history]
    steps = list(range(len(history)))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot rewards
    ax1.plot(steps, rewards, 'b-', linewidth=2, label='Reward')
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Reward')
    ax1.set_title('Reward Progression During Training')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    mean_reward = np.mean(rewards)
    max_reward = np.max(rewards)
    ax1.axhline(y=mean_reward, color='r', linestyle='--', alpha=0.7, label=f'Mean: {mean_reward:.3f}')
    ax1.axhline(y=max_reward, color='g', linestyle='--', alpha=0.7, label=f'Max: {max_reward:.3f}')
    ax1.legend()
    
    # Plot losses
    ax2.plot(steps, losses, 'r-', linewidth=2, label='Loss')
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss Progression During Training')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    mean_loss = np.mean(losses)
    min_loss = np.min(losses)
    ax2.axhline(y=mean_loss, color='b', linestyle='--', alpha=0.7, label=f'Mean: {mean_loss:.3f}')
    ax2.axhline(y=min_loss, color='orange', linestyle='--', alpha=0.7, label=f'Min: {min_loss:.3f}')
    ax2.legend()
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_path, "reward_progression.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Reward progression plot saved to: {plot_path}")
    
    plt.close('all')
    
    return plot_path


def format_array_for_prompt(arr, max_items=-1):
    """
    Format an array for inclusion in the prompt.
    
    Args:
        arr: List of items to format
        max_items: Maximum number of items to show
        
    Returns:
        Formatted string representation of the array
    """
    if not arr:
        return "[]"
    
    compacted_arr = arr
    if max_items != -1 and max_items < len(arr):
        compacted_arr = arr[:max_items]

    formatted = "\n".join([f"- {item}" for item in compacted_arr])
    formatted = f"\n{formatted}"
    return formatted


# JobMetrics is now imported from o_net_data_processor
JobMetrics = ProcessorJobMetrics

class JobKnowledgeExperiment:
    def __init__(self, input_data_path, input_data_slots_path, output_path, max_data_points=None, batch_size=10, random_seed=42):
        self.input_data_path = input_data_path
        self.input_data_slots_path = input_data_slots_path
        self.output_path = output_path
        self.data = {}
        self.max_data_points = max_data_points
        self.random_seed = random_seed
        self.batch_size = batch_size

        # Initialize shared data processor
        self.data_processor = create_shared_data_processor(random_seed=random_seed)
        self.skill_manager = self.data_processor.skill_manager
        logger.info(f"Initialized shared data processor with {len(self.skill_manager.dimensions)} dimensions")
        
        # Load and split data once at initialization
        logger.info("Loading and splitting data once at initialization...")
        self.train_queries, self.test_queries = self.data_processor.get_train_test_split(max_data_points=max_data_points, data_type=DATA_TYPE)
        logger.info(f"Stored {len(self.train_queries)} train queries and {len(self.test_queries)} test queries")
        
        # Initialize prompt after loading skill dimensions
        self.init_prompt()
        
        # Initialize token tracker with experiment-specific session ID using get_instance for proper singleton
        self.token_tracker = TokenTracker.get_instance(FULL_EXPERIMENT_ID, log_dir=token_logs, auto_save=True)
        logger.info(f"Initialized token tracker with session: {FULL_EXPERIMENT_ID}")
        logger.info(f"Token logs will be saved to: {log_dir}")
        
        # Store trainer reference for TensorBoard access
        self.trainer = None
        
    @staticmethod
    def _get_template_var_name(skill_name: str) -> str:
        """Convert a skill name to a valid template variable name."""
        return skill_name.replace(' ', '_')

    # Data reading methods are now handled by the shared data processor


    def prepare_experiment_data(self, debug=False):
        """Prepare experiment data using the stored train queries."""
        logger.info("Using stored training data...")
        
        # Use pre-loaded training queries
        queries = self.train_queries
        
        self.data = {"queries": queries}
        logger.info(f"Using {len(queries)} stored training queries")
        
        if debug and queries:
            first_query = queries[0]
            logger.info(f"Debug - First query:")
            logger.info(f"  onet_code: {first_query['x'].get('onet_code')}")
            logger.info(f"  job_title: {first_query['x'].get('job_title')}")
            logger.info(f"  sample skills: {list(first_query['x'].keys())[:10]}")
        
        return queries

    def init_prompt(self):
        """Initialize the prompt template for job knowledge prediction."""
        # Create the input slots dynamically from primary skill names
        # Use the same transformation as in get_slot_choices()
        # Create template variables with proper escaping
        input_slots = []
        if DATA_TYPE == "exp1":
            for column in self.train_queries[0]['x'].keys():
                if column in SKIP_COLUMNS:
                    continue
                slot = "        {{" + column + "}}"  # No escaping needed for literal braces in string
                input_slots.append(slot)
            
        else:    
            for primary_skill_name in self.skill_manager.get_dimension_names():
                if primary_skill_name in SKIP_COLUMNS:
                    continue
                var_name = self._get_template_var_name(primary_skill_name)
                slot = "        {{" + var_name + "}}"  # No escaping needed for literal braces in string
                input_slots.append(slot)

        input_slots = "\n".join(input_slots)
        
        self.prompt_template = f"""
        You should act as regression model that predicts numeric metrics of importance for the following "key categories" for the provided job (US Labor market) in the input.
        You will be given some derived skills extracted from O-NET job description about a given job, and you need to predict the following values in a JSON dictionary of str: integer pairs.
        
        OUTPUT FORMAT:

        {{
            "AdministrationAndManagement": <YOUR_NUMERIC_PREDICTION>,
            "Administrative": <YOUR_NUMERIC_PREDICTION>,
            "Biology": <YOUR_NUMERIC_PREDICTION>,
            "BuildingAndConstruction": <YOUR_NUMERIC_PREDICTION>,
            "Chemistry": <YOUR_NUMERIC_PREDICTION>,
            "CommunicationsAndMedia": <YOUR_NUMERIC_PREDICTION>,
            "ComputersAndElectronics": <YOUR_NUMERIC_PREDICTION>,
            "CustomerAndPersonalService": <YOUR_NUMERIC_PREDICTION>,
            "Design": <YOUR_NUMERIC_PREDICTION>,
            "EconomicsAndAccounting": <YOUR_NUMERIC_PREDICTION>,
            "EducationAndTraining": <YOUR_NUMERIC_PREDICTION>,
            "EngineeringAndTechnology": <YOUR_NUMERIC_PREDICTION>,
            "EnglishLanguage": <YOUR_NUMERIC_PREDICTION>,
            "FineArts": <YOUR_NUMERIC_PREDICTION>,
            "FoodProduction": <YOUR_NUMERIC_PREDICTION>,
            "ForeignLanguage": <YOUR_NUMERIC_PREDICTION>,
            "Geography": <YOUR_NUMERIC_PREDICTION>,
            "HistoryAndArcheology": <YOUR_NUMERIC_PREDICTION>,
            "LawAndGovernment": <YOUR_NUMERIC_PREDICTION>,
            "Mathematics": <YOUR_NUMERIC_PREDICTION>,
            "Mechanical": <YOUR_NUMERIC_PREDICTION>,
            "MedicineAndDentistry": <YOUR_NUMERIC_PREDICTION>,
            "PersonnelAndHumanResources": <YOUR_NUMERIC_PREDICTION>,
            "PhilosophyAndTheology": <YOUR_NUMERIC_PREDICTION>,
            "Physics": <YOUR_NUMERIC_PREDICTION>,
            "ProductionAndProcessing": <YOUR_NUMERIC_PREDICTION>,
            "Psychology": <YOUR_NUMERIC_PREDICTION>,
            "PublicSafetyAndSecurity": <YOUR_NUMERIC_PREDICTION>,
            "SalesAndMarketing": <YOUR_NUMERIC_PREDICTION>,
            "SociologyAndAnthropology": <YOUR_NUMERIC_PREDICTION>,
            "Telecommunications": <YOUR_NUMERIC_PREDICTION>,
            "TherapyAndCounseling": <YOUR_NUMERIC_PREDICTION>,
            "Transportation": <YOUR_NUMERIC_PREDICTION>
        }}

        ONLY OUTPUT THIS JSON. DO NOT OUTPUT ANYTHING ELSE!!!!

        (The keys in the JSON are the "key categories" for the job. You need to predict the numeric metrics of importance for each of these key categories in range 0 to 100.
        Do not skip any key categories.)

        The skill details of the job are as follows:
        
        {input_slots}
        """

    def get_slot_choices(self) -> Dict[str, Tuple[int, Any]]:
        """Get the slot choices for the prompt using the shared data processor."""
        return self.data_processor.get_slot_choices_for_p3o(data_type=DATA_TYPE)

    def forward(self, prompt: str, operation: str = "train") -> JobMetrics:
        """
        Forward pass for the job knowledge prediction pipeline.
        
        Args:
            prompt: The input prompt for the LLM
            operation: The operation type ("train" for training, "eval" for evaluation)
        """
        try:
            response: JobMetrics = call_llm_structured_output_gemini(
                prompt, 
                JobMetrics,
                track_tokens=True,
                operation=operation,
                session_id=FULL_EXPERIMENT_ID
            )
            return response
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            # Create fallback JobMetrics with reasonable default values
            fallback_data = {field_name: 0 for field_name in JobMetrics.model_fields.keys()}  # Mid-range values
            return JobMetrics(**fallback_data)

    def get_error_function(self, job_metrics_gt: JobMetrics, job_metrics_pred: JobMetrics) -> float:
        y_true: torch.Tensor = job_metrics_gt.to_torch_tensor()
        y_pred: torch.Tensor = job_metrics_pred.to_torch_tensor()

        preds = y_pred
        gt = y_true
        
        # Calculate mean squared error
        squared_error = (preds - gt) ** 2
        mse = torch.mean(squared_error)
        # Maximize objective function: higher reward for lower MSE
        f_y = (mse).item()  # Convert tensor to Python scalar
        
        # Calculate gradient of the objective function w.r.t. predictions
        # grad_f = d/d_preds (100 - MSE) = d/d_preds (100 - mean((preds - gt)^2))
        # = -2 * mean(preds - gt) = -2 * (preds - gt) / len(preds)
        grad_f_y = (2 * (preds - gt) / preds.shape[0]).numpy()  # Convert to numpy for compatibility

        return y_pred.numpy(), f_y, grad_f_y

    def get_pipeline_method(self, operation: str = "train") -> Callable[[str, Dict[str, Any]], Tuple[Any, Any, Any]]:
        def pipeline(prompt: str, query_sample: Dict[str, Any]) -> Tuple[Any, Any, Any]:
            x = query_sample["x"]
            y_true: torch.Tensor = query_sample["y"].to_torch_tensor()

            y_pred_structured = self.forward(prompt, operation=operation)

            y_pred: torch.Tensor = y_pred_structured.to_torch_tensor()

            # Implement the pseudocode: calculate MSE-based objective function
            preds = y_pred
            gt = y_true
            
            # Calculate mean squared error
            squared_error = (preds - gt) ** 2
            mse = torch.mean(squared_error)

            # logger.info(f"\033[93mpreds: {preds}\033[0m")
            # logger.info(f"\033[93mgt: {gt}\033[0m")
            # logger.info(f"\033[93mmse: {mse}\033[0m")
            
            # penalize by L1 norm heavily
            # max_element_squared_error = squared_error.max().item()
            # combined_error = max_element_squared_error * 0.7 + mse * 0.3
            

            combined_error = mse

            # Scale f_y from [0, 10000] to [-10, +10] range for better policy gradient optimization
            # Original: f_y = 10000 - combined_error (range: [0, 10000])
            # New: f_y = -10 + 20 * (10000 - combined_error) / 10000
            original_f_y = 10000 - (combined_error).item()
            f_y = -10 + 20 * (original_f_y / 10000)  # Scale to [-10, +10]
            
            # Calculate gradient of the scaled objective function w.r.t. predictions
            # Original grad: d/d_preds (10000 - MSE) = -(2 * (preds - gt) / preds.shape[0])
            # Scaled grad: d/d_preds (-10 + 20 * (10000 - MSE)/10000) = d/d_preds (-10 + 2 - 0.002*MSE)
            # = d/d_preds (-0.002 * MSE) = -0.002 * d/d_preds(MSE) = -0.002 * (2 * (preds - gt) / preds.shape[0])
            scaling_factor = 20 / 10000  # 0.002
            grad_f_y = -scaling_factor * (2 * (preds - gt) / preds.shape[0]).numpy()

            # Return additional metrics for TensorBoard logging
            additional_metrics = {
                'mse': mse.item()
            }
            
            return y_pred.numpy(), f_y, grad_f_y, additional_metrics

        return pipeline

    def forward_sample(self):
        """
        Sample a prompt with index=1 for all slots and run the first example through the forward function.
        Uses PromptGeneratorModule from p3o.py for proper prompt generation.
        """
        logger.info("Running forward sample with index=1 for all slots...")
        
        # Prepare data to get the first example
        queries = self.prepare_experiment_data()
        if not queries:
            logger.error("No queries available for forward sampling")
            return None
            
        # Get the first example
        first_example = queries[0]
        data = first_example
        x = data['x']
        
        logger.info(f"Using first example: Job Title = {x.get('job_title', 'DATA_LOADING_ISSUE')}")
        
        # Create PromptGeneratorModule instance
        slot_choices = self.get_slot_choices()
        prompt_gen = PromptGeneratorModule(self.prompt_template, slot_choices, device='cpu')
        
        # Create indices list with index=1 for all slots (bounded by num_categories)
        indices = []
        for placeholder in prompt_gen.placeholders:
            num_categories, _ = slot_choices[placeholder]
            # Use index=1 for all slots (ensure it's within bounds)
            category_index = min(1, num_categories - 1)
            indices.append(category_index)
            
        logger.info(f"\033[92mUsing indices for prompt generation: {indices}\033[0m")
        
        # Log what each placeholder will generate
        print(f"Placeholders values sampled:")
        for i, placeholder in enumerate(prompt_gen.placeholders[:10]):
            num_categories, lambda_func = slot_choices[placeholder]
            category_index = indices[i]
            placeholder_value = lambda_func(category_index, x)
            logger.info(f"Placeholder '{placeholder}' -> Category {category_index}: {placeholder_value[:100]}{'...' if len(placeholder_value) > 100 else ''}")
        
        # Generate prompt using PromptGeneratorModule
        prompt = prompt_gen.generate(indices, data_sample=data)
        
        logger.info(f"\033[94m\nGenerated prompt (first 10000 chars):\n{prompt[:10000]}{'...' if len(prompt) > 10000 else ''}\033[0m")
        
        # Run through forward function
        try:
            logger.info("\nRunning forward pass...")
            prediction = self.forward(prompt, operation="train")  # Training operation for sampling
            
            logger.info("Forward pass completed successfully!")
            logger.info(f"Prediction sample (first 5 fields):")
            field_names = list(JobMetrics.model_fields.keys())
            for field_name in field_names[:5]:
                value = getattr(prediction, field_name)
                logger.info(f"  {field_name}: {value}")
            logger.info(f"  ... and {len(field_names) - 5} more fields")
            
            # Also show the ground truth for comparison
            y_true = first_example["y"]
            logger.info(f"\nGround truth sample (first 5 fields):")
            for field_name in field_names[:5]:
                value = getattr(y_true, field_name)
                logger.info(f"  {field_name}: {value}")
            logger.info(f"  ... and {len(field_names) - 5} more fields")
            
            return {
                "prompt": prompt,
                "prediction": prediction,
                "ground_truth": y_true,
                "example_data": data,
                "indices": indices,
                "prompt_generator": prompt_gen
            }
            
        except Exception as e:
            logger.error(f"Error during forward pass: {e}")
            return None

    def pipeline_sample(self):
        """
        Sample a prompt with index=1 for all slots and run the first example through the full pipeline.
        This includes the forward pass, reward computation, and gradient calculation.
        Reuses as much code as possible from forward_sample.
        """
        logger.info("Running pipeline sample with index=1 for all slots...")
        
        # Reuse the same setup code as forward_sample
        queries = self.prepare_experiment_data()
        if not queries:
            logger.error("No queries available for pipeline sampling")
            return None
            
        # Get the first example
        first_example = queries[1]
        data = first_example
        x = data['x']
        
        logger.info(f"Using first example: Job Title = {x.get('job_title', 'DATA_LOADING_ISSUE')}")
        
        # Create PromptGeneratorModule instance
        slot_choices = self.get_slot_choices()
        prompt_gen = PromptGeneratorModule(self.prompt_template, slot_choices, device='cpu')
        
        # Create indices list with index=0 for all slots (bounded by num_categories)
        indices = []
        for placeholder in prompt_gen.placeholders:
            num_categories, _ = slot_choices[placeholder]
            # Use index=1 for all slots (ensure it's within bounds)
            category_index = min(1, num_categories - 1)
            indices.append(category_index)
            
        logger.info(f"\033[92mUsing indices for prompt generation: {indices}\033[0m")
        
        # Log what each placeholder will generate
        for i, placeholder in enumerate(prompt_gen.placeholders[:10]):
            num_categories, lambda_func = slot_choices[placeholder]
            category_index = indices[i]
            placeholder_value = lambda_func(category_index, x)
            logger.info(f"Placeholder '{placeholder}' -> Category {category_index}: {placeholder_value[:100]}{'...' if len(placeholder_value) > 100 else ''}")
        
        # Generate prompt using PromptGeneratorModule
        prompt = prompt_gen.generate(indices, data_sample=data)
        
        logger.info(f"\033[94m\nGenerated prompt (first 10000 chars):\n{prompt[:10000]}{'...' if len(prompt) > 10000 else ''}\033[0m")
        
        # Run through full pipeline (forward + reward + gradient)
        try:
            logger.info("\nRunning full pipeline (forward + reward + gradient)...")
            
            # Get the pipeline method
            pipeline = self.get_pipeline_method(operation="train")  # Training pipeline for sampling
            
            # Run the pipeline
            y_pred, f_y, grad_f_y, additional_metrics = pipeline(prompt, first_example)
            
            logger.info("Pipeline completed successfully!")
            logger.info(f"Prediction shape: {y_pred.shape}")
            logger.info(f"Reward (f_y): {f_y:.3f}")
            logger.info(f"Gradient shape: {grad_f_y.shape}")
            
            # Show prediction sample (first 5 fields)
            field_names = list(JobMetrics.model_fields.keys())
            logger.info(f"Prediction sample (first 5 fields):")
            for i, field_name in enumerate(field_names[:5]):
                value = int(y_pred[i]) if i < len(y_pred) else 0
                logger.info(f"  {field_name}: {value}")
            logger.info(f"  ... and {len(field_names) - 5} more fields")
            
            # Also show the ground truth for comparison
            y_true = first_example["y"]
            logger.info(f"\nGround truth sample (first 5 fields):")
            for field_name in field_names[:5]:
                value = getattr(y_true, field_name)
                logger.info(f"  {field_name}: {value}")
            logger.info(f"  ... and {len(field_names) - 5} more fields")
            
            # Calculate MSE for comparison
            y_true_tensor = y_true.to_torch_tensor()
            y_pred_tensor = torch.tensor(y_pred, dtype=torch.float32)
            mse = torch.mean((y_pred_tensor - y_true_tensor) ** 2)
            logger.info(f"\nMSE between prediction and ground truth: {mse.item():.3f}")
            
            return {
                "prompt": prompt,
                "prediction": y_pred,
                "ground_truth": y_true,
                "reward": f_y,
                "gradient": grad_f_y,
                "mse": additional_metrics.get('mse', mse.item()),
                "additional_metrics": additional_metrics,
                "example_data": data,
                "indices": indices,
                "prompt_generator": prompt_gen
            }
            
        except Exception as e:
            logger.error(f"Error during pipeline execution: {e}")
            import traceback
            traceback.print_exc()
            return None


    def run(self, num_steps=150, lambda_param=0.5, rho=0.9, beta=0.9, log_interval=25, log_dir=None):
        """
        Run the complete job knowledge prediction optimization process.
        
        Args:
            num_steps: Number of training steps
            lambda_param: Reward shaping weight
            rho: Output mean decay rate
            beta: Baseline decay rate
            log_interval: How often to log to console
            log_dir: Directory for logging experiment data and TensorBoard logs
        """
        logger.info("Starting Job Knowledge Prediction Optimization...")
        logger.info("=" * 60)
        logger.info(f"Experiment ID: {FULL_EXPERIMENT_ID}")
        logger.info(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(EXPERIMENT_START_TIME))}")
        logger.info("=" * 60)

        # Create prompt template and placeholder choices
        self.init_prompt()
        template = self.prompt_template
        placeholder_choices = self.get_slot_choices()
        
        logger.info(f"Template: {template}")
        logger.info(f"Placeholders: {list(placeholder_choices.keys())}")

        # Create prompt generator module
        prompt_gen = PromptGeneratorModule(template, placeholder_choices, device='cpu')

        # Create optimizer and trainer with logging using the experiment ID
        optimizer = torch.optim.Adam(prompt_gen.parameters(), lr=LEARNING_RATE)
        self.trainer = P3OTrainer(
            prompt_gen, 
            optimizer, 
            entropy_coef=0.01, 
            logger=logger,
            log_dir=log_dir,
            experiment_id=FULL_EXPERIMENT_ID,
            batch_size=self.batch_size,  # Use max_data_points as batch_size for random sampling
            max_workers=10,
            normalize_advantages=False
        )

        # Create pipeline and queries
        pipeline = self.get_pipeline_method(operation="train")  # Training pipeline
        queries = self.prepare_experiment_data()

        logger.info(f"Training with {len(queries)} job knowledge prediction queries")
        logger.info("-" * 60)

        # Get exploration configuration
        exp_config = get_exploration_config(EXPLORATION_TYPE, num_steps)
        logger.info(f"Using long exploration: {exp_config['description']}")

        # Run optimization
        history = self.trainer.train(
            queries=queries,
            pipeline=pipeline,
            num_steps=num_steps,
            lambda_param=lambda_param,
            rho=rho,
            beta=beta,
            log_interval=log_interval,
            initial_temperature=exp_config['initial_temperature'],
            final_temperature=exp_config['final_temperature'],
            temperature_decay_steps=exp_config['temperature_decay_steps']
        )

        # Close trainer to flush logs
        self.trainer.close()

        # Print token usage summary
        logger.info("\n" + "=" * 60)
        logger.info("TOKEN USAGE SUMMARY")
        logger.info("=" * 60)
        self.token_tracker.print_summary()

        # Export training token data for analysis (TokenTracker will prepend log_dir automatically)
        token_filename = f"{FULL_EXPERIMENT_ID}_tokens_training.json"
        exported_path = self.token_tracker.export_to_json(token_filename)
        logger.info(f"Training token usage exported to: {exported_path}")
        logger.info("Note: Complete token usage (including evaluation) will be exported after testing")

        # Get final results
        final_results = self.trainer.get_final_results(queries)
        
        # Add experiment metadata to final results
        final_results['experiment_metadata'] = {
            'experiment_id': FULL_EXPERIMENT_ID,
            'start_timestamp': EXPERIMENT_START_TIME,
            'start_time_iso': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(EXPERIMENT_START_TIME)),
            'base_experiment_id': EXPERIMENT_ID
        }

        # Display results
        logger.info("\n" + "=" * 60)
        logger.info("OPTIMIZATION RESULTS")
        logger.info("=" * 60)

        # Show example prompts for all queries with final learned parameters
        logger.info("\033[94m" + "\nExample prompts for all queries with learned parameters:" + "\033[0m")
        example_prompts = final_results.get('example_prompts', [])
        
        if isinstance(example_prompts, list):
            for i, example_prompt in enumerate(example_prompts[:1]):
                logger.info("\033[91m" + f"  Query {i+1}: {example_prompt['query']['x']['job_title']}" + "\033[0m")
                logger.info("\033[94m" + f"    Prompt: '{example_prompt['prompt']}'" + "\033[0m")
        

        # Analyze training progress
        rewards = [step['reward'] for step in history]
        losses = [step['loss'] for step in history]

        logger.info(f"\nTraining Statistics:")
        logger.info(f"Best reward achieved: {final_results.get('best_reward', 0):.3f}")

        logger.info(f"Average Reward: {np.mean(rewards):.3f} ± {np.std(rewards):.3f}")
        logger.info(f"Best Reward: {np.max(rewards):.3f}")
        logger.info(f"Final Reward: {rewards[-1]:.3f}")
        logger.info(f"Average Loss: {np.mean(losses):.3f} ± {np.std(losses):.3f}")

        
        # Show slot probabilities with detailed interpretation
        slot_probs = final_results.get('slot_probabilities', [])
        
        logger.info("\033[92m" + "Final slot probabilities:" + "\033[0m")

        non_zero_slots = 0
        total_slots = len(slot_probs)
        
        if isinstance(slot_probs, list):
            for i, probs in enumerate(slot_probs):
                slot_name = list(placeholder_choices.keys())[i]
                probs_np = np.array(probs)
                argmax_idx = np.argmax(probs_np)
                
                # Count non-zero slots (slots where argmax is not 0, meaning they're being used)
                if argmax_idx > 0:
                    non_zero_slots += 1
                
                logger.info(f"  {slot_name}: {probs_np}")
                logger.info("\033[92m" + f"    → Argmax: {argmax_idx} (highest probability: {probs_np[argmax_idx]:.3f})" + "\033[0m")

        logger.info(f"\n\033[93mSlot Usage Statistics:\033[0m")
        logger.info(f"  Total slots: {total_slots}")
        logger.info(f"  Active slots (argmax > 0): {non_zero_slots}")
        logger.info(f"  Inactive slots (argmax = 0): {total_slots - non_zero_slots}")
        logger.info(f"  Slot utilization: {non_zero_slots/total_slots*100:.1f}%")

        return final_results, history

    def test_optimized_model(self, final_results, max_test_points: Optional[int] = None,
                           tensorboard_writer=None, step=None):
        """
        Test the optimized prompt on held-out test data.
        
        Args:
            final_results: Results from P3O optimization containing optimized parameters
            max_test_points: Maximum number of test points to evaluate
            tensorboard_writer: Optional TensorBoard SummaryWriter for logging metrics
            step: Step number for TensorBoard logging
        """
        logger.info("\n" + "=" * 60)
        logger.info("TESTING OPTIMIZED MODEL ON HELD-OUT TEST DATA")
        logger.info("=" * 60)
        
        # Use stored test data
        test_queries = self.test_queries
        if max_test_points is not None and max_test_points < len(test_queries):
            test_queries = test_queries[:max_test_points]
        
        logger.info(f"Testing on {len(test_queries)} held-out test samples")
        
        if not test_queries:
            logger.error("No test data available!")
            return None
        
        # Create prompt generator with optimized parameters
        slot_choices = self.get_slot_choices()
        prompt_gen = PromptGeneratorModule(self.prompt_template, slot_choices, device='cpu')
        
        # Load the optimized slot probabilities
        slot_probs = final_results.get('slot_probabilities', [])
        if not slot_probs:
            logger.error("No optimized slot probabilities found in final results!")
            return None
        
        # Set the optimized parameters
        with torch.no_grad():
            # Convert parameters iterator to list for indexing
            param_list = list(prompt_gen.parameters())
            
            for i, probs in enumerate(slot_probs):
                if i < len(param_list):
                    # Convert probabilities to logits and set as parameters
                    logits = torch.log(torch.tensor(probs) + 1e-8)  # Add small epsilon to avoid log(0)
                    param_list[i].data = logits
        
        # Test on all test samples
        total_test_reward = 0.0
        total_mse = 0.0
        pipeline = self.get_pipeline_method(operation="eval")  # Evaluation pipeline
        
        logger.info("Running test evaluation...")
        test_results = []
        
        for i, test_query in enumerate(test_queries):
            try:
                # Generate optimized indices (using argmax of learned probabilities)
                indices = []
                for j, probs in enumerate(slot_probs):
                    if j < len(slot_choices):
                        indices.append(np.argmax(probs))  # Use most probable choice
                    else:
                        indices.append(0)  # Default if mismatch
                
                # Generate prompt with optimized parameters
                prompt = prompt_gen.generate(indices, data_sample=test_query)
                
                # Run pipeline to get reward and metrics
                y_pred, f_y, grad_f_y, additional_metrics = pipeline(prompt, test_query)
                
                total_test_reward += f_y
                total_mse += additional_metrics.get('mse', 0.0)
                
                test_results.append({
                    'query_idx': i,
                    'job_title': test_query['x'].get('job_title', 'Unknown'),
                    'reward': f_y,
                    'mse': additional_metrics.get('mse', 0.0),
                    'prediction': y_pred,
                    'ground_truth': test_query['y'].to_torch_tensor().numpy()
                })
                
                # Log progress every 10 samples
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(test_queries)} test samples")
                    
            except Exception as e:
                logger.error(f"Error processing test sample {i}: {e}")
                continue
        
        # Calculate test metrics
        avg_test_reward = total_test_reward / len(test_queries)
        avg_test_mse = total_mse / len(test_queries)
        
        logger.info("\n" + "-" * 40)
        logger.info("TEST RESULTS")
        logger.info("-" * 40)
        logger.info(f"Average Test Reward: {avg_test_reward:.3f}")
        logger.info(f"Average Test MSE: {avg_test_mse:.3f}")
        
        # Show sample predictions vs ground truth
        logger.info(f"\nSample Test Predictions (first 3):")
        for i, result in enumerate(test_results[:3]):
            logger.info(f"Sample {i+1}: {result['job_title']}")
            logger.info(f"  Reward: {result['reward']:.3f}, MSE: {result['mse']:.3f}")
            
            # Show first 5 predictions vs ground truth
            pred = result['prediction']
            gt = result['ground_truth']
            field_names = list(JobMetrics.model_fields.keys())
            
            logger.info(f"  Predictions vs Ground Truth (first 5 fields):")
            for j in range(min(5, len(pred))):
                field_name = field_names[j] if j < len(field_names) else f"field_{j}"
                logger.info(f"    {field_name}: pred={pred[j]:.1f}, gt={gt[j]:.1f}")
        
        test_summary = {
            'num_test_samples': len(test_queries),
            'avg_test_reward': avg_test_reward,
            'avg_test_mse': avg_test_mse,
            'test_results': test_results
        }
        
        # Log test metrics to TensorBoard
        if tensorboard_writer and step is not None:
            logger.info("Logging test metrics to TensorBoard...")
            tensorboard_writer.add_scalar('Test/Reward', avg_test_reward, step)
            tensorboard_writer.add_scalar('Test/MSE', avg_test_mse, step)
            tensorboard_writer.add_scalar('Test/NumSamples', len(test_queries), step)
            
            # Log individual test sample metrics (first few samples)
            for i, result in enumerate(test_results[:5]):  # Log first 5 samples
                tensorboard_writer.add_scalar(f'Test/Sample_{i+1}_Reward', result['reward'], step)
                tensorboard_writer.add_scalar(f'Test/Sample_{i+1}_MSE', result['mse'], step)
            
            # Calculate and log accuracy metrics if possible
            # For regression, we can log percentage of predictions within certain thresholds
            total_predictions = sum(len(r['prediction']) for r in test_results)
            if total_predictions > 0:
                # Count predictions within 10 points of ground truth (good accuracy)
                accurate_predictions = 0
                for result in test_results:
                    pred = result['prediction']
                    gt = result['ground_truth'] 
                    accurate_predictions += sum(1 for p, g in zip(pred, gt) if abs(p - g) <= 10)
                
                accuracy_within_10 = (accurate_predictions / total_predictions) * 100
                tensorboard_writer.add_scalar('Test/Accuracy_Within_10_Points', accuracy_within_10, step)
                logger.info(f"Test Accuracy (within 10 points): {accuracy_within_10:.1f}%")
            
            # Flush the writer to ensure metrics are saved
            tensorboard_writer.flush()
        
        logger.info(f"\n✅ Test evaluation complete!")
        return test_summary

    def get_trainer(self):
        """Get the P3O trainer instance for accessing TensorBoard writer."""
        return self.trainer


if __name__ == "__main__":
    # Create experiment instance and run optimization
    # Using fixed random seed to ensure reproducible results
    MAX_DATA_POINTS = 900  # Must match MIPROv2 experiment setting
    experiment = JobKnowledgeExperiment(
        input_data_path_numeric, 
        input_data_slots, 
        output_path, 
        max_data_points=MAX_DATA_POINTS,
        random_seed=42,  # Fixed seed for reproducible results
        batch_size=BATCH_SIZE
    )
    
    # Run pipeline sample with index=1 for all slots (yellow color)
    # logger.info("\033[93m" + "=" * 60 + "\033[0m")
    # logger.info("\033[93mRUNNING PIPELINE SAMPLE\033[0m")
    # logger.info("\033[93m" + "=" * 60 + "\033[0m")

    # sample_result = experiment.pipeline_sample()
    
    # if sample_result:
    #     logger.info("\033[93mPipeline sample completed successfully!\033[0m")
    #     logger.info(f"Reward: {sample_result['reward']:.3f}")
    #     logger.info(f"MSE: {sample_result['mse']:.3f}")
    # else:
    #     logger.error("\033[91mPipeline sample failed!\033[0m")
    
    logger.info("\n" + "=" * 60)
    logger.info("STARTING FULL OPTIMIZATION")
    logger.info("=" * 60)
    
    final_results, history = experiment.run(num_steps=NUM_EPOCHS, log_interval=1, log_dir=log_dir)
    
    # Plot reward progression
    plot_reward_progression(history, output_path)
    
    # Test the optimized model on held-out test data with TensorBoard logging
    # Get the trainer's TensorBoard writer and final step count for logging test metrics
    trainer = experiment.get_trainer()
    tensorboard_writer = trainer.writer if trainer and hasattr(trainer, 'writer') else None
    final_step = len(history) if history else 1
    
    test_summary = experiment.test_optimized_model(
        final_results, 
        max_test_points=MAX_TEST_POINTS,
        tensorboard_writer=tensorboard_writer,
        step=final_step
    )
    
    logger.info("\n" + "=" * 60)
    logger.info("JOB KNOWLEDGE PREDICTION OPTIMIZATION COMPLETE")
    logger.info("=" * 60)
    
    if test_summary:
        logger.info(f"Final Test Results:")
        logger.info(f"  Test Samples: {test_summary['num_test_samples']}")
        logger.info(f"  Average Test Reward: {test_summary['avg_test_reward']:.3f}")
        logger.info(f"  Average Test MSE: {test_summary['avg_test_mse']:.3f}")
        
        # Export final token data for evaluation only
        experiment.token_tracker.print_summary(operation="eval")
        
        # Export complete token data including test evaluation
        final_token_filename = f"{FULL_EXPERIMENT_ID}_tokens_complete.json"
        final_exported_path = experiment.token_tracker.export_to_json(final_token_filename)