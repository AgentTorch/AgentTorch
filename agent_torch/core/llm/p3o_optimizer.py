"""
P3O Optimizer for AgentTorch Template System
===========================================

Follows the exact patterns from p3o.py and knowledge_experience_skills.py,
with only the wiring changed to tap into our Template and LLMBackend.

Key Features:
- Uses Template.create_p3o_placeholder_choices() for slot definitions
- Uses Template.create_p3o_template_string() for P3O template format
- Implements the exact P3O optimization pipeline from the reference files
- Integrates with AgentTorch's LLM backends and ground truth data
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Callable, Optional
from concurrent.futures import ThreadPoolExecutor

# Import P3O components from the reference implementation
import sys
import os

# Add the prompt-archetype directory to Python path
prompt_archetype_dir = os.path.join(os.path.dirname(__file__), 'prompt-archetype')
if prompt_archetype_dir not in sys.path:
    sys.path.append(prompt_archetype_dir)

try:
    from p3o import PromptGeneratorModule, P3OTrainer
except ImportError:
    PromptGeneratorModule = None
    P3OTrainer = None

logger = logging.getLogger(__name__)


class P3OOptimizer:
    """
    P3O Optimizer for AgentTorch Template system.
    
    Follows the exact patterns from p3o.py and knowledge_experience_skills.py
    but integrates with our Template/Slot system and AgentTorch LLM backends.
    
    Args:
        template: Template object with learnable slots
        optimization_interval: How often to run P3O optimization (every N simulation steps)
        lr: Learning rate for P3O optimizer
        entropy_coef: Entropy coefficient for exploration
        lambda_param: Reward shaping weight (λ)
        rho: Running mean smoothing factor (ρ)
        beta: Reward baseline smoothing factor (β)
        batch_size: Number of agents to sample for P3O optimization
    """
    
    def __init__(
        self,
        template,
        optimization_interval: int = 3,
        lr: float = 0.1,
        entropy_coef: float = 0.01,
        lambda_param: float = 0.5,
        rho: float = 0.9,
        beta: float = 0.9,
        batch_size: int = 10
    ):
        # Store hyper-parameters and template
        self.template = template
        self.optimization_interval = optimization_interval
        self.lr = lr
        self.entropy_coef = entropy_coef
        self.lambda_param = lambda_param
        self.rho = rho
        self.beta = beta
        self.batch_size = batch_size
        
        # P3O components (initialized in _build_components)
        self.prompt_generator = None
        self.trainer = None
        self.step_count = 0
        
        # Ground truth and LLM pipeline
        self.ground_truth_data = {}
        self.llm_pipeline = None
        
        # Build P3O components
        self._build_components()
        
        print(f"P3OOptimizer initialized: optimization every {optimization_interval} steps, batch_size={batch_size}")
    
    def _build_components(self):
        """Build P3O components following the exact pattern from p3o.py."""
        if PromptGeneratorModule is None or P3OTrainer is None:
            print("P3O components not available, skipping initialization")
            return
        
        if self.template is None or PromptGeneratorModule is None or P3OTrainer is None:
            return

        # Get P3O-compatible template and placeholder choices
        template_str = self.template.create_p3o_template_string()
        placeholder_choices = self.template.create_p3o_placeholder_choices()

        # Initialize PromptGeneratorModule exactly as in p3o.py
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.prompt_generator = PromptGeneratorModule(template_str, placeholder_choices, device)

        # Initialize P3OTrainer exactly as in p3o.py
        optimizer = torch.optim.Adam(self.prompt_generator.parameters(), lr=self.lr)
        self.trainer = P3OTrainer(
            self.prompt_generator,
            optimizer,
            entropy_coef=self.entropy_coef,
            logger=logger
        )
    
    def step(self):
        """Increment step counter for optimization timing."""
        self.step_count += 1
        print(f"P3O step counter: {self.step_count}")
    
    def should_optimize(self, population_size: int) -> bool:
        """Check if P3O optimization should run based on step interval."""
        should_run = self.step_count % self.optimization_interval == 0
        print(f"P3O should optimize? {should_run} (step {self.step_count} % interval {self.optimization_interval} = {self.step_count % self.optimization_interval})")
        return should_run
    
    def set_pipeline(self, llm_pipeline: Callable[[str, Dict[str, Any]], Tuple[Any, Any, Any]]):
        """Set the LLM pipeline function."""
        self.llm_pipeline = llm_pipeline
    
    def set_ground_truth_data(self, ground_truth_data: Dict[str, Any]):
        """Set ground truth data for P3O optimization."""
        self.ground_truth_data = ground_truth_data
    
    def _make_pipeline(self, archetype, kwargs):
        """
        Create P3O pipeline following the exact pattern from knowledge_experience_skills.py.
        
        Args:
            archetype: Agent archetype with LLM backend
            kwargs: Additional configuration data
            
        Returns:
            Pipeline function with signature: (prompt: str, query_sample: Dict[str, Any]) -> (y_pred, f_y, grad_f_y)
        """
        def pipeline(prompt: str, query_sample: Dict[str, Any]):
            try:
                # Debug: Print query_sample structure
                # print(f"🔍 P3O Pipeline - query_sample type: {type(query_sample)}")
                # print(f"🔍 P3O Pipeline - query_sample keys: {query_sample.keys() if isinstance(query_sample, dict) else 'Not a dict'}")
                
                # 1) Forward call - get LLM prediction using __call__ method
                y_pred_structured = archetype(prompt_list=[prompt], last_k=2)
                
                # Handle the archetype response structure
                if isinstance(y_pred_structured, list) and len(y_pred_structured) > 0:
                    if isinstance(y_pred_structured[0], list) and len(y_pred_structured[0]) > 0:
                        # Nested list structure [[{...}]]
                        y_pred = y_pred_structured[0][0]["text"] if isinstance(y_pred_structured[0][0], dict) and "text" in y_pred_structured[0][0] else "0.5"
                    elif isinstance(y_pred_structured[0], dict) and "text" in y_pred_structured[0]:
                        # Direct list structure [{...}]
                        y_pred = y_pred_structured[0]["text"]
                    else:
                        y_pred = "0.5"
                else:
                    y_pred = "0.5"
                
                y_pred = float(y_pred)
                
                # 2) Compute f_y (reward = 100*100 - MSE)
                if isinstance(query_sample, dict) and "y" in query_sample:
                    y_true = float(query_sample["y"])
                else:
                    print(f"P3O Pipeline - Invalid query_sample structure: {query_sample}")
                    y_true = 0.5
                
                mse = (y_pred - y_true) ** 2
                f_y = (100*100 - mse)
                
                # 3) Gradient of f wrt y_pred
                grad_f = -2 * (y_pred - y_true)
                
                return y_pred, f_y, grad_f
                
            except Exception as e:
                print(f"Error in P3O pipeline: {e}")
                # Return fallback values
                return 0.5, 0.0, 0.0
        
        return pipeline
    
    def create_agent_queries(self, archetype, kwargs: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create P3O-compatible queries with representative sampling.
        
        Follows the exact pattern from knowledge_experience_skills.py.
        
        Args:
            archetype: Agent archetype with population data
            kwargs: Additional configuration data
            
        Returns:
            List of queries in P3O format: [{"x": agent_data, "y": ground_truth}, ...]
        """
        print(f"P3O DEBUG: create_agent_queries called with archetype={type(archetype)}")
        queries = []
        
        # Get population size
        population_size = getattr(archetype.population, 'population_size', 18) if hasattr(archetype, 'population') and archetype.population else 18
        
        # Sample representative batch (not always first N)
        sample_size = min(population_size, self.batch_size)
        if sample_size == population_size:
            # Use all agents if batch_size >= population_size
            agent_ids = list(range(population_size))
        else:
            # Random sample for unbiased gradient
            agent_ids = np.random.choice(population_size, size=sample_size, replace=False)
        
        for agent_id in agent_ids:
            # Provide full data to ensure early slot lambdas have values (prevents placeholders)
            try:
                # Get population from behavior if archetype doesn't have it
                population = None
                if hasattr(archetype, 'population') and archetype.population:
                    population = archetype.population
                elif hasattr(self, 'behavior_population'):
                    population = self.behavior_population
                
                # Get mapping from behavior if available
                mapping = getattr(self, 'behavior_mapping', {}) if hasattr(self, 'behavior_mapping') else {}
                
                assembled = self.template.assemble_data(
                    agent_id=agent_id,
                    population=population,
                    mapping=mapping,
                    config_kwargs=kwargs
                )
            except Exception as e:
                assembled = {"agent_id": agent_id}
            agent_data = assembled
            
            # Get ground truth - NO FALLBACKS
            if "ground_truth" not in self.ground_truth_data:
                raise ValueError("Ground truth data not available for P3O optimization")
            
            if agent_id not in self.ground_truth_data["ground_truth"]:
                raise ValueError(f"Ground truth missing for agent {agent_id}")
                
            ground_truth = self.ground_truth_data["ground_truth"][agent_id]
            
            query = {"x": agent_data, "y": ground_truth}
            queries.append(query)
        
        print(f"P3O using {len(queries)} agents for optimization (representative sample)")
        return queries
    
    def optimize(self, archetype, kwargs: Dict[str, Any], num_steps: int = 100, log_interval: int = 20) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Run P3O optimization following the exact pattern from p3o.py.
        
        Args:
            archetype: Agent archetype with LLM backend
            kwargs: Additional configuration data
            num_steps: Number of optimization steps
            log_interval: Logging interval
            
        Returns:
            Tuple of (final_results, training_history)
        """
        if not self.trainer or not self.prompt_generator:
            print("P3O components not available, skipping optimization")
            return {}, {}
        
        # Set up pipeline and ground truth
        pipeline = self._make_pipeline(archetype, kwargs)
        ground_truth_data = self.template.load_ground_truth_dict()
        self.set_ground_truth_data(ground_truth_data)

        # Create queries
        queries = self.create_agent_queries(archetype, kwargs)

        # Run training exactly as in p3o.py
        history = self.trainer.train(
            queries=queries,
            pipeline=pipeline,
            num_steps=num_steps,
            lambda_param=self.lambda_param,
            rho=self.rho,
            beta=self.beta,
            log_interval=log_interval
        )

        # Get final results
        final_results = self.trainer.get_final_results(queries)

        # Extract optimized slot choices and apply to template
        if "slot_probabilities" in final_results:
            slot_probs = final_results["slot_probabilities"]
            optimized_slots = {}

            # Get placeholder names from the prompt generator
            placeholder_names = getattr(self.prompt_generator, 'placeholder_names', None)
            if placeholder_names is None:
                # Fallback: get names from template slots
                slots = self.template.create_slots()
                placeholder_names = list(slots.keys())

            for i, name in enumerate(placeholder_names):
                if i < len(slot_probs):
                    optimized_slots[name] = torch.argmax(slot_probs[i]).item()

            # Apply optimized slots to template
            if optimized_slots:
                self.template.set_optimized_slots(optimized_slots)

        return final_results, history
    
    def get_optimized_prompt(self, agent_data: Dict[str, Any]) -> str:
        """
        Get optimized prompt using current slot choices.
        
        Args:
            agent_data: Agent data dictionary
            
        Returns:
            Optimized prompt string
        """
        if not self.prompt_generator:
            return ""
        
        try:
            # Use prompt generator to create optimized prompt
            optimized_prompt = self.prompt_generator.generate_prompt(agent_data)
            return optimized_prompt
        except Exception as e:
            print(f"Error generating optimized prompt: {e}")
            return ""