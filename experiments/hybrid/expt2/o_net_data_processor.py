"""
O-NET Data Processor
====================

Common data processing utilities for O-NET job analysis experiments.
This module ensures that both P3O and MIPROv2 experiments use the same
dataset and train/test splits for fair comparison.

Features:
- Deterministic data loading and splitting with fixed random seeds
- Skill dimension management and normalization
- Common data transformation utilities
- Support for both P3O and DSPy experiment formats
"""

import json
import logging
import os
from typing import Dict, Any, List, Tuple, Optional, Union
import pandas as pd
import numpy as np
from pydantic import BaseModel

# Configure logging
logger = logging.getLogger(__name__)

data_input_columns = [
    'Tasks', 'TechnologySkills',
    'WorkActivities', 'DetailedWorkActivities', 'WorkContext', 'Skills',
    'Knowledge', 'Abilities', 'Interests', 'WorkValues', 'WorkStyles',
    'RelatedOccupations', 'ProfessionalAssociations'
]

class JobMetrics(BaseModel):
    """Model for job knowledge metrics across different domains."""
    AdministrationAndManagement: int
    Administrative: int
    Biology: int
    BuildingAndConstruction: int
    Chemistry: int
    CommunicationsAndMedia: int
    ComputersAndElectronics: int
    CustomerAndPersonalService: int
    Design: int
    EconomicsAndAccounting: int
    EducationAndTraining: int
    EngineeringAndTechnology: int
    EnglishLanguage: int
    FineArts: int
    FoodProduction: int
    ForeignLanguage: int
    Geography: int
    HistoryAndArcheology: int
    LawAndGovernment: int
    Mathematics: int
    Mechanical: int
    MedicineAndDentistry: int
    PersonnelAndHumanResources: int
    PhilosophyAndTheology: int
    Physics: int
    ProductionAndProcessing: int
    Psychology: int
    PublicSafetyAndSecurity: int
    SalesAndMarketing: int
    SociologyAndAnthropology: int
    Telecommunications: int
    TherapyAndCounseling: int
    Transportation: int

    def to_torch_tensor(self):
        """Convert JobMetrics to torch tensor with consistent field ordering."""
        import torch
        field_names = list(JobMetrics.model_fields.keys())
        values = [getattr(self, field_name) for field_name in field_names]
        return torch.tensor(values, dtype=torch.float32)


class SkillDimension(BaseModel):
    """Model for storing skill dimensions from the CSV file."""
    primary_skill_name: str
    skill_name: str
    description: str
    confidence: float
    usage_count: int
    last_updated: str
    is_primary: bool


class SkillDimensionManager:
    """Manager class for handling skill dimensions."""
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.dimensions: Dict[str, SkillDimension] = {}  # Maps skill_name -> SkillDimension
        self.skill_to_primary: Dict[str, str] = {}  # Maps skill_name -> primary_skill_name
        self.primary_skills: List[str] = []  # List of unique primary skill names
        self._load_dimensions()
    
    def _load_dimensions(self):
        """Load dimensions from CSV file."""
        try:
            df = pd.read_csv(self.csv_path)
            logger.info(f"Loaded skill dimensions CSV with {len(df)} rows")
            
            primary_skills_set = set()
            
            for _, row in df.iterrows():
                # Create SkillDimension from row data
                dim = SkillDimension(
                    primary_skill_name=row['primary_skill_name'],
                    skill_name=row['skill_name'],
                    description=row['description'],
                    confidence=float(row['confidence']),
                    usage_count=int(row['usage_count']),
                    last_updated=str(row['last_updated']) if pd.notna(row['last_updated']) else '',
                    is_primary=bool(row['is_primary'])
                )
                
                # Store dimension by skill_name
                self.dimensions[dim.skill_name] = dim
                
                # Map skill_name to primary_skill_name
                self.skill_to_primary[dim.skill_name] = dim.primary_skill_name
                
                # Collect unique primary skills
                primary_skills_set.add(dim.primary_skill_name)
            
            # Store sorted list of primary skills
            self.primary_skills = sorted(list(primary_skills_set))
            
            logger.info(f"Successfully loaded {len(self.dimensions)} skill dimensions")
            logger.info(f"Found {len(self.primary_skills)} unique primary skills")
            
        except Exception as e:
            logger.error(f"Error loading skill dimensions: {e}")
            self.dimensions = {}
            self.skill_to_primary = {}
            self.primary_skills = []
    
    def get_dimension(self, skill_name: str) -> Optional[SkillDimension]:
        """Get a specific dimension by skill name."""
        return self.dimensions.get(skill_name)
    
    def get_all_dimensions(self) -> List[Tuple[str, SkillDimension]]:
        """Get all dimensions with their names."""
        return [(name, dim) for name, dim in self.dimensions.items()]
    
    def get_dimension_names(self) -> List[str]:
        """Get list of unique primary skill names."""
        return self.primary_skills.copy()
    
    def get_normalized_skill_name(self, skill_name: str) -> Optional[str]:
        """Get the primary skill name for a given skill name."""
        return self.skill_to_primary.get(skill_name)
    
    def get_all_skill_names(self) -> List[str]:
        """Get list of all skill names (both primary and synonyms)."""
        return list(self.dimensions.keys())


class ONetDataProcessor:
    """
    Main data processor class that handles O-NET job data loading, 
    processing, and splitting for both P3O and MIPROv2 experiments.
    """
    
    def __init__(self, 
                 numeric_data_path: str,
                 skill_vector_path: str,
                 skill_dimensions_path: str,
                 output_column_prefix: str = "NUMERIC_knowledge_",
                 random_seed: int = 42):
        """
        Initialize the data processor.
        
        Args:
            numeric_data_path: Path to the processed job data CSV
            skill_vector_path: Path to the skill vectors JSON
            skill_dimensions_path: Path to the skill dimensions CSV
            output_column_prefix: Prefix for output metric columns
            random_seed: Fixed seed for deterministic data splitting
        """
        self.numeric_data_path = numeric_data_path  # stores path to the files where numeric skill values live
        self.skill_vector_path = skill_vector_path  # stores path to the files where skill vectors live for all the jobs
        self.skill_dimensions_path = skill_dimensions_path  # stores path to the files where normalized skill dimensions live
        self.output_column_prefix = output_column_prefix
        self.random_seed = random_seed
        
        # Initialize skill manager
        self.skill_manager = SkillDimensionManager(skill_dimensions_path)
        
        # Data storage
        self._raw_data = None
        self._processed_queries = None
        self._train_indices = None
        self._test_indices = None
        
        logger.info(f"Initialized ONetDataProcessor with seed={random_seed}")
    
    @staticmethod
    def _get_template_var_name(skill_name: str) -> str:
        """Convert a skill name to a valid template variable name."""
        return skill_name.replace(' ', '_')
    
    def _safe_str_convert(self, val):
        """Safely convert value to string, handling pandas null values."""
        if pd.isna(val):
            return ""
        return str(val)
    
    def _safe_array_convert(self, val):
        """Safely convert value to array, splitting by newlines."""
        if pd.isna(val):
            return []
        str_val = str(val)
        if not str_val.strip():
            return []
        return [item.strip() for item in str_val.split('\n') if item.strip()]
    
    def _load_slot_data(self) -> pd.DataFrame:
        """Load skill vector data from JSON file."""
        data = []
        with open(self.skill_vector_path, 'r') as f:
            slot_data = json.load(f)
            for item in slot_data:
                data.append({
                    "onet_code": item["onet_code"],
                    "skill_vector": item["skill_vector"]
                })
        return pd.DataFrame(data)
    
    def load_raw_data(self) -> pd.DataFrame:
        """
        Load and merge all raw data sources.
        
        Returns:
            DataFrame with merged job data and skill vectors
        """
        if self._raw_data is not None:
            return self._raw_data
        
        logger.info(f"Loading job data from: {self.numeric_data_path}")
        
        # Load numeric job data
        job_df = pd.read_csv(self.numeric_data_path)
        logger.info(f"Loaded {len(job_df)} job samples")
        
        # Load skill vector data
        slots_df = self._load_slot_data()
        logger.info(f"Loaded {len(slots_df)} skill vector records")
        
        # Merge data
        merged_df = job_df.merge(slots_df, left_on="soc_code", right_on="onet_code", how="left")
        logger.info(f"Merged dataset has {len(merged_df)} records")
        
        # Log column information
        logger.info(f"CSV columns: {list(merged_df.columns)}")
        
        self._raw_data = merged_df
        return merged_df
    
    def process_job_queries(self, max_data_points: Optional[int] = None, data_type: str = "exp1", debug: bool = False) -> List[Dict[str, Any]]:
        """
        Process raw data into query format suitable for both experiments.
        
        Args:
            max_data_points: Maximum number of data points to process (None for all)
            data_type: Experiment ID to determine format ("exp1" or "exp2")
            debug: Enable debug logging
            
        Returns:
            List of query dictionaries with 'x' and 'y' keys
        """
        if self._processed_queries is not None:
            if max_data_points is None:
                return self._processed_queries
            else:
                return self._processed_queries[:max_data_points]
        
        # Load raw data
        df = self.load_raw_data()
        
        # Set random seed for reproducible processing
        np.random.seed(self.random_seed)
        
        queries = []
        
        
        for i, (_, row) in enumerate(df.iterrows()):
            # Extract basic job information
            soc_code = self._safe_str_convert(row['soc_code'])
            job_name = self._safe_str_convert(row['job_name'])

            ########## Process skill vector (Y) ##########
            skill_vector_raw = row.get("skill_vector", {})
            
            
            # Handle invalid skill vectors
            if pd.isna(skill_vector_raw) or not isinstance(skill_vector_raw, dict):
                logger.warning(f"Invalid skill_vector for job_code {soc_code} (row {i}): {type(skill_vector_raw)}")
                skill_vector_raw = {}
            elif len(skill_vector_raw) == 0:
                logger.warning(f"Empty skill_vector for job_code {soc_code} (row {i})")
            
            # Create target metrics (y)
            y_data = {}
            for field_name in JobMetrics.model_fields.keys():
                column_name = f"{self.output_column_prefix}{field_name}"
                if column_name in row:
                    try:
                        y_data[field_name] = int(row[column_name])
                    except (ValueError, TypeError):
                        y_data[field_name] = 0
                else:
                    y_data[field_name] = 0
            
            # Create JobMetrics instance
            y = JobMetrics(**y_data)
            ########## Process skill vector (Y) ########## END

            ########## Process skill vector (X) ##########
            if data_type == "exp1":
                # read all the columns in data_input_columns
                global data_input_columns
                data = {
                    col:row.get(col, '') for col in data_input_columns
                }
                x = {
                    "onet_code": soc_code,
                    "job_title": job_name,
                    **data
                }
            else:
                # Transform skill names to their primary equivalents
                skill_vector_transformed = {}
                skills_not_normalized = []
                for skill_name, value in skill_vector_raw.items():
                    # Get normalized (primary) skill name
                    skill_name_lower = skill_name.lower()
                    normalized_skill = self.skill_manager.get_normalized_skill_name(skill_name_lower)
                    
                    if normalized_skill:
                        # Use primary skill name as template variable
                        var_name = self._get_template_var_name(normalized_skill)
                        skill_vector_transformed[var_name] = value
                    else:
                        skills_not_normalized.append(skill_name)
                        var_name = self._get_template_var_name(skill_name)
                        skill_vector_transformed[var_name] = value
                
                # Only log if there are many unnormalized skills (reduce log spam)
                if len(skills_not_normalized) > 3:
                    logger.warning(f"Job {i+1}: {len(skills_not_normalized)}/{len(skill_vector_raw)} skills not normalized")
                
                # Create input features (x)
                x = {
                    "onet_code": soc_code,
                    "job_title": job_name,
                    **skill_vector_transformed
                }
            ########## Process skill vector (X) ########## END
            
            
            # Debug logging for first row
            if debug and i == 0:
                logger.info(f"Processed first row:")
                logger.info(f"  onet_code: {soc_code}")
                logger.info(f"  job_title: {job_name}")
                logger.info(f"  skill_vector_transformed: {skill_vector_transformed}")
                logger.info(f"  y_data sample: {dict(list(y_data.items())[:5])}")
            
            query = {"x": x, "y": y}
            queries.append(query)
        
        self._processed_queries = queries
        logger.info(f"Created {len(queries)} processed queries from job data")
        
        # Log overall normalization statistics
        total_skills_processed = sum(len(q['x']) - 2 for q in queries)  # -2 for onet_code and job_title
        logger.info(f"Processed {total_skills_processed} total skill entries across all jobs")
        
        # Apply max_data_points limit if specified
        if max_data_points is not None:
            return queries[:max_data_points]
        return queries
    
    def get_train_test_split(self, 
                           max_data_points: Optional[int] = None,
                           test_ratio: float = 0.3,
                           data_type: str = "exp1") -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Get deterministic train/test split of the data.
        Default: 30% train, 70% test (common for prompt optimization).
        
        Args:
            max_data_points: Maximum total data points to use
            test_ratio: Fraction of data to use for testing (default 0.7 = 70% test)
            data_type: Experiment ID to determine format ("exp1" or "exp2")
        Returns:
            Tuple of (train_queries, test_queries)
        """
        # Get processed queries
        queries = self.process_job_queries(max_data_points=max_data_points, data_type=data_type)
        
        # Set seed for deterministic splitting
        np.random.seed(self.random_seed)
        
        # Create indices and shuffle
        indices = np.arange(len(queries))
        np.random.shuffle(indices)
        
        # Split indices
        test_size = int(len(queries) * test_ratio)
        test_indices = indices[:test_size]
        train_indices = indices[test_size:]
        
        # Store for reproducibility
        self._train_indices = train_indices
        self._test_indices = test_indices
        
        # Create splits
        train_queries = [queries[i] for i in train_indices]
        test_queries = [queries[i] for i in test_indices]
        
        logger.info(f"Data split: {len(train_queries)} train, {len(test_queries)} test samples")
        logger.info(f"Train/test split seed: {self.random_seed}")
        
        return train_queries, test_queries
    
    def get_slot_choices_for_p3o(self, data_type: str = "exp1") -> Dict[str, Tuple[int, Any]]:  #TODO
        """
        Get slot choices configuration for P3O experiment.
        
        Returns:
            Dictionary mapping slot names to (num_categories, lambda_function) tuples
            data_type: Experiment ID to determine format ("exp1" or "exp2")
        """
        def make_custom_lambda_exp1(skill_name):
            """Create a lambda that captures the skill_name value."""
            return lambda cat, data: "" if cat == 0 else (
                f"{skill_name} : {data[skill_name]}\n" 
            )

        def make_custom_lambda_exp2(skill_name):
            """Create a lambda that captures the skill_name value."""
            return lambda cat, data: "" if cat == 0 else (
                f"Skill: {skill_name}\n" if skill_name in data and data[skill_name] == 1 else ""
            )
        
        slot_choices = {}
        skills_columns = []
        if data_type == "exp1":
            skills_columns = data_input_columns
        else:
            skills_columns = self.skill_manager.get_dimension_names()
        
        # For each primary skill
        for slot_name in skills_columns:
            # Get template variable name
            if data_type == "exp1":
                make_custom_lambda = make_custom_lambda_exp1
            else:
                slot_name = self._get_template_var_name(slot_name)
                make_custom_lambda = make_custom_lambda_exp2
            
            slot_choices[slot_name] = (
                2,  # Binary choice: 0 for nothing, 1 for skill info
                make_custom_lambda(slot_name)
            )
        
        return slot_choices
    
    def convert_p3o_to_dspy_format(self, queries: List[Dict[str, Any]], data_type: str = "exp1") -> List:
        import dspy
        """
        Convert P3O format queries to DSPy format examples.
        
        Args:
            queries: List of P3O format queries
            data_type: Experiment ID to determine format ("exp1" or "exp2")
            
        Returns:
            List of DSPy examples
        """
       
        examples = []
        for query in queries:
            x = query["x"]
            y = query["y"]
            
            if data_type == "exp1":
                # For exp1: Use original work activity and skills strings
                job_info = self._get_original_job_info_for_exp1(x)
            else:
                # For exp2: Use resolved binary skill values (default behavior)
                job_info = self._get_binary_skill_info_for_exp2(x)
            
            # Create DSPy example
            example = dspy.Example(
                job_info=job_info,
                job_metrics=y
            ).with_inputs("job_info")
            
            examples.append(example)
        
        return examples
    
    def _get_binary_skill_info_for_exp2(self, x: Dict[str, Any]) -> str:
        """Get job info string from binary skill data for exp2."""
        # Create job info string from ONLY skill data (no job title or ONET code)
        skill_info = []
        for key, value in x.items():
            if key not in ['job_title', 'onet_code'] and value == 1:
                skill_info.append(key.replace('_', ' ').title())
        
        # Create job info with only skills - no identifying information
        if skill_info:
            return f"Job Description via skills: {', '.join(skill_info)}"
        else:
            return "No specific skills identified"
    
    def _get_original_job_info_for_exp1(self, x: Dict[str, Any]) -> str:
        """Get job info string from original work activity and skills strings for exp1."""
        # do not expose onet_code and job_title
        job_parts = [f'{col}: {x.get(col, "")}' for col in data_input_columns]
        if len(job_parts) == 0:
            return "No job information available"
        return " | ".join(job_parts)
    
    def verify_data_consistency(self, 
                              p3o_queries: List[Dict[str, Any]], 
                              dspy_examples: List) -> bool:
        """
        Verify that P3O and DSPy experiments are using the same underlying data.
        
        Args:
            p3o_queries: Queries used by P3O experiment
            dspy_examples: Examples used by DSPy experiment
            
        Returns:
            True if data is consistent, False otherwise
        """
        if len(p3o_queries) != len(dspy_examples):
            logger.error(f"Data length mismatch: P3O={len(p3o_queries)}, DSPy={len(dspy_examples)}")
            return False
        
        # Check a sample of the data for consistency
        sample_size = min(5, len(p3o_queries))
        for i in range(sample_size):
            p3o_job_title = p3o_queries[i]["x"]["job_title"]
            p3o_onet_code = p3o_queries[i]["x"]["onet_code"]
            
            # DSPy examples don't contain job titles directly, but we can check 
            # by comparing the JobMetrics outputs
            p3o_metrics = p3o_queries[i]["y"]
            dspy_metrics = dspy_examples[i].job_metrics
            
            if p3o_metrics.model_dump() != dspy_metrics.model_dump():
                logger.error(f"Metrics mismatch at index {i}")
                logger.error(f"  P3O job: {p3o_job_title} ({p3o_onet_code})")
                logger.error(f"  P3O metrics: {p3o_metrics.model_dump()}")
                logger.error(f"  DSPy metrics: {dspy_metrics.model_dump()}")
                return False
        
        logger.info(f"Data consistency verified for {len(p3o_queries)} samples")
        return True
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get a summary of the loaded data."""
        queries = self.process_job_queries()
        
        summary = {
            "total_samples": len(queries),
            "random_seed": self.random_seed,
            "skill_dimensions": len(self.skill_manager.dimensions),
            "primary_skills": len(self.skill_manager.primary_skills),
            "data_sources": {
                "numeric_data": self.numeric_data_path,
                "skill_vectors": self.skill_vector_path,
                "skill_dimensions": self.skill_dimensions_path
            }
        }
        
        # Sample job titles
        sample_titles = [q["x"]["job_title"] for q in queries[:5]]
        summary["sample_job_titles"] = sample_titles
        
        # Get train/test split info
        if self._train_indices is not None:
            summary["train_samples"] = len(self._train_indices)
            summary["test_samples"] = len(self._test_indices)
        
        return summary


def create_shared_data_processor(random_seed: int = 42) -> ONetDataProcessor:
    """
    Create a shared data processor instance with standard paths.
    
    Args:
        random_seed: Fixed seed for reproducible results across experiments
        
    Returns:
        Configured ONetDataProcessor instance
    """
    # Resolve paths relative to the experiment working directory (experiments/hybrid)
    # Resolve paths relative to the module location (stable regardless of CWD)
    module_dir = os.path.dirname(__file__)  # .../experiments/hybrid/expt2
    hybrid_root = os.path.dirname(module_dir)
    # Numeric CSV under expt2_data
    numeric_data_path = os.path.join(hybrid_root, "expt2_data", "job_data_processed.csv")
    # Skill vectors JSON under expt2_data
    skill_vector_path = os.path.join(hybrid_root, "expt2_data", "skill_dimensions_updated", "updated_job_vectors.json")
    # Skill dimensions CSV: prefer expt2_data; fallback to legacy location at hybrid root
    skill_dimensions_primary = os.path.join(hybrid_root, "expt2_data", "skill_dimensions_updated", "all_skills.csv")
    skill_dimensions_fallback = os.path.join(hybrid_root, "all_skills.csv")
    skill_dimensions_path = skill_dimensions_primary if os.path.exists(skill_dimensions_primary) else skill_dimensions_fallback
    
    processor = ONetDataProcessor(
        numeric_data_path=numeric_data_path,
        skill_vector_path=skill_vector_path,
        skill_dimensions_path=skill_dimensions_path,
        random_seed=random_seed
    )
    
    return processor


if __name__ == "__main__":
    # Example usage and testing
    pass