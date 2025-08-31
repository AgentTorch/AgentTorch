#!/usr/bin/env python3
"""
Pydantic Class for Mock Test 18 Population Mapping
==================================================

This file contains a Pydantic class that mirrors the exact structure of our mapping.json file.
Used for testing and validation of the mapping system.
"""

from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field, field_validator
from pathlib import Path


class PopulationFieldConfig(BaseModel):
    """Configuration for a population field."""
    source: str = Field(..., description="Path to the pickle file containing the field data")
    categorical: Optional[List[str]] = Field(None, description="List of categorical values for decoding")


class ExternalSourceConfig(BaseModel):
    """Configuration for an external data source."""
    source_path: str = Field(..., description="Path to the external data file")
    lookup_field: str = Field(..., description="Field name used for lookup in the external data")
    fields: List[str] = Field(..., description="List of fields to extract from the external data")


class FieldSourcesConfig(BaseModel):
    """Configuration for field sources."""
    population_fields: Dict[str, PopulationFieldConfig] = Field(..., description="Population field configurations")
    external_sources: Dict[str, ExternalSourceConfig] = Field(..., description="External data source configurations")


class MockTest18Mapping(BaseModel):
    """
    Pydantic model that mirrors the exact structure of mock_test_18/mapping.json.
    
    This model represents the complete mapping configuration for the mock_test_18 population,
    including all population fields and external data sources.
    """
    
    population_size: int = Field(..., description="Number of agents in the population")
    created_by: str = Field(..., description="Script or process that created this population")
    description: str = Field(..., description="Description of the population")
    field_sources: FieldSourcesConfig = Field(..., description="Configuration for all field sources")
    
    @field_validator('population_size')
    @classmethod
    def validate_population_size(cls, v):
        """Validate that population size is positive."""
        if v <= 0:
            raise ValueError('Population size must be positive')
        return v
    
    def get_population_field_names(self) -> List[str]:
        """Get all population field names."""
        return list(self.field_sources.population_fields.keys())
    
    def get_external_source_names(self) -> List[str]:
        """Get all external source names."""
        return list(self.field_sources.external_sources.keys())
    
    def get_lookup_field_for_external_source(self, source_name: str) -> Optional[str]:
        """Get the lookup field for a specific external source."""
        if source_name in self.field_sources.external_sources:
            return self.field_sources.external_sources[source_name].lookup_field
        return None
    
    def get_external_source_for_field(self, field_name: str) -> Optional[str]:
        """Find which external source provides a specific field."""
        for source_name, source_config in self.field_sources.external_sources.items():
            if field_name in source_config.fields:
                return source_name
        return None
    
    def validate_file_paths(self) -> List[str]:
        """Validate that all file paths exist and return list of missing files."""
        missing_files = []
        
        # Check population field files
        for field_name, field_config in self.field_sources.population_fields.items():
            file_path = Path(f"agent_torch/populations/mock_test_18/{field_config.source}")
            if not file_path.exists():
                missing_files.append(str(file_path))
        
        # Check external source files
        for source_name, source_config in self.field_sources.external_sources.items():
            file_path = Path(source_config.source_path)
            if not file_path.exists():
                missing_files.append(str(file_path))
        
        return missing_files


# Example usage and testing
if __name__ == "__main__":
    # Test with our actual mapping.json structure
    test_mapping_data = {
        "population_size": 18,
        "created_by": "create_mock_18_population_fixed.py",
        "description": "Mock 18-agent population for P3O testing - numeric encoded",
        "field_sources": {
            "population_fields": {
                "age": {
                    "source": "age.pickle"
                },
                "gender": {
                    "source": "gender.pickle",
                    "categorical": ["male", "female"]
                },
                "ethnicity": {
                    "source": "ethnicity.pickle",
                    "categorical": ["other", "white", "hispanic", "black", "asian"]
                },
                "area": {
                    "source": "area.pickle",
                    "categorical": ["urban", "suburban", "rural"]
                },
                "household_size": {
                    "source": "household.pickle"
                },
                "region": {
                    "source": "region.pickle",
                    "categorical": ["Northeast", "Southeast", "Midwest", "West"]
                },
                "soc_code": {
                    "source": "soc_code.pickle",
                    "categorical": [
                        "13-1074.00", "13-2081.00", "25-4012.00", "27-2091.00",
                        "29-1229.05", "35-9031.00", "39-6011.00", "47-2081.00",
                        "47-2121.00", "49-9081.00", "53-1044.00", "53-3033.00",
                        "53-7011.00", "53-7121.00"
                    ]
                },
                "hobby_code": {
                    "source": "hobby_code.pickle",
                    "categorical": [
                        "H001", "H002", "H003", "H004", "H005",
                        "H006", "H007", "H008", "H009", "H010"
                    ]
                }
            },
            "external_sources": {
                "job_profiles": {
                    "source_path": "agent_torch/core/llm/data/job_data_processed/job_data_processed.pkl",
                    "lookup_field": "soc_code",
                    "fields": ["job_name", "Skills", "Tasks", "Abilities", "WorkContext", "TechnologySkills", "WorkActivities", "DetailedWorkActivities", "Knowledge", "Interests", "WorkValues", "WorkStyles", "Education"]
                },
                "hobbies": {
                    "source_path": "agent_torch/core/llm/data/hobbies_data/hobbies_data.pkl",
                    "lookup_field": "hobby_code",
                    "fields": ["hobby_name", "HobbyActivities", "HobbySkills", "Equipment", "Time_Commitment", "Cost_Level", "Social_Aspect", "Indoor_Outdoor"]
                }
            }
        }
    }
    
    try:
        # Test validation
        mapping = MockTest18Mapping(**test_mapping_data)
        print("✅ Pydantic validation successful!")
        print(f"📊 Population size: {mapping.population_size}")
        print(f"📋 Population fields: {mapping.get_population_field_names()}")
        print(f"🔗 External sources: {mapping.get_external_source_names()}")
        print(f"🔍 Lookup field for hobbies: {mapping.get_lookup_field_for_external_source('hobbies')}")
        print(f"🔍 External source for hobby_name: {mapping.get_external_source_for_field('hobby_name')}")
        
        # Test file path validation
        missing_files = mapping.validate_file_paths()
        if missing_files:
            print(f"⚠️ Missing files: {missing_files}")
        else:
            print("✅ All file paths exist!")
            
    except Exception as e:
        print(f"❌ Validation failed: {e}") 