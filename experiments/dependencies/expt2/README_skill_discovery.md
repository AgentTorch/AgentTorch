# Skill Dimension Discovery for Job Analysis

This project implements an algorithm to discover unique skill dimensions that identify jobs by analyzing job descriptions and using LLM to extract new skill dimensions.

## Algorithm Overview

The algorithm follows this process:

1. **Initialize**: Start with an empty set of skill dimensions S = {}
2. **Iterate**: For each job j in jobs:
   - Analyze j['DetailedWorkActivities'] and j['Skills']
   - Use LLM to identify new skill dimensions (≤5 words) not in S
   - Add new dimensions to S
3. **Return**: The complete set of unique skill dimensions

This creates a sparse vector representation for each job based on discovered dimensions.

## Files

### Core Scripts

1. **`skill_dimension_discovery.py`** - Basic implementation
   - Discovers unique skill dimensions
   - Creates simple sparse vectors (random scoring)
   - Good for testing and understanding the concept

2. **`skill_dimension_discovery_advanced.py`** - Advanced implementation
   - Discovers unique skill dimensions
   - Creates sparse vectors using LLM-based scoring
   - Provides comprehensive analysis and visualization
   - Saves results to JSON files

3. **`skill_dimension_discovery_with_updates.py`** - Enhanced implementation with updates
   - Discovers AND updates skill dimensions
   - Can merge similar dimensions
   - Can remove redundant dimensions
   - Iterative refinement over multiple passes
   - Tracks usage counts and confidence scores

4. **`incremental_skill_updates.py`** - Incremental updates
   - Loads existing skill dimensions from file
   - Processes new jobs to update existing dimensions
   - Conservative approach - focuses on improving existing dimensions
   - Useful for maintaining and refining dimensions over time

5. **`test_data_reading.py`** - Testing script
   - Verifies data reading functionality
   - Tests CSV file structure
   - Validates data quality

### Data Requirements

The scripts expect a CSV file at `./data/job_data_processed/job_data_processed.csv` with these columns:
- `soc_code` - ONET code for the job
- `job_name` - Job title
- `Skills` - Multi-line text of job skills
- `DetailedWorkActivities` - Multi-line text of detailed work activities

## Usage

### 1. Test Data Reading

First, test that your data can be read correctly:

```bash
python test_data_reading.py
```

This will verify:
- CSV file exists and can be read
- Required columns are present
- Data quality (null values, etc.)
- Sample data processing

### 2. Basic Skill Dimension Discovery

Run the basic implementation:

```bash
python skill_dimension_discovery.py
```

This will:
- Process the first 50 jobs (configurable)
- Discover unique skill dimensions
- Create simple sparse vectors
- Display results in console

### 3. Advanced Skill Dimension Discovery

Run the advanced implementation:

```bash
python skill_dimension_discovery_advanced.py
```

This will:
- Process the first 20 jobs (configurable)
- Discover unique skill dimensions
- Create LLM-scored sparse vectors
- Save results to `./expt2_data/skill_dimensions/`
- Provide comprehensive analysis

### 4. Enhanced Discovery with Updates

Run the enhanced implementation that can update existing dimensions:

```bash
python skill_dimension_discovery_with_updates.py
```

This will:
- Process jobs with multiple iterations
- Discover new dimensions AND update existing ones
- Merge similar dimensions
- Remove redundant dimensions
- Track usage counts and confidence scores
- Save results to `./expt2_data/skill_dimensions_updated/`

### 5. Incremental Updates

Update existing skill dimensions with new job data:

```bash
python incremental_skill_updates.py
```

This will:
- Load existing skill dimensions from file
- Process new jobs to refine existing dimensions
- Be conservative about adding new dimensions
- Focus on improving existing ones
- Save updated dimensions to `./expt2_data/skill_dimensions_incremental/`

## Output Files

### Advanced Version
The advanced script creates these output files:

- **`skill_dimensions.json`** - List of all discovered skill dimensions
- **`job_vectors.json`** - Job data with sparse vectors
- **`summary.json`** - Summary statistics

### Enhanced Version with Updates
The enhanced script creates these output files:

- **`updated_skill_dimensions.json`** - Skill dimensions with metadata (usage counts, confidence, etc.)
- **`updated_job_vectors.json`** - Job data with sparse vectors
- **`updated_summary.json`** - Summary statistics including top dimensions by usage

### Incremental Updates
The incremental script creates these output files:

- **`incremental_skill_dimensions.json`** - Updated skill dimensions after processing new jobs

## Configuration

### Input Data Path

Change the input data path in any script:

```python
input_data_path = "./data/job_data_processed/job_data_processed.csv"
```

### Number of Jobs to Process

Modify the `max_jobs` parameter in the main functions:

```python
# In skill_dimension_discovery.py
jobs = read_job_data(input_data_path, max_jobs=50)  # Process first 50 jobs

# In skill_dimension_discovery_advanced.py
jobs = read_job_data(input_data_path, max_jobs=20)  # Process first 20 jobs
```

### LLM Model

Change the LLM model in the scripts:

```python
# Default is Claude 3.5 Haiku
model='claude-3-5-haiku-latest'

# You can change to other models like:
# model='gpt-4o-mini'
# model='claude-3-sonnet-latest'
```

## Key Functions

### `update_skill_dimensions(job, existing_dimensions)`

This is the core function that implements the algorithm:

```python
def update_skill_dimensions(job: Dict[str, Any], existing_dimensions: Set[str]) -> Set[str]:
    """
    Update skill dimensions set by analyzing a job and discovering new dimensions.
    
    Args:
        job: Job dictionary with skills and work activities
        existing_dimensions: Current set of discovered skill dimensions
        
    Returns:
        Updated set of skill dimensions
    """
```

### `update_skill_dimensions_enhanced(job, existing_dimensions)`

Enhanced version that can update existing dimensions:

```python
def update_skill_dimensions_enhanced(job: Dict[str, Any], existing_dimensions: Dict[str, SkillDimension]) -> Dict[str, SkillDimension]:
    """
    Enhanced function to update skill dimensions by analyzing a job.
    Can add new dimensions, update existing ones, merge similar ones, or remove redundant ones.
    
    Args:
        job: Job dictionary with skills and work activities
        existing_dimensions: Current dictionary of skill dimensions
        
    Returns:
        Updated dictionary of skill dimensions
    """
```

### `discover_skill_dimensions(jobs)`

Main function that runs the discovery process:

```python
def discover_skill_dimensions(jobs: List[Dict[str, Any]]) -> Set[str]:
    """
    Discover unique skill dimensions across all jobs.
    
    Args:
        jobs: List of job dictionaries
        
    Returns:
        Set of unique skill dimensions
    """
```

## Example Output

### Discovered Skill Dimensions

```
Discovered skill dimensions:
  1. Analytical Thinking
  2. Communication Skills
  3. Customer Service
  4. Data Analysis
  5. Equipment Operation
  6. Leadership
  7. Problem Solving
  8. Project Management
  9. Safety Compliance
 10. Technical Programming
```

### Job Vector Example

```
Job: Software Developer
Top 5 skill dimensions:
  Technical Programming: 0.95
  Problem Solving: 0.87
  Analytical Thinking: 0.82
  Data Analysis: 0.78
  Project Management: 0.65
```

## Dependencies

Required Python packages:
- `pandas` - Data manipulation
- `pydantic` - Data validation
- `llm_utils` - LLM calling utilities
- `numpy` - Numerical operations
- `logging` - Logging functionality

## Troubleshooting

### Common Issues

1. **File not found**: Ensure the CSV file exists at the specified path
2. **Missing columns**: Verify the CSV has required columns (`soc_code`, `job_name`, `Skills`, `DetailedWorkActivities`)
3. **LLM API errors**: Check your API keys and model availability
4. **Memory issues**: Reduce `max_jobs` parameter for large datasets

### Debug Mode

Enable debug logging by modifying the logging level:

```python
logging.basicConfig(level=logging.DEBUG, ...)
```

## Future Enhancements

Potential improvements:
1. **Batch processing** - Process jobs in batches for efficiency
2. **Caching** - Cache LLM responses to avoid repeated calls
3. **Parallel processing** - Use multiple workers for faster processing
4. **Visualization** - Add plots and charts for better analysis
5. **Clustering** - Group similar skill dimensions
6. **Validation** - Add validation metrics for discovered dimensions
