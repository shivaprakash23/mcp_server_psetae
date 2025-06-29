# Main Context Document (MCD): Sentinel-1 PSETAE Multi-Agent System

## Project Overview

PSETAE is a crop classification system using satellite imagery with deep learning. This MCD defines the implementation of a Model Context Protocol (MCP) multi-agent system to orchestrate the Sentinel-1 PSETAE workflow. The initial implementation focuses exclusively on Sentinel-1 data to establish a stable baseline before expanding to other data sources.

## System Architecture

### Existing PSETAE Components
- **Data Extraction**: GEE-based scripts for Sentinel-1 data
  - Requires GeoJSON input files for processing
- **Model Training**: PSETAE implementation for Sentinel-1
- **Inference**: Prediction scripts for trained Sentinel-1 models

### MCP Agent Structure
- **MCP Server**: Central coordination point with database (`mcp_state.db`)
- **Admin Agent**: Workflow orchestrator and task manager
- **Worker Agents**:
  1. **Sentinel1DataExtractionAgent**: Handles GEE data retrieval and processing for Sentinel-1
  2. **Sentinel1ModelTrainingAgent**: Manages Sentinel-1 model training and hyperparameter tuning
  3. **Sentinel1InferenceAgent**: Applies Sentinel-1 models to new data
  4. **Sentinel1TileCoverageAgent**: Analyzes satellite tile coverage for study areas using Sentinel-1 data

## Workflow Definition

### End-to-End Process
1. **Project Initialization**: Define study area, time period for Sentinel-1 data
2. **Data Preparation**: Convert shapefiles to GeoJSON format
3. **Tile Coverage Analysis**: Analyze satellite tile coverage for study areas
4. **Data Extraction**: Retrieve and process Sentinel-1 imagery using GEE
5. **Model Training**: Train Sentinel-1 PSETAE model with specified parameters
6. **Inference**: Apply model to new data

### Agent Interactions
- Agents communicate through the MCP server
- Data references (file paths) are passed between agents
- Status updates and task completion notifications are shared
- Knowledge is stored in the central database

## Implementation Plan

### Directory Structure
```
mcp_server_psetae/
├── server/                  # MCP server implementation
├── agents/                  # Agent implementations
├── utils/                   # Utility functions
│   ├── shapefile_converter.py  # Shapefile to GeoJSON converter
├── config/                  # Configuration files
├── logs/                    # Log files (only check if errors occur)
└── tests/                   # Test scripts
```

### Agent Responsibilities

#### Admin Agent
- Load project parameters
- Create and assign tasks to worker agents
- Monitor progress and handle exceptions
- Generate reports and visualizations

#### Sentinel1DataExtractionAgent
- Authenticate with GEE
- Convert shapefiles (.shp) to GeoJSON format for GEE compatibility
- Execute Sentinel-1 data extraction scripts with appropriate parameters
- Process and normalize Sentinel-1 imagery
- Store data in standardized format
- **SEQUENTIAL EXTRACTION WORKFLOW ARCHITECTURE**: Create a single extraction task for all data splits (test, training, validation)
  - Use `create_sequential_extraction_workflow.py` with the following parameters:
    ```bash
    python create_sequential_extraction_workflow.py --geojson "output/geojson/croptype_KA28_wgs84_test_622.geojson" --start-date <start_date> --end-date <end_date>
    ```
  - The task will automatically determine the correct output directories for each split based on the GeoJSON filename
- **GEE Authentication Requirements**:
  - Must use correct GEE project ID in `sentinel_extraction.py` (not 'your-project-id')
  - Uses the script in `tools/satellite_data_extraction_gee/sentinel_extraction.py`
- **Task Switching Mechanism**:
  - Agent tracks processed tasks locally to prevent reprocessing
  - Updates task status to "completed" after successful processing
  - Filters for incomplete tasks to ensure sequential processing

#### Sentinel1ModelTrainingAgent
- Configure and execute Sentinel-1 training scripts
- Manage hyperparameter tuning
- Track and report training metrics
- Store model checkpoints

#### Sentinel1InferenceAgent
- Load trained Sentinel-1 models
- Apply models to new data
- Format and store prediction results
- Calculate performance metrics

#### Sentinel1TileCoverageAgent
- Analyze satellite tile coverage for study areas
- Determine optimal tiles for data extraction
- Generate coverage reports and visualizations
- Identify gaps in coverage
- Calculate performance metrics

## Operational Guidelines

### Log Monitoring Policy
- Log files should only be checked if there is an error in the workflow
- Do not routinely inspect logs (e.g., sentinel1_data_extraction_agent.log) unless a failure is detected
- If checking logs, focus on recent entries (current timestamp) to identify relevant errors

## Execution Instructions

1. Start MCP server using start_mcp_server.py
2. Initialize Admin Agent with this MCD
3. **REQUIRED**: Run Admin Agent with `--setup-workflow` flag to create all worker agents
   - This step is mandatory and must be performed every time the Admin Agent starts
   - Use `python agents/admin_agent.py --token <admin_token> --server-url <server_url> --setup-workflow`
   - Save the generated worker agent tokens for future use
4. Define project parameters
5. **SEQUENTIAL EXTRACTION WORKFLOW ARCHITECTURE**: Create a single extraction task for all data splits (test, training, validation)
   - Use `create_sequential_extraction_workflow.py` with the following parameters:
     ```bash
     python create_sequential_extraction_workflow.py --start-date <start_date> --end-date <end_date>
     ```
   - The script uses hardcoded GeoJSON files for each split (test, validation, training)
   - The task will automatically determine the correct output directories for each split based on the GeoJSON filename
6. **IMPORTANT - SENTINEL1 DATA EXTRACTION AGENT CONSIDERATIONS**:
   - **GEE Authentication**: Ensure the Google Earth Engine project ID is correctly set to `ee-shivaprakashssy-psetae-ka28` in the `sentinel_extraction.py` script
   - **Script Location**: The agent must use the correct version of `sentinel_extraction.py` from `mcp_server_psetae/tools/satellite_data_extraction_gee/`
   - **Task Switching**: The agent implements task tracking to prevent task repetition and ensure proper sequential processing of test, validation, and training splits
   - If tasks are being repeated or overwritten, check that task status updates are working correctly
7. **CRITICAL - SENTINEL1 MODEL TRAINING AGENT CONSIDERATIONS**:
   - **Temporal Encoding Parameters**: The agent MUST pass the following parameters to avoid errors in the temporal attention encoder:
     - `--T 366` (temporal period, must be an integer)
     - `--lms 180` (maximum sequence length, must be an integer)
     - `--positions order` (position encoding type)
   - **Attention Parameters**: The agent MUST explicitly pass these parameters to ensure proper initialization:
     - `--n_head 4` (number of attention heads)
     - `--d_k 32` (dimension of key and query vectors)
   - **Geometric Features**: Set `--geomfeat 0` to disable geometric features and avoid pandas errors
   - **CPU Fallback**: The training script has been modified to automatically use CPU if CUDA is not available
   - **Parameter Type Enforcement**: All numeric parameters must be properly passed as strings in the subprocess call, which will be correctly parsed as integers by the argparse module
8. Execute workflow through Admin Agent

## References
- PSETAE GitHub Repository
- Agent-MCP Framework Documentation
- Google Earth Engine API Documentation
