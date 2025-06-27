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
- **IMPORTANT**: Requires separate extraction tasks for test, training, and validation data splits
  - Each split must use its corresponding GeoJSON file (containing 'test', 'training', or 'validation' in filename)
  - Output is placed in dataset-specific directories (testdirectory, traindirectory, validationdirectory)

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

## Execution Instructions

1. Start MCP server
2. Initialize Admin Agent with this MCD
3. **REQUIRED**: Run Admin Agent with `--setup-workflow` flag to create all worker agents
   - This step is mandatory and must be performed every time the Admin Agent starts
   - Use `python agents/admin_agent.py --token <admin_token> --server-url <server_url> --setup-workflow`
   - Save the generated worker agent tokens for future use
4. Define project parameters
5. **REQUIRED**: Create separate extraction tasks for test, training, and validation splits
   - **CRITICAL WORKFLOW REQUIREMENT**: Before running any extraction workflow, ALWAYS check if all three extraction tasks (test, training, validation) exist on the MCP server. If any are missing, they MUST be created.
   - Use `create_data_extraction_task.py` with appropriate GeoJSON files:
     ```bash
     # For test data
     python create_data_extraction_task.py --geojson "output/geojson/croptype_KA28_wgs84_test_622.geojson" --start-date <start_date> --end-date <end_date>
     
     # For training data
     python create_data_extraction_task.py --geojson "output/geojson/croptype_KA28_wgs84_training_622.geojson" --start-date <start_date> --end-date <end_date>
     
     # For validation data
     python create_data_extraction_task.py --geojson "output/geojson/croptype_KA28_wgs84_validation_622.geojson" --start-date <start_date> --end-date <end_date>
     ```
   - Each task will automatically determine the correct output directory based on the GeoJSON filename
   - To check existing tasks, use: `python -c "import requests; response = requests.get('http://localhost:8080/api/tasks/list', params={'token': '<admin_token>'}); print(response.text)"` and verify that tasks for all three splits are present
6. Execute workflow through Admin Agent

## References
- PSETAE GitHub Repository
- Agent-MCP Framework Documentation
- Google Earth Engine API Documentation
