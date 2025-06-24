# Main Context Document (MCD): PSETAE Multi-Agent System

## Project Overview

PSETAE is a crop classification system using satellite imagery from multiple sources (HLS L30/S30, Sentinel-1/2) with deep learning. This MCD defines the implementation of a Model Context Protocol (MCP) multi-agent system to orchestrate the PSETAE workflow.

## System Architecture

### Existing PSETAE Components
- **Data Extraction**: GEE-based scripts for HLS and Sentinel data
  - Requires GeoJSON input files for processing
- **Model Training**: PSETAE implementations for different satellite sources
- **Inference**: Prediction scripts for trained models
- **Tile Analysis**: Coverage analysis for satellite data

### MCP Agent Structure
- **MCP Server**: Central coordination point with database (`mcp_state.db`)
- **Admin Agent**: Workflow orchestrator and task manager
- **Worker Agents**:
  1. **DataExtractionAgent**: Handles GEE data retrieval and processing
  2. **ModelTrainingAgent**: Manages model training and hyperparameter tuning
  3. **InferenceAgent**: Applies models to new data
  4. **TileCoverageAgent**: Analyzes satellite coverage

## Workflow Definition

### End-to-End Process
1. **Project Initialization**: Define study area, time period, and data sources
2. **Data Preparation**: Convert shapefiles to GeoJSON format
3. **Data Extraction**: Retrieve and process satellite imagery using GEE
4. **Model Training**: Train PSETAE models with specified parameters
5. **Inference**: Apply models to new data
6. **Coverage Analysis**: Analyze satellite coverage for study areas

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

#### DataExtractionAgent
- Authenticate with GEE
- Convert shapefiles (.shp) to GeoJSON format for GEE compatibility
- Execute data extraction scripts with appropriate parameters
- Process and normalize satellite imagery
- Store data in standardized format

#### ModelTrainingAgent
- Configure and execute training scripts
- Manage hyperparameter tuning
- Track and report training metrics
- Store model checkpoints

#### InferenceAgent
- Load trained models
- Apply models to new data
- Format and store prediction results
- Calculate performance metrics

#### TileCoverageAgent
- Analyze satellite coverage for study areas
- Generate coverage reports
- Provide recommendations for data acquisition

## Execution Instructions

1. Start MCP server
2. Initialize Admin Agent with this MCD
3. Admin creates worker agents
4. Define project parameters
5. Execute workflow through Admin Agent

## References
- PSETAE GitHub Repository
- Agent-MCP Framework Documentation
- Google Earth Engine API Documentation
