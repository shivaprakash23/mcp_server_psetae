# MCP Server for Sentinel-1 PSETAE

A multi-agent system implementation using the Model Context Protocol (MCP) to orchestrate Sentinel-1 PSETAE crop classification workflows.

## Overview

This repository implements a Model Context Protocol (MCP) server that coordinates multiple AI agents to execute PSETAE (Pixel Set Encoder with Temporal Attention Encoder) workflows for crop classification using Sentinel-1 satellite imagery. The system integrates with Google Earth Engine (GEE) for data extraction and leverages the existing PSETAE codebase for model training and inference. This implementation focuses exclusively on Sentinel-1 data to establish a stable baseline before expanding to other data sources.

## Features

- **Multi-Agent Architecture**: Hierarchical agent structure with Admin and specialized Worker agents
- **Workflow Orchestration**: End-to-end automation of PSETAE tasks from data extraction to inference
- **GEE Integration**: Seamless interaction with Google Earth Engine for satellite data retrieval
- **Modular Design**: Agents encapsulate specific functionality of the PSETAE pipeline
- **Central Knowledge Base**: Shared context and memory through MCP database

## System Architecture

![MCP PSETAE Architecture](docs/images/architecture.png)

### Agent Structure

- **Admin Agent**: Orchestrates the workflow and delegates tasks
- **Worker Agents**:
  - **Sentinel1DataExtractionAgent**: Handles GEE data retrieval and processing for Sentinel-1
  - **Sentinel1ModelTrainingAgent**: Manages model training and hyperparameter tuning for Sentinel-1
  - **Sentinel1InferenceAgent**: Applies Sentinel-1 models to new data
  - **Sentinel1TileCoverageAgent**: Analyzes satellite tile coverage for study areas
  - **Sentinel1DocumentationAgent**: Prepares the report based on the results of the PSETAE model that got trained.

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/mcp_server_psetae.git
cd mcp_server_psetae
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up Google Earth Engine authentication:
```bash
earthengine authenticate
```

## Usage

### Starting the MCP Server

```bash
python server/server.py --port 8080 --project-dir /path/to/project
```

### Initializing the Admin Agent

```bash
# First, retrieve the admin token from the database
# Then initialize the admin agent with:
# "Initialize as an admin agent with this token: [admin-token] Please add the MCD.md file to the project context. Don't summarize it."
```

### Creating Worker Agents

Worker agents are created through the Admin Agent:

1. Ask the Admin Agent to create a worker (e.g., "Create a worker agent with ID 'data-extraction-worker'")
2. The Admin will provide a worker token
3. Initialize the worker with the provided token

## Workflow Example

1. **Project Initialization**:
   - Define study area, time period, and satellite data sources

2. **Data Extraction**:
   - Extract HLS or Sentinel data using Google Earth Engine
   - Process and normalize the satellite imagery

3. **Model Training**:
   - Train PSETAE models with the extracted data
   - Tune hyperparameters for optimal performance

4. **Inference**:
   - Apply trained models to new data
   - Generate crop classification maps

5. **Coverage Analysis**:
   - Analyze satellite tile coverage for the study area
   - Generate coverage reports

## Directory Structure

```
mcp_server_psetae/
├── server/                  # MCP server implementation
├── agents/                  # Agent implementations
├── utils/                   # Utility functions
├── config/                  # Configuration files
├── docs/                    # Documentation
└── tests/                   # Test scripts
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- PSETAE codebase by Shivaprakash Yaragal, Lund University
- Agent-MCP framework by rinadelph
- Google Earth Engine team for the Python API
