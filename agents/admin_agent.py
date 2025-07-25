#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Admin Agent for Sentinel-1 PSETAE MCP Server
This script implements the Admin Agent that orchestrates the Sentinel-1 PSETAE workflow.
"""

import os
import sys
import json
import argparse
import logging
import requests
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.config import *

# Initialize logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(BASE_DIR, "logs", "admin_agent.log"), mode='a') 
        if os.path.exists(os.path.join(BASE_DIR, "logs")) else logging.StreamHandler()
    ]
)
logger = logging.getLogger("admin_agent")

class AdminAgent:
    """Admin Agent for orchestrating the PSETAE workflow"""
    
    def __init__(self, token, server_url):
        """Initialize the Admin Agent
        
        Args:
            token (str): Admin token for authentication
            server_url (str): URL of the MCP server
        """
        self.token = token
        self.server_url = server_url
        self.agent_id = ADMIN_AGENT_ID
        self.agent_type = "admin"
        
        # Validate token
        self.validate_token()
        
        logger.info(f"Admin Agent initialized with ID: {self.agent_id}")
    
    def validate_token(self):
        """Validate the admin token with the MCP server"""
        try:
            response = requests.post(
                f"{self.server_url}/api/validate-token",
                json={
                    "agent_id": self.agent_id,
                    "agent_type": self.agent_type,
                    "token": self.token
                }
            )
            
            if response.status_code != 200:
                logger.error(f"Token validation failed: {response.text}")
                raise ValueError("Invalid admin token")
            
            logger.info("Admin token validated successfully")
        except Exception as e:
            logger.error(f"Error validating token: {str(e)}")
            raise
    
    def load_mcd(self, mcd_path):
        """Load the Main Context Document (MCD) into the project context
        
        Args:
            mcd_path (str): Path to the MCD file
        """
        try:
            with open(mcd_path, 'r') as f:
                mcd_content = f.read()
            
            # Add MCD to memory
            response = requests.post(
                f"{self.server_url}/api/memory/add",
                params={"token": self.token},
                json={
                    "agent_id": self.agent_id,
                    "content": mcd_content,
                    "metadata": {
                        "type": "mcd",
                        "filename": os.path.basename(mcd_path),
                        "timestamp": datetime.now().isoformat()
                    }
                }
            )
            
            if response.status_code != 200:
                logger.error(f"Failed to load MCD: {response.text}")
                raise ValueError("Failed to load MCD")
            
            logger.info(f"MCD loaded successfully from {mcd_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading MCD: {str(e)}")
            raise
    
    def create_worker_agent(self, agent_id, agent_type):
        """Create a new worker agent
        
        Args:
            agent_id (str): ID for the new agent
            agent_type (str): Type of agent to create
            
        Returns:
            dict: Worker agent information including token
        """
        try:
            response = requests.post(
                f"{self.server_url}/api/create-worker",
                params={"token": self.token},
                json={
                    "agent_id": agent_id,
                    "agent_type": agent_type,
                    "token": self.token
                }
            )
            
            if response.status_code != 200:
                logger.error(f"Failed to create worker agent: {response.text}")
                raise ValueError(f"Failed to create worker agent: {response.text}")
            
            worker_info = response.json()
            logger.info(f"Created worker agent: {agent_id} of type {agent_type}")
            
            return worker_info
        except Exception as e:
            logger.error(f"Error creating worker agent: {str(e)}")
            raise
    
    def create_task(self, title, description, assigned_to, priority=1, metadata=None):
        """Create a new task
        
        Args:
            title (str): Task title
            description (str): Task description
            assigned_to (str): Agent ID to assign the task to
            priority (int): Task priority (1-5)
            metadata (dict): Additional metadata for the task
            
        Returns:
            dict: Created task information
        """
        try:
            response = requests.post(
                f"{self.server_url}/api/tasks/create",
                params={"token": self.token},
                json={
                    "title": title,
                    "description": description,
                    "assigned_to": assigned_to,
                    "priority": priority,
                    "metadata": metadata or {}
                }
            )
            
            if response.status_code != 200:
                logger.error(f"Failed to create task: {response.text}")
                raise ValueError(f"Failed to create task: {response.text}")
            
            task_info = response.json()
            logger.info(f"Created task: {title} assigned to {assigned_to}")
            
            return task_info
        except Exception as e:
            logger.error(f"Error creating task: {str(e)}")
            raise
    
    def get_tasks(self, agent_id=None):
        """Get tasks for an agent
        
        Args:
            agent_id (str): Agent ID to get tasks for, defaults to admin agent
            
        Returns:
            list: List of tasks
        """
        try:
            agent_id = agent_id or self.agent_id
            response = requests.get(
                f"{self.server_url}/api/tasks/{agent_id}",
                params={"token": self.token}
            )
            
            if response.status_code != 200:
                logger.error(f"Failed to get tasks: {response.text}")
                raise ValueError(f"Failed to get tasks: {response.text}")
            
            tasks = response.json().get("tasks", [])
            logger.info(f"Retrieved {len(tasks)} tasks for agent {agent_id}")
            
            return tasks
        except Exception as e:
            logger.error(f"Error getting tasks: {str(e)}")
            raise
    
    def setup_psetae_workflow(self):
        """Set up the Sentinel-1 PSETAE workflow with all required agents"""
        # Create Sentinel-1 data extraction agent
        data_agent = self.create_worker_agent("sentinel1-data-extraction-agent", "sentinel1-data-extraction")
        logger.info(f"Sentinel-1 Data Extraction Agent created with token: {data_agent['token']}")
        
        # Create Sentinel-1 model training agent
        training_agent = self.create_worker_agent("sentinel1-model-training-agent", "sentinel1-model-training")
        logger.info(f"Sentinel-1 Model Training Agent created with token: {training_agent['token']}")
        
        # Create Sentinel-1 inference agent
        inference_agent = self.create_worker_agent("sentinel1-inference-agent", "sentinel1-inference")
        logger.info(f"Sentinel-1 Inference Agent created with token: {inference_agent['token']}")
        
        # Create Sentinel-1 tile coverage agent
        coverage_agent = self.create_worker_agent("sentinel1-tile-coverage-agent", "sentinel1-tile-coverage")
        logger.info(f"Sentinel-1 Tile Coverage Agent created with token: {coverage_agent['token']}")
        
        # Create Sentinel-1 reporting agent
        reporting_agent = self.create_worker_agent("sentinel1-reporting-agent", "sentinel1-reporting")
        logger.info(f"Sentinel-1 Reporting Agent created with token: {reporting_agent['token']}")
        
        # Return all agent information
        return {
            "data_agent": data_agent,
            "training_agent": training_agent,
            "inference_agent": inference_agent,
            "coverage_agent": coverage_agent,
            "reporting_agent": reporting_agent
        }

    def create_sentinel1_workflow(self, shapefile_path, output_base_dir, start_date, end_date, model_config=None):
        """Create a complete Sentinel-1 workflow from shapefile to inference results
        
        Args:
            shapefile_path (str): Path to the shapefile defining the study area
            output_base_dir (str): Base directory for all outputs
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            model_config (str): Optional path to model configuration file
            
        Returns:
            dict: Workflow information including task IDs
        """
        try:
            # Create output directories
            geojson_dir = os.path.join(output_base_dir, "geojson")
            coverage_dir = os.path.join(output_base_dir, "tile_coverage")
            data_dir = os.path.join(output_base_dir, "data")
            model_dir = os.path.join(output_base_dir, "model")
            inference_dir = os.path.join(output_base_dir, "inference")
            
            os.makedirs(geojson_dir, exist_ok=True)
            os.makedirs(coverage_dir, exist_ok=True)
            os.makedirs(data_dir, exist_ok=True)
            os.makedirs(model_dir, exist_ok=True)
            os.makedirs(inference_dir, exist_ok=True)
            
            # Generate output file paths
            geojson_path = os.path.join(geojson_dir, os.path.splitext(os.path.basename(shapefile_path))[0] + ".geojson")
            coverage_report = os.path.join(coverage_dir, "coverage_report.json")
            
            # Create workflow tasks
            workflow = {}
            
            # 1. Create shapefile conversion task
            workflow["conversion_task"] = self.create_task(
                title="Shapefile to GeoJSON Conversion",
                description=f"Convert shapefile {shapefile_path} to GeoJSON format",
                assigned_to="sentinel1-data-extraction-agent",
                priority=1,
                metadata={
                    "shapefile_path": shapefile_path,
                    "output_path": geojson_path
                }
            )
            
            # 2. Create tile coverage analysis task
            workflow["coverage_task"] = self.create_task(
                title="Sentinel-1 Tile Coverage Analysis",
                description=f"Analyze Sentinel-1 tile coverage for area in {geojson_path} from {start_date} to {end_date}",
                assigned_to="sentinel1-tile-coverage-agent",
                priority=2,
                metadata={
                    "geojson_path": geojson_path,
                    "output_dir": coverage_dir,
                    "start_date": start_date,
                    "end_date": end_date,
                    "collection": SENTINEL1_COLLECTION
                }
            )
            
            # 3. Create data extraction task
            workflow["extraction_task"] = self.create_task(
                title="Sentinel-1 Data Extraction",
                description=f"Extract Sentinel-1 data for area in {geojson_path} from {start_date} to {end_date}",
                assigned_to="sentinel1-data-extraction-agent",
                priority=3,
                metadata={
                    "geojson_path": geojson_path,
                    "output_dir": data_dir,
                    "start_date": start_date,
                    "end_date": end_date,
                    "bands": ["VV", "VH"]
                }
            )
            
            # 4. Create model training task
            workflow["training_task"] = self.create_task(
                title="Sentinel-1 Model Training",
                description=f"Train Sentinel-1 PSETAE model using data in {data_dir}",
                assigned_to="sentinel1-model-training-agent",
                priority=4,
                metadata={
                    "data_dir": data_dir,
                    "output_dir": model_dir,
                    "config_file": model_config,
                    "epochs": 100,
                    "batch_size": 32,
                    "learning_rate": 0.001
                }
            )
            
            # 5. Create inference task
            workflow["inference_task"] = self.create_task(
                title="Sentinel-1 Model Inference",
                description=f"Run inference using trained model on data in {data_dir}",
                assigned_to="sentinel1-inference-agent",
                priority=5,
                metadata={
                    "model_path": os.path.join(model_dir, "models", "best_model.pth"),  # This will be updated by the training agent
                    "data_dir": data_dir,
                    "output_dir": inference_dir,
                    "config_file": model_config
                }
            )
            
            # Add workflow to memory
            self.add_memory(
                f"Created Sentinel-1 workflow for {shapefile_path} from {start_date} to {end_date}",
                {
                    "action": "create_workflow",
                    "shapefile_path": shapefile_path,
                    "output_base_dir": output_base_dir,
                    "start_date": start_date,
                    "end_date": end_date,
                    "model_config": model_config,
                    "workflow_tasks": {
                        "coverage": workflow["coverage_task"]["id"],
                        "extraction": workflow["extraction_task"]["id"],
                        "training": workflow["training_task"]["id"],
                        "inference": workflow["inference_task"]["id"]
                    },
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            logger.info(f"Created complete Sentinel-1 workflow with {len(workflow)} tasks")
            return workflow
            
        except Exception as e:
            logger.error(f"Error creating Sentinel-1 workflow: {str(e)}")
            raise
    
    def create_sequential_extraction_workflow(self, geojson_files, output_base_dir, start_date, end_date, bands=None):
        """Create a sequential multi-split extraction workflow for test, validation, and training data.
        
        This method creates extraction tasks one at a time to avoid cross-task interference
        and ensure each split's data goes to the correct output directory.
        
        Args:
            geojson_files (dict): Dictionary mapping split names to GeoJSON file paths
                                 e.g., {"test": "test.geojson", "validation": "val.geojson", "training": "train.geojson"}
            output_base_dir (str): Base directory for all outputs
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            bands (list): List of bands to extract (default: ["VV", "VH"])
            
        Returns:
            dict: Workflow information including task IDs for each split
        """
        if bands is None:
            bands = ["VV", "VH"]
            
        try:
            # Create output directories for each split
            sentinel1_data_dir = os.path.join(output_base_dir, "sentinel1_data")
            split_directories = {
                "test": os.path.join(sentinel1_data_dir, "testdirectory"),
                "validation": os.path.join(sentinel1_data_dir, "validationdirectory"), 
                "training": os.path.join(sentinel1_data_dir, "traindirectory")
            }
            
            # Create all directories
            for split_dir in split_directories.values():
                os.makedirs(split_dir, exist_ok=True)
                logger.info(f"Created directory: {split_dir}")
            
            # Read tile coverage information if available
            coverage_file = os.path.join(output_base_dir, "tile_coverage", "sentinel1_coverage.txt")
            tile_info = {"tracks": ["63"], "acquisitions": {"63": 16}}  # Default values
            
            if os.path.exists(coverage_file):
                try:
                    with open(coverage_file, 'r') as f:
                        content = f.read()
                        # Parse tile information from coverage file
                        tracks = []
                        acquisitions = {}
                        for line in content.split('\n'):
                            if 'Track' in line and 'acquisitions' in line:
                                parts = line.split()
                                if len(parts) >= 4:
                                    track = parts[1].rstrip(':')
                                    acq_count = int(parts[2])
                                    tracks.append(track)
                                    acquisitions[track] = acq_count
                        
                        if tracks:
                            tile_info = {"tracks": tracks, "acquisitions": acquisitions}
                            logger.info(f"Found tile information: {len(tracks)} tracks")
                except Exception as e:
                    logger.warning(f"Could not read tile coverage file: {e}")
            
            # Create workflow tasks for each split
            workflow = {}
            
            # Define processing order: test -> validation -> training
            processing_order = ["test", "validation", "training"]
            priority = 1
            
            for split in processing_order:
                if split not in geojson_files:
                    logger.warning(f"No GeoJSON file provided for {split} split, skipping")
                    continue
                    
                geojson_path = geojson_files[split]
                output_dir = split_directories[split]
                
                # Verify GeoJSON file exists
                if not os.path.exists(geojson_path):
                    logger.error(f"GeoJSON file not found for {split} split: {geojson_path}")
                    continue
                
                # Create extraction task for this split
                task_metadata = {
                    "geojson_path": geojson_path,
                    "output_dir": output_dir,
                    "start_date": start_date,
                    "end_date": end_date,
                    "bands": bands,
                    "task_type": "data_extraction",
                    "tile_information": tile_info,
                    "coverage_file": coverage_file,
                    "split": split,  # Explicitly mark the split type
                    "workflow_type": "sequential_extraction",  # Mark as sequential workflow
                    "processing_order": processing_order.index(split) + 1  # Order in sequence
                }
                
                task_title = f"Sentinel-1 {split.upper()} Data Extraction {start_date} to {end_date}"
                task_description = f"Extract Sentinel-1 data for {split.upper()} split from {start_date} to {end_date} using {os.path.basename(geojson_path)}"
                
                extraction_task = self.create_task(
                    title=task_title,
                    description=task_description,
                    assigned_to="sentinel1-data-extraction-agent",
                    priority=priority,
                    metadata=task_metadata
                )
                
                workflow[f"{split}_extraction_task"] = extraction_task
                logger.info(f"Created {split} extraction task with ID: {extraction_task.get('id')}")
                logger.info(f"  GeoJSON: {geojson_path}")
                logger.info(f"  Output: {output_dir}")
                
                priority += 1
            
            # Add workflow to memory
            self.add_memory(
                f"Created sequential extraction workflow for {len(workflow)} splits from {start_date} to {end_date}",
                {
                    "action": "create_sequential_extraction_workflow",
                    "splits": list(geojson_files.keys()),
                    "output_base_dir": output_base_dir,
                    "start_date": start_date,
                    "end_date": end_date,
                    "bands": bands,
                    "workflow": workflow,
                    "processing_order": processing_order
                }
            )
            
            logger.info(f"Sequential extraction workflow created successfully with {len(workflow)} tasks")
            return workflow
            
        except Exception as e:
            logger.error(f"Error creating sequential extraction workflow: {str(e)}")
            raise

    def add_memory(self, content, metadata=None):
        """Add a memory entry
        
        Args:
            content (str): Memory content
            metadata (dict): Additional metadata
            
        Returns:
            dict: Memory information
        """
        try:
            response = requests.post(
                f"{self.server_url}/api/memory/add",
                params={"token": self.token},
                json={
                    "agent_id": self.agent_id,
                    "content": content,
                    "metadata": metadata or {}
                }
            )
            
            if response.status_code != 200:
                logger.error(f"Failed to add memory: {response.text}")
                raise ValueError(f"Failed to add memory: {response.text}")
            
            memory_info = response.json()
            logger.info(f"Added memory with ID: {memory_info.get('memory_id')}")
            
            return memory_info
        except Exception as e:
            logger.error(f"Error adding memory: {str(e)}")
            raise

def main():
    """Main function to initialize the Admin Agent"""
    parser = argparse.ArgumentParser(description='Initialize the Admin Agent for PSETAE')
    parser.add_argument('--token', type=str, required=True, help='Admin token for authentication')
    parser.add_argument('--server-url', type=str, default=f"http://localhost:{SERVER_PORT}", help='MCP server URL')
    parser.add_argument('--mcd', type=str, default=os.path.join(BASE_DIR, "MCD.md"), help='Path to the Main Context Document')
    parser.add_argument('--setup-workflow', action='store_true', help='Set up the complete PSETAE workflow')
    
    # Workflow creation arguments
    parser.add_argument('--create-workflow', action='store_true', help='Create a complete Sentinel-1 workflow')
    parser.add_argument('--shapefile', type=str, help='Path to the shapefile defining the study area')
    parser.add_argument('--output-dir', type=str, help='Base directory for all outputs')
    parser.add_argument('--start-date', type=str, help='Start date for data extraction (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date for data extraction (YYYY-MM-DD)')
    parser.add_argument('--model-config', type=str, help='Path to model configuration file')
    
    args = parser.parse_args()
    
    try:
        # Initialize Admin Agent
        admin_agent = AdminAgent(args.token, args.server_url)
        
        # Load MCD
        admin_agent.load_mcd(args.mcd)
        
        # Set up workflow if requested
        if args.setup_workflow:
            agents = admin_agent.setup_psetae_workflow()
            
            # Print agent tokens
            print("\n" + "="*50)
            print("Sentinel-1 PSETAE Workflow Initialized")
            print("="*50)
            print("\nAgent Tokens (save these for initializing worker agents):")
            print(f"Sentinel-1 Data Extraction Agent: {agents['data_agent']['token']}")
            print(f"Sentinel-1 Model Training Agent: {agents['training_agent']['token']}")
            print(f"Sentinel-1 Inference Agent: {agents['inference_agent']['token']}")
            print(f"Sentinel-1 Tile Coverage Agent: {agents['coverage_agent']['token']}")
            print(f"Sentinel-1 Reporting Agent: {agents['reporting_agent']['token']}")
            print("\n" + "="*50)
        
        # Create workflow if requested
        if args.create_workflow:
            if not all([args.shapefile, args.output_dir, args.start_date, args.end_date]):
                raise ValueError("Missing required parameters for workflow creation")
            
            workflow = admin_agent.create_sentinel1_workflow(
                args.shapefile,
                args.output_dir,
                args.start_date,
                args.end_date,
                args.model_config
            )
            
            # Print workflow information
            print("\n" + "="*50)
            print("Sentinel-1 Workflow Created")
            print("="*50)
            print(f"\nWorkflow for {os.path.basename(args.shapefile)} from {args.start_date} to {args.end_date}")
            print(f"\nTask IDs:")
            print(f"Tile Coverage Analysis: {workflow['coverage_task']['id']}")
            print(f"Data Extraction: {workflow['extraction_task']['id']}")
            print(f"Model Training: {workflow['training_task']['id']}")
            print(f"Model Inference: {workflow['inference_task']['id']}")
            print("\n" + "="*50)
        
        print("\nAdmin Agent initialized successfully!")
        
    except Exception as e:
        logger.error(f"Error initializing Admin Agent: {str(e)}")
        sys.exit(1)
        sys.exit(1)

if __name__ == "__main__":
    main()
