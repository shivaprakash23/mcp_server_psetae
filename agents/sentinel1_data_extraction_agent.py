#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sentinel-1 Data Extraction Agent for PSETAE MCP Server
This agent handles GEE data retrieval and processing for Sentinel-1 data.
It acts as a wrapper around the existing data extraction scripts in the PSETAE repository.
"""

import os
import sys
import json
import argparse
import logging
import requests
import subprocess
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
        logging.FileHandler(os.path.join(BASE_DIR, "logs", "sentinel1_data_extraction_agent.log"), mode='a') 
        if os.path.exists(os.path.join(BASE_DIR, "logs")) else logging.StreamHandler()
    ]
)
logger = logging.getLogger("sentinel1_data_extraction_agent")

class Sentinel1DataExtractionAgent:
    """Sentinel-1 Data Extraction Agent for handling GEE data retrieval and processing"""
    
    def __init__(self, token, server_url, agent_id):
        """Initialize the Sentinel-1 Data Extraction Agent
        
        Args:
            token (str): Agent token for authentication
            server_url (str): URL of the MCP server
            agent_id (str): ID of this agent
        """
        self.token = token
        self.server_url = server_url
        self.agent_id = agent_id
        self.agent_type = "sentinel1-data-extraction"
        
        # Validate token
        self.validate_token()
        
        logger.info(f"Sentinel-1 Data Extraction Agent initialized with ID: {self.agent_id}")
    
    def validate_token(self):
        """Validate the agent token with the MCP server"""
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
                raise ValueError("Invalid agent token")
            
            logger.info("Agent token validated successfully")
        except Exception as e:
            logger.error(f"Error validating token: {str(e)}")
            raise
    
    def get_tasks(self):
        """Get tasks assigned to this agent
        
        Returns:
            list: List of tasks
        """
        try:
            response = requests.get(
                f"{self.server_url}/api/tasks/{self.agent_id}",
                params={"token": self.token}
            )
            
            if response.status_code != 200:
                logger.error(f"Failed to get tasks: {response.text}")
                raise ValueError(f"Failed to get tasks: {response.text}")
            
            tasks = response.json().get("tasks", [])
            logger.info(f"Retrieved {len(tasks)} tasks")
            
            return tasks
        except Exception as e:
            logger.error(f"Error getting tasks: {str(e)}")
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
    
    def convert_shapefile(self, shapefile_path, output_path=None):
        """Convert a shapefile to GeoJSON format using the existing utility script
        
        Args:
            shapefile_path (str): Path to the shapefile
            output_path (str): Path to save the GeoJSON file (optional)
            
        Returns:
            str: Path to the GeoJSON file
        """
        try:
            # Use the shapefile converter script from the PSETAE repository
            converter_script = os.path.join(PSETAE_BASE_DIR, "utils", "shapefile_to_geojson.py")
            
            # Build command
            cmd = [sys.executable, converter_script, "--shapefile", shapefile_path]
            if output_path:
                cmd.extend(["--output", output_path])
            
            # Execute command
            logger.info(f"Executing shapefile conversion: {' '.join(cmd)}")
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                logger.error(f"Shapefile conversion failed: {stderr}")
                raise RuntimeError(f"Shapefile conversion failed: {stderr}")
            
            # Parse output to get the path of the generated GeoJSON file
            for line in stdout.splitlines():
                if line.strip().endswith(".geojson"):
                    geojson_path = line.strip()
                    break
            else:
                # If no .geojson file found in output, assume it's in the same directory with .geojson extension
                geojson_path = os.path.splitext(shapefile_path)[0] + ".geojson"
                if output_path:
                    geojson_path = output_path
            
            logger.info(f"Converted shapefile {shapefile_path} to GeoJSON {geojson_path}")
            
            # Add memory entry
            self.add_memory(
                f"Converted shapefile {shapefile_path} to GeoJSON {geojson_path}",
                {
                    "action": "shapefile_conversion",
                    "input_file": shapefile_path,
                    "output_file": geojson_path,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            return geojson_path
        except Exception as e:
            logger.error(f"Error converting shapefile: {str(e)}")
            raise
    
    def batch_convert_shapefiles(self, input_dir, output_dir=None):
        """Convert all shapefiles in a directory to GeoJSON format using the existing utility script
        
        Args:
            input_dir (str): Directory containing shapefiles
            output_dir (str): Directory to save GeoJSON files (optional)
            
        Returns:
            list: Paths to the GeoJSON files
        """
        try:
            # Use the batch shapefile converter script from the PSETAE repository
            converter_script = os.path.join(PSETAE_BASE_DIR, "utils", "batch_shapefile_to_geojson.py")
            
            # Build command
            cmd = [sys.executable, converter_script, "--input_dir", input_dir]
            if output_dir:
                cmd.extend(["--output_dir", output_dir])
            
            # Execute command
            logger.info(f"Executing batch shapefile conversion: {' '.join(cmd)}")
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                logger.error(f"Batch shapefile conversion failed: {stderr}")
                raise RuntimeError(f"Batch shapefile conversion failed: {stderr}")
            
            # Parse output to get the paths of the generated GeoJSON files
            geojson_paths = []
            for line in stdout.splitlines():
                if line.strip().endswith(".geojson"):
                    geojson_paths.append(line.strip())
            
            logger.info(f"Converted {len(geojson_paths)} shapefiles from {input_dir}")
            
            # Add memory entry
            self.add_memory(
                f"Batch converted {len(geojson_paths)} shapefiles from {input_dir}",
                {
                    "action": "batch_shapefile_conversion",
                    "input_dir": input_dir,
                    "output_dir": output_dir or os.path.join(input_dir, "geojson"),
                    "file_count": len(geojson_paths),
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            return geojson_paths
        except Exception as e:
            logger.error(f"Error batch converting shapefiles: {str(e)}")
            raise
    
    def extract_sentinel1_data(self, geojson_path, output_dir, start_date, end_date, bands=None):
        """Extract Sentinel-1 data using GEE
        
        Args:
            geojson_path (str): Path to the GeoJSON file
            output_dir (str): Directory to save extracted data
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            bands (list): List of bands to extract (default: VV, VH)
            
        Returns:
            str: Path to the extracted data
        """
        try:
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Use the existing Sentinel-1 extraction script
            sentinel_script = os.path.join(DATA_EXTRACTION_DIR, "sentinel_extraction.py")
            
            # Default bands if not specified
            if bands is None:
                bands = SENTINEL1_BANDS
            
            # Build command
            cmd = [
                sys.executable,
                sentinel_script,
                "--geojson", geojson_path,
                "--output", output_dir,
                "--start_date", start_date,
                "--end_date", end_date,
                "--bands", ",".join(bands),
                "--collection", SENTINEL1_COLLECTION,
                "--orbit", SENTINEL1_ORBIT
            ]
            
            # Execute command
            logger.info(f"Executing Sentinel-1 extraction: {' '.join(cmd)}")
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                logger.error(f"Sentinel-1 extraction failed: {stderr}")
                raise RuntimeError(f"Sentinel-1 extraction failed: {stderr}")
            
            logger.info(f"Sentinel-1 extraction completed successfully")
            logger.debug(f"Extraction output: {stdout}")
            
            # Add memory entry
            self.add_memory(
                f"Extracted Sentinel-1 data for area in {geojson_path} from {start_date} to {end_date}",
                {
                    "action": "sentinel1_extraction",
                    "geojson_path": geojson_path,
                    "output_dir": output_dir,
                    "start_date": start_date,
                    "end_date": end_date,
                    "bands": bands,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            return output_dir
        except Exception as e:
            logger.error(f"Error extracting Sentinel-1 data: {str(e)}")
            raise
    
    def process_task(self, task):
        """Process a task assigned to this agent
        
        Args:
            task (dict): Task information
            
        Returns:
            bool: True if task was processed successfully
        """
        try:
            task_id = task.get("id")
            metadata = json.loads(task.get("metadata", "{}"))
            
            logger.info(f"Processing task {task_id}: {task.get('title')}")
            
            # Check task type
            if "shapefile_conversion" in task.get("title", "").lower():
                # Shapefile conversion task
                shapefile_path = metadata.get("shapefile_path")
                output_path = metadata.get("output_path")
                
                if not shapefile_path:
                    raise ValueError("Missing shapefile_path in task metadata")
                
                self.convert_shapefile(shapefile_path, output_path)
                
            elif "batch_conversion" in task.get("title", "").lower():
                # Batch shapefile conversion task
                input_dir = metadata.get("input_dir")
                output_dir = metadata.get("output_dir")
                
                if not input_dir:
                    raise ValueError("Missing input_dir in task metadata")
                
                self.batch_convert_shapefiles(input_dir, output_dir)
                
            elif "data_extraction" in task.get("title", "").lower():
                # Data extraction task
                geojson_path = metadata.get("geojson_path")
                output_dir = metadata.get("output_dir")
                start_date = metadata.get("start_date")
                end_date = metadata.get("end_date")
                bands = metadata.get("bands")
                
                if not all([geojson_path, output_dir, start_date, end_date]):
                    raise ValueError("Missing required parameters in task metadata")
                
                self.extract_sentinel1_data(geojson_path, output_dir, start_date, end_date, bands)
                
            else:
                logger.warning(f"Unknown task type: {task.get('title')}")
                return False
            
            logger.info(f"Task {task_id} processed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error processing task: {str(e)}")
            return False
    
    def run(self):
        """Run the agent to process tasks"""
        try:
            # Get tasks
            tasks = self.get_tasks()
            
            if not tasks:
                logger.info("No tasks assigned to this agent")
                return
            
            # Process each task
            for task in tasks:
                self.process_task(task)
                
        except Exception as e:
            logger.error(f"Error running agent: {str(e)}")
            raise

def main():
    """Main function to run the Sentinel-1 Data Extraction Agent"""
    parser = argparse.ArgumentParser(description='Run the Sentinel-1 Data Extraction Agent')
    parser.add_argument('--token', type=str, required=True, help='Agent token for authentication')
    parser.add_argument('--server-url', type=str, default=f"http://localhost:{SERVER_PORT}", help='MCP server URL')
    parser.add_argument('--agent-id', type=str, default="sentinel1-data-extraction-agent", help='Agent ID')
    
    # Task-specific arguments
    parser.add_argument('--shapefile', type=str, help='Path to shapefile for conversion')
    parser.add_argument('--output', type=str, help='Output path for converted GeoJSON')
    parser.add_argument('--input-dir', type=str, help='Input directory for batch conversion')
    parser.add_argument('--output-dir', type=str, help='Output directory for batch conversion or data extraction')
    parser.add_argument('--geojson', type=str, help='Path to GeoJSON file for data extraction')
    parser.add_argument('--start-date', type=str, help='Start date for data extraction (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date for data extraction (YYYY-MM-DD)')
    parser.add_argument('--bands', type=str, help='Comma-separated list of bands for extraction')
    
    args = parser.parse_args()
    
    try:
        # Initialize agent
        agent = Sentinel1DataExtractionAgent(args.token, args.server_url, args.agent_id)
        
        # Check for direct command execution
        if args.shapefile:
            # Convert shapefile
            agent.convert_shapefile(args.shapefile, args.output)
        elif args.input_dir:
            # Batch convert shapefiles
            agent.batch_convert_shapefiles(args.input_dir, args.output_dir)
        elif args.geojson and args.start_date and args.end_date:
            # Extract data
            bands = args.bands.split(",") if args.bands else None
            agent.extract_sentinel1_data(
                args.geojson, 
                args.output_dir or os.path.dirname(args.geojson), 
                args.start_date, 
                args.end_date, 
                bands
            )
        else:
            # Run agent to process tasks
            agent.run()
        
        print("\nAgent execution completed successfully!")
        
    except Exception as e:
        logger.error(f"Error executing agent: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
