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
import re
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
        
        # Track processed tasks to avoid reprocessing
        self.processed_tasks = set()
        
        # Track processed task parameters to avoid reprocessing duplicate tasks
        self.processed_task_params = set()
        
        # Create a file to persist processed tasks across restarts
        self.processed_tasks_file = os.path.join(BASE_DIR, "logs", f"{self.agent_id}_processed_tasks.json")
        self._load_processed_tasks()
        
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
            
    def get_incomplete_tasks(self):
        """Get only incomplete tasks assigned to this agent
        
        Returns:
            list: List of incomplete tasks
        """
        try:
            all_tasks = self.get_tasks()
            incomplete_tasks = []
            
            for task in all_tasks:
                # Check if task status is not 'completed' or 'failed'
                status = task.get("status", "").lower()
                if status != "completed" and status != "failed":
                    incomplete_tasks.append(task)
                else:
                    logger.info(f"Skipping task {task.get('id')} with status '{status}'")
            
            logger.info(f"Found {len(incomplete_tasks)} incomplete tasks out of {len(all_tasks)} total tasks")
            return incomplete_tasks
        except Exception as e:
            logger.error(f"Error getting incomplete tasks: {str(e)}")
            # Return empty list on error to avoid processing tasks when we can't determine status
            return []
    
    def get_task_info(self, task_id):
        """Get information about a specific task
        
        Args:
            task_id (str): ID of the task to retrieve
            
        Returns:
            dict: Task information or None if not found
        """
        try:
            response = requests.get(
                f"{self.server_url}/api/tasks/{task_id}",
                params={"token": self.token}
            )
            
            if response.status_code != 200:
                logger.error(f"Failed to get task info: {response.text}")
                return None
            
            task_info = response.json()
            logger.info(f"Retrieved information for task {task_id}")
            
            return task_info
        except Exception as e:
            logger.error(f"Error getting task info: {str(e)}")
            return None
    
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
            
            # Let sentinel_extraction.py handle the creation of DATA and META subdirectories
            # through its prepare_output function
            
            # Use the sentinel_extraction.py script from tools/satellite_data_extraction_gee
            sentinel_script = os.path.join(BASE_DIR, "tools", "satellite_data_extraction_gee", "sentinel_extraction.py")
            
            # Default bands if not specified
            if bands is None:
                bands = ["VV", "VH"]
            
            # Build command based on sentinel_extraction.py parameters
            # Note: The script requires rpg_file and output_dir as positional arguments
            cmd = [
                sys.executable,
                sentinel_script,
                geojson_path,  # First positional argument is rpg_file (GeoJSON)
                output_dir,    # Second positional argument is output_dir
                "--col_id", "COPERNICUS/S1_GRD",  # Sentinel-1 collection
                "--start_date", start_date,
                "--end_date", end_date,
                "--speckle_filter", "temporal",  # Default speckle filter
                "--kernel_size", "5"  # Default kernel size
            ]
            
            # Log the command for debugging
            logger.info(f"Command: {' '.join(cmd)}")
            
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
        """Process a single data extraction task"""
        try:
            task_id = task.get("id")
            title = task.get("title", "")
            metadata = task.get("metadata", {})
            
            # Handle both string and dict metadata
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse metadata JSON for task {task_id}")
                    metadata = {}
            
            # Ensure task_type is set correctly
            if "task_type" not in metadata:
                if "extraction" in title.lower() or "data" in title.lower():
                    metadata["task_type"] = "data_extraction"
                    logger.info(f"Set task_type to 'data_extraction' based on title: {title}")
            
            logger.info(f"Processing task {task_id}: {task.get('title')}")
            logger.info(f"Task metadata: {metadata}")
            
            # Check if this is a sequential workflow task
            workflow_type = metadata.get("workflow_type")
            if workflow_type == "sequential_extraction":
                return self.process_sequential_extraction_task(task, metadata)
            else:
                # Legacy processing for backward compatibility
                return self.process_legacy_extraction_task(task, metadata)
                
        except Exception as e:
            logger.error(f"Error processing task {task_id}: {str(e)}")
            return False

    def process_sequential_extraction_task(self, task, metadata):
        """Process a sequential extraction task with proper isolation"""
        try:
            task_id = task.get("id")
            split = metadata.get("split", "unknown")
            processing_order = metadata.get("processing_order", 0)
            
            logger.info(f"Processing sequential extraction task for {split.upper()} split (order: {processing_order})")
            
            # Extract parameters directly from metadata (no modification needed)
            geojson_path = metadata.get("geojson_path")
            output_dir = metadata.get("output_dir")
            start_date = metadata.get("start_date")
            end_date = metadata.get("end_date")
            bands = metadata.get("bands", ["VV", "VH"])
            
            # Validate required parameters
            if not all([geojson_path, output_dir, start_date, end_date]):
                logger.error(f"Missing required parameters for task {task_id}")
                logger.error(f"geojson_path: {geojson_path}")
                logger.error(f"output_dir: {output_dir}")
                logger.error(f"start_date: {start_date}")
                logger.error(f"end_date: {end_date}")
                return False
            
            # Verify GeoJSON file exists
            if not os.path.exists(geojson_path):
                logger.error(f"GeoJSON file not found: {geojson_path}")
                return False
            
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            logger.info(f"Sequential extraction parameters:")
            logger.info(f"  Split: {split}")
            logger.info(f"  GeoJSON: {geojson_path}")
            logger.info(f"  Output: {output_dir}")
            logger.info(f"  Date range: {start_date} to {end_date}")
            logger.info(f"  Bands: {bands}")
            
            # Execute the extraction
            success = self.execute_extraction(geojson_path, output_dir, start_date, end_date, bands)
            
            if success:
                logger.info(f"Sequential extraction completed successfully for {split} split")
                logger.info(f"Results saved to: {output_dir}")
                
                # Update task status with metadata for parameter tracking
                self.update_task_status(task_id, "completed", metadata)
                return True
            else:
                logger.error(f"Sequential extraction failed for {split} split")
                self.update_task_status(task_id, "failed", f"Sequential extraction failed for {split} split")
                return False
                
        except Exception as e:
            logger.error(f"Error in sequential extraction task: {str(e)}")
            return False

    def process_legacy_extraction_task(self, task, metadata):
        """Process legacy extraction task (backward compatibility)"""
        try:
            task_id = task.get("id")
            
            # Remove any date-specific subdirectory from output_dir if present
            output_dir = metadata.get("output_dir", "")
            if output_dir:
                # Check if the output_dir ends with a date pattern like 20240901_20250331
                output_dir_parts = output_dir.split(os.sep)
                if len(output_dir_parts) > 0:
                    last_part = output_dir_parts[-1]
                    # Check if the last part matches a date pattern (YYYYMMDD_YYYYMMDD)
                    if re.match(r'\d{8}_\d{8}', last_part):
                        # Remove the date-specific subdirectory
                        output_dir = os.path.dirname(output_dir)
                        metadata["output_dir"] = output_dir
                        logger.info(f"Removed date-specific subdirectory. Using output_dir: {output_dir}")
                        
            # Ensure output directory contains dataset-specific directory (testdirectory, traindirectory, validationdirectory)
            output_dir = metadata.get("output_dir", "")
            if output_dir:
                # First, remove any date-specific subdirectory if present
                # This ensures we're working with the base sentinel1_data directory
                output_dir_parts = output_dir.split(os.sep)
                if len(output_dir_parts) >= 1 and re.match(r'\d{8}_\d{8}', output_dir_parts[-1]):
                    output_dir = os.sep.join(output_dir_parts[:-1])
                    logger.info(f"Removed date-specific subdirectory. Using output_dir: {output_dir}")
                
                # Now determine the dataset type from the GeoJSON filename
                geojson_path = metadata.get("geojson_path", "")
                
                # Always use GeoJSON files from /output/geojson
                if geojson_path:
                    # Check if the geojson_path is not from /output/geojson
                    if "output/geojson" not in geojson_path and "output\\geojson" not in geojson_path:
                        # Extract the filename and look for it in /output/geojson
                        geojson_filename = os.path.basename(geojson_path)
                        correct_geojson_path = os.path.join(BASE_DIR, "output", "geojson", geojson_filename)
                        
                        if os.path.exists(correct_geojson_path):
                            logger.info(f"Replacing GeoJSON path {geojson_path} with {correct_geojson_path}")
                            geojson_path = correct_geojson_path
                            metadata["geojson_path"] = geojson_path
                        else:
                            # Look for a matching GeoJSON file in /output/geojson
                            geojson_dir = os.path.join(BASE_DIR, "output", "geojson")
                            if os.path.exists(geojson_dir):
                                for file in os.listdir(geojson_dir):
                                    if file.endswith(".geojson"):
                                        if ("test" in file.lower() and "test" in geojson_filename.lower()) or \
                                           ("train" in file.lower() and "train" in geojson_filename.lower()) or \
                                           (("valid" in file.lower() or "val" in file.lower()) and \
                                            ("valid" in geojson_filename.lower() or "val" in geojson_filename.lower())):
                                            correct_geojson_path = os.path.join(geojson_dir, file)
                                            logger.info(f"Found matching GeoJSON file: {correct_geojson_path}")
                                            geojson_path = correct_geojson_path
                                            metadata["geojson_path"] = geojson_path
                                            break
                
                # Determine dataset type from GeoJSON filename
                if geojson_path:
                    geojson_filename = os.path.basename(geojson_path).lower()
                    
                    # Force dataset type based on GeoJSON filename
                    if "test" in geojson_filename:
                        dataset_type = "testdirectory"
                    elif "train" in geojson_filename:
                        dataset_type = "traindirectory"
                    elif "valid" in geojson_filename or "val" in geojson_filename:
                        dataset_type = "validationdirectory"
                    else:
                        # Default to test if can't determine
                        dataset_type = "testdirectory"
                        logger.info(f"Could not determine dataset type from filename '{geojson_filename}'. Using '{dataset_type}' as default.")
                    
                    # Set output directory to the correct dataset directory without nesting
                    sentinel1_data_dir = os.path.join(BASE_DIR, "output", "sentinel1_data")
                    output_dir = os.path.join(sentinel1_data_dir, dataset_type)
                    metadata["output_dir"] = output_dir
                    os.makedirs(output_dir, exist_ok=True)
                    logger.info(f"Using dataset-specific directory: {output_dir}")

            
            # Extract parameters from metadata
            geojson_path = metadata.get("geojson_path")
            output_dir = metadata.get("output_dir")
            start_date = metadata.get("start_date")
            end_date = metadata.get("end_date")
            bands = metadata.get("bands", ["VV", "VH"])
            
            # Validate required parameters
            if not all([geojson_path, output_dir, start_date, end_date]):
                logger.error(f"Missing required parameters for task {task_id}")
                return False
            
            # Execute the extraction
            success = self.execute_extraction(geojson_path, output_dir, start_date, end_date, bands)
            
            if success:
                logger.info(f"Legacy extraction completed successfully")
                self.update_task_status(task_id, "completed", "Data extraction completed successfully")
                return True
            else:
                logger.error(f"Legacy extraction failed")
                self.update_task_status(task_id, "failed", "Data extraction failed")
                return False
                
        except Exception as e:
            logger.error(f"Error in legacy extraction task: {str(e)}")
            return False

    def execute_extraction(self, geojson_path, output_dir, start_date, end_date, bands):
        """Execute the actual data extraction"""
        try:
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Let sentinel_extraction.py handle the creation of DATA and META subdirectories
            # through its prepare_output function
            
            # Use the sentinel_extraction.py script from tools/satellite_data_extraction_gee
            sentinel_script = os.path.join(BASE_DIR, "tools", "satellite_data_extraction_gee", "sentinel_extraction.py")
            
            # Default bands if not specified
            if bands is None:
                bands = ["VV", "VH"]
            
            # Build command based on sentinel_extraction.py parameters
            # Note: The script requires rpg_file and output_dir as positional arguments
            cmd = [
                sys.executable,
                sentinel_script,
                geojson_path,  # First positional argument is rpg_file (GeoJSON)
                output_dir,    # Second positional argument is output_dir
                "--col_id", "COPERNICUS/S1_GRD",  # Sentinel-1 collection
                "--start_date", start_date,
                "--end_date", end_date,
                "--speckle_filter", "temporal",  # Default speckle filter
                "--kernel_size", "5"  # Default kernel size
            ]
            
            # Log the command for debugging
            logger.info(f"Command: {' '.join(cmd)}")
            
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
            
            return True
        except Exception as e:
            logger.error(f"Error executing extraction: {str(e)}")
            return False
    
    def update_task_status(self, task_id, status, message=None, max_retries=3):
        """Update the status of a task with retry logic
        
        Args:
            task_id (str): ID of the task to update
            status (str): New status for the task (e.g., 'completed', 'failed')
            message (str): Optional status message
            max_retries (int): Maximum number of retry attempts
            
        Returns:
            dict: Updated task information or None if all retries failed
        """
        retries = 0
        while retries <= max_retries:
            try:
                payload = {
                    "status": status
                }
                
                if message:
                    payload["status_message"] = message
                
                logger.info(f"Updating task {task_id} status to '{status}' (attempt {retries+1}/{max_retries+1})")
                response = requests.put(
                    f"{self.server_url}/api/tasks/{task_id}/status",
                    params={"token": self.token},
                    json=payload
                )
                
                if response.status_code == 200:
                    task_info = response.json()
                    logger.info(f"Successfully updated task {task_id} status to '{status}'")
                    
                    # Add to processed tasks tracking
                    self._add_to_processed_tasks(task_id, status, message)
                    
                    return task_info
                elif response.status_code == 404 and "Not Found" in response.text:
                    # Special handling for 'Not Found' errors
                    logger.warning(f"Task {task_id} not found on server. Marking as processed locally.")
                    self._add_to_processed_tasks(task_id, status, message)
                    return None
                else:
                    logger.error(f"Failed to update task status: {response.text}")
                    
                    # Only retry on server errors (5xx) or specific 4xx errors
                    if response.status_code >= 500 or response.status_code in [429, 408]:
                        retries += 1
                        if retries <= max_retries:
                            logger.info(f"Retrying update for task {task_id} in 2 seconds...")
                            import time
                            time.sleep(2)  # Wait before retry
                            continue
                    
                    # For other errors, mark locally and return
                    logger.warning(f"Could not update task status on server. Marking as processed locally.")
                    self._add_to_processed_tasks(task_id, status, message)
                    return None
                    
            except Exception as e:
                logger.error(f"Error updating task status: {str(e)}")
                retries += 1
                if retries <= max_retries:
                    logger.info(f"Retrying update for task {task_id} in 2 seconds...")
                    import time
                    time.sleep(2)  # Wait before retry
                else:
                    # After all retries, mark locally and return
                    logger.warning(f"Failed all retries to update task status. Marking as processed locally.")
                    self._add_to_processed_tasks(task_id, status, message)
                    return None
        
        return None
    
    def _load_processed_tasks(self):
        """Load the set of processed tasks from disk"""
        try:
            if os.path.exists(self.processed_tasks_file):
                with open(self.processed_tasks_file, 'r') as f:
                    data = json.load(f)
                    self.processed_tasks = set(data.get('processed_task_ids', []))
                    self.processed_task_params = set(tuple(item) for item in data.get('processed_task_params', []))
                    logger.info(f"Loaded {len(self.processed_tasks)} previously processed tasks and {len(self.processed_task_params)} task parameter sets")
            else:
                logger.info("No processed tasks file found, starting with empty set")
                self.processed_tasks = set()
                self.processed_task_params = set()
        except Exception as e:
            logger.error(f"Error loading processed tasks: {str(e)}")
            self.processed_tasks = set()
            self.processed_task_params = set()
    
    def _save_processed_tasks(self):
        """Save the set of processed tasks to disk"""
        try:
            os.makedirs(os.path.dirname(self.processed_tasks_file), exist_ok=True)
            with open(self.processed_tasks_file, 'w') as f:
                json.dump({
                    'processed_task_ids': list(self.processed_tasks),
                    'processed_task_params': [list(params) for params in self.processed_task_params],
                    'last_updated': datetime.now().isoformat()
                }, f)
            logger.info(f"Saved {len(self.processed_tasks)} processed tasks and {len(self.processed_task_params)} task parameter sets to disk")
        except Exception as e:
            logger.error(f"Error saving processed tasks: {str(e)}")
    
    def _add_to_processed_tasks(self, task_id, status, metadata=None):
        """Add a task to the processed tasks set
        
        Args:
            task_id (str): ID of the task
            status (str): Status of the task (e.g., 'completed', 'failed')
            metadata (dict): Task metadata for parameter tracking
        """
        self.processed_tasks.add(task_id)
        
        # Also track task parameters to detect duplicates
        if metadata and isinstance(metadata, dict):
            # Create a parameter fingerprint for this task
            param_keys = ['geojson_path', 'output_dir', 'start_date', 'end_date', 'split']
            param_values = []
            
            for key in param_keys:
                if key in metadata:
                    # For paths, just use the basename to handle different path formats
                    if key in ['geojson_path', 'output_dir'] and metadata[key]:
                        param_values.append(os.path.basename(metadata[key]))
                    else:
                        param_values.append(str(metadata[key]))
                else:
                    param_values.append(None)
            
            # Add the parameter fingerprint to the set
            param_fingerprint = tuple(param_values)
            self.processed_task_params.add(param_fingerprint)
            logger.info(f"Added parameter fingerprint {param_fingerprint} to processed task parameters")
        
        logger.info(f"Added task {task_id} to processed tasks (status: {status})")
        self._save_processed_tasks()
    
    def _is_task_processed(self, task_id, metadata=None):
        """Check if a task has already been processed
        
        Args:
            task_id (str): ID of the task
            metadata (dict): Task metadata for parameter-based duplicate detection
            
        Returns:
            bool: True if the task has been processed, False otherwise
        """
        # First check by ID
        if task_id in self.processed_tasks:
            return True
            
        # Then check by parameters to catch duplicates
        if metadata and isinstance(metadata, dict):
            # Create a parameter fingerprint for this task
            param_keys = ['geojson_path', 'output_dir', 'start_date', 'end_date', 'split']
            param_values = []
            
            for key in param_keys:
                if key in metadata:
                    # For paths, just use the basename to handle different path formats
                    if key in ['geojson_path', 'output_dir'] and metadata[key]:
                        param_values.append(os.path.basename(metadata[key]))
                    else:
                        param_values.append(str(metadata[key]))
                else:
                    param_values.append(None)
            
            # Check if this parameter fingerprint exists
            param_fingerprint = tuple(param_values)
            if param_fingerprint in self.processed_task_params:
                logger.info(f"Task {task_id} has identical parameters to a previously processed task")
                return True
                
        return False
            
    def run(self):
        """Run the agent to process tasks"""
        try:
            # Get only incomplete tasks
            tasks = self.get_incomplete_tasks()
            
            if not tasks:
                logger.info("No incomplete tasks assigned to this agent")
                return
            
            # Sort tasks by processing_order if available
            tasks.sort(key=lambda t: int(t.get('metadata', {}).get('processing_order', 999)) 
                      if isinstance(t.get('metadata'), dict) else 999)
            
            # Process each task, skipping already processed ones
            for task in tasks:
                task_id = task.get('id')
                
                # Get task metadata for parameter-based duplicate detection
                metadata = task.get('metadata', {})
                
                if self._is_task_processed(task_id, metadata):
                    logger.info(f"Skipping already processed task {task_id}")
                    continue
                    
                logger.info(f"Processing task {task_id} (not in processed tasks list)")
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
