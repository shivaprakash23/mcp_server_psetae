#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sentinel-1 Tile Coverage Agent for PSETAE MCP Server
This agent handles satellite tile coverage analysis for study areas using Sentinel-1 data.
It acts as a wrapper around the existing satellite_tiles_information_extraction scripts in the PSETAE repository.
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
from server.config import *

# Initialize logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(BASE_DIR, "logs", "sentinel1_tile_coverage_agent.log"), mode='a') 
        if os.path.exists(os.path.join(BASE_DIR, "logs")) else logging.StreamHandler()
    ]
)
logger = logging.getLogger("sentinel1_tile_coverage_agent")

class Sentinel1TileCoverageAgent:
    """Sentinel-1 Tile Coverage Agent for analyzing satellite tile coverage for study areas"""
    
    def __init__(self, token, server_url, agent_id):
        """Initialize the Sentinel-1 Tile Coverage Agent
        
        Args:
            token (str): Agent token for authentication
            server_url (str): URL of the MCP server
            agent_id (str): ID of this agent
        """
        self.token = token
        self.server_url = server_url
        self.agent_id = agent_id
        self.agent_type = "sentinel1-tile-coverage"
        
        # Validate token
        self.validate_token()
        
        logger.info(f"Sentinel-1 Tile Coverage Agent initialized with ID: {self.agent_id}")
    
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
    
    def analyze_tile_coverage(self, geojson_path, output_dir, start_date, end_date, collection=SENTINEL1_COLLECTION):
        """Analyze satellite tile coverage for a study area
        
        Args:
            geojson_path (str): Path to the GeoJSON file defining the study area
            output_dir (str): Directory to save coverage analysis outputs
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            collection (str): Satellite collection name (default: Sentinel-1)
            
        Returns:
            str: Path to the coverage analysis results
        """
        try:
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Use the existing tile coverage analysis script from the PSETAE repository
            coverage_script = os.path.join(TILES_INFO_DIR, "sentinel_tile_coverage.py")
            
            # Build command
            cmd = [
                sys.executable,
                coverage_script,
                "--geojson", geojson_path,
                "--output", output_dir,
                "--start_date", start_date,
                "--end_date", end_date,
                "--collection", collection
            ]
            
            # Execute command
            logger.info(f"Executing tile coverage analysis: {' '.join(cmd)}")
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=TILES_INFO_DIR  # Set working directory to the script's directory
            )
            
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                logger.error(f"Tile coverage analysis failed: {stderr}")
                raise RuntimeError(f"Tile coverage analysis failed: {stderr}")
            
            logger.info(f"Tile coverage analysis completed successfully")
            logger.debug(f"Analysis output: {stdout}")
            
            # Add memory entry
            self.add_memory(
                f"Analyzed Sentinel-1 tile coverage for area in {geojson_path} from {start_date} to {end_date}",
                {
                    "action": "tile_coverage_analysis",
                    "geojson_path": geojson_path,
                    "output_dir": output_dir,
                    "start_date": start_date,
                    "end_date": end_date,
                    "collection": collection,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            return output_dir
        except Exception as e:
            logger.error(f"Error analyzing tile coverage: {str(e)}")
            raise
    
    def generate_coverage_report(self, coverage_dir, output_file):
        """Generate a coverage report from tile coverage analysis results
        
        Args:
            coverage_dir (str): Directory containing coverage analysis results
            output_file (str): Path to save the coverage report
            
        Returns:
            str: Path to the coverage report
        """
        try:
            # Use the existing report generation script from the PSETAE repository
            report_script = os.path.join(TILES_INFO_DIR, "generate_coverage_report.py")
            
            # Build command
            cmd = [
                sys.executable,
                report_script,
                "--input", coverage_dir,
                "--output", output_file
            ]
            
            # Execute command
            logger.info(f"Generating coverage report: {' '.join(cmd)}")
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=TILES_INFO_DIR  # Set working directory to the script's directory
            )
            
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                logger.error(f"Coverage report generation failed: {stderr}")
                raise RuntimeError(f"Coverage report generation failed: {stderr}")
            
            logger.info(f"Coverage report generated successfully at {output_file}")
            
            # Add memory entry
            self.add_memory(
                f"Generated coverage report at {output_file} from analysis in {coverage_dir}",
                {
                    "action": "coverage_report_generation",
                    "coverage_dir": coverage_dir,
                    "output_file": output_file,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            return output_file
        except Exception as e:
            logger.error(f"Error generating coverage report: {str(e)}")
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
            if "tile_coverage" in task.get("title", "").lower() or "coverage_analysis" in task.get("title", "").lower():
                # Tile coverage analysis task
                geojson_path = metadata.get("geojson_path")
                output_dir = metadata.get("output_dir")
                start_date = metadata.get("start_date")
                end_date = metadata.get("end_date")
                collection = metadata.get("collection", SENTINEL1_COLLECTION)
                
                if not all([geojson_path, output_dir, start_date, end_date]):
                    raise ValueError("Missing required parameters in task metadata")
                
                self.analyze_tile_coverage(geojson_path, output_dir, start_date, end_date, collection)
                
            elif "coverage_report" in task.get("title", "").lower():
                # Coverage report generation task
                coverage_dir = metadata.get("coverage_dir")
                output_file = metadata.get("output_file")
                
                if not all([coverage_dir, output_file]):
                    raise ValueError("Missing required parameters in task metadata")
                
                self.generate_coverage_report(coverage_dir, output_file)
                
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
    """Main function to run the Sentinel-1 Tile Coverage Agent"""
    parser = argparse.ArgumentParser(description='Run the Sentinel-1 Tile Coverage Agent')
    parser.add_argument('--token', type=str, required=True, help='Agent token for authentication')
    parser.add_argument('--server-url', type=str, default=f"http://localhost:{SERVER_PORT}", help='MCP server URL')
    parser.add_argument('--agent-id', type=str, default="sentinel1-tile-coverage-agent", help='Agent ID')
    
    # Task-specific arguments
    parser.add_argument('--geojson', type=str, help='Path to GeoJSON file defining the study area')
    parser.add_argument('--output-dir', type=str, help='Directory to save coverage analysis outputs')
    parser.add_argument('--start-date', type=str, help='Start date for coverage analysis (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date for coverage analysis (YYYY-MM-DD)')
    parser.add_argument('--collection', type=str, default=SENTINEL1_COLLECTION, help='Satellite collection name')
    parser.add_argument('--coverage-dir', type=str, help='Directory containing coverage analysis results')
    parser.add_argument('--output-file', type=str, help='Path to save the coverage report')
    
    args = parser.parse_args()
    
    try:
        # Initialize agent
        agent = Sentinel1TileCoverageAgent(args.token, args.server_url, args.agent_id)
        
        # Check for direct command execution
        if args.geojson and args.output_dir and args.start_date and args.end_date:
            # Analyze tile coverage
            agent.analyze_tile_coverage(
                args.geojson,
                args.output_dir,
                args.start_date,
                args.end_date,
                args.collection
            )
        elif args.coverage_dir and args.output_file:
            # Generate coverage report
            agent.generate_coverage_report(
                args.coverage_dir,
                args.output_file
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
