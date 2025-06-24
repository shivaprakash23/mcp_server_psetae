#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sentinel-1 Inference Agent for PSETAE MCP Server
This agent handles model inference for Sentinel-1 PSETAE models.
It acts as a wrapper around the existing inference scripts in the PSETAE repository.
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
        logging.FileHandler(os.path.join(BASE_DIR, "logs", "sentinel1_inference_agent.log"), mode='a') 
        if os.path.exists(os.path.join(BASE_DIR, "logs")) else logging.StreamHandler()
    ]
)
logger = logging.getLogger("sentinel1_inference_agent")

class Sentinel1InferenceAgent:
    """Sentinel-1 Inference Agent for handling PSETAE model inference"""
    
    def __init__(self, token, server_url, agent_id):
        """Initialize the Sentinel-1 Inference Agent
        
        Args:
            token (str): Agent token for authentication
            server_url (str): URL of the MCP server
            agent_id (str): ID of this agent
        """
        self.token = token
        self.server_url = server_url
        self.agent_id = agent_id
        self.agent_type = "sentinel1-inference"
        
        # Validate token
        self.validate_token()
        
        logger.info(f"Sentinel-1 Inference Agent initialized with ID: {self.agent_id}")
    
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
    
    def run_inference(self, model_path, data_dir, output_dir, config_file=None):
        """Run inference using a trained Sentinel-1 PSETAE model
        
        Args:
            model_path (str): Path to the trained model
            data_dir (str): Directory containing data for inference
            output_dir (str): Directory to save inference outputs
            config_file (str): Path to model configuration file (optional)
            
        Returns:
            str: Path to the inference results
        """
        try:
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Use the existing Sentinel-1 inference script from the PSETAE repository
            inference_script = os.path.join(SENTINEL_DIR, "inference.py")
            
            # Build command with appropriate arguments for the existing script
            cmd = [
                sys.executable,
                inference_script,
                "--model_path", model_path,
                "--data_path", data_dir,
                "--output_dir", output_dir
            ]
            
            # Add config file if provided
            if config_file:
                cmd.extend(["--config_file", config_file])
            
            # Execute command
            logger.info(f"Executing model inference: {' '.join(cmd)}")
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=SENTINEL_DIR  # Set working directory to the script's directory
            )
            
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                logger.error(f"Model inference failed: {stderr}")
                raise RuntimeError(f"Model inference failed: {stderr}")
            
            logger.info(f"Model inference completed successfully")
            logger.debug(f"Inference output: {stdout}")
            
            # Add memory entry
            self.add_memory(
                f"Ran inference with model {model_path} on data in {data_dir}",
                {
                    "action": "model_inference",
                    "model_path": model_path,
                    "data_dir": data_dir,
                    "output_dir": output_dir,
                    "config_file": config_file,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            return output_dir
        except Exception as e:
            logger.error(f"Error running inference: {str(e)}")
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
            if "inference" in task.get("title", "").lower():
                # Inference task
                model_path = metadata.get("model_path")
                data_dir = metadata.get("data_dir")
                output_dir = metadata.get("output_dir")
                config_file = metadata.get("config_file")
                
                if not all([model_path, data_dir, output_dir]):
                    raise ValueError("Missing required parameters in task metadata")
                
                self.run_inference(model_path, data_dir, output_dir, config_file)
                
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
    """Main function to run the Sentinel-1 Inference Agent"""
    parser = argparse.ArgumentParser(description='Run the Sentinel-1 Inference Agent')
    parser.add_argument('--token', type=str, required=True, help='Agent token for authentication')
    parser.add_argument('--server-url', type=str, default=f"http://localhost:{SERVER_PORT}", help='MCP server URL')
    parser.add_argument('--agent-id', type=str, default="sentinel1-inference-agent", help='Agent ID')
    
    # Task-specific arguments
    parser.add_argument('--model-path', type=str, help='Path to the trained model')
    parser.add_argument('--data-dir', type=str, help='Directory containing data for inference')
    parser.add_argument('--output-dir', type=str, help='Directory to save inference outputs')
    parser.add_argument('--config-file', type=str, help='Path to model configuration file')
    
    args = parser.parse_args()
    
    try:
        # Initialize agent
        agent = Sentinel1InferenceAgent(args.token, args.server_url, args.agent_id)
        
        # Check for direct command execution
        if args.model_path and args.data_dir and args.output_dir:
            # Run inference
            agent.run_inference(
                args.model_path,
                args.data_dir,
                args.output_dir,
                args.config_file
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
