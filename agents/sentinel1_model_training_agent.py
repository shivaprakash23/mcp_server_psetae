#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sentinel-1 Model Training Agent for PSETAE MCP Server
This agent handles model training for Sentinel-1 PSETAE models.
It acts as a wrapper around the existing model training scripts in the PSETAE repository.
"""

import os
import sys
import json
import time
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
        logging.FileHandler(os.path.join(BASE_DIR, "logs", "sentinel1_model_training_agent.log"), mode='a') 
        if os.path.exists(os.path.join(BASE_DIR, "logs")) else logging.StreamHandler()
    ]
)
logger = logging.getLogger("sentinel1_model_training_agent")

class Sentinel1ModelTrainingAgent:
    """Sentinel-1 Model Training Agent for handling PSETAE model training"""
    
    def __init__(self, token, server_url, agent_id):
        """Initialize the Sentinel-1 Model Training Agent
        
        Args:
            token (str): Agent token for authentication
            server_url (str): URL of the MCP server
            agent_id (str): ID of this agent
        """
        self.token = token
        self.server_url = server_url
        self.agent_id = agent_id
        self.agent_type = "sentinel1-model-training"
        
        # Validate token
        self.validate_token()
        
        logger.info(f"Sentinel-1 Model Training Agent initialized with ID: {self.agent_id}")
    
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
            
    def update_task_status(self, task_id, status, max_retries=4):
        """Update the status of a task
        
        Args:
            task_id (str): ID of the task to update
            status (str): New status ('in_progress', 'completed', 'failed')
            max_retries (int): Maximum number of retry attempts
            
        Returns:
            bool: True if update was successful
        """
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"Updating task {task_id} status to '{status}' (attempt {attempt}/{max_retries})")
                
                # Use PUT method for task status updates
                response = requests.put(
                    f"{self.server_url}/api/tasks/{task_id}",
                    params={"token": self.token},
                    json={"status": status}
                )
                
                if response.status_code == 200:
                    logger.info(f"Successfully updated task {task_id} status to '{status}'")
                    return True
                else:
                    logger.warning(f"Failed to update task status: {response.text}")
                    
            except Exception as e:
                logger.error(f"Error updating task status: {str(e)}")
                
            # Wait before retrying (exponential backoff)
            if attempt < max_retries:
                time.sleep(2 ** attempt)
                
        logger.error(f"Failed to update task {task_id} status after {max_retries} attempts")
        return False
    
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
    
    def train_model(self, train_dir, val_dir, test_dir, output_dir, epochs=100, batch_size=32, 
                  learning_rate=0.0001, sensor='S1', input_dim=2, mlp1=None, num_classes=10):
        """Train a Sentinel-1 PSETAE model using the existing training script
        
        Args:
            train_dir (str): Directory containing training data
            val_dir (str): Directory containing validation data
            test_dir (str): Directory containing test data
            output_dir (str): Directory to save model outputs
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            learning_rate (float): Learning rate for training
            sensor (str): Sensor type (S1 or S2)
            input_dim (int): Number of input dimensions (2 for S1: VV, VH)
            mlp1 (list): MLP1 architecture (e.g. [2,32,64])
            num_classes (int): Number of output classes
            
        Returns:
            str: Path to the trained model
        """
        try:
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Use the existing PSETAE training script
            train_script = os.path.join(BASE_DIR, "tools", "psetae_model", "single_sensor", "train.py")
            
            # Default MLP1 architecture if not provided
            if mlp1 is None:
                mlp1 = [2, 32, 64]
            
            # Format MLP1 as string
            mlp1_str = f"[{','.join(map(str, mlp1))}]"
            
            # Build command with appropriate arguments for the PSETAE script
            cmd = [
                sys.executable,
                train_script,
                "--dataset_folder", train_dir,
                "--val_folder", val_dir,
                "--test_folder", test_dir,
                "--res_dir", output_dir,
                "--epochs", str(epochs),
                "--batch_size", str(batch_size),
                "--lr", str(learning_rate),
                "--sensor", sensor,
                "--input_dim", str(input_dim),
                "--mlp1", mlp1_str,
                "--num_classes", str(num_classes),
                "--geomfeat", "0"  # Disable geometric features to avoid pandas error
            ]
            
            # Execute command
            logger.info(f"Executing model training: {' '.join(cmd)}")
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=SENTINEL_DIR  # Set working directory to the script's directory
            )
            
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                logger.error(f"Model training failed: {stderr}")
                raise RuntimeError(f"Model training failed: {stderr}")
            
            logger.info(f"Model training completed successfully")
            logger.debug(f"Training output: {stdout}")
            
            # Find the model file (usually the last checkpoint in the models subdirectory)
            model_dir = os.path.join(output_dir, "models")
            if os.path.exists(model_dir):
                model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
                if model_files:
                    model_path = os.path.join(model_dir, sorted(model_files)[-1])
                else:
                    # Look in the output directory itself
                    model_files = [f for f in os.listdir(output_dir) if f.endswith('.pth')]
                    if not model_files:
                        raise FileNotFoundError(f"No model files found in {output_dir} or {model_dir}")
                    model_path = os.path.join(output_dir, sorted(model_files)[-1])
            else:
                # Look in the output directory itself
                model_files = [f for f in os.listdir(output_dir) if f.endswith('.pth')]
                if not model_files:
                    raise FileNotFoundError(f"No model files found in {output_dir}")
                model_path = os.path.join(output_dir, sorted(model_files)[-1])
            
            # Add memory entry
            self.add_memory(
                f"Trained Sentinel-1 PSETAE model saved to {model_path}",
                {
                    "action": "model_training",
                    "data_dir": data_dir,
                    "output_dir": output_dir,
                    "model_path": model_path,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "config_file": config_file,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            return model_path
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
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
            
            # Check task type - recognize both 'model training' and 'Model Training' in the title
            if "model training" in task.get("title", "").lower() or "Model Training" in task.get("title", ""):
                # Model training task
                train_dir = metadata.get("train_dir")
                val_dir = metadata.get("val_dir")
                test_dir = metadata.get("test_dir")
                output_dir = metadata.get("output_dir")
                epochs = metadata.get("epochs", 100)
                batch_size = metadata.get("batch_size", 32)
                learning_rate = metadata.get("learning_rate", 0.0001)
                sensor = metadata.get("sensor", "S1")
                input_dim = metadata.get("input_dim", 2)
                mlp1 = metadata.get("mlp1", [2, 32, 64])
                num_classes = metadata.get("num_classes", 10)
                
                if not all([train_dir, val_dir, test_dir, output_dir]):
                    raise ValueError("Missing required parameters in task metadata")
                
                # Update task status to 'in_progress'
                self.update_task_status(task_id, "in_progress")
                
                self.train_model(
                    train_dir, 
                    val_dir,
                    test_dir,
                    output_dir, 
                    epochs, 
                    batch_size, 
                    learning_rate,
                    sensor,
                    input_dim,
                    mlp1,
                    num_classes
                )
                
                # Update task status to 'completed'
                self.update_task_status(task_id, "completed")
                
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
    """Main function to run the Sentinel-1 Model Training Agent"""
    parser = argparse.ArgumentParser(description='Run the Sentinel-1 Model Training Agent')
    parser.add_argument('--token', type=str, required=True, help='Agent token for authentication')
    parser.add_argument('--server-url', type=str, default=f"http://localhost:{SERVER_PORT}", help='MCP server URL')
    parser.add_argument('--agent-id', type=str, default="sentinel1-model-training-agent", help='Agent ID')
    
    # Task-specific arguments
    parser.add_argument('--train-dir', type=str, help='Directory containing training data')
    parser.add_argument('--val-dir', type=str, help='Directory containing validation data')
    parser.add_argument('--test-dir', type=str, help='Directory containing test data')
    parser.add_argument('--output-dir', type=str, help='Directory to save model outputs')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.0001, help='Learning rate for training')
    parser.add_argument('--sensor', type=str, default='S1', help='Sensor type (S1 or S2)')
    parser.add_argument('--input-dim', type=int, default=2, help='Number of input dimensions')
    parser.add_argument('--num-classes', type=int, default=10, help='Number of output classes')
    
    args = parser.parse_args()
    
    try:
        # Initialize agent
        agent = Sentinel1ModelTrainingAgent(args.token, args.server_url, args.agent_id)
        
        # Check for direct command execution
        if args.train_dir and args.val_dir and args.test_dir and args.output_dir:
            # Train model
            agent.train_model(
                args.train_dir,
                args.val_dir,
                args.test_dir,
                args.output_dir,
                args.epochs,
                args.batch_size,
                args.learning_rate,
                args.sensor,
                args.input_dim,
                None,  # Use default MLP1
                args.num_classes
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
