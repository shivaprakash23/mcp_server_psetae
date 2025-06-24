#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MCP Server Configuration
This module contains configuration settings for the MCP server.
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Server settings
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8080

# Database settings
DB_NAME = ".agent/mcp_state.db"
DB_PATH = os.path.join(BASE_DIR, DB_NAME)

# Agent settings
ADMIN_AGENT_ID = "admin-agent"
AGENT_TYPES = {
    "data-extraction": "DataExtractionAgent",
    "model-training": "ModelTrainingAgent",
    "inference": "InferenceAgent",
    "tile-coverage": "TileCoverageAgent"
}

# PSETAE paths
PSETAE_BASE_DIR = os.path.join(BASE_DIR.parent, "psetae_github_publish")
DATA_EXTRACTION_DIR = os.path.join(PSETAE_BASE_DIR, "data_extraction")
HLS_L30_DIR = os.path.join(PSETAE_BASE_DIR, "hls_l30_psetae")
HLS_S30_DIR = os.path.join(PSETAE_BASE_DIR, "hls_s30_psetae")
SENTINEL_DIR = os.path.join(PSETAE_BASE_DIR, "sentinel_psetae")
TILES_INFO_DIR = os.path.join(PSETAE_BASE_DIR, "satellite_tiles_information_extraction")

# Logging settings
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = os.path.join(BASE_DIR, "logs", "mcp_server.log")

# Token settings
TOKEN_EXPIRY_DAYS = 30

# Memory settings
MAX_MEMORY_ENTRIES = 1000
MEMORY_CHUNK_SIZE = 1000

# API rate limits
RATE_LIMIT_REQUESTS = 100
RATE_LIMIT_PERIOD_SECONDS = 60
