"""
Secure configuration file for Bikolpo Backend
Loads all secrets from environment variables.
NEVER store real credentials inside this file.
"""

import os

# -------------------------------
# 🔐 API KEYS (Loaded from .env / Railway Variables)
# -------------------------------
NASA_USERNAME = os.getenv("NASA_USERNAME")
NASA_PASSWORD = os.getenv("NASA_PASSWORD")

# Path to your GEE JSON (or raw JSON content)
GEE_SERVICE_ACCOUNT_JSON = os.getenv("GEE_SERVICE_ACCOUNT_JSON")

# OpenRouteService API Key
ORS_API_KEY = os.getenv("ORS_API_KEY")

# -------------------------------
# 📁 DATA STORAGE
# -------------------------------
DATA_DIR = "data"
RAW_DATA_DIR = f"{DATA_DIR}/raw"
PROCESSED_DATA_DIR = f"{DATA_DIR}/processed"
MODEL_DIR = "models"

# -------------------------------
# ⚙️ MODEL PARAMETERS
# -------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2

# -------------------------------
# 🛣️ ROUTING WEIGHTS
# -------------------------------
LAMBDA_SAFEST = 10.0   # High weight on flood risk
LAMBDA_BALANCED = 3.0  # Balanced
LAMBDA_FASTEST = 0.5   # Mostly speed-based
