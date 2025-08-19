"""Main entry point for the RAG pipeline."""

import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent))

from src.api import run_server

if __name__ == "__main__":
    run_server()
