import sys
from pathlib import Path

# Make the repo root importable so tests can reach the training/ package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
