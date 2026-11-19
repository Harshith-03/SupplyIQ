"""CLI wrapper: see supplyiq/data/agent_benchmark.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from supplyiq.data.agent_benchmark import main

if __name__ == "__main__":
    main()
