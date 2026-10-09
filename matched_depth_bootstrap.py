#!/usr/bin/env python3
"""
Wrapper to execute unified matched-depth analysis.
Ensures matched_depth_bootstrap.py uses the exact same sweep grid, interpolation,
seed, and B=10,000 as run_updated_multi_depth.py.
"""
from scripts.recompute_matched_and_multi_depth import main

if __name__ == "__main__":
    main()
