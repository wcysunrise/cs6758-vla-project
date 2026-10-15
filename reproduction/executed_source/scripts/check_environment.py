#!/usr/bin/env python3
"""Check the target environment using the same preflight as evaluation."""
import sys
from run_baseline import main

if __name__ == "__main__":
    raise SystemExit(main(["--preflight", *sys.argv[1:]]))
