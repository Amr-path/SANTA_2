#!/usr/bin/env python3
"""
run_kaggle.py - Main entry point for Santa 2025 Solver
Usage: python run_kaggle.py [--mode quick|full] [--seed 42]
"""
import sys
import os
import time
import argparse

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from src.geometry import compute_bounding_square_side
from src.packing import PackingSolver, validate_and_fix_solution
from src.optimize import OptimizationConfig
from src.validate import validate_all_solutions, compute_score
from src.io_utils import find_data_path, create_submission, print_solution_summary, get_output_path, validate_submission_format

MODE = "quick"
RANDOM_SEED = 42
MAX_N = 200

def get_config(mode: str) -> OptimizationConfig:
    return OptimizationConfig.quick_mode() if mode == "quick" else OptimizationConfig.full_mode()

def main(mode: str = MODE, seed: int = RANDOM_SEED, max_n: int = MAX_N):
    print("=" * 70)
    print("SANTA 2025 CHRISTMAS TREE PACKING SOLVER")
    print("=" * 70)
    print(f"Mode: {mode}, Seed: {seed}, Max N: {max_n}\n")
    
    try:
        data_path = find_data_path()
        print(f"✓ Data path: {data_path}")
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        return None
    
    start_time = time.time()
    config = get_config(mode)
    config.seed = seed
    
    print(f"\nSolving n=1 to {max_n}...")
    print("-" * 40)
    
    solver = PackingSolver(config=config, seed=seed)
    solutions = solver.solve_all(max_n=max_n, verbose=True)
    
    print("-" * 40)
    print(f"Solving completed in {time.time() - start_time:.1f} seconds")
    
    print("\nValidating solutions...")
    validation = validate_all_solutions(solutions, max_n=max_n, verbose=True)
    
    if not validation.valid:
        print("\n⚠ Attempting to fix invalid solutions...")
        for n in validation.invalid_ns:
            if n in solutions:
                fixed, _ = validate_and_fix_solution(solutions[n])
                solutions[n] = fixed
        validate_all_solutions(solutions, max_n=max_n, verbose=True)
    
    print()
    print_solution_summary(solutions, max_n=max_n)
    
    print("\nCreating submission file...")
    output_path = get_output_path("submission.csv")
    
    try:
        created_path = create_submission(solutions, output_path=output_path)
        print(f"✓ Submission saved to: {created_path}")
        
        is_valid, error = validate_submission_format(created_path)
        if is_valid:
            print("✓ Submission format validated")
        else:
            print(f"⚠ Format issue: {error}")
    except Exception as e:
        print(f"✗ Error creating submission: {e}")
        return None
    
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Estimated score: {compute_score(solutions, max_n):.2f}")
    print(f"Output: {created_path}")
    
    return solutions

def parse_args():
    parser = argparse.ArgumentParser(description="Santa 2025 Solver")
    parser.add_argument("--mode", choices=["quick", "full"], default=MODE)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--max-n", type=int, default=MAX_N)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(mode=args.mode, seed=args.seed, max_n=args.max_n)
