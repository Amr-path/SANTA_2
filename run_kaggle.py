#!/usr/bin/env python3
"""
run_kaggle.py - Main entry point for Santa 2025 Solver (Advanced Edition)

This solver uses state-of-the-art optimization techniques:
- Simulated Annealing with adaptive cooling and reheating
- Gradient-based optimization (L-BFGS-B)
- Differential Evolution for global optimization
- Basin Hopping for escaping local minima
- Multi-resolution search strategies
- Optimal rotation discovery through angular sweeps
- Advanced interlocking placement patterns

Usage: python run_kaggle.py [--mode quick|full|ultra] [--seed 42]
"""
import sys
import os
import time
import argparse

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from src.geometry import compute_bounding_square_side
from src.packing import AdvancedPackingSolver, validate_and_fix_solution
from src.advanced_optimize import AdvancedConfig
from src.validate import validate_all_solutions, compute_score
from src.io_utils import (
    find_data_path, create_submission, print_solution_summary,
    get_output_path, validate_submission_format
)

MODE = "full"
RANDOM_SEED = 42
MAX_N = 200


def get_config(mode: str) -> AdvancedConfig:
    """Get configuration based on mode."""
    if mode == "quick":
        return AdvancedConfig.fast_mode()
    elif mode == "ultra":
        return AdvancedConfig.ultra_mode()
    else:  # full
        return AdvancedConfig()


def main(mode: str = MODE, seed: int = RANDOM_SEED, max_n: int = MAX_N):
    print("=" * 70)
    print("SANTA 2025 CHRISTMAS TREE PACKING SOLVER - ADVANCED EDITION")
    print("=" * 70)
    print(f"Mode: {mode}, Seed: {seed}, Max N: {max_n}")
    print()
    print("Optimization techniques enabled:")
    print("  - Simulated Annealing with adaptive cooling")
    print("  - L-BFGS-B gradient optimization")
    print("  - Differential Evolution (for n <= 30)")
    print("  - Basin Hopping (for n <= 50)")
    print("  - Multi-resolution search")
    print("  - Angular sweep rotation optimization")
    print("  - Advanced interlocking patterns")
    print()

    try:
        data_path = find_data_path()
        print(f"Data path: {data_path}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None

    start_time = time.time()
    config = get_config(mode)
    config.seed = seed

    print(f"\nSolving n=1 to {max_n} with advanced optimization...")
    print("-" * 40)

    solver = AdvancedPackingSolver(config=config, seed=seed)
    solutions = solver.solve_all(max_n=max_n, verbose=True)

    print("-" * 40)
    print(f"Solving completed in {time.time() - start_time:.1f} seconds")

    print("\nValidating solutions...")
    validation = validate_all_solutions(solutions, max_n=max_n, verbose=True)

    if not validation.valid:
        print("\nAttempting to fix invalid solutions...")
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
        print(f"Submission saved to: {created_path}")

        is_valid, error = validate_submission_format(created_path)
        if is_valid:
            print("Submission format validated")
        else:
            print(f"Format issue: {error}")
    except Exception as e:
        print(f"Error creating submission: {e}")
        return None

    total_time = time.time() - start_time
    final_score = compute_score(solutions, max_n)

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"FINAL SCORE: {final_score:.2f}")
    print(f"Output: {created_path}")

    return solutions


def parse_args():
    parser = argparse.ArgumentParser(description="Santa 2025 Advanced Solver")
    parser.add_argument("--mode", choices=["quick", "full", "ultra"], default=MODE,
                        help="Optimization mode: quick (fast), full (balanced), ultra (best)")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED,
                        help="Random seed for reproducibility")
    parser.add_argument("--max-n", type=int, default=MAX_N,
                        help="Maximum n to solve")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(mode=args.mode, seed=args.seed, max_n=args.max_n)
