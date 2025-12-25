"""
io_utils.py - File I/O utilities for Santa 2025
"""
import os
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional

def find_data_path() -> str:
    candidates = ["/kaggle/input/santa-2025", "./data", "../data", "."]
    for path in candidates:
        if os.path.isdir(path):
            sample = os.path.join(path, "sample_submission.csv")
            if os.path.isfile(sample):
                return path
    if os.path.isfile("sample_submission.csv"):
        return "."
    raise FileNotFoundError("Could not find competition data")

def get_sample_submission_path() -> str:
    return os.path.join(find_data_path(), "sample_submission.csv")

def get_output_path(filename: str = "submission.csv") -> str:
    if os.path.isdir("/kaggle/working"):
        return f"/kaggle/working/{filename}"
    return filename

def parse_s_value(s: str) -> float:
    if isinstance(s, str) and s.startswith('s'):
        return float(s[1:])
    return float(s)

def format_s_value(value: float, decimals: int = 6) -> str:
    return f"s{value:.{decimals}f}"

def parse_sample_submission(path: Optional[str] = None) -> Dict[int, List[str]]:
    if path is None:
        path = get_sample_submission_path()
    df = pd.read_csv(path)
    puzzle_ids = {}
    for _, row in df.iterrows():
        id_str = row['id']
        n = int(id_str.split('_')[0])
        if n not in puzzle_ids:
            puzzle_ids[n] = []
        puzzle_ids[n].append(id_str)
    return puzzle_ids

def load_existing_solutions(path: Optional[str] = None) -> Dict[int, List[Tuple[float, float, float]]]:
    if path is None:
        path = get_sample_submission_path()
    df = pd.read_csv(path)
    solutions = {}
    for _, row in df.iterrows():
        parts = row['id'].split('_')
        n, idx = int(parts[0]), int(parts[1])
        x, y, deg = parse_s_value(row['x']), parse_s_value(row['y']), parse_s_value(row['deg'])
        if n not in solutions:
            solutions[n] = []
        while len(solutions[n]) <= idx:
            solutions[n].append(None)
        solutions[n][idx] = (x, y, deg)
    return solutions

def create_submission(
    solutions: Dict[int, List[Tuple[float, float, float]]],
    output_path: Optional[str] = None, decimals: int = 6
) -> str:
    if output_path is None:
        output_path = get_output_path()
    
    rows = []
    for n in range(1, 201):
        if n not in solutions:
            raise ValueError(f"Missing solution for n={n}")
        placements = solutions[n]
        if len(placements) != n:
            raise ValueError(f"Wrong count for n={n}: expected {n}, got {len(placements)}")
        for i, (x, y, deg) in enumerate(placements):
            rows.append({
                'id': f"{n:03d}_{i}",
                'x': format_s_value(x, decimals),
                'y': format_s_value(y, decimals),
                'deg': format_s_value(deg, decimals),
            })
    
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path

def validate_submission_format(path: str) -> Tuple[bool, Optional[str]]:
    try:
        df = pd.read_csv(path)
    except Exception as e:
        return False, f"Failed to read CSV: {e}"
    
    expected_cols = {'id', 'x', 'y', 'deg'}
    if set(df.columns) != expected_cols:
        return False, f"Wrong columns"
    
    expected_rows = 200 * 201 // 2
    if len(df) != expected_rows:
        return False, f"Wrong row count: expected {expected_rows}, got {len(df)}"
    
    for _, row in df.iterrows():
        for col in ['x', 'y', 'deg']:
            val = row[col]
            if not isinstance(val, str) or not val.startswith('s'):
                return False, f"Invalid format in {col}"
        x, y = parse_s_value(row['x']), parse_s_value(row['y'])
        if not (-100 <= x <= 100) or not (-100 <= y <= 100):
            return False, "Value out of range"
    
    return True, None

def print_solution_summary(solutions: Dict[int, List[Tuple[float, float, float]]], max_n: int = 200):
    from .geometry import compute_bounding_square_side
    from .validate import compute_score
    
    print("=" * 60)
    print("SOLUTION SUMMARY")
    print("=" * 60)
    
    n_solutions = len([n for n in range(1, max_n + 1) if n in solutions])
    print(f"Solutions: {n_solutions}/{max_n}")
    
    sides = {n: compute_bounding_square_side(solutions[n]) for n in range(1, max_n + 1) if n in solutions}
    
    if sides:
        print(f"\nBounding square stats:")
        print(f"  Min: {min(sides.values()):.4f} (n={min(sides, key=sides.get)})")
        print(f"  Max: {max(sides.values()):.4f} (n={max(sides, key=sides.get)})")
        print(f"  Mean: {np.mean(list(sides.values())):.4f}")
    
    print(f"\nEstimated score: {compute_score(solutions, max_n):.2f}")
    print("=" * 60)
