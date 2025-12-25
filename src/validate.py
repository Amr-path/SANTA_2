"""
validate.py - Validation utilities for Santa 2025
"""
import math
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from .geometry import (
    transform_tree, check_all_overlaps, compute_bounding_box,
    placement_in_bounds, CollisionDetector, compute_bounding_square_side
)

@dataclass
class ValidationResult:
    valid: bool
    n: int
    n_trees: int
    overlaps: List[Tuple[int, int]]
    out_of_bounds: List[int]
    bounding_square: float
    error_message: Optional[str] = None

@dataclass
class FullValidationResult:
    valid: bool
    n_valid: int
    n_invalid: int
    invalid_ns: List[int]
    total_overlaps: int
    total_out_of_bounds: int
    error_messages: List[str]

def validate_solution(
    solution: List[Tuple[float, float, float]], n: int,
    strict: bool = False, bounds_limit: float = 100.0
) -> ValidationResult:
    if len(solution) != n:
        return ValidationResult(
            valid=False, n=n, n_trees=len(solution), overlaps=[], out_of_bounds=[],
            bounding_square=0.0, error_message=f"Expected {n} trees, got {len(solution)}"
        )
    
    if n == 0:
        return ValidationResult(valid=True, n=n, n_trees=0, overlaps=[], out_of_bounds=[], bounding_square=0.0)
    
    out_of_bounds = [i for i, (x, y, d) in enumerate(solution) if not placement_in_bounds(x, y, d, bounds_limit)]
    overlaps = check_all_overlaps(solution)
    
    min_x, min_y, max_x, max_y = compute_bounding_box(solution)
    bounding_square = max(max_x - min_x, max_y - min_y)
    
    valid = len(overlaps) == 0 and len(out_of_bounds) == 0
    error_msg = None
    if not valid:
        errors = []
        if overlaps:
            errors.append(f"{len(overlaps)} overlap(s)")
        if out_of_bounds:
            errors.append(f"{len(out_of_bounds)} out of bounds")
        error_msg = "; ".join(errors)
    
    return ValidationResult(
        valid=valid, n=n, n_trees=len(solution), overlaps=overlaps,
        out_of_bounds=out_of_bounds, bounding_square=bounding_square, error_message=error_msg
    )

def validate_solution_fast(solution: List[Tuple[float, float, float]], bounds_limit: float = 100.0) -> bool:
    if not solution:
        return True
    
    for x, y, d in solution:
        if not placement_in_bounds(x, y, d, bounds_limit):
            return False
    
    detector = CollisionDetector()
    for i, (x, y, d) in enumerate(solution):
        poly = transform_tree(x, y, d)
        if detector.check_collision(poly):
            return False
        detector.add_polygon(poly)
    
    return True

def validate_all_solutions(
    solutions: Dict[int, List[Tuple[float, float, float]]],
    max_n: int = 200, strict: bool = False, verbose: bool = True
) -> FullValidationResult:
    n_valid, n_invalid = 0, 0
    invalid_ns = []
    total_overlaps, total_out_of_bounds = 0, 0
    error_messages = []
    
    for n in range(1, max_n + 1):
        if n not in solutions:
            n_invalid += 1
            invalid_ns.append(n)
            error_messages.append(f"n={n}: Missing solution")
            continue
        
        result = validate_solution(solutions[n], n, strict=strict)
        if result.valid:
            n_valid += 1
        else:
            n_invalid += 1
            invalid_ns.append(n)
            total_overlaps += len(result.overlaps)
            total_out_of_bounds += len(result.out_of_bounds)
            error_messages.append(f"n={n}: {result.error_message}")
    
    valid = n_invalid == 0
    
    if verbose:
        if valid:
            print(f"✓ All {n_valid} solutions are valid")
        else:
            print(f"✗ {n_invalid} invalid solution(s)")
            for msg in error_messages[:10]:
                print(f"  - {msg}")
    
    return FullValidationResult(
        valid=valid, n_valid=n_valid, n_invalid=n_invalid, invalid_ns=invalid_ns,
        total_overlaps=total_overlaps, total_out_of_bounds=total_out_of_bounds, error_messages=error_messages
    )

def compute_score(solutions: Dict[int, List[Tuple[float, float, float]]], max_n: int = 200) -> float:
    total = 0.0
    for n in range(1, max_n + 1):
        if n not in solutions or not solutions[n]:
            continue
        side = compute_bounding_square_side(solutions[n])
        tree_area = 0.25
        efficiency = 0.6
        ref = math.sqrt(n * tree_area / efficiency)
        if ref > 0:
            total += n * (side ** 2) / (ref ** 2)
    return total

def compute_side_lengths(solutions: Dict[int, List[Tuple[float, float, float]]], max_n: int = 200) -> Dict[int, float]:
    return {n: compute_bounding_square_side(solutions[n]) for n in range(1, max_n + 1) if n in solutions and solutions[n]}
