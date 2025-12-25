"""
packing.py - Advanced packing strategies for Santa 2025

This module implements state-of-the-art packing algorithms:
1. Multiple initial placement strategies (hexagonal, diamond, spiral)
2. Incremental placement with optimal insertion
3. Global optimization using multiple metaheuristics
4. Solution caching and progressive refinement
"""
import numpy as np
import math
import random
from typing import List, Tuple, Dict, Optional
from .geometry import (
    transform_tree, CollisionDetector, compute_bounding_square_side,
    center_placements, placement_in_bounds, compute_bounding_box,
    check_overlap, normalize_angle, get_centroid
)
from .advanced_optimize import (
    AdvancedOptimizer, AdvancedConfig,
    create_hexagonal_interlocking, create_optimal_interlocking,
    create_diamond_pattern, create_spiral_compact,
    find_best_initial_placement, optimize_placement_advanced
)
from .optimize import OptimizationConfig

Placement = Tuple[float, float, float]
Solution = List[Placement]


class AdvancedPackingSolver:
    """
    State-of-the-art packing solver with multiple optimization phases.
    """

    def __init__(self, config=None, seed: int = 42):
        # Handle both OptimizationConfig and AdvancedConfig
        if config is None:
            self.config = AdvancedConfig()
        elif isinstance(config, OptimizationConfig):
            self.config = config.to_advanced_config()
        else:
            self.config = config
        self.config.seed = seed
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        self.optimizer = AdvancedOptimizer(self.config)
        self.solutions: Dict[int, Solution] = {}
        self.scores: Dict[int, float] = {}

        # Precomputed optimal configurations for small n
        self._precompute_small_solutions()

    def _precompute_small_solutions(self):
        """Precompute optimal solutions for very small n."""
        # n=1: Single tree at origin
        self.solutions[1] = [(0.0, 0.0, 0.0)]
        self.scores[1] = compute_bounding_square_side(self.solutions[1])

        # n=2: Two trees side by side with optimal rotation
        best = None
        best_score = float('inf')
        for d1 in [0, 180]:
            for d2 in [0, 180]:
                for dx in [0.35, 0.4, 0.45, 0.5]:
                    sol = [(-dx/2, 0, d1), (dx/2, 0, d2)]
                    if self._is_valid(sol):
                        score = compute_bounding_square_side(sol)
                        if score < best_score:
                            best_score = score
                            best = sol
        if best:
            self.solutions[2] = center_placements(best)
            self.scores[2] = best_score

        # n=3: Triangle arrangement
        best = None
        best_score = float('inf')
        for d1, d2, d3 in [(0, 0, 180), (0, 180, 0), (180, 0, 0), (0, 180, 180)]:
            for spacing in [0.35, 0.4, 0.45]:
                sol = [
                    (0, spacing * 0.6, d1),
                    (-spacing * 0.5, -spacing * 0.3, d2),
                    (spacing * 0.5, -spacing * 0.3, d3)
                ]
                if self._is_valid(sol):
                    score = compute_bounding_square_side(sol)
                    if score < best_score:
                        best_score = score
                        best = sol
        if best:
            self.solutions[3] = center_placements(best)
            self.scores[3] = best_score

    def _is_valid(self, solution: Solution) -> bool:
        """Check if solution has no overlaps and is in bounds."""
        detector = CollisionDetector()
        for x, y, d in solution:
            if not placement_in_bounds(x, y, d):
                return False
            poly = transform_tree(x, y, d)
            if detector.check_collision(poly):
                return False
            detector.add_polygon(poly)
        return True

    def _find_best_insertion_advanced(self, existing: Solution,
                                       detector: CollisionDetector) -> Placement:
        """
        Find the best position to insert a new tree using multiple strategies.
        """
        n = len(existing)
        if n == 0:
            return (0.0, 0.0, 0.0)

        min_x, min_y, max_x, max_y = compute_bounding_box(existing)
        current_side = compute_bounding_square_side(existing)

        best: Optional[Placement] = None
        best_score = float('inf')

        # Strategy 1: Grid search with fine resolution
        margin = 0.3
        grid_size = max(10, min(20, n))

        xs = np.linspace(min_x - margin, max_x + margin, grid_size)
        ys = np.linspace(min_y - margin, max_y + margin, grid_size)

        # Optimal rotations for interlocking
        rotations = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]

        for x in xs:
            for y in ys:
                for deg in rotations:
                    poly = transform_tree(x, y, deg)
                    if detector.check_collision(poly):
                        continue
                    if not placement_in_bounds(x, y, deg):
                        continue

                    test = existing + [(x, y, deg)]
                    score = compute_bounding_square_side(test)
                    if score < best_score:
                        best_score = score
                        best = (x, y, deg)

        # Strategy 2: Search near existing trees (gaps between trees)
        for i, (px, py, pd) in enumerate(existing):
            for angle_offset in np.linspace(0, 2 * math.pi, 12, endpoint=False):
                for radius in [0.4, 0.5, 0.6]:
                    x = px + radius * math.cos(angle_offset)
                    y = py + radius * math.sin(angle_offset)

                    for deg in [0, 60, 120, 180, 240, 300]:
                        poly = transform_tree(x, y, deg)
                        if detector.check_collision(poly):
                            continue
                        if not placement_in_bounds(x, y, deg):
                            continue

                        test = existing + [(x, y, deg)]
                        score = compute_bounding_square_side(test)
                        if score < best_score:
                            best_score = score
                            best = (x, y, deg)

        # Strategy 3: Center of mass approach
        cx, cy = get_centroid(existing)
        for angle in np.linspace(0, 2 * math.pi, 24, endpoint=False):
            for r in [0.3, 0.4, 0.5, 0.6, 0.7]:
                x = cx + r * math.cos(angle)
                y = cy + r * math.sin(angle)

                for deg in [0, 90, 180, 270]:
                    poly = transform_tree(x, y, deg)
                    if detector.check_collision(poly):
                        continue
                    if not placement_in_bounds(x, y, deg):
                        continue

                    test = existing + [(x, y, deg)]
                    score = compute_bounding_square_side(test)
                    if score < best_score:
                        best_score = score
                        best = (x, y, deg)

        # Fallback: place outside bounding box
        if best is None:
            for angle in np.linspace(0, 2 * math.pi, 32, endpoint=False):
                r = current_side / 2 + 0.6
                x, y = r * math.cos(angle), r * math.sin(angle)
                for deg in [0, 180]:
                    poly = transform_tree(x, y, deg)
                    if not detector.check_collision(poly):
                        best = (x, y, deg)
                        break
                if best:
                    break

        return best if best else (current_side + 1, 0.0, 0.0)

    def solve_single(self, n: int, verbose: bool = True) -> Solution:
        """
        Solve for a single n using incremental building + optimization.
        """
        if n <= 0:
            return []

        if n in self.solutions:
            return self.solutions[n]

        # Base cases already precomputed
        if n <= 3 and n in self.solutions:
            return self.solutions[n]

        # Ensure we have solution for n-1
        if n - 1 not in self.solutions:
            self.solve_single(n - 1, verbose=False)

        # Get previous solution
        prev = list(self.solutions[n - 1])

        # Build collision detector
        detector = CollisionDetector()
        for x, y, d in prev:
            detector.add_polygon(transform_tree(x, y, d))

        # Find best insertion point
        new_p = self._find_best_insertion_advanced(prev, detector)
        solution = prev + [new_p]

        # Optimization intensity increases with n
        if n <= 10:
            mult = 1.5
        elif n <= 50:
            mult = 1.0 + (n / 100)
        elif n <= 100:
            mult = 1.5 + (n / 100)
        else:
            mult = 2.0 + (n / 100)

        # Apply advanced optimization
        optimized = self.optimizer.optimize(solution, mult, verbose=False)
        optimized = center_placements(optimized)

        # Verify validity
        if not self._is_valid(optimized):
            # Fall back to less aggressive optimization
            optimized = center_placements(solution)

        self.solutions[n] = optimized
        self.scores[n] = compute_bounding_square_side(optimized)

        if verbose and n % 10 == 0:
            print(f"  n={n}: side={self.scores[n]:.6f}")

        return optimized

    def solve_with_global_restart(self, n: int, verbose: bool = True) -> Solution:
        """
        Solve with multiple global restarts to escape local minima.
        """
        if n <= 1:
            return [(0.0, 0.0, 0.0)] if n == 1 else []

        best_solution = None
        best_score = float('inf')

        # Strategy 1: Incremental building
        self.solutions.clear()
        self.scores.clear()
        self._precompute_small_solutions()
        sol1 = self.solve_single(n, verbose=False)
        if self._is_valid(sol1):
            score1 = compute_bounding_square_side(sol1)
            if score1 < best_score:
                best_score = score1
                best_solution = sol1

        # Strategy 2: Start from hexagonal pattern
        try:
            hex_init = create_hexagonal_interlocking(n)
            if self._is_valid(hex_init):
                hex_opt = self.optimizer.optimize(hex_init, 2.0, verbose=False)
                if self._is_valid(hex_opt):
                    score2 = compute_bounding_square_side(hex_opt)
                    if score2 < best_score:
                        best_score = score2
                        best_solution = hex_opt
        except Exception:
            pass

        # Strategy 3: Start from diamond pattern
        try:
            diamond_init = create_diamond_pattern(n)
            if self._is_valid(diamond_init):
                diamond_opt = self.optimizer.optimize(diamond_init, 2.0, verbose=False)
                if self._is_valid(diamond_opt):
                    score3 = compute_bounding_square_side(diamond_opt)
                    if score3 < best_score:
                        best_score = score3
                        best_solution = diamond_opt
        except Exception:
            pass

        # Strategy 4: Start from optimal interlocking
        try:
            interlock_init = create_optimal_interlocking(n)
            if self._is_valid(interlock_init):
                interlock_opt = self.optimizer.optimize(interlock_init, 2.0, verbose=False)
                if self._is_valid(interlock_opt):
                    score4 = compute_bounding_square_side(interlock_opt)
                    if score4 < best_score:
                        best_score = score4
                        best_solution = interlock_opt
        except Exception:
            pass

        if best_solution is None:
            best_solution = sol1

        self.solutions[n] = center_placements(best_solution)
        self.scores[n] = compute_bounding_square_side(self.solutions[n])

        return self.solutions[n]

    def solve_all(self, max_n: int = 200, verbose: bool = True) -> Dict[int, Solution]:
        """
        Solve for all n from 1 to max_n.
        """
        import sys
        if verbose:
            print(f"Solving for n=1 to {max_n} with advanced optimization...", flush=True)

        for n in range(1, max_n + 1):
            # Use global restart for milestone n values
            if n in [1, 10, 25, 50, 75, 100, 125, 150, 175, 200]:
                self.solve_with_global_restart(n, verbose=verbose)
            else:
                self.solve_single(n, verbose=verbose)

            # Print progress every n to show it's working
            if verbose and n % 10 == 0:
                sys.stdout.flush()

        if verbose:
            score = self.compute_total_score()
            print(f"\nTotal score estimate: {score:.2f}", flush=True)

        return self.solutions

    def compute_total_score(self) -> float:
        """Compute the total Kaggle score."""
        total = 0.0
        for n, sol in self.solutions.items():
            side = self.scores.get(n, compute_bounding_square_side(sol))
            tree_area = 0.25
            efficiency = 0.6
            ref = math.sqrt(n * tree_area / efficiency)
            if ref > 0:
                total += n * (side ** 2) / (ref ** 2)
        return total


# Backward compatibility aliases
PackingSolver = AdvancedPackingSolver
create_spiral_placement = create_spiral_compact


def create_initial_solution(n: int, strategy: str = "optimal") -> Solution:
    """Create initial solution using specified strategy."""
    if strategy == "hexagonal":
        return create_hexagonal_interlocking(n)
    elif strategy == "diamond":
        return create_diamond_pattern(n)
    elif strategy == "spiral":
        return create_spiral_compact(n)
    elif strategy == "optimal":
        return find_best_initial_placement(n)
    else:
        return create_optimal_interlocking(n)


def validate_and_fix_solution(solution: Solution) -> Tuple[Solution, bool]:
    """Validate and attempt to fix overlapping solutions."""
    from .geometry import check_all_overlaps

    overlaps = check_all_overlaps(solution)
    if not overlaps:
        return solution, False

    fixed = list(solution)
    max_iterations = 100

    for iteration in range(max_iterations):
        overlaps = check_all_overlaps(fixed)
        if not overlaps:
            break

        for i, j in overlaps:
            x1, y1, d1 = fixed[i]
            x2, y2, d2 = fixed[j]

            dx, dy = x2 - x1, y2 - y1
            dist = math.sqrt(dx * dx + dy * dy)

            if dist < 0.001:
                angle = random.uniform(0, 2 * math.pi)
                dx, dy = math.cos(angle), math.sin(angle)
            else:
                dx, dy = dx / dist, dy / dist

            # Push apart
            push = 0.05 * (1 + iteration * 0.1)
            fixed[i] = (x1 - dx * push, y1 - dy * push, d1)
            fixed[j] = (x2 + dx * push, y2 + dy * push, d2)

    return center_placements(fixed), len(overlaps) > 0
