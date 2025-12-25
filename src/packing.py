"""
packing.py - Main packing strategies for Santa 2025
"""
import numpy as np
import math
import random
from typing import List, Tuple, Dict, Optional
from .geometry import (
    transform_tree, CollisionDetector, compute_bounding_square_side,
    center_placements, placement_in_bounds, compute_bounding_box
)
from .optimize import optimize_placement, OptimizationConfig

Placement = Tuple[float, float, float]
Solution = List[Placement]

def create_spiral_placement(n: int, spacing: float = 0.7) -> Solution:
    if n == 0:
        return []
    
    placements = [(0.0, 0.0, 0.0)]
    if n == 1:
        return placements
    
    base_spacing = 0.6 * spacing
    angle = 0.0
    radius = base_spacing
    angle_inc = 2.4
    radius_inc = base_spacing * 0.3
    
    for i in range(1, n):
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        deg = 0.0 if i % 2 == 0 else 180.0
        placements.append((x, y, deg))
        angle += angle_inc
        radius += radius_inc / (1 + i * 0.01)
    
    return center_placements(placements)

def create_grid_placement(n: int, spacing: float = 0.8) -> Solution:
    if n == 0:
        return []
    
    cols = int(math.ceil(math.sqrt(n)))
    placements = []
    
    for i in range(n):
        row, col = i // cols, i % cols
        x, y = col * spacing, row * spacing
        deg = 0.0 if (row + col) % 2 == 0 else 180.0
        placements.append((x, y, deg))
    
    return center_placements(placements)

def create_motif_placement(n: int) -> Solution:
    if n == 0:
        return []
    
    dx, dy = 0.55, 0.50
    placements = []
    idx, row = 0, 0
    
    while idx < n:
        x_offset = (row % 2) * (dx / 2)
        col = 0
        while idx < n:
            x = col * dx + x_offset
            y = row * dy
            deg = (0.0 if col % 2 == 0 else 180.0) if row % 2 == 0 else (180.0 if col % 2 == 0 else 0.0)
            placements.append((x, y, deg))
            idx += 1
            col += 1
            if col * dx > math.sqrt(n) * dx * 1.2:
                break
        row += 1
    
    return center_placements(placements)

def find_best_insertion(existing: Solution, detector: CollisionDetector, n_candidates: int = 100) -> Placement:
    if not existing:
        return (0.0, 0.0, 0.0)
    
    min_x, min_y, max_x, max_y = compute_bounding_box(existing)
    margin = 0.6
    rotations = [0, 60, 90, 120, 180, 240, 270, 300]
    
    best, best_score = None, float('inf')
    grid = int(math.sqrt(n_candidates))
    
    xs = np.linspace(min_x - margin, max_x + margin, grid)
    ys = np.linspace(min_y - margin, max_y + margin, grid)
    
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
    
    if best is None:
        # Fallback: place outside
        side = compute_bounding_square_side(existing)
        for angle in np.linspace(0, 2*math.pi, 16, endpoint=False):
            r = side/2 + 0.6
            x, y = r * math.cos(angle), r * math.sin(angle)
            for deg in rotations:
                poly = transform_tree(x, y, deg)
                if not detector.check_collision(poly):
                    return (x, y, deg)
        return (side + 1, 0, 0)
    
    return best

class PackingSolver:
    def __init__(self, config: Optional[OptimizationConfig] = None, seed: int = 42):
        self.config = config or OptimizationConfig()
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        self.solutions: Dict[int, Solution] = {}
        self.scores: Dict[int, float] = {}
    
    def solve_single(self, n: int, verbose: bool = True) -> Solution:
        if n <= 0:
            return []
        
        if n == 1:
            sol = [(0.0, 0.0, 0.0)]
            self.solutions[1] = sol
            self.scores[1] = compute_bounding_square_side(sol)
            return sol
        
        if n - 1 not in self.solutions:
            self.solve_single(n - 1, verbose=False)
        
        prev = self.solutions[n - 1].copy()
        
        detector = CollisionDetector()
        for x, y, d in prev:
            detector.add_polygon(transform_tree(x, y, d))
        
        new_p = find_best_insertion(prev, detector)
        solution = prev + [new_p]
        
        mult = 1 + (n / 50)
        optimized = optimize_placement(solution, self.config, mult, verbose=False)
        optimized = center_placements(optimized)
        
        self.solutions[n] = optimized
        self.scores[n] = compute_bounding_square_side(optimized)
        
        if verbose and n % 10 == 0:
            print(f"  n={n}: side={self.scores[n]:.6f}")
        
        return optimized
    
    def solve_all(self, max_n: int = 200, verbose: bool = True) -> Dict[int, Solution]:
        if verbose:
            print(f"Solving for n=1 to {max_n}...")
        
        for n in range(1, max_n + 1):
            self.solve_single(n, verbose=verbose)
        
        if verbose:
            score = self.compute_total_score()
            print(f"\nTotal score estimate: {score:.2f}")
        
        return self.solutions
    
    def compute_total_score(self) -> float:
        total = 0.0
        for n, sol in self.solutions.items():
            side = self.scores.get(n, compute_bounding_square_side(sol))
            baseline = math.sqrt(n) * 0.6
            if baseline > 0:
                total += n * (side ** 2) / (baseline ** 2)
        return total

def create_initial_solution(n: int, strategy: str = "motif") -> Solution:
    if strategy == "spiral":
        return create_spiral_placement(n)
    elif strategy == "grid":
        return create_grid_placement(n)
    return create_motif_placement(n)

def validate_and_fix_solution(solution: Solution) -> Tuple[Solution, bool]:
    from .geometry import check_all_overlaps
    overlaps = check_all_overlaps(solution)
    if not overlaps:
        return solution, False
    
    fixed = list(solution)
    for i, j in overlaps:
        x1, y1, d1 = fixed[i]
        x2, y2, d2 = fixed[j]
        dx, dy = x2 - x1, y2 - y1
        dist = math.sqrt(dx*dx + dy*dy)
        if dist < 0.001:
            angle = random.uniform(0, 2*math.pi)
            dx, dy = math.cos(angle), math.sin(angle)
        else:
            dx, dy = dx/dist, dy/dist
        push = 0.1
        fixed[i] = (x1 - dx*push, y1 - dy*push, d1)
        fixed[j] = (x2 + dx*push, y2 + dy*push, d2)
    
    return fixed, True
