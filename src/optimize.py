"""
optimize.py - Optimization algorithms for Santa 2025
"""
import numpy as np
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional
from .geometry import (
    transform_tree, CollisionDetector, compute_bounding_square_side,
    center_placements, placement_in_bounds, normalize_angle, get_centroid
)

@dataclass
class OptimizationConfig:
    sa_iterations_base: int = 1000
    sa_temp_initial: float = 0.5
    sa_temp_final: float = 0.001
    translation_step: float = 0.03
    translation_step_min: float = 0.005
    rotation_step: float = 5.0
    rotation_step_min: float = 1.0
    prob_translate: float = 0.5
    prob_rotate: float = 0.25
    prob_swap: float = 0.15
    prob_compact: float = 0.10
    adaptive_steps: bool = True
    step_decay: float = 0.99
    seed: int = 42
    
    @classmethod
    def quick_mode(cls):
        return cls(sa_iterations_base=500, translation_step=0.05, rotation_step=5.0)
    
    @classmethod
    def full_mode(cls):
        return cls(sa_iterations_base=2000, sa_temp_final=0.0001, translation_step=0.02, rotation_step=2.0)

def make_translation_move(p: Tuple, step: float) -> Tuple:
    x, y, deg = p
    return (x + random.uniform(-step, step), y + random.uniform(-step, step), deg)

def make_rotation_move(p: Tuple, step: float) -> Tuple:
    x, y, deg = p
    return (x, y, normalize_angle(deg + random.uniform(-step, step)))

def make_compaction_move(p: Tuple, centroid: Tuple, step: float) -> Tuple:
    x, y, deg = p
    cx, cy = centroid
    dx, dy = cx - x, cy - y
    dist = math.sqrt(dx*dx + dy*dy)
    if dist < 0.001:
        return p
    return (x + dx/dist*step, y + dy/dist*step, deg)

def simulated_annealing(
    initial: List[Tuple], config: OptimizationConfig, n_iterations: int
) -> List[Tuple]:
    n = len(initial)
    if n <= 1:
        return initial
    
    current = list(initial)
    best = list(current)
    current_score = compute_bounding_square_side(current)
    best_score = current_score
    
    detector = CollisionDetector()
    for x, y, deg in current:
        detector.add_polygon(transform_tree(x, y, deg))
    
    temp = config.sa_temp_initial
    temp_factor = (config.sa_temp_final / config.sa_temp_initial) ** (1.0 / max(n_iterations, 1))
    trans_step = config.translation_step
    rot_step = config.rotation_step
    
    for _ in range(n_iterations):
        idx = random.randint(0, n - 1)
        
        r = random.random()
        if r < config.prob_translate:
            new_p = make_translation_move(current[idx], trans_step)
        elif r < config.prob_translate + config.prob_rotate:
            new_p = make_rotation_move(current[idx], rot_step)
        elif r < config.prob_translate + config.prob_rotate + config.prob_compact:
            new_p = make_compaction_move(current[idx], get_centroid(current), trans_step)
        else:
            # Swap
            idx2 = random.randint(0, n - 1)
            if idx2 != idx:
                x1, y1, d1 = current[idx]
                x2, y2, d2 = current[idx2]
                candidate = list(current)
                candidate[idx] = (x2, y2, d1)
                candidate[idx2] = (x1, y1, d2)
                
                valid = True
                for i in [idx, idx2]:
                    px, py, pd = candidate[i]
                    if not placement_in_bounds(px, py, pd):
                        valid = False
                        break
                
                if valid:
                    # Check collisions
                    test_detector = CollisionDetector()
                    for x, y, d in candidate:
                        test_detector.add_polygon(transform_tree(x, y, d))
                    
                    has_collision = False
                    for i in range(len(candidate)):
                        poly = transform_tree(*candidate[i])
                        if test_detector.check_collision(poly, exclude_index=i):
                            has_collision = True
                            break
                    
                    if not has_collision:
                        cand_score = compute_bounding_square_side(center_placements(candidate))
                        delta = cand_score - current_score
                        if delta < 0 or (temp > 0 and random.random() < math.exp(-delta/temp)):
                            current = candidate
                            current_score = cand_score
                            detector.clear()
                            for x, y, d in current:
                                detector.add_polygon(transform_tree(x, y, d))
                            if current_score < best_score:
                                best = list(current)
                                best_score = current_score
                
                temp *= temp_factor
                continue
        
        x, y, deg = new_p
        if not placement_in_bounds(x, y, deg):
            temp *= temp_factor
            continue
        
        new_poly = transform_tree(x, y, deg)
        old_poly = detector.polygons[idx]
        detector.update_polygon(idx, new_poly)
        
        if detector.check_collision(new_poly, exclude_index=idx):
            detector.update_polygon(idx, old_poly)
            temp *= temp_factor
            continue
        
        candidate = list(current)
        candidate[idx] = new_p
        cand_score = compute_bounding_square_side(center_placements(candidate))
        delta = cand_score - current_score
        
        if delta < 0 or (temp > 0 and random.random() < math.exp(-delta/temp)):
            current = candidate
            current_score = cand_score
            if current_score < best_score:
                best = list(current)
                best_score = current_score
        else:
            detector.update_polygon(idx, old_poly)
        
        temp *= temp_factor
        if config.adaptive_steps:
            trans_step = max(config.translation_step_min, trans_step * config.step_decay)
            rot_step = max(config.rotation_step_min, rot_step * config.step_decay)
    
    return center_placements(best)

def local_search(solution: List[Tuple], config: OptimizationConfig, max_iter: int = 1000) -> List[Tuple]:
    n = len(solution)
    if n <= 1:
        return solution
    
    current = list(solution)
    current_score = compute_bounding_square_side(center_placements(current))
    
    detector = CollisionDetector()
    for x, y, d in current:
        detector.add_polygon(transform_tree(x, y, d))
    
    step = config.translation_step / 2
    no_improve = 0
    
    for _ in range(max_iter):
        if no_improve > 100:
            break
        
        idx = random.randint(0, n - 1)
        new_p = make_translation_move(current[idx], step) if random.random() < 0.7 else make_rotation_move(current[idx], config.rotation_step / 2)
        
        x, y, deg = new_p
        if not placement_in_bounds(x, y, deg):
            no_improve += 1
            continue
        
        new_poly = transform_tree(x, y, deg)
        old_poly = detector.polygons[idx]
        detector.update_polygon(idx, new_poly)
        
        if detector.check_collision(new_poly, exclude_index=idx):
            detector.update_polygon(idx, old_poly)
            no_improve += 1
            continue
        
        candidate = list(current)
        candidate[idx] = new_p
        cand_score = compute_bounding_square_side(center_placements(candidate))
        
        if cand_score < current_score:
            current = candidate
            current_score = cand_score
            no_improve = 0
        else:
            detector.update_polygon(idx, old_poly)
            no_improve += 1
    
    return center_placements(current)

def optimize_placement(
    solution: List[Tuple], config: Optional[OptimizationConfig] = None,
    iterations_multiplier: float = 1.0, verbose: bool = False
) -> List[Tuple]:
    if config is None:
        config = OptimizationConfig()
    
    n = len(solution)
    if n <= 1:
        return solution
    
    random.seed(config.seed + n)
    np.random.seed(config.seed + n)
    
    n_iter = int(config.sa_iterations_base * iterations_multiplier)
    optimized = simulated_annealing(solution, config, n_iter)
    optimized = local_search(optimized, config, n_iter // 2)
    
    return optimized
