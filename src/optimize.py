"""
optimize.py - Optimization algorithms for Santa 2025

This module now primarily delegates to advanced_optimize.py for
state-of-the-art optimization. This file maintains backward compatibility.
"""
import numpy as np
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

# Import advanced optimization
from .advanced_optimize import (
    AdvancedOptimizer, AdvancedConfig,
    optimize_placement_advanced
)
from .geometry import (
    transform_tree, CollisionDetector, compute_bounding_square_side,
    center_placements, placement_in_bounds, normalize_angle, get_centroid
)


@dataclass
class OptimizationConfig:
    """Legacy configuration - maps to AdvancedConfig."""
    sa_iterations_base: int = 5000
    sa_temp_initial: float = 2.0
    sa_temp_final: float = 1e-6
    translation_step: float = 0.05
    translation_step_min: float = 0.001
    rotation_step: float = 15.0
    rotation_step_min: float = 0.5
    prob_translate: float = 0.35
    prob_rotate: float = 0.30
    prob_swap: float = 0.10
    prob_compact: float = 0.15
    adaptive_steps: bool = True
    step_decay: float = 0.995
    seed: int = 42

    @classmethod
    def quick_mode(cls):
        """Quick mode for fast testing."""
        return cls(sa_iterations_base=2000, translation_step=0.08, rotation_step=20.0)

    @classmethod
    def full_mode(cls):
        """Full mode for best results."""
        return cls(sa_iterations_base=10000, sa_temp_final=1e-8, translation_step=0.03, rotation_step=10.0)

    def to_advanced_config(self) -> AdvancedConfig:
        """Convert to AdvancedConfig."""
        return AdvancedConfig(
            seed=self.seed,
            sa_iterations_base=self.sa_iterations_base,
            sa_temp_initial=self.sa_temp_initial,
            sa_temp_final=self.sa_temp_final,
            translation_step_initial=self.translation_step,
            translation_step_final=self.translation_step_min,
            rotation_step_initial=self.rotation_step,
            rotation_step_final=self.rotation_step_min,
            prob_translate=self.prob_translate,
            prob_rotate=self.prob_rotate,
            prob_swap=self.prob_swap,
            prob_compact=self.prob_compact,
        )


def optimize_placement(
    solution: List[Tuple],
    config: Optional[OptimizationConfig] = None,
    iterations_multiplier: float = 1.0,
    verbose: bool = False
) -> List[Tuple]:
    """
    Main optimization entry point.
    Delegates to advanced optimization.
    """
    if config is None:
        config = OptimizationConfig()

    n = len(solution)
    if n <= 1:
        return center_placements(solution)

    random.seed(config.seed + n)
    np.random.seed(config.seed + n)

    # Use advanced optimization
    advanced_config = config.to_advanced_config()
    return optimize_placement_advanced(solution, advanced_config, iterations_multiplier, verbose)


# Legacy functions for backward compatibility
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
