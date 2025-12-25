"""
advanced_optimize.py - State-of-the-Art Optimization for Santa 2025

Implements:
1. Simulated Annealing with adaptive cooling and reheating
2. Gradient-based continuous optimization (L-BFGS-B)
3. Multi-resolution search strategies
4. Optimal rotation discovery through angular sweeps
5. Advanced interlocking patterns
6. Differential Evolution for global optimization
7. Basin Hopping for escaping local minima
8. Particle Swarm Optimization concepts
"""
import numpy as np
import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Callable
from scipy.optimize import minimize, differential_evolution, basinhopping
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

from .geometry import (
    transform_tree, CollisionDetector, compute_bounding_square_side,
    center_placements, placement_in_bounds, normalize_angle, get_centroid,
    compute_bounding_box, transform_vertices, TREE_VERTICES, check_overlap
)


@dataclass
class AdvancedConfig:
    """Advanced optimization configuration with all parameters."""
    # Core settings
    seed: int = 42
    max_n: int = 200

    # Simulated Annealing settings
    sa_iterations_base: int = 5000
    sa_temp_initial: float = 2.0
    sa_temp_final: float = 1e-6
    sa_reheat_threshold: float = 0.15
    sa_reheat_factor: float = 2.0
    sa_adaptive_cooling: bool = True

    # Move probabilities
    prob_translate: float = 0.35
    prob_rotate: float = 0.30
    prob_swap: float = 0.10
    prob_compact: float = 0.15
    prob_jiggle: float = 0.10

    # Step sizes
    translation_step_initial: float = 0.1
    translation_step_final: float = 0.001
    rotation_step_initial: float = 30.0
    rotation_step_final: float = 0.5

    # L-BFGS-B settings
    lbfgsb_enabled: bool = True
    lbfgsb_iterations: int = 100
    lbfgsb_penalty: float = 1e6

    # Differential Evolution settings
    de_enabled: bool = True
    de_pop_size: int = 15
    de_max_iter: int = 50
    de_mutation: Tuple[float, float] = (0.5, 1.0)
    de_recombination: float = 0.7

    # Basin Hopping settings
    basin_enabled: bool = True
    basin_niter: int = 20
    basin_step_size: float = 0.3

    # Multi-resolution settings
    multi_res_enabled: bool = True
    multi_res_levels: int = 4

    # Angular sweep settings
    angular_sweep_points: int = 72  # Every 5 degrees
    angular_fine_tune: bool = True

    # Interlocking pattern settings
    interlocking_enabled: bool = True
    optimal_angles: List[float] = field(default_factory=lambda: [0, 60, 120, 180, 240, 300])

    @classmethod
    def ultra_mode(cls):
        """Maximum optimization - for best scores."""
        return cls(
            sa_iterations_base=10000,
            sa_temp_initial=5.0,
            de_pop_size=25,
            de_max_iter=100,
            basin_niter=50,
            lbfgsb_iterations=200,
            angular_sweep_points=360,  # Every degree
        )

    @classmethod
    def fast_mode(cls):
        """Fast mode for quick testing."""
        return cls(
            sa_iterations_base=2000,
            de_enabled=False,
            basin_enabled=False,
            lbfgsb_iterations=50,
            angular_sweep_points=36,
        )


class AdvancedOptimizer:
    """
    State-of-the-art optimizer combining multiple advanced techniques.
    """

    def __init__(self, config: Optional[AdvancedConfig] = None):
        self.config = config or AdvancedConfig()
        self.rng = np.random.default_rng(self.config.seed)
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)

        # Precompute optimal interlocking angles
        self._precompute_interlocking_angles()

        # Cache for expensive computations
        self._angle_cache: Dict[Tuple[float, float], float] = {}

    def _precompute_interlocking_angles(self):
        """Precompute angles that allow optimal tree interlocking."""
        # The tree shape has specific angles where it interlocks best
        # Top angle is 53.13 degrees (arctan(0.5/0.25))
        # This affects optimal rotation combinations
        self.interlocking_pairs = [
            (0, 180),    # Head-to-head upside down
            (0, 0),      # Same orientation (offset)
            (60, 240),   # 60-degree offset
            (90, 270),   # Perpendicular
            (30, 210),   # 30-degree offset
            (45, 225),   # 45-degree offset
        ]

    def _solution_to_array(self, solution: List[Tuple]) -> np.ndarray:
        """Convert solution to numpy array for optimization."""
        return np.array(solution).flatten()

    def _array_to_solution(self, arr: np.ndarray, n: int) -> List[Tuple]:
        """Convert numpy array back to solution."""
        arr = arr.reshape(n, 3)
        return [(float(arr[i, 0]), float(arr[i, 1]), float(arr[i, 2]) % 360) for i in range(n)]

    def _compute_penalty_score(self, solution: List[Tuple], penalty_weight: float = 1e6) -> float:
        """
        Compute score with heavy penalties for overlaps and out-of-bounds.
        Used for gradient-based optimization.
        """
        n = len(solution)
        if n == 0:
            return 0.0

        # Base score: bounding square side
        centered = center_placements(solution)
        base_score = compute_bounding_square_side(centered)

        # Overlap penalty
        overlap_penalty = 0.0
        polys = [transform_tree(x, y, d) for x, y, d in solution]
        for i in range(n):
            for j in range(i + 1, n):
                if polys[i].intersects(polys[j]):
                    intersection = polys[i].intersection(polys[j])
                    overlap_penalty += intersection.area * penalty_weight

        # Out-of-bounds penalty
        bounds_penalty = 0.0
        for x, y, d in solution:
            verts = transform_vertices(x, y, d)
            min_v, max_v = verts.min(), verts.max()
            if min_v < -100:
                bounds_penalty += (-100 - min_v) ** 2 * penalty_weight
            if max_v > 100:
                bounds_penalty += (max_v - 100) ** 2 * penalty_weight

        return base_score + overlap_penalty + bounds_penalty

    def _is_valid_solution(self, solution: List[Tuple], eps: float = 1e-9) -> bool:
        """Fast validity check."""
        n = len(solution)
        if n == 0:
            return True

        # Check bounds
        for x, y, d in solution:
            if not placement_in_bounds(x, y, d):
                return False

        # Check overlaps using spatial indexing
        detector = CollisionDetector()
        for i, (x, y, d) in enumerate(solution):
            poly = transform_tree(x, y, d)
            if detector.check_collision(poly):
                return False
            detector.add_polygon(poly)

        return True

    # ==================== SIMULATED ANNEALING ====================

    def _adaptive_temperature(self, iteration: int, max_iter: int,
                               acceptance_rate: float, current_temp: float) -> float:
        """
        Adaptive cooling schedule based on acceptance rate.
        If acceptance is too low, cool slower; if too high, cool faster.
        """
        target_rate = 0.44 * np.exp(-3 * iteration / max_iter)  # Decreasing target

        if acceptance_rate < target_rate * 0.5:
            # Too cold - slow down cooling
            factor = 0.999
        elif acceptance_rate > target_rate * 1.5:
            # Too hot - speed up cooling
            factor = 0.99
        else:
            # Just right - standard cooling
            factor = 0.995

        new_temp = current_temp * factor
        return max(new_temp, self.config.sa_temp_final)

    def _make_move(self, solution: List[Tuple], temp: float,
                   trans_step: float, rot_step: float) -> Tuple[List[Tuple], str]:
        """Make a random move based on current temperature."""
        n = len(solution)
        candidate = list(solution)

        r = random.random()
        move_type = ""

        if r < self.config.prob_translate:
            # Translation move
            idx = random.randint(0, n - 1)
            x, y, d = candidate[idx]
            dx = random.gauss(0, trans_step)
            dy = random.gauss(0, trans_step)
            candidate[idx] = (x + dx, y + dy, d)
            move_type = "translate"

        elif r < self.config.prob_translate + self.config.prob_rotate:
            # Rotation move - either random or to optimal angle
            idx = random.randint(0, n - 1)
            x, y, d = candidate[idx]

            if random.random() < 0.3 and self.config.interlocking_enabled:
                # Snap to optimal interlocking angle
                optimal_angles = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
                new_d = random.choice(optimal_angles)
            else:
                new_d = normalize_angle(d + random.gauss(0, rot_step))

            candidate[idx] = (x, y, new_d)
            move_type = "rotate"

        elif r < self.config.prob_translate + self.config.prob_rotate + self.config.prob_swap:
            # Swap positions of two pieces
            if n >= 2:
                i, j = random.sample(range(n), 2)
                x1, y1, d1 = candidate[i]
                x2, y2, d2 = candidate[j]
                # Swap positions, keep rotations
                candidate[i] = (x2, y2, d1)
                candidate[j] = (x1, y1, d2)
                move_type = "swap"

        elif r < self.config.prob_translate + self.config.prob_rotate + \
                 self.config.prob_swap + self.config.prob_compact:
            # Compaction move - move toward center of mass
            idx = random.randint(0, n - 1)
            x, y, d = candidate[idx]
            cx, cy = get_centroid(candidate)
            dx, dy = cx - x, cy - y
            dist = math.sqrt(dx*dx + dy*dy)
            if dist > 0.001:
                step = random.uniform(0.01, trans_step * 2)
                candidate[idx] = (x + dx/dist*step, y + dy/dist*step, d)
            move_type = "compact"

        else:
            # Jiggle move - small random perturbation to all pieces
            scale = trans_step * 0.3
            for i in range(n):
                x, y, d = candidate[i]
                candidate[i] = (
                    x + random.gauss(0, scale),
                    y + random.gauss(0, scale),
                    d
                )
            move_type = "jiggle"

        return candidate, move_type

    def simulated_annealing(self, initial: List[Tuple], n_iterations: int) -> List[Tuple]:
        """
        Advanced simulated annealing with:
        - Adaptive cooling
        - Reheating mechanism
        - Multiple move types
        - Best solution tracking
        """
        n = len(initial)
        if n <= 1:
            return center_placements(initial)

        current = list(initial)
        best = list(current)
        current_score = compute_bounding_square_side(center_placements(current))
        best_score = current_score

        temp = self.config.sa_temp_initial
        trans_step = self.config.translation_step_initial
        rot_step = self.config.rotation_step_initial

        # Tracking for adaptive cooling
        accept_count = 0
        window_size = 100
        recent_accepts = []

        # Tracking for reheating
        no_improve_count = 0
        reheat_threshold = int(n_iterations * self.config.sa_reheat_threshold)

        for iteration in range(n_iterations):
            # Make a move
            candidate, move_type = self._make_move(current, temp, trans_step, rot_step)

            # Check validity
            if not self._is_valid_solution(candidate):
                recent_accepts.append(0)
                if len(recent_accepts) > window_size:
                    recent_accepts.pop(0)
                continue

            # Compute score
            cand_score = compute_bounding_square_side(center_placements(candidate))
            delta = cand_score - current_score

            # Metropolis acceptance
            accept = False
            if delta < 0:
                accept = True
            elif temp > 0:
                accept = random.random() < math.exp(-delta / temp)

            if accept:
                current = candidate
                current_score = cand_score
                accept_count += 1
                recent_accepts.append(1)

                if current_score < best_score:
                    best = list(current)
                    best_score = current_score
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                recent_accepts.append(0)
                no_improve_count += 1

            if len(recent_accepts) > window_size:
                recent_accepts.pop(0)

            # Adaptive cooling
            if self.config.sa_adaptive_cooling and len(recent_accepts) >= window_size:
                acceptance_rate = sum(recent_accepts) / len(recent_accepts)
                temp = self._adaptive_temperature(iteration, n_iterations, acceptance_rate, temp)
            else:
                # Standard exponential cooling
                temp *= (self.config.sa_temp_final / self.config.sa_temp_initial) ** (1.0 / n_iterations)

            # Reheating mechanism
            if no_improve_count >= reheat_threshold:
                temp = min(temp * self.config.sa_reheat_factor, self.config.sa_temp_initial * 0.5)
                no_improve_count = 0
                # Also reset to best solution
                current = list(best)
                current_score = best_score

            # Adaptive step sizes
            progress = iteration / n_iterations
            trans_step = self.config.translation_step_initial * (1 - progress) + \
                        self.config.translation_step_final * progress
            rot_step = self.config.rotation_step_initial * (1 - progress) + \
                      self.config.rotation_step_final * progress

        return center_placements(best)

    # ==================== L-BFGS-B OPTIMIZATION ====================

    def _objective_function(self, x: np.ndarray, n: int) -> float:
        """Objective function for scipy optimization."""
        solution = self._array_to_solution(x, n)
        return self._compute_penalty_score(solution, self.config.lbfgsb_penalty)

    def lbfgsb_optimize(self, solution: List[Tuple]) -> List[Tuple]:
        """
        Use L-BFGS-B for gradient-based local refinement.
        This finds the nearest local minimum efficiently.
        """
        n = len(solution)
        if n <= 1:
            return solution

        x0 = self._solution_to_array(solution)

        # Bounds: x,y in [-100, 100], angle in [0, 360]
        bounds = []
        for _ in range(n):
            bounds.extend([(-99, 99), (-99, 99), (0, 360)])

        try:
            result = minimize(
                lambda x: self._objective_function(x, n),
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={
                    'maxiter': self.config.lbfgsb_iterations,
                    'ftol': 1e-9,
                    'gtol': 1e-7,
                }
            )

            candidate = self._array_to_solution(result.x, n)

            # Only accept if valid and better
            if self._is_valid_solution(candidate):
                cand_score = compute_bounding_square_side(center_placements(candidate))
                orig_score = compute_bounding_square_side(center_placements(solution))
                if cand_score < orig_score:
                    return center_placements(candidate)
        except Exception:
            pass

        return solution

    # ==================== DIFFERENTIAL EVOLUTION ====================

    def differential_evolution_optimize(self, solution: List[Tuple]) -> List[Tuple]:
        """
        Use Differential Evolution for global optimization.
        Good for escaping local minima.
        """
        n = len(solution)
        if n <= 1 or n > 30:  # Only for smaller problems due to complexity
            return solution

        # Get current bounding box to set search bounds
        min_x, min_y, max_x, max_y = compute_bounding_box(solution)
        margin = 0.5

        bounds = []
        for _ in range(n):
            bounds.extend([
                (min_x - margin, max_x + margin),
                (min_y - margin, max_y + margin),
                (0, 360)
            ])

        try:
            result = differential_evolution(
                lambda x: self._objective_function(x, n),
                bounds,
                seed=self.config.seed,
                maxiter=self.config.de_max_iter,
                popsize=self.config.de_pop_size,
                mutation=self.config.de_mutation,
                recombination=self.config.de_recombination,
                polish=False,  # We'll do our own polishing
                workers=1,
                updating='deferred',
                init='latinhypercube',
            )

            candidate = self._array_to_solution(result.x, n)

            if self._is_valid_solution(candidate):
                cand_score = compute_bounding_square_side(center_placements(candidate))
                orig_score = compute_bounding_square_side(center_placements(solution))
                if cand_score < orig_score:
                    return center_placements(candidate)
        except Exception:
            pass

        return solution

    # ==================== BASIN HOPPING ====================

    def basin_hopping_optimize(self, solution: List[Tuple]) -> List[Tuple]:
        """
        Basin hopping for escaping local minima through random perturbations
        followed by local minimization.
        """
        n = len(solution)
        if n <= 1 or n > 50:  # Only for medium-sized problems
            return solution

        x0 = self._solution_to_array(solution)

        class RandomDisplacement:
            def __init__(self, stepsize=0.2):
                self.stepsize = stepsize

            def __call__(self, x):
                n = len(x) // 3
                x_new = x.copy()
                for i in range(n):
                    x_new[3*i] += np.random.uniform(-self.stepsize, self.stepsize)
                    x_new[3*i + 1] += np.random.uniform(-self.stepsize, self.stepsize)
                    # Occasionally change rotation
                    if np.random.random() < 0.3:
                        x_new[3*i + 2] = np.random.choice([0, 60, 90, 120, 180, 240, 270, 300])
                return x_new

        try:
            result = basinhopping(
                lambda x: self._objective_function(x, n),
                x0,
                niter=self.config.basin_niter,
                T=1.0,
                stepsize=self.config.basin_step_size,
                take_step=RandomDisplacement(self.config.basin_step_size),
                minimizer_kwargs={
                    'method': 'L-BFGS-B',
                    'options': {'maxiter': 30}
                },
                seed=self.config.seed,
            )

            candidate = self._array_to_solution(result.x, n)

            if self._is_valid_solution(candidate):
                cand_score = compute_bounding_square_side(center_placements(candidate))
                orig_score = compute_bounding_square_side(center_placements(solution))
                if cand_score < orig_score:
                    return center_placements(candidate)
        except Exception:
            pass

        return solution

    # ==================== ANGULAR SWEEP ====================

    def angular_sweep_optimize(self, solution: List[Tuple]) -> List[Tuple]:
        """
        Systematically try different rotation angles for each piece
        to find optimal orientations.
        """
        n = len(solution)
        if n <= 1:
            return solution

        current = list(solution)
        current_score = compute_bounding_square_side(center_placements(current))

        angles_to_try = np.linspace(0, 360, self.config.angular_sweep_points, endpoint=False)

        improved = True
        while improved:
            improved = False
            for i in range(n):
                x, y, _ = current[i]
                best_angle = current[i][2]
                best_local_score = current_score

                for angle in angles_to_try:
                    candidate = list(current)
                    candidate[i] = (x, y, angle)

                    if not self._is_valid_solution(candidate):
                        continue

                    cand_score = compute_bounding_square_side(center_placements(candidate))
                    if cand_score < best_local_score - 1e-9:
                        best_local_score = cand_score
                        best_angle = angle

                if best_angle != current[i][2]:
                    current[i] = (x, y, best_angle)
                    current_score = best_local_score
                    improved = True

        # Fine-tune with smaller steps if enabled
        if self.config.angular_fine_tune:
            for i in range(n):
                x, y, d = current[i]
                for delta in [-2, -1, 1, 2]:
                    candidate = list(current)
                    candidate[i] = (x, y, normalize_angle(d + delta))

                    if self._is_valid_solution(candidate):
                        cand_score = compute_bounding_square_side(center_placements(candidate))
                        if cand_score < current_score - 1e-9:
                            current[i] = (x, y, normalize_angle(d + delta))
                            current_score = cand_score

        return center_placements(current)

    # ==================== MULTI-RESOLUTION SEARCH ====================

    def multi_resolution_optimize(self, solution: List[Tuple]) -> List[Tuple]:
        """
        Multi-resolution optimization: start with coarse moves, refine progressively.
        """
        n = len(solution)
        if n <= 1:
            return solution

        current = list(solution)

        for level in range(self.config.multi_res_levels):
            # Scale decreases with each level
            scale = 2.0 ** (self.config.multi_res_levels - level - 1)
            trans_step = 0.1 * scale
            rot_step = 30 * scale

            # Run SA at this resolution level
            iterations = self.config.sa_iterations_base // (level + 1)

            temp = self.config.sa_temp_initial / (level + 1)
            current_score = compute_bounding_square_side(center_placements(current))

            for _ in range(iterations):
                # Random move
                idx = random.randint(0, n - 1)
                x, y, d = current[idx]

                if random.random() < 0.6:
                    # Translation
                    new_p = (x + random.gauss(0, trans_step),
                            y + random.gauss(0, trans_step), d)
                else:
                    # Rotation
                    new_p = (x, y, normalize_angle(d + random.gauss(0, rot_step)))

                candidate = list(current)
                candidate[idx] = new_p

                if not self._is_valid_solution(candidate):
                    continue

                cand_score = compute_bounding_square_side(center_placements(candidate))
                delta = cand_score - current_score

                if delta < 0 or (temp > 0 and random.random() < math.exp(-delta / temp)):
                    current = candidate
                    current_score = cand_score

                temp *= 0.999

        return center_placements(current)

    # ==================== ADVANCED LOCAL SEARCH ====================

    def gradient_descent_local(self, solution: List[Tuple], max_iter: int = 500) -> List[Tuple]:
        """
        Custom gradient-free local search with intelligent neighbor generation.
        """
        n = len(solution)
        if n <= 1:
            return solution

        current = list(solution)
        current_score = compute_bounding_square_side(center_placements(current))

        step = 0.02
        no_improve = 0

        for iteration in range(max_iter):
            if no_improve > 50:
                break

            improved = False

            # Try each piece
            for idx in range(n):
                x, y, d = current[idx]

                # Try 8 directions + 4 rotations
                moves = [
                    (x + step, y, d),
                    (x - step, y, d),
                    (x, y + step, d),
                    (x, y - step, d),
                    (x + step * 0.7, y + step * 0.7, d),
                    (x - step * 0.7, y + step * 0.7, d),
                    (x + step * 0.7, y - step * 0.7, d),
                    (x - step * 0.7, y - step * 0.7, d),
                    (x, y, normalize_angle(d + 5)),
                    (x, y, normalize_angle(d - 5)),
                    (x, y, normalize_angle(d + 15)),
                    (x, y, normalize_angle(d - 15)),
                ]

                for move in moves:
                    candidate = list(current)
                    candidate[idx] = move

                    if not self._is_valid_solution(candidate):
                        continue

                    cand_score = compute_bounding_square_side(center_placements(candidate))
                    if cand_score < current_score - 1e-9:
                        current[idx] = move
                        current_score = cand_score
                        improved = True
                        break

                if improved:
                    break

            if improved:
                no_improve = 0
            else:
                no_improve += 1
                step *= 0.95  # Reduce step size
                step = max(step, 0.001)

        return center_placements(current)

    # ==================== MASTER OPTIMIZATION ====================

    def optimize(self, solution: List[Tuple], iterations_mult: float = 1.0,
                 verbose: bool = False) -> List[Tuple]:
        """
        Master optimization routine combining all techniques.
        """
        n = len(solution)
        if n <= 1:
            return center_placements(solution)

        current = list(solution)
        best = list(current)
        best_score = compute_bounding_square_side(center_placements(best))

        n_iter = int(self.config.sa_iterations_base * iterations_mult)

        # Phase 1: Multi-resolution search (optional)
        if self.config.multi_res_enabled and n <= 100:
            current = self.multi_resolution_optimize(current)
            score = compute_bounding_square_side(center_placements(current))
            if score < best_score:
                best = list(current)
                best_score = score

        # Phase 2: Simulated Annealing (main optimization)
        current = self.simulated_annealing(best, n_iter)
        score = compute_bounding_square_side(center_placements(current))
        if score < best_score:
            best = list(current)
            best_score = score

        # Phase 3: Differential Evolution (global search)
        if self.config.de_enabled and n <= 30:
            current = self.differential_evolution_optimize(best)
            score = compute_bounding_square_side(center_placements(current))
            if score < best_score:
                best = list(current)
                best_score = score

        # Phase 4: Basin Hopping (escape local minima)
        if self.config.basin_enabled and n <= 50:
            current = self.basin_hopping_optimize(best)
            score = compute_bounding_square_side(center_placements(current))
            if score < best_score:
                best = list(current)
                best_score = score

        # Phase 5: L-BFGS-B (gradient-based refinement)
        if self.config.lbfgsb_enabled:
            current = self.lbfgsb_optimize(best)
            score = compute_bounding_square_side(center_placements(current))
            if score < best_score:
                best = list(current)
                best_score = score

        # Phase 6: Angular sweep (rotation optimization)
        current = self.angular_sweep_optimize(best)
        score = compute_bounding_square_side(center_placements(current))
        if score < best_score:
            best = list(current)
            best_score = score

        # Phase 7: Final local search polish
        current = self.gradient_descent_local(best)
        score = compute_bounding_square_side(center_placements(current))
        if score < best_score:
            best = list(current)
            best_score = score

        return center_placements(best)


# ==================== ADVANCED PLACEMENT STRATEGIES ====================

def create_hexagonal_interlocking(n: int) -> List[Tuple]:
    """
    Create initial placement using hexagonal interlocking pattern.
    Trees alternate orientation to interlock efficiently.
    """
    if n == 0:
        return []
    if n == 1:
        return [(0.0, 0.0, 0.0)]

    placements = []

    # Hexagonal packing parameters
    # Tree is ~0.5 wide, ~1.0 tall
    dx = 0.45  # Horizontal spacing
    dy = 0.42  # Vertical spacing (tighter due to interlocking)

    # Calculate grid dimensions
    cols = int(math.ceil(math.sqrt(n * 1.2)))  # Slightly wider than tall

    idx = 0
    row = 0
    while idx < n:
        # Offset for hexagonal pattern
        x_offset = (row % 2) * (dx / 2)

        col = 0
        while idx < n and col < cols:
            x = col * dx + x_offset
            y = row * dy

            # Rotation pattern for interlocking
            # Alternate between 0 and 180 degrees based on position
            if row % 2 == 0:
                deg = 0.0 if col % 2 == 0 else 180.0
            else:
                deg = 180.0 if col % 2 == 0 else 0.0

            placements.append((x, y, deg))
            idx += 1
            col += 1

        row += 1

    return center_placements(placements)


def create_diamond_pattern(n: int) -> List[Tuple]:
    """
    Create initial placement using diamond/rhombic pattern.
    Good for minimizing bounding square.
    """
    if n == 0:
        return []
    if n == 1:
        return [(0.0, 0.0, 0.0)]

    placements = []

    # Diamond pattern parameters
    spacing = 0.48

    # Build in a diamond shape from center outward
    layers = int(math.ceil(math.sqrt(n)))

    idx = 0
    for layer in range(layers + 1):
        if idx >= n:
            break

        if layer == 0:
            placements.append((0.0, 0.0, 0.0))
            idx += 1
        else:
            # Add pieces in a ring around center
            for side in range(4):
                for pos in range(layer):
                    if idx >= n:
                        break

                    # Calculate position on diamond edge
                    if side == 0:
                        x = (layer - pos) * spacing
                        y = pos * spacing
                    elif side == 1:
                        x = -pos * spacing
                        y = (layer - pos) * spacing
                    elif side == 2:
                        x = -(layer - pos) * spacing
                        y = -pos * spacing
                    else:
                        x = pos * spacing
                        y = -(layer - pos) * spacing

                    # Skip if already placed nearby
                    if any(abs(p[0] - x) < spacing/2 and abs(p[1] - y) < spacing/2
                           for p in placements):
                        continue

                    deg = 180.0 if (idx % 2 == 1) else 0.0
                    placements.append((x, y, deg))
                    idx += 1

    return center_placements(placements[:n])


def create_optimal_interlocking(n: int) -> List[Tuple]:
    """
    Create placement optimized for tree shape interlocking.
    Uses the tree geometry to find optimal arrangements.
    """
    if n == 0:
        return []
    if n == 1:
        return [(0.0, 0.0, 0.0)]

    placements = []

    # The tree shape: top triangle + trunk
    # Key insight: trees can nest when one is rotated 180 degrees
    # The trunk of one can fit between the branches of another

    # Tight packing parameters
    dx = 0.40  # Trees are 0.5 wide, slight overlap possible with rotation
    dy = 0.45  # Vertical spacing

    # Use a brick pattern (offset rows)
    cols = int(math.ceil(math.sqrt(n * 1.1)))

    idx = 0
    row = 0
    while idx < n:
        x_offset = (row % 2) * (dx * 0.5)

        col = 0
        while idx < n and col < cols:
            x = col * dx + x_offset
            y = row * dy

            # Interlocking rotation: alternate to allow nesting
            # Row 0: 0, 180, 0, 180, ...
            # Row 1: 180, 0, 180, 0, ...
            base_rot = 0 if (row + col) % 2 == 0 else 180

            placements.append((x, y, float(base_rot)))
            idx += 1
            col += 1

        row += 1

    return center_placements(placements)


def create_spiral_compact(n: int) -> List[Tuple]:
    """
    Compact spiral placement that maintains tight packing.
    """
    if n == 0:
        return []
    if n == 1:
        return [(0.0, 0.0, 0.0)]

    placements = [(0.0, 0.0, 0.0)]

    # Archimedean spiral with tight spacing
    a = 0.0  # Start at center
    b = 0.12  # Spiral growth rate (tighter)

    angle = 0.0
    angle_step = 0.8  # Radians between placements

    for i in range(1, n):
        # Find valid position along spiral
        for _ in range(100):  # Try up to 100 times
            angle += angle_step
            r = a + b * angle
            x = r * math.cos(angle)
            y = r * math.sin(angle)

            # Rotation for interlocking
            deg = 0.0 if i % 2 == 0 else 180.0

            # Check if valid (no collision with existing)
            poly = transform_tree(x, y, deg)
            valid = True
            for px, py, pd in placements:
                if check_overlap(poly, transform_tree(px, py, pd)):
                    valid = False
                    break

            if valid:
                placements.append((x, y, deg))
                break

            # Increase angle step if collision
            angle_step *= 1.02
        else:
            # Fallback: place at larger radius
            r = max(r * 1.5, 2.0)
            x = r * math.cos(angle)
            y = r * math.sin(angle)
            placements.append((x, y, 0.0))

    return center_placements(placements)


def find_best_initial_placement(n: int) -> List[Tuple]:
    """
    Try multiple placement strategies and return the best one.
    """
    if n <= 1:
        return [(0.0, 0.0, 0.0)] if n == 1 else []

    strategies = [
        create_hexagonal_interlocking,
        create_diamond_pattern,
        create_optimal_interlocking,
        create_spiral_compact,
    ]

    best = None
    best_score = float('inf')

    for strategy in strategies:
        try:
            placement = strategy(n)
            if len(placement) != n:
                continue

            # Validate
            detector = CollisionDetector()
            valid = True
            for x, y, d in placement:
                poly = transform_tree(x, y, d)
                if detector.check_collision(poly):
                    valid = False
                    break
                detector.add_polygon(poly)

            if not valid:
                continue

            score = compute_bounding_square_side(placement)
            if score < best_score:
                best_score = score
                best = placement
        except Exception:
            continue

    return best if best else create_hexagonal_interlocking(n)


# ==================== HELPER FUNCTIONS ====================

def optimize_placement_advanced(
    solution: List[Tuple],
    config: Optional[AdvancedConfig] = None,
    iterations_multiplier: float = 1.0,
    verbose: bool = False
) -> List[Tuple]:
    """
    Main entry point for advanced optimization.
    """
    if config is None:
        config = AdvancedConfig()

    optimizer = AdvancedOptimizer(config)
    return optimizer.optimize(solution, iterations_multiplier, verbose)
