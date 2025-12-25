"""
geometry.py - Core geometry for Santa 2025 Christmas Tree Packing
"""
import numpy as np
import math
from shapely.geometry import Polygon
from shapely.affinity import rotate, translate
from shapely.prepared import prep
from shapely import STRtree
from typing import List, Tuple, Optional

# Tree polygon vertices (centered at origin)
TREE_VERTICES = np.array([
    [0.0, 0.5],      # Top point
    [-0.25, 0.0],    # Left corner
    [-0.125, 0.0],   # Left trunk top
    [-0.125, -0.5],  # Left trunk bottom
    [0.125, -0.5],   # Right trunk bottom
    [0.125, 0.0],    # Right trunk top
    [0.25, 0.0],     # Right corner
])

def get_tree_polygon() -> Polygon:
    """Get the base tree polygon at origin."""
    return Polygon(TREE_VERTICES)

def get_tree_area() -> float:
    return get_tree_polygon().area

def transform_tree(x: float, y: float, deg: float) -> Polygon:
    """Create transformed tree polygon."""
    tree = get_tree_polygon()
    tree = rotate(tree, deg, origin=(0, 0))
    tree = translate(tree, x, y)
    return tree

def transform_vertices(x: float, y: float, deg: float) -> np.ndarray:
    """Transform tree vertices using numpy."""
    rad = math.radians(deg)
    cos_r, sin_r = math.cos(rad), math.sin(rad)
    rot = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
    return TREE_VERTICES @ rot.T + np.array([x, y])

class CollisionDetector:
    """Efficient collision detection using STRtree."""
    EPSILON = 1e-9
    
    def __init__(self):
        self.polygons: List[Polygon] = []
        self.prepared_polys = []
        self.tree: Optional[STRtree] = None
        self._dirty = True
    
    def clear(self):
        self.polygons = []
        self.prepared_polys = []
        self.tree = None
        self._dirty = True
    
    def add_polygon(self, poly: Polygon):
        self.polygons.append(poly)
        self.prepared_polys.append(prep(poly))
        self._dirty = True
    
    def update_polygon(self, index: int, new_poly: Polygon):
        self.polygons[index] = new_poly
        self.prepared_polys[index] = prep(new_poly)
        self._dirty = True
    
    def _rebuild_tree(self):
        if self._dirty and self.polygons:
            self.tree = STRtree(self.polygons)
            self._dirty = False
    
    def check_collision(self, poly: Polygon, exclude_index: int = -1) -> bool:
        self._rebuild_tree()
        if not self.polygons:
            return False
        
        candidates = self.tree.query(poly)
        for idx in candidates:
            if idx == exclude_index:
                continue
            if self.prepared_polys[idx].intersects(poly):
                intersection = self.polygons[idx].intersection(poly)
                if intersection.area > self.EPSILON:
                    return True
        return False

def check_overlap(p1: Polygon, p2: Polygon, eps: float = 1e-9) -> bool:
    if not p1.intersects(p2):
        return False
    return p1.intersection(p2).area > eps

def check_all_overlaps(placements: List[Tuple[float, float, float]]) -> List[Tuple[int, int]]:
    polys = [transform_tree(x, y, d) for x, y, d in placements]
    overlaps = []
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if check_overlap(polys[i], polys[j]):
                overlaps.append((i, j))
    return overlaps

def compute_bounding_box(placements: List[Tuple[float, float, float]]) -> Tuple[float, float, float, float]:
    if not placements:
        return (0, 0, 0, 0)
    all_v = np.vstack([transform_vertices(x, y, d) for x, y, d in placements])
    return all_v[:, 0].min(), all_v[:, 1].min(), all_v[:, 0].max(), all_v[:, 1].max()

def compute_bounding_square_side(placements: List[Tuple[float, float, float]]) -> float:
    if not placements:
        return 0.0
    min_x, min_y, max_x, max_y = compute_bounding_box(placements)
    return max(max_x - min_x, max_y - min_y)

def center_placements(placements: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
    if not placements:
        return placements
    min_x, min_y, max_x, max_y = compute_bounding_box(placements)
    cx, cy = (min_x + max_x) / 2, (min_y + max_y) / 2
    return [(x - cx, y - cy, d) for x, y, d in placements]

def get_centroid(placements: List[Tuple[float, float, float]]) -> Tuple[float, float]:
    if not placements:
        return (0, 0)
    return sum(p[0] for p in placements) / len(placements), sum(p[1] for p in placements) / len(placements)

def normalize_angle(deg: float) -> float:
    return deg % 360

def placement_in_bounds(x: float, y: float, deg: float, limit: float = 100.0) -> bool:
    v = transform_vertices(x, y, deg)
    return np.all(v >= -limit) and np.all(v <= limit)
