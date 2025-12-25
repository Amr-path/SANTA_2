# fix_overlaps.py - Auto-fix script
import os

# Fix geometry.py
print("Fixing geometry.py...")
geometry_code = '''"""FIXED Geometry - Strict collision detection"""
import numpy as np
from shapely.geometry import Polygon
from shapely.affinity import rotate, translate
from shapely.prepared import prep
from shapely.strtree import STRtree

EPSILON = 1e-5  # FIXED
SAFETY_MARGIN = 0.01

def get_tree_polygon():
    coords = [(0.0,1.0),(-0.5,0.0),(-0.12,0.0),(-0.12,-0.3),(0.12,-0.3),(0.12,0.0),(0.5,0.0)]
    poly = Polygon(coords)
    return poly if poly.is_valid else poly.buffer(0)

def transform_tree(tree_polygon, x, y, deg):
    rotated = rotate(tree_polygon, deg, origin=(0,0), use_radians=False)
    return translate(rotated, xoff=x, yoff=y)

def get_bounding_square_side(placements):
    if not placements: return 0.0
    tree_base = get_tree_polygon()
    min_x = max_x = min_y = max_y = None
    for x,y,deg in placements:
        tree = transform_tree(tree_base, x, y, deg)
        bounds = tree.bounds
        if min_x is None:
            min_x, min_y, max_x, max_y = bounds[0], bounds[1], bounds[2], bounds[3]
        else:
            min_x, min_y = min(min_x, bounds[0]), min(min_y, bounds[1])
            max_x, max_y = max(max_x, bounds[2]), max(max_y, bounds[3])
    return max(max_x - min_x, max_y - min_y)

def check_collision_strict(tree1, tree2):
    b1, b2 = tree1.bounds, tree2.bounds
    m = SAFETY_MARGIN
    if (b1[2]+m < b2[0] or b1[0] > b2[2]+m or b1[3]+m < b2[1] or b1[1] > b2[3]+m):
        return False
    if not tree1.intersects(tree2): return False
    return tree1.intersection(tree2).area > EPSILON

def check_all_collisions_fast(placements):
    if len(placements) <= 1: return False, []
    tree_base = get_tree_polygon()
    trees = [(i, transform_tree(tree_base, x, y, deg)) for i, (x, y, deg) in enumerate(placements)]
    tree_index = STRtree([tree for _, tree in trees])
    collision_pairs = []
    for i, tree_i in trees:
        for idx in tree_index.query(tree_i):
            j, tree_j = trees[idx]
            if j > i and check_collision_strict(tree_i, tree_j):
                collision_pairs.append((i, j))
    return len(collision_pairs) > 0, collision_pairs

def check_single_collision_fast(new_tree, existing_trees, spatial_index=None):
    if not existing_trees: return False
    if spatial_index is None: spatial_index = STRtree(existing_trees)
    for idx in spatial_index.query(new_tree):
        if check_collision_strict(new_tree, existing_trees[idx]): return True
    return False

def separate_overlapping_trees(placements, separation=0.2):
    tree_base = get_tree_polygon()
    fixed = list(placements)
    for _ in range(50):
        has_collision, pairs = check_all_collisions_fast(fixed)
        if not has_collision: return fixed
        i, j = pairs[0]
        ti, tj = transform_tree(tree_base, *fixed[i]), transform_tree(tree_base, *fixed[j])
        ci, cj = ti.centroid, tj.centroid
        dx, dy = cj.x - ci.x, cj.y - ci.y
        dist = np.sqrt(dx**2 + dy**2)
        if dist < 0.001: dx, dy, dist = np.cos(np.random.uniform(0, 2*np.pi)), np.sin(np.random.uniform(0, 2*np.pi)), 1.0
        dx, dy = dx/dist, dy/dist
        fixed[i] = (fixed[i][0] - dx*separation/2, fixed[i][1] - dy*separation/2, fixed[i][2])
        fixed[j] = (fixed[j][0] + dx*separation/2, fixed[j][1] + dy*separation/2, fixed[j][2])
    return placements

def is_within_bounds(x, y, max_coord=100): return abs(x) <= max_coord and abs(y) <= max_coord
def normalize_angle(deg): return deg % 360 if deg % 360 >= 0 else (deg % 360) + 360
def get_polygon_center(polygon): return (polygon.centroid.x, polygon.centroid.y)
def polygons_to_spatial_index(polygons): return STRtree(polygons) if polygons else None
def get_minimum_safe_distance():
    tree = get_tree_polygon()
    b = tree.bounds
    return np.sqrt((b[2]-b[0])**2 + (b[3]-b[1])**2) * 1.2
'''

with open('src/geometry.py', 'w') as f:
    f.write(geometry_code)
print("✓ Fixed geometry.py")

# Fix packing.py
print("Fixing packing.py...")
with open('src/packing.py', 'r') as f:
    packing = f.read()

packing = packing.replace('* 1.3', '* 1.6  # FIXED')
with open('src/packing.py', 'w') as f:
    f.write(packing)
print("✓ Fixed packing.py")

print("\n✅ ALL FIXES APPLIED!")
print("\nTest with: python test_local.py")