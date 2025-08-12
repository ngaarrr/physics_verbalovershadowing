from __future__ import division
import numpy as np
import json
import convert_coordinate
import csv
import scipy.stats as st
import config
from scipy.spatial import ConvexHull


def generate_ngon(n, rad):
    """
    Generate a regular n-sided polygon (n-gon) with specified number of sides and radius.

    This function creates the vertices of a regular polygon inscribed in a circle.
    The polygon is centered at the origin (0,0) and all vertices lie on a circle
    with the specified radius. The first vertex is placed at the top of the circle
    (positive y-axis) and subsequent vertices are placed clockwise around the circle.

    Special case: When n=4, creates a rectangle instead of a square.

    Parameters:
    ----------
    n : int
            The number of sides/vertices of the polygon. Common values:
            - n=3: triangle
            - n=4: rectangle (width = 1.5 * radius, height = radius)
            - n=5: pentagon
            - n=6: hexagon
            - etc.

    rad : float
            The radius of the circle that circumscribes the polygon.
            For rectangles (n=4), this becomes the height, and width = 1.5 * radius.
            For other polygons, this is the distance from center to each vertex.

    Returns:
    --------
    list of tuples
            A list of (x, y) coordinate tuples representing the vertices of the polygon.
            The vertices are ordered clockwise starting from the top (positive y-axis).
            Each tuple contains (x, y) coordinates where:
            - x: horizontal coordinate (positive = right, negative = left)
            - y: vertical coordinate (positive = up, negative = down)

    Example:
    --------
    >>> generate_ngon(3, 50)  # Triangle with radius 50
    [(0.0, 50.0), (-43.3, -25.0), (43.3, -25.0)]

    >>> generate_ngon(4, 30)  # Rectangle: width=45, height=30
    [(-22.5, 15.0), (22.5, 15.0), (22.5, -15.0), (-22.5, -15.0)]

    Algorithm:
    ----------
    For n=4 (rectangle):
    1. Create rectangle with width = 1.5 * radius, height = radius
    2. Place vertices at corners: top-left, top-right, bottom-right, bottom-left

    For other polygons:
    1. Calculate the angle between consecutive vertices: 2π/n radians
    2. For each vertex i (0 to n-1):
       - Calculate angle = (2π/n) * i
       - x = sin(angle) * radius
       - y = cos(angle) * radius
    3. Return list of (x, y) coordinate pairs

    Note:
    -----
    - For n=4: Creates a rectangle (not a square)
    - For other n: Creates regular polygons (all sides equal length, all angles equal)
    - The polygon is always convex
    - The first vertex is always at the top
    - Vertices are ordered clockwise from the top

    Usage in Physics Simulation:
    ---------------------------
    This function is primarily used to generate obstacle shapes in the Plinko-style
    physics simulation. The obstacles are defined in config.py with properties like:
    - 'n_sides': number of sides (3=triangle, 4=rectangle, 5=pentagon)
    - 'size': radius of the circumscribed circle (for rectangles: height)

    The generated vertices are then used by the Pymunk physics engine to create
    collision shapes for the obstacles that the ball can bounce off of.
    """
    pts = []

    if n == 4:
        width = 3.0 * rad  # Rectangle width (increased from 2.5 to 3.0)
        height = rad  # Rectangle height
        # Create rectangle vertices: top-left, top-right, bottom-right, bottom-left
        pts = [
            (-width / 2, height / 2),  # top-left
            (width / 2, height / 2),  # top-right
            (width / 2, -height / 2),  # bottom-right
            (-width / 2, -height / 2),  # bottom-left
        ]
    else:
        # Regular polygon generation
        ang = 2 * np.pi / n  # Calculate angle between consecutive vertices
        for i in range(n):
            # Calculate position of vertex i
            # sin(angle) gives x-coordinate, cos(angle) gives y-coordinate
            # Multiply by radius to scale the polygon
            pts.append((np.sin(ang * i) * rad, np.cos(ang * i) * rad))

    return pts


def get_vertices(shape):
    vertices = []
    if hasattr(shape, "get_vertices"):
        # Polygon shape
        for v in shape.get_vertices():
            x, y = v.rotated(shape.body.angle) + shape.body.position
            vertices.append((x, y))
    else:
        # Circle shape - create a polygon approximation
        center = shape.body.position
        radius = shape.radius
        num_points = 8  # 8-sided polygon approximation
        for i in range(num_points):
            angle = 2 * np.pi * i / num_points
            x = center.x + radius * np.cos(angle)
            y = center.y + radius * np.sin(angle)
            vertices.append((x, y))
    return vertices


def radius_reg_poly(side_length, n):
    return side_length / (2 * (np.pi / n))


def get_points_on_segment(x1, y1, x2, y2, bound_1, bound_2):
    # divide the segment into 6 parts
    num = 20
    pts = []
    for i in range(num):
        x, y = x1 / num * i + x2 / num * (num - i), y1 / num * i + y2 / num * (num - i)
        if bound_1 <= x <= bound_2:
            x, y = convert_coordinate.convertCoordinate(x, y)
            pts.append({"x": x, "y": y})
    return pts


def get_top_surfaces(pts, center):
    x, y = center
    top_surfaces = []

    for i in range(len(pts) - 1):
        x1, y1, x2, y2 = pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1]
        mid_point = (y1 + y2) / 2
        if mid_point > y:
            if pts[i][0] <= pts[i + 1][0]:
                top_surfaces.append([pts[i], pts[i + 1]])
            else:
                top_surfaces.append([pts[i + 1], pts[i]])

    if len(pts) > 2:
        x1, y1, x2, y2 = pts[0][0], pts[0][1], pts[-1][0], pts[-1][1]
        mid_point = (y1 + y2) / 2
        if mid_point > y:
            if pts[0][0] <= pts[-1][0]:
                top_surfaces.append([pts[0], pts[-1]])
            else:
                top_surfaces.append([pts[-1], pts[0]])

    return top_surfaces


def overlap(s1, s2):
    # check if the x ranges of two surfaces overlap with each other
    x10, x11 = s1[0][0], s1[1][0]
    x20, x21 = s2[0][0], s2[1][0]
    if x10 < x20 < x11:
        return True
    elif x20 < x10 < x21:
        return True
    return False


def remove_overlap(surfaces):
    # if two the x ranges of two surfaces overlap, keep only the one above
    top_surfaces = []

    for i, s1 in enumerate(surfaces):
        bound_1, bound_2 = s1[0][0], s1[1][0]

        # check if its x range overlaps with any other surfaces
        for j, s2 in enumerate(surfaces):
            if i != j:
                # no-op if candidate is higher than s2
                if s1[0][1] > s2[0][1]:
                    continue
                #  ---   s2
                # ---    s1
                if s1[0][0] < s2[0][0] < s1[1][0] < s2[1][0]:
                    bound_2 = min(bound_2, s2[0][0])
                #  ---   s2
                #   ---  s1
                elif s2[0][0] < s1[0][0] < s2[1][0] < s1[1][0]:
                    bound_1 = max(bound_1, s2[1][0])
                #  ----   s2
                #   --    s1
                elif s2[0][0] < s1[0][0] < s1[1][0] < s2[1][0]:
                    bound_1, bound_2 = 1, -1
                    break
        if bound_1 < bound_2:
            top_surfaces.append([s1, (bound_1, bound_2)])

    result = []
    for s in top_surfaces:
        x1, y1, x2, y2, bound_1, bound_2 = (
            s[0][0][0],
            s[0][0][1],
            s[0][1][0],
            s[0][1][1],
            s[1][0],
            s[1][1],
        )
        result += get_points_on_segment(x1, y1, x2, y2, bound_1, bound_2)

    # result is of the form [{'x': x1, 'y': y1}, {'x': x2, 'y': y2},...]
    return result


def gaussian_noise(mean=0, sd=1):
    """Apply gaussian noise"""
    return np.random.normal(mean, sd)


def loss(prop, target, sd=100):
    """Gaussian loss"""
    return -(((prop - target) / sd) ** 2)


def flipy(c, y):
    """Small hack to convert chipmunk physics to pygame coordinates"""
    return -y + c["screen_size"]["height"]


def load_config(name):
    c = config.get_config()

    with open(name, "rb") as f:
        ob = json.load(f)

        # PARAMETERS
    c["drop_noise"] = ob["parameters"]["drop_noise"]
    c["collision_noise_mean"] = ob["parameters"]["collision_noise_mean"]
    c["collision_noise_sd"] = ob["parameters"]["collision_noise_sd"]
    # c['loss_sd_vision'] = ob['parameters']['loss_sd_vision']
    # c['loss_sd_sound'] = ob['parameters']['loss_sd_sound']
    # c['loss_penalty_sound'] = ob['parameters']['loss_penalty_sound']

    # GLOBAL SETTINGS
    c["dt"] = ob["global"]["timestep"]
    c["substeps_per_frame"] = ob["global"]["substeps"]
    c["med"] = ob["global"]["midpoint"]
    c["gravity"] = ob["global"]["gravity"]
    c["screen_size"] = ob["global"]["screen_size"]
    c["hole_dropped_into"] = ob["global"]["hole_dropped_into"] - 1
    # c['hole_dropped_into'] = ob['global']['hole_dropped_into']

    # PLINKO BOX SETTINGS
    c["width"] = ob["box"]["width"]
    c["height"] = ob["box"]["height"]
    c["hole_width"] = ob["box"]["holes"]["width"]
    c["hole_positions"] = ob["box"]["holes"]["positions"]
    c["wall_elasticity"] = ob["box"]["walls"]["elasticity"]
    c["wall_friction"] = ob["box"]["walls"]["friction"]
    c["ground_elasticity"] = ob["box"]["ground"]["elasticity"]
    c["ground_friction"] = ob["box"]["ground"]["friction"]
    c["ground_y"] = ob["box"]["ground"]["position"]["y"]

    # BALL SETTINGS
    c["ball_radius"] = ob["ball"]["radius"]
    c["ball_mass"] = ob["ball"]["mass"]
    c["ball_elasticity"] = ob["ball"]["elasticity"]
    c["ball_friction"] = ob["ball"]["friction"]

    # OBSTACLE SETTINGS
    c["obstacles"] = ob["obstacles"]

    # BALL FINAL POSITION
    c["ball_final_position"] = ob["simulation"][c["hole_dropped_into"]]["ball"][
        "position"
    ][-1]
    c["paths"] = [x["ball"]["position"] for x in ob["simulation"]]

    return c


def combine_video_and_audio(video, audio, filename="test_with_sound.mp4"):
    """
    Function to combine video and audio.
    video = path to mp4 file
    audio = path to wav file
    filename = path to created file
    """
    import subprocess as sp

    sp.call(
        "ffmpeg -i {video} -i {audio} -c:v copy -c:a aac -strict experimental {filename}".format(
            video=video, audio=audio, filename=filename
        ),
        shell=True,
    )


def load_cache_eye_data_heatmap(world):
    """
    Load the cached heatmap of
    """
    z = np.loadtxt(
        open(
            "../../data/cached_eye_data_heatmap_matrix/world_"
            + str(int(world))
            + ".csv",
            "rb",
        ),
        delimiter=",",
        skiprows=0,
    )
    return z


def save_eye_data_matrix(z, world):
    """
    Save an eye data matrix to csv
    """
    np.savetxt(
        "../../data/cached_eye_data_heatmap_matrix/world_" + str(int(world)) + ".csv",
        z,
        delimiter=",",
    )


def kde_scipy(vals1, vals2, r1, r2, N1, N2, w):
    # vals1, vals2 are the values of two variables (columns)
    # (a,b) interval for vals1; usually larger than (np.min(vals1), np.max(vals1))
    # (c,d) -"-          vals2
    (a, b) = r1
    (c, d) = r2
    x = np.linspace(a, b, N1)
    y = np.linspace(c, d, N2)
    X, Y = np.meshgrid(x, y)
    positions = np.vstack([Y.ravel(), X.ravel()])

    values = np.vstack([vals1, vals2])
    kernel = st.gaussian_kde(values, weights=w)

    Z = np.reshape(kernel(positions).T, X.shape)
    return [x, y, Z]


def get_heatmap(df, ybins=600, xbins=500):
    df = df[(df["x"] >= 0) & (df["x"] < 600) & (df["y"] >= 0) & (df["y"] < 500)].copy()
    weights = None
    if "dur" in df:
        weights = df["dur"]
    return kde_scipy(df["y"], df["x"], (0, 600), (0, 500), ybins, xbins, weights)


def get_average_zvalue(df, exps):
    # avg kde across all worlds for vision experiment
    # df_vision = df[(df['experiment']=='vision']
    # _, _, z_vision = get_heatmap(df_vision, ybins=60, xbins=50)
    z_vision = None

    # WARNINGS!! COMMENT OUT TO SAVE TIME
    z_vision_sound = None
    # avg kde across all worlds for vision_sound experiment
    df_combined = df[(df["experiment"] == "vision_sound")]
    _, _, z_vision_sound = get_heatmap(df_combined, ybins=60, xbins=50)

    return z_vision, z_vision_sound


def get_adjusted_heatmap(df, z_vision, z_vision_sound, exp, ybins=600, xbins=500):
    x, y, z = get_heatmap(df, ybins, xbins)

    # subtract mean kde map from current world's kde, create mask: 0 - noise, 1 - non noise
    if exp == "vision":
        z_adjusted = z - z_vision
    elif exp == "vision_sound":
        z_adjusted = z - z_vision_sound
    mask = np.where(z_adjusted < 0, 0, 1)

    # filter fixation data using mask
    df = df[(df["x"] >= 0) & (df["x"] < 600) & (df["y"] >= 0) & (df["y"] < 500)].copy()
    df["mask"] = df.apply(
        lambda row: mask[int(row["y"] / 10)][int(row["x"] / 10)] > 0, axis=1
    )
    df = df[df["mask"] > 0]
    weights = None
    if "dur" in df:
        weights = df["dur"]
    xp, yp, zp = kde_scipy(df["y"], df["x"], (0, 600), (0, 500), 600, 500, weights)

    return xp, yp, zp, df


def generate_abstract_pentagon(
    center=(0, 0), n_branches=5, min_len=60, max_len=120, min_angle=20, max_angle=100
):
    """
    Generate a random, abstract pentagon-like set of 5 points (not connected).
    Each point is a random distance and angle from the center, forming a pentagon-like scatter.
    Returns a list of (x, y) tuples representing the points.
    """
    import random
    import math

    cx, cy = center
    angles = []
    angle = random.uniform(0, 360)
    for _ in range(n_branches):
        angles.append(angle)
        angle += random.uniform(min_angle, max_angle)
    angles = [a % 360 for a in angles]
    angles.sort()
    points = []
    for a in angles:
        length = random.uniform(min_len, max_len)
        rad = math.radians(a)
        x = cx + length * math.cos(rad)
        y = cy + length * math.sin(rad)
        points.append((x, y))
    return points


def generate_abstract_obstacle(center, box_width, box_height, size=45, steps=32):
    """
    Generate a closed polygon using Fourier descriptor logic (natural shape).
    The polygon is centered at 'center' and scaled so its maximum radius is 'size'.
    Returns a list of (x, y) tuples suitable for pymunk.Poly.
    """

    def build_fd_pattern(total_frequencies=14):
        amp = np.zeros(total_frequencies)
        phase = np.zeros(total_frequencies)
        even_pool = lambda n: np.random.choice(
            np.arange(2, total_frequencies + 1, 2), size=n, replace=False
        )
        for idx in even_pool(4):
            amp[idx - 1] = np.random.uniform(0.6, 1.4)
            phase[idx - 1] = np.pi / 2
        for k in range(8, total_frequencies + 1):
            amp[k - 1] *= 0.5 * (8 / k) ** 2
        return {"Amplitude": amp, "Phase": phase}

    def cumbend(fd, t):
        amp = fd["Amplitude"]
        phase = fd["Phase"]
        theta = -t
        for k, (A, P) in enumerate(zip(amp, phase), start=1):
            theta += A * np.cos(k * t - P)
        return theta

    def cumbend_to_points(fd, steps=steps):
        t_vals = np.linspace(0, 2 * np.pi, steps, endpoint=False)
        x, y = [0.0], [0.0]
        cx, cy = 0.0, 0.0
        for t in t_vals:
            angle = cumbend(fd, t)
            cx += np.cos(angle)
            cy += np.sin(angle)
            x.append(cx)
            y.append(cy)
        x = np.array(x)
        y = np.array(y)
        x -= np.mean(x)
        y -= np.mean(y)
        # Scale so max radius is 1
        radii = np.sqrt(x**2 + y**2)
        max_radius = np.max(radii)
        x = x / max_radius
        y = y / max_radius
        return x, y

    fd = build_fd_pattern()
    x, y = cumbend_to_points(fd, steps=steps)
    # Scale to desired size
    x = x * size + center[0]
    y = y * size + center[1]
    return list(zip(x, y))


def create_abstract_collision_shape(arms):
    """
    Create a collision shape from abstract arms by creating a convex hull.
    This ensures the ball will collide with the visual abstract obstacle.

    Parameters:
    -----------
    arms : list
        List of arms, each arm is [(x0, y0), (x1, y1), (x2, y2)]

    Returns:
    --------
    list of tuples
        Vertices of the collision polygon
    """
    # Collect all points from the arms
    all_points = []
    for arm in arms:
        # Add the hub point (center)
        all_points.append(arm[0])
        # Add the tip point (end of arm)
        all_points.append(arm[2])

    # Create a simple convex hull by finding the extreme points
    if len(all_points) < 3:
        # If we don't have enough points, create a small circle
        center = arms[0][0] if arms else (0, 0)
        radius = 20
        return [
            (center[0] + radius, center[1]),
            (center[0], center[1] + radius),
            (center[0] - radius, center[1]),
            (center[0], center[1] - radius),
        ]

    # Find the center (average of all points)
    center_x = sum(p[0] for p in all_points) / len(all_points)
    center_y = sum(p[1] for p in all_points) / len(all_points)
    center = (center_x, center_y)

    # Find the point furthest from center
    max_dist = 0
    max_point = center
    for point in all_points:
        dist = ((point[0] - center[0]) ** 2 + (point[1] - center[1]) ** 2) ** 0.5
        if dist > max_dist:
            max_dist = dist
            max_point = point

    # Create a polygon by connecting points at regular angles around the center
    # Use 8 points to create a reasonable collision shape
    num_points = 8
    vertices = []
    for i in range(num_points):
        angle = 2 * np.pi * i / num_points
        # Use the max distance to ensure we cover all arms
        x = center[0] + max_dist * np.cos(angle)
        y = center[1] + max_dist * np.sin(angle)
        vertices.append((x, y))

    return vertices


def create_abstract_collision_shape_relative(arms, center):
    """
    Create a simple circular collision shape from abstract arms.
    This ensures reliable collision detection with Pymunk.

    Parameters:
    -----------
    arms : list
        List of arms, each arm is [(x0, y0), (x1, y1), (x2, y2)]
    center : tuple
        The center position (x, y) of the pentagon

    Returns:
    --------
    list of tuples
        Vertices of a circular collision polygon, relative to (0,0)
    """
    cx, cy = center

    # Find the maximum distance from center to any arm tip
    max_dist = 0
    for arm in arms:
        # Check distance from center to tip of arm
        tip_x, tip_y = arm[2]
        dist = ((tip_x - cx) ** 2 + (tip_y - cy) ** 2) ** 0.5
        if dist > max_dist:
            max_dist = dist

    # Add some padding to ensure we catch all collisions
    collision_radius = max_dist + 10

    # Create a simple octagon (8-sided polygon) for collision
    # This is simpler and more reliable than a complex shape
    num_points = 8
    vertices = []
    for i in range(num_points):
        angle = 2 * np.pi * i / num_points
        x = collision_radius * np.cos(angle)
        y = collision_radius * np.sin(angle)
        vertices.append((x, y))

    return vertices


def get_abstract_collision_radius(arms, center):
    """
    Calculate the collision radius for a abstract shape.

    Parameters:
    -----------
    arms : list
        List of arms, each arm is [(x0, y0), (x1, y1), (x2, y2)]
    center : tuple
        The center position (x, y) of the pentagon

    Returns:
    --------
    float
        The collision radius
    """
    cx, cy = center

    # Find the maximum distance from center to any arm tip
    max_dist = 0
    for arm in arms:
        # Check distance from center to tip of arm
        tip_x, tip_y = arm[2]
        dist = ((tip_x - cx) ** 2 + (tip_y - cy) ** 2) ** 0.5
        if dist > max_dist:
            max_dist = dist

    # Add some padding to ensure we catch all collisions
    return max_dist + 15


def abstract_tips_polygon(arms, center):
    """
    Create a precise collision shape that matches the visual abstract arms.
    Uses the arm tips to create a polygon that closely matches the visual shape.
    """
    cx, cy = center

    # Get the tip of each arm, relative to the center
    tips = [(arm[2][0] - cx, arm[2][1] - cy) for arm in arms]

    # Sort the tips by angle to ensure a valid polygon
    tips_sorted = sorted(tips, key=lambda p: np.arctan2(p[1], p[0]))

    # Add the center point to create a star-like polygon
    # This creates a shape that follows the arm tips but doesn't create a huge invisible area
    vertices = []
    for tip in tips_sorted:
        # Add a point closer to center (not the full tip) to make the collision shape smaller
        # This prevents the invisible force field effect
        scale_factor = 0.7  # Make the collision shape 70% of the visual size
        x = tip[0] * scale_factor
        y = tip[1] * scale_factor
        vertices.append((x, y))

    return vertices


def line_intersection(p1, d1, p2, d2):
    """Find intersection of two lines: p1 + t*d1 and p2 + s*d2"""
    # Solve: p1 + t*d1 = p2 + s*d2
    # => t*d1 - s*d2 = p2 - p1
    # We'll solve for t and s
    A = np.array([d1, -d2]).T
    b = np.array(p2) - np.array(p1)
    if np.linalg.matrix_rank(A) < 2:
        return (p1 + p2) / 2  # fallback: average if lines are parallel
    t, s = np.linalg.solve(A, b)
    return np.array(p1) + t * np.array(d1)


def create_individual_arm_collision_shapes(arms, center, width=24, overlap=0):
    """
    For each arm (hub→bend→tip), create a 6-point polygon that wraps the thick arm,
    using miter joins at the bend for a perfect match. Returns a list of dicts: {'poly': polygon}
    Mirrors the polygons vertically about the center y-coordinate.
    """
    cx, cy = center
    arm_shapes = []
    half_w = width / 2
    for arm in arms:
        # Points in world coordinates
        hub = np.array(arm[0])
        bend = np.array(arm[1])
        tip = np.array(arm[2])
        # Directions
        dir_hb = bend - hub
        dir_bt = tip - bend
        dir_hb = dir_hb / np.linalg.norm(dir_hb)
        dir_bt = dir_bt / np.linalg.norm(dir_bt)
        # Perpendiculars
        perp_hb = np.array([-dir_hb[1], dir_hb[0]])
        perp_bt = np.array([-dir_bt[1], dir_bt[0]])
        # Left and right offset lines at hub, bend, tip
        left_hub = hub + perp_hb * half_w
        right_hub = hub - perp_hb * half_w
        left_tip = tip + perp_bt * half_w
        right_tip = tip - perp_bt * half_w
        # At bend, compute miter join (intersection of offset lines)
        # Left side
        left_bend = line_intersection(
            bend + perp_hb * half_w, dir_hb, bend + perp_bt * half_w, dir_bt
        )
        # Right side
        right_bend = line_intersection(
            bend - perp_hb * half_w, dir_hb, bend - perp_bt * half_w, dir_bt
        )
        # 6-point polygon: left_hub, left_bend, left_tip, right_tip, right_bend, right_hub
        poly = [
            tuple(left_hub),
            tuple(left_bend),
            tuple(left_tip),
            tuple(right_tip),
            tuple(right_bend),
            tuple(right_hub),
        ]
        # Mirror vertically about center y-coordinate
        mirrored_poly = [(x, 2 * cy - y) for (x, y) in poly]
        arm_shapes.append({"poly": mirrored_poly})
    return arm_shapes


def abstract_skeleton_polygon(arms, center):
    """
    Create a single polygon that follows the skeleton of the abstract:
    hub → bend → tip → next bend → next hub, etc., for all arms, forming a closed shape.
    """
    cx, cy = center
    n = len(arms)
    # Collect points: for each arm, get bend and tip
    bends = []
    tips = []
    for arm in arms:
        bends.append((arm[1][0] - cx, arm[1][1] - cy))
        tips.append((arm[2][0] - cx, arm[2][1] - cy))
    # The hub is the center (0,0)
    # Build the polygon: alternate between bends and tips
    poly = []
    for i in range(n):
        poly.append(bends[i])
        poly.append(tips[i])
    return poly


def abstract_convex_hull_polygon(arms, center, width=24):
    """
    Create a single convex hull polygon that covers all thick arm segments of the abstract.
    """
    cx, cy = center
    all_points = []
    for arm in arms:
        pts = [tuple(np.array(p) - np.array([cx, cy])) for p in arm]
        for seg_start, seg_end in zip(pts[:2], pts[1:]):
            dx, dy = seg_end[0] - seg_start[0], seg_end[1] - seg_start[1]
            length = np.hypot(dx, dy)
            angle = np.arctan2(dy, dx)
            cx_seg = (seg_start[0] + seg_end[0]) / 2
            cy_seg = (seg_start[1] + seg_end[1]) / 2
            # Rectangle centered at (0,0), width along y, length along x
            rect = [
                (-length / 2, -width / 2),
                (length / 2, -width / 2),
                (length / 2, width / 2),
                (-length / 2, width / 2),
            ]
            # Rotate and translate rectangle to segment position
            for x, y in rect:
                x_rot = x * np.cos(angle) - y * np.sin(angle)
                y_rot = x * np.sin(angle) + y * np.cos(angle)
                x_final = x_rot + cx_seg
                y_final = y_rot + cy_seg
                all_points.append([x_final, y_final])
    # Compute convex hull
    if len(all_points) < 3:
        return all_points
    hull = ConvexHull(all_points)
    poly = [tuple(all_points[i]) for i in hull.vertices]
    return poly
