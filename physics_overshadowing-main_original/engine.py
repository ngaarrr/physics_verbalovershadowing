# import libraries
from __future__ import division
import sys
from glob import glob
import numpy as np
from scipy.stats import truncnorm
import pymunk
from pymunk import Vec2d
import time

# import files
import utils
import visual
import config
import convert_coordinate

shape_code = {
    "walls": 0,
    "ground": 1,
    "ball": 2,
    "rectangle": 3,
    "triangle": 4,
    "pentagon": 5,
}

inverse_shape_code = {shape_code[key]: key for key in shape_code}

TIMEOUT = 1

VELOCITY_SD = 0.0


def main():
    # c = config.get_config() # generate config
    c = utils.load_config("configs/world_1.json")  # load a config file
    # c = utils.load_config("data/json/world_6541.json")  # load a config file
    c["hole_dropped_into"] = 0
    c["drop_noise"] = 0
    c["falling_noise"] = 0
    c["collision_noise_mean"] = 0
    c["collision_noise_sd"] = 0
    # for i in range(0,5):
    data = run_simulation(c)
    visual.visualize(c, data)


def run_simulation(c, convert_coordinates=False, distorted=False):

    # PHYSICS PARAMETERS
    space = pymunk.Space()
    space.gravity = (0.0, c["gravity"])
    space.convert_coordinates = convert_coordinates

    # CREATE OBJECTS
    make_walls(c, space)
    top_surfaces = make_obstacles(c, space, distorted=distorted)
    ball = make_ball(c, space)
    make_ground(c, space)

    ###############
    ## MAIN LOOP ##
    ###############
    timestep = 0

    all_data = {}

    ball_pos = []
    ball_vel = []

    start = time.time()
    while ball.position.y > 130:  # run into ground collision check (approximately)
        ### Update physics

        for _ in range(c["substeps_per_frame"]):
            # add gaussian noise to ball's velocity at each time step
            space.step(c["dt"] / c["substeps_per_frame"])

        timestep += 1

        # Convert to 3d scene coordinates when flag is set true
        x, y = ball.position.x, ball.position.y
        if convert_coordinates:
            x, y = convert_coordinate.convertCoordinate(x, y)

        ball_pos.append({"x": x, "y": y})
        ball_vel.append({"x": ball.velocity.x, "y": ball.velocity.y})
        # h.data['current_timestep'] = timestep

        if time.time() > start + TIMEOUT:
            drop = {
                "pos": ball_pos[0],
                "step": 0,
                "sd": c["drop_noise"],
                "angle": c["ball_initial_angle"],
            }
            # collisions = clean_collisions(collisions=h.data['collisions'])
            all_data["drop"] = drop
            # all_data['collisions'] = collisions
            all_data["ball_position"] = ball_pos
            all_data["ball_velocity"] = ball_vel
            all_data["top_surfaces"] = top_surfaces
            return all_data

    drop = {
        "pos": ball_pos[0],
        "step": 0,
        "sd": c["drop_noise"],
        "angle": c["ball_initial_angle"],
    }

    all_data["drop"] = drop

    all_data["ball_position"] = ball_pos
    all_data["ball_velocity"] = ball_vel
    all_data["top_surfaces"] = top_surfaces

    return all_data


def make_ball(c, space):
    inertia = pymunk.moment_for_circle(c["ball_mass"], 0, c["ball_radius"], (0, 0))
    body = pymunk.Body(c["ball_mass"], inertia)
    x = c["hole_positions"][c["hole_dropped_into"]]
    y = c["med"] + c["height"] / 2 + c["ball_radius"]
    body.position = x, y
    shape = pymunk.Circle(body, c["ball_radius"], (0, 0))
    shape.elasticity = c["ball_elasticity"]
    shape.friction = c["ball_friction"]

    shape.collision_type = shape_code["ball"]

    space.add(body, shape)

    # used for setting initial velocity (should not be part of the ball definition)
    ang = c["ball_initial_angle"]
    amp = 100
    off = 3 * np.pi / 2
    # so that clockwise is negative
    body.velocity = Vec2d(amp * -np.cos(ang + off), amp * np.sin(ang + off))

    return body


def make_ground(c, space):

    sz = (c["width"], 10)  # 4is for border

    body = pymunk.Body(body_type=pymunk.Body.STATIC)
    body.position = (c["med"], c["ground_y"])

    shape = pymunk.Poly.create_box(body, sz)
    shape.elasticity = 1
    shape.friction = 1

    shape.collision_type = shape_code["ground"]
    space.add(body, shape)


def make_walls(c, space):

    walls = pymunk.Body(body_type=pymunk.Body.STATIC)

    topwall_y = c["med"] + c["height"] / 2
    num_holes = len(c["hole_positions"])

    static_lines = [
        pymunk.Segment(
            walls,
            a=(c["med"] - c["width"] / 2, topwall_y),
            b=(c["hole_positions"][0] - c["hole_width"] / 2, topwall_y),
            radius=2.0,
        ),
    ]
    for wall in range(0, num_holes - 1):
        static_lines.append(
            pymunk.Segment(
                walls,
                a=(c["hole_positions"][wall] + c["hole_width"] / 2, topwall_y),
                b=(c["hole_positions"][wall + 1] - c["hole_width"] / 2, topwall_y),
                radius=2.0,
            )
        )

    # top horizontal: 4
    static_lines.append(
        pymunk.Segment(
            walls,
            a=(c["hole_positions"][num_holes - 1] + c["hole_width"] / 2, topwall_y),
            b=(c["med"] + c["width"] / 2, topwall_y),
            radius=2.0,
        )
    )

    # static_lines = [
    # 	# top horizontal: 1
    # 	pymunk.Segment(walls,
    # 				a = (c['med'] - c['width']/2, topwall_y),
    # 				b = (c['hole_positions'][0] - c['hole_width']/2, topwall_y),
    # 				radius = 2.0),
    # 	# top horizontal: 2
    # 	pymunk.Segment(walls,
    # 				a = (c['hole_positions'][0] + c['hole_width']/2, topwall_y),
    # 				b = (c['hole_positions'][1] - c['hole_width']/2, topwall_y),
    # 				radius = 2.0),
    # 	# top horizontal: 3
    # 	pymunk.Segment(walls,
    # 				a = (c['hole_positions'][1] + c['hole_width']/2, topwall_y),
    # 				b = (c['hole_positions'][2] - c['hole_width']/2, topwall_y),
    # 				radius = 2.0),
    # 	# top horizontal: 4
    # 	pymunk.Segment(walls,
    # 				a = (c['hole_positions'][2] + c['hole_width']/2, topwall_y),
    # 				b = (c['med'] + c['width']/2, topwall_y),
    # 				radius = 2.0),

    # left vertical
    static_lines.append(
        pymunk.Segment(
            walls,
            a=(c["med"] - c["width"] / 2, c["med"] - c["height"] / 2),
            b=(c["med"] - c["width"] / 2, c["med"] + c["height"] / 2),
            radius=2.0,
        )
    )

    # right vertical
    static_lines.append(
        pymunk.Segment(
            walls,
            a=(c["med"] + c["width"] / 2, c["med"] - c["height"] / 2),
            b=(c["med"] + c["width"] / 2, c["med"] + c["height"] / 2),
            radius=2.0,
        )
    )
    space.add(walls)
    for line in static_lines:
        line.elasticity = c["wall_elasticity"]
        line.friction = c["wall_friction"]
        line.collision_type = shape_code["walls"]
        space.add(line)


def make_obstacles(c, space, distorted=False):
    # make obstacles in the space, and return a list of sample points on top surfaces of the obstacles
    # note that if two obstacles stack over each other, only the top surfaces of the higher one would be returned
    # the output would be used to generate topological looking in the attention heatmap
    top_surfaces = []
    for ob, ob_dict in c["obstacles"].items():

        rigid_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        if not distorted:
            polygon = utils.generate_ngon(
                c["obstacles"][ob]["n_sides"], c["obstacles"][ob]["size"]
            )
        else:
            polygon = ob_dict["shape"]

        shape = pymunk.Poly(rigid_body, polygon)
        shape.elasticity = c["obstacles"][ob]["elasticity"]
        shape.friction = c["obstacles"][ob]["friction"]

        pos = c["obstacles"][ob]["position"]

        # add gaussian noises to x and y
        position_noise_x = utils.gaussian_noise(0, 0)
        position_noise_y = utils.gaussian_noise(0, 0)
        rigid_body.position = pos["x"] + position_noise_x, pos["y"] + position_noise_y
        # add gaussian noise to rotation
        position_noise_r = utils.gaussian_noise(0, 0)
        rigid_body.angle = c["obstacles"][ob]["rotation"] + position_noise_r

        shape.collision_type = shape_code[ob]  # key ob is the name

        space.add(shape, rigid_body)

        vertices = utils.get_vertices(shape)
        ## TODO: FIX THE SURFACE LOOK
        top_surfaces += utils.get_top_surfaces(
            vertices, rigid_body.position
        )  # [[a1, a2], [b1, b2]]
    top_surfaces_non_overlap = utils.remove_overlap(top_surfaces)
    return top_surfaces_non_overlap


def generate_distorted_shape(ob_dict, eye_pos, perturb=10, divider=4, version=1):
    n = ob_dict["n_sides"]
    side_length = ob_dict["size"]
    rot = ob_dict["rotation"]

    if version == 0:
        # Generates a shape with verticies randomly distributed around the circle
        # that circumscribes the polygon

        radius = utils.radius_reg_poly(side_length, n)

        points_with_angles = []

        for i in range(n):
            random_angle = np.random.rand() * 2 * np.pi
            random_radius = np.random.normal(radius, radius / divider)
            x = random_radius * np.cos(random_angle)
            y = random_radius * np.sin(random_angle)

            points_with_angles.append([(x, y), random_angle])

        sorted_points = sorted(points_with_angles, key=lambda x: x[1])

        return [pt[0] for pt in sorted_points]

    elif version == 1:
        # Perturb the verticies of the actual object
        actual_shape = utils.generate_ngon(n, side_length)

        distorted_shape = [
            [coordinate + np.random.normal(scale=perturb) for coordinate in pt]
            for pt in actual_shape
        ]

        return distorted_shape


if __name__ == "__main__":
    main()
