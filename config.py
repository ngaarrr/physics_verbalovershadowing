# import libraries
from __future__ import division
from math import pi
import sys
from random import choice, randint, shuffle, uniform, sample


def main():
    pass


def get_config():
    """
    Creates the main configuration file.
    Random choices: Obstacle positions and in what hole the ball is dropped
    """
    c = {}

    # PARAMETERS
    c["drop_noise"] = 0
    c["collision_noise_mean"] = 1
    c["collision_noise_sd"] = 0

    # GLOBAL SETTINGS
    c["dt"] = 1 / 60  # time step in physics engine (reverted to reasonable value)
    c["substeps_per_frame"] = 4
    c["med"] = 450  # midpoint (adjusted for larger screen)
    c["gravity"] = -1000  # gravity (reduced for slower ball movement)
    c["screen_size"] = {"width": 900, "height": 900}

    # PLINKO BOX SETTINGS
    c["width"] = 800
    c["height"] = 600
    # Create 10 evenly spaced holes across the width
    num_holes = 10
    fill_ratio = 0.8  # holes take up 80% of slot width
    slot_width = c["width"] / (num_holes + 1)

    c["hole_width"] = min(80, slot_width * fill_ratio)
    c["hole_positions"] = [
        c["med"] - c["width"] / 2 + slot_width * (i + 1) for i in range(num_holes)
    ]
    c["wall_elasticity"] = 0.2
    c["wall_friction"] = 0.9
    c["ground_elasticity"] = 0
    c["ground_friction"] = 2
    c["ground_y"] = c["med"] - c["height"] / 2 - 5

    # DIVIDER SETTINGS
    c["divider_thickness"] = 4.4  # Thickness of the bin divider

    # BALL SETTINGS
    c["ball_radius"] = 15
    c["ball_mass"] = 10
    c["ball_elasticity"] = 0.2
    c["ball_friction"] = 0.5
    c["ball_initial_angle"] = 0

    # OBSTACLE SETTINGS
    # Place the middle shape (abstract) at the center, slightly larger
    mid_x = 350
    mid_y = 350
    abstract_size = 55 * 1.2  # 20% larger than original (assuming original was 55)
    # Rectangle positions: offset from center, one upper third, one lower third
    offset_x = 100
    upper_y = mid_y + 120
    lower_y = mid_y - 120
    rectangle_size = 50 * 1.2
    rectangle_rotation = 0.7  # Use original rotation
    # Set all obstacles to the same size, as a whole number (90)
    unified_size = 90
    c["obstacles"] = {
        # "rectangle1": {
        #     "type": "rectangle",
        #     "n_sides": 4,
        #     "size": unified_size,
        #     "position": {"x": mid_x - offset_x, "y": upper_y},
        #     "rotation": rectangle_rotation,
        #     "elasticity": 0,
        #     "friction": 0,
        # },
        # "rectangle2": {
        #     "type": "rectangle",
        #     "n_sides": 4,
        #     "size": unified_size,
        #     "position": {"x": mid_x + offset_x, "y": lower_y},
        #     "rotation": -rectangle_rotation,
        #     "elasticity": 0,
        #     "friction": 0,
        # },
        "abstract": {
            "type": "abstract",
            "size": unified_size,
            "position": {"x": mid_x, "y": mid_y},
            "rotation": 0,
            "elasticity": 0,
            "friction": 0,
        },
        "abstract_upper": {
            "type": "abstract",
            "size": unified_size,
            "position": {"x": mid_x - offset_x, "y": upper_y},
            "rotation": rectangle_rotation,
            "elasticity": 0,
            "friction": 0,
        },
        "abstract_lower": {
            "type": "abstract",
            "size": unified_size,
            "position": {"x": mid_x + offset_x, "y": lower_y},
            "rotation": -rectangle_rotation,
            "elasticity": 0,
            "friction": 0,
        },
    }

    # random choices
    c = random_choices(c)

    return c


def random_choices(c):
    # CONFIG
    gap = 100
    wall_margin = 80
    min_spacing = 80  # Minimum distance between any two shapes

    # SIZES
    box_height = c["height"]
    usable_height = box_height - 2 * gap
    section_height = usable_height / 3

    box_width = c["width"]
    usable_width = box_width - 2 * wall_margin

    # Define the usable horizontal range (with wall margins)
    x_min = c["med"] - box_width / 2 + wall_margin
    x_max = c["med"] + box_width / 2 - wall_margin

    # Generate three random X positions within the usable range
    # We'll generate them and then ensure proper spacing
    x_positions = []
    attempts = 0
    max_attempts = 1000

    while len(x_positions) < 3 and attempts < max_attempts:
        new_x = uniform(x_min, x_max)

        # Check if this position is far enough from existing positions
        valid_position = True
        for existing_x in x_positions:
            if abs(new_x - existing_x) < min_spacing:
                valid_position = False
                break

        if valid_position:
            x_positions.append(new_x)

        attempts += 1

    # If we couldn't find 3 positions with proper spacing, use fallback
    if len(x_positions) < 3:
        # Fallback: use evenly spaced positions
        x_positions = [
            x_min + usable_width * 0.2,
            x_min + usable_width * 0.5,
            x_min + usable_width * 0.8,
        ]

    # Sort positions from left to right
    x_positions.sort()

    # Randomly assign the three X positions to upper, middle, and lower shapes
    from random import shuffle

    shape_x_positions = x_positions.copy()
    shuffle(shape_x_positions)

    # Y positions for three sections (centers) - ensure shapes stay above green bins
    ground_y = c["ground_y"]  # Bottom of green bins
    min_shape_y = ground_y + 50  # Minimum 50 pixels above green bins

    upper_center_y = c["med"] + box_height / 2 - gap - section_height / 2
    middle_center_y = c["med"]
    lower_center_y = c["med"] - box_height / 2 + gap + section_height / 2

    # Range for y randomization (±section_height/4)
    y_range = section_height / 4

    # Randomize y within each third, but ensure minimum distance from ground
    upper_y = uniform(upper_center_y - y_range, upper_center_y + y_range)
    middle_y = uniform(middle_center_y - y_range, middle_center_y + y_range)
    lower_y = uniform(lower_center_y - y_range, lower_center_y + y_range)

    # Ensure all shapes stay above the green bins
    upper_y = max(upper_y, min_shape_y)
    middle_y = max(middle_y, min_shape_y)
    lower_y = max(lower_y, min_shape_y)

    # Set obstacle positions and random rotations
    # Assign the shuffled X positions to the shapes
    c["obstacles"]["abstract_upper"].update(
        {
            "position": {"x": shape_x_positions[0], "y": upper_y},
            "rotation": uniform(0, 2 * pi),
        }
    )
    c["obstacles"]["abstract"].update(
        {
            "position": {"x": shape_x_positions[1], "y": middle_y},
            "rotation": uniform(0, 2 * pi),
        }
    )
    c["obstacles"]["abstract_lower"].update(
        {
            "position": {"x": shape_x_positions[2], "y": lower_y},
            "rotation": uniform(0, 2 * pi),
        }
    )

    # Random hole drop
    c["hole_dropped_into"] = randint(0, len(c["hole_positions"]) - 1)
    return c


if __name__ == "__main__":
    main()
