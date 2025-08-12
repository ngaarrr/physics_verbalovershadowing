# =============================================================================
# PHYSICS SIMULATION ENGINE FOR PLINKO-STYLE GAME
# =============================================================================

# Import libraries
from __future__ import division
import sys
from glob import glob
import numpy as np
from scipy.stats import truncnorm
import pymunk
from pymunk import Vec2d
import time
import os
import json
import copy
from scipy.spatial import ConvexHull

# Import project files
import utils
import visual
import config
import convert_coordinate

# =============================================================================
# SCENE IMAGE SAVING FUNCTION
# =============================================================================


def save_scene_image(c, filename, simplified=False):
    """Save a static image of the scene without the ball"""
    import pygame

    # Setup pygame for headless rendering
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    pygame.init()
    screen = pygame.display.set_mode(
        (c["screen_size"]["width"], c["screen_size"]["height"])
    )

    # Set up the rotated obstacles
    rotated = visual.rotate_shapes(c)

    # Draw the scene without ball
    screen.fill(pygame.color.THECOLORS["white"])

    # Draw obstacles
    colors = [
        pygame.color.THECOLORS["blue"],
        pygame.color.THECOLORS["red"],
        pygame.color.THECOLORS["green"],
    ]
    visual.draw_obstacles(rotated, screen, colors, c)

    # Draw walls, ground, and bins
    visual.draw_walls(c, screen)
    visual.draw_ground(c, screen)
    visual.draw_bins(c, screen)

    # Save the image
    pygame.image.save(screen, filename)
    pygame.quit()

    print(f"Saved scene image: {filename}")


def save_scene_with_ball_no_obstacles(c, filename, simplified=False):
    """Save a static image of the scene with ball in initial position but no obstacles"""
    import pygame

    # Setup pygame for headless rendering
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    pygame.init()
    screen = pygame.display.set_mode(
        (c["screen_size"]["width"], c["screen_size"]["height"])
    )

    # Draw the scene without obstacles
    screen.fill(pygame.color.THECOLORS["white"])

    # Draw walls, ground, and bins (no obstacles)
    visual.draw_walls(c, screen)
    visual.draw_ground(c, screen)
    visual.draw_bins(c, screen)

    # Calculate ball's initial position (same as in make_ball function)
    ball_x = c["hole_positions"][c["hole_dropped_into"]]
    ball_y = c["med"] + c["height"] / 2 + c["ball_radius"]

    # Draw the ball in its initial position
    pygame.draw.circle(
        screen,
        pygame.color.THECOLORS["blue"],
        (int(ball_x), int(visual.utils.flipy(c, ball_y))),
        c["ball_radius"],
    )

    # Save the image
    pygame.image.save(screen, filename)
    pygame.quit()

    print(f"Saved scene with ball (no obstacles): {filename}")


# =============================================================================
# VIDEO SAVING FUNCTION
# =============================================================================


def save_video(c, sim_data, filename):
    """Save simulation as MP4 video using pygame and ffmpeg"""
    import pygame
    import subprocess
    import tempfile
    import shutil

    # Setup pygame for headless rendering
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    pygame.init()
    screen = pygame.display.set_mode(
        (c["screen_size"]["width"], c["screen_size"]["height"])
    )

    # Set up the rotated obstacles
    rotated = visual.rotate_shapes(c)

    # Create temporary directory for frames
    temp_dir = tempfile.mkdtemp()

    try:
        # Limit frames for video
        max_frames = min(200, len(sim_data["ball_position"]))
        step = max(1, len(sim_data["ball_position"]) // max_frames)

        frame_count = 0
        for t in range(0, len(sim_data["ball_position"]), step):
            frame = sim_data["ball_position"][t]

            # Draw frame
            screen.fill(pygame.color.THECOLORS["white"])
            colors = [
                pygame.color.THECOLORS["blue"],
                pygame.color.THECOLORS["red"],
                pygame.color.THECOLORS["green"],
            ]

            visual.draw_obstacles(rotated, screen, colors, c)
            visual.draw_ground(c, screen)
            visual.draw_bins(c, screen)
            visual.draw_ball(c, screen, frame)
            visual.draw_walls(c, screen)

            # Save frame as PNG
            frame_path = os.path.join(temp_dir, f"frame_{frame_count:05d}.png")
            pygame.image.save(screen, frame_path)
            frame_count += 1

        # Create video using ffmpeg
        if frame_count > 0:
            try:
                cmd = [
                    "ffmpeg",
                    "-y",  # Overwrite output file
                    "-framerate",
                    "30",  # 30 FPS
                    "-i",
                    os.path.join(temp_dir, "frame_%05d.png"),
                    "-c:v",
                    "libx264",
                    "-profile:v",
                    "high",
                    "-crf",
                    "20",  # Good quality
                    "-pix_fmt",
                    "yuv420p",
                    filename,
                ]

                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"Saved video: {filename}")
                else:
                    print(f"Error saving video {filename}: {result.stderr}")
            except Exception as e:
                print(f"Error running ffmpeg for {filename}: {e}")

    finally:
        # Cleanup
        pygame.quit()
        shutil.rmtree(temp_dir, ignore_errors=True)


# =============================================================================
# PHYSICS ENGINE CONFIGURATION
# =============================================================================

# Collision type definitions for physics engine
shape_code = {
    "walls": 0,
    "ground": 1,
    "ball": 2,
    "rectangle": 3,
    "rectangle1": 3,  # Same collision type as rectangle
    "rectangle2": 3,  # Same collision type as rectangle
    "triangle": 4,
    "pentagon": 5,
}

inverse_shape_code = {shape_code[key]: key for key in shape_code}

# Simulation parameters
TIMEOUT = 10  # Maximum simulation time in seconds
VELOCITY_SD = 0.0  # Standard deviation for velocity noise (unused)


# =============================================================================
# MAIN SIMULATION FUNCTIONS
# =============================================================================


def main():
    """Main entry point - run 100 attempts without visualizations for each scene, auto-close when ball reaches ground."""
    print(f"\n--- Running 100 Attempts without Visualizations ---")

    for attempt in range(1, 101):  # Run 100 attempts
        print(f"\n=== ATTEMPT {attempt}/100 ===")
        c = config.get_config()
        num_holes = len(c["hole_positions"])
        stuck_count = 0
        bin_results = []

        # Generate the abstract shapes ONCE and store them in the config
        if "abstract" in c["obstacles"] and "shape" not in c["obstacles"]["abstract"]:
            size = c["obstacles"]["abstract"].get(
                "size", 90
            )  # Use correct default size
            c["obstacles"]["abstract"]["shape"] = utils.generate_abstract_obstacle(
                (0, 0), c["width"], c["height"], size=size
            )
        if (
            "abstract_upper" in c["obstacles"]
            and "shape" not in c["obstacles"]["abstract_upper"]
        ):
            size = c["obstacles"]["abstract_upper"].get(
                "size", 90
            )  # Use correct default size
            c["obstacles"]["abstract_upper"]["shape"] = (
                utils.generate_abstract_obstacle(
                    (0, 0), c["width"], c["height"], size=size
                )
            )
        if (
            "abstract_lower" in c["obstacles"]
            and "shape" not in c["obstacles"]["abstract_lower"]
        ):
            size = c["obstacles"]["abstract_lower"].get(
                "size", 90
            )  # Use correct default size
            c["obstacles"]["abstract_lower"]["shape"] = (
                utils.generate_abstract_obstacle(
                    (0, 0), c["width"], c["height"], size=size
                )
            )

        # Run abstract simulation for all holes WITHOUT visualization
        print("\n=== ABSTRACT VERSION ===")
        for hole_idx in range(num_holes):
            c["hole_dropped_into"] = hole_idx
            print(
                f"\nDropping ball from hole {hole_idx + 1} (x = {c['hole_positions'][hole_idx]:.1f}) [Abstract]"
            )
            data = run_simulation(c, auto_close=False, simplified=False)
            final_pos = data["ball_position"][-1]
            final_x = final_pos["x"]
            final_y = final_pos["y"]
            bin_y_threshold = 165
            center_x = c["med"]
            divider_half_thickness = (
                c["divider_thickness"] / 2
            )  # Calculate dynamically from config
            if final_y > bin_y_threshold:
                stuck_count += 1
                bin_results.append("stuck")
                # Stop immediately if ball gets stuck
                print(
                    f"\nBall got stuck at hole {hole_idx + 1}. Stopping abstract simulation."
                )
                break
            else:
                if final_x < center_x - divider_half_thickness:
                    bin_results.append("left")
                else:
                    bin_results.append("right")

        # Fill remaining holes with "stuck" if we stopped early
        while len(bin_results) < num_holes:
            bin_results.append("stuck")

        summary = " ".join([f"{i+1}: {res}" for i, res in enumerate(bin_results)])
        if stuck_count != 0:
            print(
                f"\nScene was UNSUCCESSFUL: Ball got stuck {stuck_count} time(s) out of {num_holes} drops."
            )
            print(f"Summary: {summary}")
            print("Moving to next attempt...")
            continue

        # --- Generate non-abstract version ---
        abstract_ob = c["obstacles"]["abstract"]
        abstract_points = abstract_ob["shape"]
        points_np = np.array(abstract_points)
        hull = ConvexHull(points_np)
        hull_points = [tuple(points_np[v]) for v in hull.vertices]

        c_nonabstract = copy.deepcopy(c)
        c_nonabstract["obstacles"]["abstract"]["shape"] = hull_points

        if "abstract_upper" in c["obstacles"]:
            abstract_upper_points = c["obstacles"]["abstract_upper"]["shape"]
            points_np_upper = np.array(abstract_upper_points)
            hull_upper = ConvexHull(points_np_upper)
            hull_points_upper = [tuple(points_np_upper[v]) for v in hull_upper.vertices]
            c_nonabstract["obstacles"]["abstract_upper"]["shape"] = hull_points_upper

        if "abstract_lower" in c["obstacles"]:
            abstract_lower_points = c["obstacles"]["abstract_lower"]["shape"]
            points_np_lower = np.array(abstract_lower_points)
            hull_lower = ConvexHull(points_np_lower)
            hull_points_lower = [tuple(points_np_lower[v]) for v in hull_lower.vertices]
            c_nonabstract["obstacles"]["abstract_lower"]["shape"] = hull_points_lower

        # Run non-abstract simulation for all holes WITHOUT visualization
        print("\n=== NON-ABSTRACT VERSION ===")
        stuck_count_nonabs = 0
        bin_results_nonabs = []
        for hole_idx in range(num_holes):
            c_nonabstract["hole_dropped_into"] = hole_idx
            print(
                f"\nDropping ball from hole {hole_idx + 1} (x = {c_nonabstract['hole_positions'][hole_idx]:.1f}) [Non-abstract]"
            )
            data = run_simulation(c_nonabstract, auto_close=False, simplified=True)
            final_pos = data["ball_position"][-1]
            final_x = final_pos["x"]
            final_y = final_pos["y"]
            bin_y_threshold = 165
            center_x = c_nonabstract["med"]
            divider_half_thickness = (
                c_nonabstract["divider_thickness"] / 2
            )  # Calculate dynamically from config
            if final_y > bin_y_threshold:
                stuck_count_nonabs += 1
                bin_results_nonabs.append("stuck")
                # Stop immediately if ball gets stuck
                print(
                    f"\nBall got stuck at hole {hole_idx + 1}. Stopping non-abstract simulation."
                )
                break
            else:
                if final_x < center_x - divider_half_thickness:
                    bin_results_nonabs.append("left")
                else:
                    bin_results_nonabs.append("right")

        # Fill remaining holes with "stuck" if we stopped early
        while len(bin_results_nonabs) < num_holes:
            bin_results_nonabs.append("stuck")

        summary_nonabs = " ".join(
            [f"{i+1}: {res}" for i, res in enumerate(bin_results_nonabs)]
        )

        if stuck_count_nonabs != 0:
            print(
                f"\nNon-abstract scene was UNSUCCESSFUL: Ball got stuck {stuck_count_nonabs} time(s) out of {num_holes} drops."
            )
            print(f"Summary: {summary_nonabs}")
            print("Moving to next attempt...")
            continue

        # Always print both summaries first
        print("\nAbstract Summary:")
        print(summary)
        print("\nNon-abstract Summary:")
        print(summary_nonabs)

        # --- Compare bin results for switches ---
        print("\n=== COMPARISON DEBUG ===")
        print("Abstract results:", bin_results)
        print("Non-abstract results:", bin_results_nonabs)

        switch_holes = []
        for i, (abs_res, nonabs_res) in enumerate(zip(bin_results, bin_results_nonabs)):
            print(f"Hole {i+1}: Abstract={abs_res}, Non-abstract={nonabs_res}")
            if abs_res != nonabs_res and abs_res != "stuck" and nonabs_res != "stuck":
                switch_holes.append(i + 1)
                print(f"  → SWITCH DETECTED!")

        if switch_holes:
            switch_msg = f"Bin position switched for hole(s): {', '.join(str(h) for h in switch_holes)}"
            print("Found the scene!")
            print(switch_msg)

            # Store results for summary (without visualization)
            abstract_results = []
            nonabstract_results = []

            print("Abstract version:")
            for hole_idx in range(num_holes):
                c["hole_dropped_into"] = hole_idx
                print(
                    f"\nDropping ball from hole {hole_idx + 1} (x = {c['hole_positions'][hole_idx]:.1f}) [Abstract]"
                )
                data = run_simulation(c, auto_close=False, simplified=False)
                final_pos = data["ball_position"][-1]
                final_x = final_pos["x"]
                final_y = final_pos["y"]
                center_x = c["med"]
                divider_half_thickness = c["divider_thickness"] / 2
                if final_y <= 165:
                    if final_x < center_x - divider_half_thickness:
                        result = "LEFT"
                    else:
                        result = "RIGHT"
                else:
                    result = "STUCK"
                abstract_results.append(result)

            print("\nNon-abstract version:")
            for hole_idx in range(num_holes):
                c_nonabstract["hole_dropped_into"] = hole_idx
                print(
                    f"\nDropping ball from hole {hole_idx + 1} (x = {c_nonabstract['hole_positions'][hole_idx]:.1f}) [Non-abstract]"
                )
                data = run_simulation(c_nonabstract, auto_close=False, simplified=True)
                final_pos = data["ball_position"][-1]
                final_x = final_pos["x"]
                final_y = final_pos["y"]
                center_x = c_nonabstract["med"]
                divider_half_thickness = c_nonabstract["divider_thickness"] / 2
                if final_y <= 165:
                    if final_x < center_x - divider_half_thickness:
                        result = "LEFT"
                    else:
                        result = "RIGHT"
                else:
                    result = "STUCK"
                nonabstract_results.append(result)

            # Print comprehensive summary
            print("\n" + "=" * 60)
            print("COMPREHENSIVE SUMMARY")
            print("=" * 60)
            print("Hole | Abstract | Non-Abstract | Switch?")
            print("-" * 40)
            for i in range(num_holes):
                switch_indicator = "YES" if i + 1 in switch_holes else "NO"
                print(
                    f"  {i+1:2d} | {abstract_results[i]:8s} | {nonabstract_results[i]:12s} | {switch_indicator}"
                )

            print(f"\nBIN SWITCHES FOUND: {len(switch_holes)}")
            print(f"Switched holes: {', '.join(str(h) for h in switch_holes)}")

            # Save files
            import json
            import os
            from datetime import datetime

            # Create output directory structure
            successful_dir = "SUCCESSFUL"
            os.makedirs(successful_dir, exist_ok=True)

            # Find next available scene number
            scene_num = 1
            while os.path.exists(
                os.path.join(successful_dir, f"SUCCESSFUL_SCENE_{scene_num}")
            ):
                scene_num += 1

            output_dir = os.path.join(successful_dir, f"SUCCESSFUL_SCENE_{scene_num}")
            os.makedirs(output_dir, exist_ok=True)

            # Save JSON files
            with open(f"{output_dir}/abstract_config.json", "w") as f:
                json.dump(c, f, indent=2)

            with open(f"{output_dir}/nonabstract_config.json", "w") as f:
                json.dump(c_nonabstract, f, indent=2)

            # Save README
            with open(f"{output_dir}/README.md", "w") as f:
                f.write(f"# Successful Scene {scene_num} - Bin Switch Detected\n\n")
                f.write(
                    f"**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                )
                f.write(f"**Attempt:** {attempt}/100\n\n")
                f.write(f"## Summary\n\n")
                f.write(
                    f"Found {len(switch_holes)} bin switch(es) between abstract and non-abstract shapes.\n\n"
                )
                f.write(f"## Results by Hole\n\n")
                f.write(f"| Hole | Abstract | Non-Abstract | Switch? |\n")
                f.write(f"|------|----------|--------------|---------|\n")
                for i in range(num_holes):
                    switch_indicator = "YES" if i + 1 in switch_holes else "NO"
                    f.write(
                        f"| {i+1} | {abstract_results[i]} | {nonabstract_results[i]} | {switch_indicator} |\n"
                    )
                f.write(f"\n## Switched Holes\n")
                f.write(
                    f"Holes where bin position changed: {', '.join(str(h) for h in switch_holes)}\n\n"
                )
                f.write(f"## Files\n")
                f.write(
                    f"- `abstract_config.json`: Configuration for abstract shapes\n"
                )
                f.write(
                    f"- `nonabstract_config.json`: Configuration for non-abstract shapes\n"
                )
                f.write(f"- `README.md`: This file\n")
                f.write(
                    f"- `hole_X_abstract.png`: Scene image for abstract version of switched holes\n"
                )
                f.write(
                    f"- `hole_X_nonabstract.png`: Scene image for non-abstract version of switched holes\n"
                )
                f.write(
                    f"- `hole_X_abstract_ball.png`: Scene with ball (no obstacles) for abstract version\n"
                )
                f.write(
                    f"- `hole_X_nonabstract_ball.png`: Scene with ball (no obstacles) for non-abstract version\n"
                )
                f.write(
                    f"- `hole_X_abstract.mp4`: Video for abstract version of switched holes\n"
                )
                f.write(
                    f"- `hole_X_nonabstract.mp4`: Video for non-abstract version of switched holes\n"
                )

            print(f"\nFiles saved to: {output_dir}/")
            print(f"- abstract_config.json")
            print(f"- nonabstract_config.json")
            print(f"- README.md")

            # Save scene images for switched holes
            print(f"\nSaving scene images for switched holes...")
            for hole_idx in switch_holes:
                hole_num = hole_idx - 1  # Convert to 0-based index

                # Save abstract version image
                c["hole_dropped_into"] = hole_num
                save_scene_image(
                    c, f"{output_dir}/hole_{hole_idx}_abstract.png", simplified=False
                )

                # Save non-abstract version image
                c_nonabstract["hole_dropped_into"] = hole_num
                save_scene_image(
                    c_nonabstract,
                    f"{output_dir}/hole_{hole_idx}_nonabstract.png",
                    simplified=True,
                )

                # Set hole for both configurations
                c["hole_dropped_into"] = hole_num
                c_nonabstract["hole_dropped_into"] = hole_num

                # Save scene image with ball in initial position (no obstacles) - abstract
                save_scene_with_ball_no_obstacles(
                    c,
                    f"{output_dir}/hole_{hole_idx}_abstract_ball.png",
                    simplified=False,
                )

                # Save scene image with ball in initial position (no obstacles) - non-abstract
                save_scene_with_ball_no_obstacles(
                    c_nonabstract,
                    f"{output_dir}/hole_{hole_idx}_nonabstract_ball.png",
                    simplified=True,
                )

                print(f"Saved scene images for hole {hole_idx}")

            # Save videos for switched holes
            print(f"\nSaving videos for switched holes...")
            for hole_idx in switch_holes:
                hole_num = hole_idx - 1  # Convert to 0-based index

                # Save abstract version video
                c["hole_dropped_into"] = hole_num
                data_abstract = run_simulation(c, auto_close=False, simplified=False)
                save_video(
                    c, data_abstract, f"{output_dir}/hole_{hole_idx}_abstract.mp4"
                )

                # Save non-abstract version video
                c_nonabstract["hole_dropped_into"] = hole_num
                data_nonabstract = run_simulation(
                    c_nonabstract, auto_close=False, simplified=True
                )
                save_video(
                    c_nonabstract,
                    data_nonabstract,
                    f"{output_dir}/hole_{hole_idx}_nonabstract.mp4",
                )

                print(f"Saved videos for hole {hole_idx}")

            print(f"\nAttempt {attempt} complete!")
            return  # Exit the loop when switch is found
        else:
            switch_msg = "No bin position switches between abstract and non-abstract."
            print(switch_msg)

        print(f"\nAttempt {attempt} complete!")

    print("\nAll 100 attempts completed!")


# --- Two-abstract comparison logic is commented out below for easy reversion ---
# (see previous version for two-abstract comparison and saving)


def run_simulation(
    c, convert_coordinates=False, distorted=False, auto_close=False, simplified=False
):
    """Main simulation loop - runs physics and tracks ball trajectory"""
    # Setup physics space with gravity
    space = pymunk.Space()
    space.gravity = (0.0, c["gravity"])
    space.convert_coordinates = convert_coordinates

    # Create all objects in the scene
    make_walls(c, space)
    top_surfaces = make_obstacles(c, space, distorted=distorted, simplified=simplified)
    ball = make_ball(c, space)
    make_ground(c, space)

    ###############
    ## MAIN LOOP ##
    ###############
    timestep = 0

    all_data = {}

    ball_pos = []
    ball_vel = []

    start = time.time()  # Track simulation start time for timeout
    rest_counter = 0  # Count consecutive frames where ball is resting
    rest_threshold = (
        10  # Number of consecutive frames required to consider ball at rest
    )
    bin_y_threshold = 165  # y value for the top of the bin area
    reached_bin = False  # Track if ball ever reached the bin area

    # Track if ball is stuck (not moving for many frames)
    stuck_counter = 0  # Count consecutive frames where ball barely moves
    stuck_threshold = 500  # If ball doesn't move for 500 frames, consider it stuck
    last_position = None  # Store previous ball position to detect movement

    # Track positions for evenly spaced printing
    positions_to_print = []  # Will store (timestep, x, y) tuples for printing
    print_count = 0  # Count how many positions we've printed so far
    target_prints = 10  # Total number of positions we want to print

    while True:
        # Update physics
        for _ in range(c["substeps_per_frame"]):
            space.step(c["dt"] / c["substeps_per_frame"])

        timestep += 1

        x, y = ball.position.x, ball.position.y

        # Store position for potential printing
        positions_to_print.append((timestep, x, y))

        # Print first position immediately
        if timestep == 1:
            print(f"Frame {timestep}: x = {x:.2f}, y = {y:.2f} (START)")
            print_count += 1

        # Progress indicator every 10000 frames
        if timestep % 10000 == 0:
            print(
                f"Progress: Frame {timestep}, x={x:.2f}, y={y:.2f}, vx={ball.velocity.x:.3f}, vy={ball.velocity.y:.3f}"
            )

        if convert_coordinates:
            x, y = convert_coordinate.convertCoordinate(x, y)

        ball_pos.append({"x": x, "y": y})
        ball_vel.append({"x": ball.velocity.x, "y": ball.velocity.y})

        # Track if the ball ever reached the bin area
        if y <= bin_y_threshold:
            reached_bin = True

        # Check if ball is stuck (not moving)
        current_position = (x, y)
        if last_position is not None:
            position_change = abs(current_position[0] - last_position[0]) + abs(
                current_position[1] - last_position[1]
            )
            if position_change < 0.1:  # Ball moved less than 0.1 pixels
                stuck_counter += 1
            else:
                stuck_counter = 0
        last_position = current_position

        # Only increment rest_counter if ball is near ground and moving slowly
        near_ground = y <= bin_y_threshold
        moving_slowly = abs(ball.velocity.y) < 5
        if near_ground and moving_slowly:
            rest_counter += 1
        else:
            rest_counter = 0

        if rest_counter >= rest_threshold:
            break

        # Check if ball is stuck (not moving for many frames)
        if stuck_counter >= stuck_threshold:
            print(
                f"BALL STUCK: No movement for {stuck_threshold} frames at x={x:.2f}, y={y:.2f}"
            )
            break

        # Timeout: only report stuck if the ball never reached the bin area
        if time.time() > start + TIMEOUT:
            if not reached_bin:
                # Ball never reached the bin area, treat as stuck
                print(
                    f"TIMEOUT: Ball stuck at x={x:.2f}, y={y:.2f} after {timestep} frames"
                )
                # Print intermediate frames even when stuck
                total_frames = len(positions_to_print)
                if total_frames > 2:  # Only if we have more than start and end
                    step_size = (total_frames - 1) / (target_prints - 1)
                    for i in range(
                        1, target_prints - 1
                    ):  # Skip first (already printed) and last
                        frame_idx = int(i * step_size)
                        if frame_idx < total_frames:
                            frame_num, pos_x, pos_y = positions_to_print[frame_idx]
                            print(
                                f"Frame {frame_num}: x = {pos_x:.2f}, y = {pos_y:.2f}"
                            )

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
            else:
                # Ball reached the bin at some point, let it finish
                break

    # Print evenly spaced intermediate positions
    total_frames = len(positions_to_print)
    if total_frames > 2:  # Only if we have more than start and end
        step_size = (total_frames - 1) / (
            target_prints - 1
        )  # -1 because we already printed start
        for i in range(1, target_prints - 1):  # Skip first (already printed) and last
            frame_idx = int(i * step_size)
            if frame_idx < total_frames:
                frame_num, pos_x, pos_y = positions_to_print[frame_idx]
                print(f"Frame {frame_num}: x = {pos_x:.2f}, y = {pos_y:.2f}")

    # Always record the final true resting position
    final_x, final_y = ball.position.x, ball.position.y
    if convert_coordinates:
        final_x, final_y = convert_coordinate.convertCoordinate(final_x, final_y)

    ball_pos.append({"x": final_x, "y": final_y})
    ball_vel.append({"x": ball.velocity.x, "y": ball.velocity.y})
    print(f"Final settled position: x={final_x:.1f}, y={final_y:.1f}")

    # Determine outcome and print result
    center_x = c["med"]  # Middle divider position
    if final_y <= bin_y_threshold:
        # Ball reached the bin area
        if final_x < center_x:
            print(f"Ball reached bin: LEFT (x={final_x:.1f}, y={final_y:.1f})")
        else:
            print(f"Ball reached bin: RIGHT (x={final_x:.1f}, y={final_y:.1f})")
    else:
        # Ball got stuck
        print(f"Ball got STUCK at x={final_x:.1f}, y={final_y:.1f}")

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

    # Show visualization if auto_close is enabled
    if auto_close:
        visual.visualize(c, all_data, auto_close=True)

    return all_data


# =============================================================================
# PHYSICS OBJECT CREATION FUNCTIONS
# =============================================================================


def make_ball(c, space):
    """Create the ball that drops through the obstacles"""
    inertia = pymunk.moment_for_circle(c["ball_mass"], 0, c["ball_radius"], (0, 0))
    body = pymunk.Body(c["ball_mass"], inertia)
    x = c["hole_positions"][c["hole_dropped_into"]]
    y = c["med"] + c["height"] / 2 + c["ball_radius"]
    body.position = x, y
    body.sleep_threshold = (
        None  # Prevent sleeping - keeps ball active even when stationary
    )
    shape = pymunk.Circle(body, c["ball_radius"], (0, 0))
    shape.elasticity = c["ball_elasticity"]
    shape.friction = c["ball_friction"]

    shape.collision_type = shape_code["ball"]

    space.add(body, shape)

    # Set initial velocity downward
    ang = c["ball_initial_angle"]
    amp = 60  # Reduced from 100 to slow down the ball
    off = 3 * np.pi / 2
    # so that clockwise is negative
    body.velocity = Vec2d(amp * -np.cos(ang + off), amp * np.sin(ang + off))

    return body


def make_ground(c, space):
    """Create the ground surface and bin divider"""
    ground_body = pymunk.Body(body_type=pymunk.Body.STATIC)
    ground_body.position = (c["med"], c["ground_y"])
    ground_shape = pymunk.Poly.create_box(ground_body, (c["width"], 10))
    ground_shape.elasticity = 1
    ground_shape.friction = 1
    ground_shape.collision_type = shape_code["ground"]
    space.add(ground_body, ground_shape)

    # Create divider between bins
    bin_height = 30  # Should match visual.py (reduced from 60 to 30)
    divider_body = pymunk.Body(body_type=pymunk.Body.STATIC)
    divider_shape = pymunk.Segment(
        divider_body,
        (c["med"], c["ground_y"]),
        (c["med"], c["ground_y"] + bin_height),
        c["divider_thickness"],  # Use config value instead of hardcoded
    )
    divider_shape.elasticity = 1
    divider_shape.friction = 1
    divider_shape.collision_type = shape_code["walls"]
    space.add(divider_body, divider_shape)


def make_walls(c, space):
    """Create the box walls with holes at the top for ball entry"""
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


def make_obstacles(c, space, distorted=False, simplified=False):
    """Create obstacles (rectangles and abstract shape) with collision shapes. Returns list of top surfaces for visualization"""
    top_surfaces = []
    for ob, ob_dict in c["obstacles"].items():
        rigid_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        if not distorted:
            if ob_dict.get("type") == "abstract":
                # Use the pre-generated shape from the config
                polygon = ob_dict["shape"]
            else:
                polygon = utils.generate_ngon(ob_dict["n_sides"], ob_dict["size"])
                ob_dict["shape"] = polygon  # Ensure shape is stored for abstract
        else:
            polygon = ob_dict["shape"]

        # Set position and rotation from config
        rigid_body.position = ob_dict["position"]["x"], ob_dict["position"]["y"]
        rigid_body.angle = ob_dict["rotation"]

        if simplified:
            # Use convex hull (simplified collision)
            sh = pymunk.Poly(rigid_body, polygon)
            sh.elasticity = ob_dict["elasticity"]
            sh.friction = ob_dict["friction"]
            sh.collision_type = shape_code.get(ob, 3)
            space.add(rigid_body, sh)
        else:
            # Convert polygons to line segments for exact visual match
            segments = []

            # Create line segments for each polygon
            for i in range(len(polygon)):
                p1 = polygon[i]
                p2 = polygon[(i + 1) % len(polygon)]  # Connect last point to first
                seg = pymunk.Segment(rigid_body, p1, p2, radius=1.0)
                seg.collision_type = shape_code.get(ob, 3)
                seg.elasticity = ob_dict["elasticity"]
                seg.friction = ob_dict["friction"]
                segments.append(seg)

            space.add(rigid_body, *segments)

        # For top surface calculation, use the original polygon vertices
        # Convert polygon vertices to world coordinates
        world_vertices = []
        for vertex in polygon:
            # Apply rotation and translation to get world coordinates
            x, y = vertex
            cos_a = np.cos(rigid_body.angle)
            sin_a = np.sin(rigid_body.angle)
            world_x = x * cos_a - y * sin_a + rigid_body.position.x
            world_y = x * sin_a + y * cos_a + rigid_body.position.y
            world_vertices.append((world_x, world_y))

        top_surfaces += utils.get_top_surfaces(world_vertices, rigid_body.position)

    top_surfaces_non_overlap = utils.remove_overlap(top_surfaces)
    return top_surfaces_non_overlap


# UNUSED FUNCTION - COMMENTED OUT
# def generate_distorted_shape(ob_dict, eye_pos, perturb=10, divider=4, version=1):
#     """Generate distorted versions of obstacles for experimental conditions (unused)"""
#     n = ob_dict["n_sides"]
#     side_length = ob_dict["size"]
#     rot = ob_dict["rotation"]
#
#     if version == 0:
#         # Generates a shape with verticies randomly distributed around the circle
#         # that circumscribes the polygon
#
#         radius = utils.radius_reg_poly(side_length, n)
#
#         points_with_angles = []
#
#         for i in range(n):
#             random_angle = np.random.rand() * 2 * np.pi
#             random_radius = np.random.normal(radius, radius / divider)
#             x = random_radius * np.cos(random_angle)
#             y = random_radius * np.sin(random_angle)
#
#             points_with_angles.append([(x, y), random_angle])
#
#         sorted_points = sorted(points_with_angles, key=lambda x: x[1])
#
#         return [pt[0] for pt in sorted_points]
#
#     elif version == 1:
#         # Perturb the verticies of the actual object
#         actual_shape = utils.generate_ngon(n, side_length)
#
#         distorted_shape = [
#             [coordinate + np.random.normal(scale=perturb) for coordinate in pt]
#             for pt in actual_shape
#         ]
#
#         return distorted_shape


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, help="Path to a saved scene JSON file")
    parser.add_argument("--hole", type=int, help="Hole index to drop from (optional)")
    args = parser.parse_args()

    if args.scene:
        import json

        with open(args.scene, "r") as f:
            c = json.load(f)
        if args.hole is not None:
            c["hole_dropped_into"] = args.hole
        else:
            c["hole_dropped_into"] = 0  # default to first hole
        data = run_simulation(c)
        visual.visualize(c, data)
    else:
        main()
