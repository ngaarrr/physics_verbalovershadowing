#!/usr/bin/env python3
"""
Ball drop analysis functions for physics simulation.
"""

import config
import engine
import time
import math


def simulate_all_hole_drops(c):
    """
    Simulate ball drops from all holes (left to right) and record which bin the ball ends up in.
    Returns (results, scene_valid) where scene_valid is False if any ball gets stuck.
    """
    results = {}
    center_divider_x = c["med"]
    print(f"Simulating drops from all {len(c['hole_positions'])} holes...")
    print(f"Center divider at x = {center_divider_x}")
    for hole_idx in range(len(c["hole_positions"])):
        print(
            f"  Dropping ball from hole {hole_idx + 1}/{len(c['hole_positions'])} (x = {c['hole_positions'][hole_idx]:.1f})"
        )
        c["hole_dropped_into"] = hole_idx
        try:
            sim_data = engine.run_simulation(c)
            if sim_data["ball_position"]:
                final_pos = sim_data["ball_position"][-1]
                final_x = final_pos["x"]
                final_y = final_pos["y"]
                bin_threshold = 165
                if final_y <= bin_threshold:
                    if final_x < center_divider_x:
                        bin_result = "left"
                    else:
                        bin_result = "right"
                    results[hole_idx] = {
                        "bin": bin_result,
                        "final_position": (final_x, final_y),
                        "path_length": len(sim_data["ball_position"]),
                    }
                    print(
                        f"    Ball ended at ({final_x:.1f}, {final_y:.1f}) -> {bin_result} bin"
                    )
                else:
                    results[hole_idx] = {
                        "bin": "stuck",
                        "final_position": (final_x, final_y),
                        "path_length": len(sim_data["ball_position"]),
                    }
                    print(f"    Ball got stuck at ({final_x:.1f}, {final_y:.1f})")
                    print("    Ball trajectory (y values):")
                    for i, pos in enumerate(sim_data["ball_position"]):
                        print(f"      Step {i}: y = {pos['y']:.1f}")
                    return results, False
            else:
                results[hole_idx] = {
                    "bin": "stuck",
                    "final_position": None,
                    "path_length": 0,
                }
                print(f"    Ball got stuck - no final position")
                return results, False
        except Exception as e:
            print(f"    Error in simulation: {e}")
            results[hole_idx] = {
                "bin": "error",
                "final_position": None,
                "path_length": 0,
            }
            return results, False
    return results, True


def detect_stuck_ball(sim_data, c):
    """
    Detect if the ball got stuck (did not reach the bottom bins).
    Parameters:
    -----------
    sim_data : dict
        Simulation data from engine.run_simulation()
    c : dict
        Configuration dictionary
    Returns:
    --------
    bool
        True if ball is stuck, False otherwise
    """
    if not sim_data["ball_position"]:
        return True
    final_pos = sim_data["ball_position"][-1]
    final_y = final_pos["y"]
    bin_threshold = 165
    if final_y > bin_threshold:  # Ball did NOT reach the bins
        return True
    return False


def filter_scenes_without_stuck_balls(num_scenes=100):
    """
    Generate multiple scenes and filter out those where the ball gets stuck in the abstract obstacle.

    Parameters:
    -----------
    num_scenes : int
        Number of scenes to generate and test

    Returns:
    --------
    tuple
        (valid_scenes, stuck_scenes) where each contains scene configs and their results
    """
    valid_scenes = []
    stuck_scenes = []

    print(f"Generating and testing {num_scenes} scenes...")

    for scene_idx in range(num_scenes):
        print(f"  Testing scene {scene_idx + 1}/{num_scenes}")

        # Generate new scene configuration
        c = config.get_config()

        # Test ball drops from all holes
        hole_results, scene_valid = simulate_all_hole_drops(c)

        # Check if any ball got stuck
        any_stuck = False
        for hole_idx, result in hole_results.items():
            if result["bin"] in ["stuck", "error"]:
                any_stuck = True
                break

        if any_stuck:
            stuck_scenes.append({"config": c, "results": hole_results})
            print(f"    Scene {scene_idx + 1}: REJECTED (ball stuck)")
        else:
            valid_scenes.append({"config": c, "results": hole_results})
            print(f"    Scene {scene_idx + 1}: ACCEPTED")

    print(f"\nResults:")
    print(f"  Valid scenes: {len(valid_scenes)}")
    print(f"  Stuck scenes: {len(stuck_scenes)}")
    print(f"  Acceptance rate: {len(valid_scenes)/num_scenes*100:.1f}%")

    return valid_scenes, stuck_scenes


def print_scene_results(scenes, title="Scene Results"):
    """
    Print detailed results for a list of scenes.

    Parameters:
    -----------
    scenes : list
        List of scene dictionaries with config and results
    title : str
        Title for the output
    """
    print(f"\n{title}")
    print("=" * 60)

    for i, scene in enumerate(scenes):
        print(f"\nScene {i + 1}:")
        print(
            f"  Rectangle1: ({scene['config']['obstacles']['rectangle1']['position']['x']:.1f}, {scene['config']['obstacles']['rectangle1']['position']['y']:.1f})"
        )
        print(
            f"  Rectangle2: ({scene['config']['obstacles']['rectangle2']['position']['x']:.1f}, {scene['config']['obstacles']['rectangle2']['position']['y']:.1f})"
        )
        print(
            f"  Pentagon: ({scene['config']['obstacles']['pentagon']['position']['x']:.1f}, {scene['config']['obstacles']['pentagon']['position']['y']:.1f})"
        )
        print("  Hole results:")

        for hole_idx in range(len(scene["config"]["hole_positions"])):
            result = scene["results"][hole_idx]
            hole_x = scene["config"]["hole_positions"][hole_idx]
            print(f"    Hole {hole_idx + 1} (x={hole_x:.1f}): {result['bin']}")


def main():
    """
    Main function to demonstrate the ball drop analysis.
    """
    print("Ball Drop Analysis Demo")
    print("=" * 50)

    # Generate a single scene and test all holes
    print("\n1. Testing single scene with all holes:")
    c = config.get_config()
    results, scene_valid = simulate_all_hole_drops(c)

    print(f"\nResults for single scene:")
    for hole_idx, result in results.items():
        hole_x = c["hole_positions"][hole_idx]
        print(f"  Hole {hole_idx + 1} (x={hole_x:.1f}): {result['bin']}")

    # Filter multiple scenes
    print(f"\n2. Filtering multiple scenes:")
    valid_scenes, stuck_scenes = filter_scenes_without_stuck_balls(num_scenes=20)

    # Print results for valid scenes
    if valid_scenes:
        print_scene_results(
            valid_scenes[:3], "Sample Valid Scenes"
        )  # Show first 3 valid scenes

    if stuck_scenes:
        print_scene_results(
            stuck_scenes[:2], "Sample Stuck Scenes"
        )  # Show first 2 stuck scenes


if __name__ == "__main__":
    main()
