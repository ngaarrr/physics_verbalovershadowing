#!/usr/bin/env python3
"""
Script to show professor ball drops from all ten holes with automatic closing.
This demonstrates the difference between abstract and non-abstract shapes.

Usage:
    python show_professor.py                                    # Use default files
    python show_professor.py abstract.json nonabstract.json    # Use custom files
"""

import json
import sys
import time
import os

# Add the current directory to Python path to import our modules
sys.path.append(".")

from engine import run_simulation


def show_drops_for_config(config_file, config_name, auto_close=True):
    """Show ball drops for all ten holes with a specific configuration."""
    print(f"\n{'='*80}")
    print(f"SHOWING {config_name.upper()} CONFIGURATION")
    print(f"Config file: {config_file}")
    print(f"{'='*80}")

    # Load the configuration
    with open(config_file, "r") as f:
        config = json.load(f)

    # Test all ten holes
    for hole_idx in range(10):  # 10 holes
        config["hole_dropped_into"] = hole_idx

        print(f"\n{'='*60}")
        print(f"HOLE {hole_idx + 1} - {config_name.upper()}")
        print(f"Starting position: x = {config['hole_positions'][hole_idx]:.1f}")
        print(f"{'='*60}")

        # Run simulation with visualization but NO auto-close for smooth transitions
        data = run_simulation(
            config, auto_close=False, simplified=(config_name == "nonabstract")
        )

        # Get final position
        final_pos = data["ball_position"][-1]
        final_x = final_pos["x"]
        final_y = final_pos["y"]

        # Determine bin
        center_x = config["med"]
        divider_half_thickness = config["divider_thickness"] / 2

        if final_y <= 165:
            if final_x < center_x - divider_half_thickness:
                bin_result = "LEFT"
            else:
                bin_result = "RIGHT"
        else:
            bin_result = "STUCK"

        print(f"\nFinal position: x={final_x:.1f}, y={final_y:.1f}")
        print(f"Result: {bin_result}")

        # Brief pause before next hole (no window closing)
        if hole_idx < 9:  # Don't pause after the last hole
            print("Moving to next hole...")
            time.sleep(1)  # Just 1 second pause


def main():
    """Main function to show both configurations."""

    # Get file paths from command line arguments or use defaults
    if len(sys.argv) == 3:
        # User provided both files
        abstract_file = sys.argv[1]
        nonabstract_file = sys.argv[2]
        print(f"Using custom files:")
        print(f"  Abstract: {abstract_file}")
        print(f"  Non-abstract: {nonabstract_file}")
    elif len(sys.argv) == 1:
        # Use default files
        config_dir = "SUCCESSFUL/SUCCESSFUL_SCENE_1"
        abstract_file = f"{config_dir}/abstract_config.json"
        nonabstract_file = f"{config_dir}/nonabstract_config.json"
        print(f"Using default files:")
        print(f"  Abstract: {abstract_file}")
        print(f"  Non-abstract: {nonabstract_file}")
    else:
        print("Usage:")
        print(
            "  python show_professor.py                                    # Use default files"
        )
        print(
            "  python show_professor.py abstract.json nonabstract.json    # Use custom files"
        )
        return

    print("\nPROFESSOR DEMONSTRATION SCRIPT")
    print("This will show ball drops from all ten holes for both configurations.")
    print("Each simulation will close automatically after completion.")
    print("\nPress Enter to start...")
    input()

    # Check if files exist
    if not os.path.exists(abstract_file):
        print(f"Error: Abstract config file not found: {abstract_file}")
        return

    if not os.path.exists(nonabstract_file):
        print(f"Error: Non-abstract config file not found: {nonabstract_file}")
        return

    # Show abstract configuration
    show_drops_for_config(abstract_file, "abstract", auto_close=True)

    print("\n" + "=" * 80)
    print("ABSTRACT CONFIGURATION COMPLETE")
    print("Now showing non-abstract configuration...")
    print("=" * 80)
    input("Press Enter to continue...")

    # Show non-abstract configuration
    show_drops_for_config(nonabstract_file, "nonabstract", auto_close=True)

    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE!")
    print("Both abstract and non-abstract configurations have been shown.")
    print("=" * 80)


if __name__ == "__main__":
    main()
