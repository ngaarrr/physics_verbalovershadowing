# Physics Verbal Overshadowing

A research project investigating verbal overshadowing effects in intuitive physics using a Plinko-style ball drop simulation. This project explores how different shape representations (abstract vs. non-abstract) affect physics predictions and behavior.

## Project Overview

This repository contains a physics simulation engine that studies how different visual representations of obstacles influence intuitive physics judgments. The simulation drops balls through a Plinko-style apparatus with either abstract or non-abstract shapes, measuring whether participants' predictions change based on the representation type.

## Features

- **Physics Simulation Engine**: Realistic ball physics using Pymunk physics engine
- **Dual Shape Representations**: Abstract (geometric) vs. non-abstract (natural) obstacle shapes
- **Automated Analysis**: Batch processing of multiple scenes with statistical analysis
- **Video Generation**: Automatic creation of demonstration videos for each condition
- **Shape Generation**: Fourier descriptor-based shape generation for structured artificial shapes
- **Scene Validation**: Automatic detection of valid physics scenarios without stuck balls

## Installation

### Prerequisites
1. Install [Miniconda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/macos.html)

### Setup Environment
```bash
# Create new conda environment
conda create -n physics_overshadowing python=3.8

# Activate environment
conda activate physics_overshadowing

# Install dependencies
pip install numpy matplotlib scipy KDEPy pymunk==6.0.0 pygame
```

## Usage

### Basic Simulation
```bash
# Run a single simulation
python engine.py

# Run with demonstration mode
python demonstrate.py
```

### Analysis and Batch Processing
```bash
# Analyze ball drops from all holes
python ball_drop_analysis.py

# Generate new shapes using Fourier descriptors
python fourier_descriptor_shape.py
```

### Configuration
- Modify `config.py` to adjust physics parameters, obstacle positions, and simulation settings
- Use JSON configuration files in the `configs/` directory for different experimental setups
- Pre-configured successful scenes are available in the `SUCCESSFUL/` directory

## Research Design

The project investigates verbal overshadowing by comparing:
- **Abstract shapes**: Geometric, easily describable obstacles
- **Non-abstract shapes**: Natural, less easily describable obstacles

Participants predict where balls will land, and the research measures whether the type of shape representation affects prediction accuracy or consistency.

## File Structure

- `engine.py` - Main physics simulation engine
- `demonstrate.py` - Interactive demonstration script
- `ball_drop_analysis.py` - Analysis and validation tools
- `fourier_descriptor_shape.py` - Shape generation using Fourier descriptors
- `config.py` - Configuration and parameter management
- `visual.py` - Visualization and rendering utilities
- `utils.py` - Helper functions and utilities
- `SUCCESSFUL/` - Pre-validated experimental scenes with results
- `fdshapes_structured_*` - Generated shape image collections

## Output

The simulation generates:
- Physics simulation data (ball trajectories, final positions)
- Video recordings of ball drops
- Statistical analysis of bin landing patterns
- Scene validation results

## Research Applications

This tool is designed for:
- Cognitive psychology research on intuitive physics
- Studies of verbal overshadowing in visual perception/problem solving.
- Educational research on physics learning

## Citation

If you use this code in your research, please cite the original physics overshadowing project and include appropriate acknowledgments for the physics simulation framework.
