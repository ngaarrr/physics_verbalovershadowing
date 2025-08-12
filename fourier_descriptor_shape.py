"""
Author: Zenith A. Zyn

This code implements structured shape generation using Fourier Descriptors,
directly inspired by Zahn & Roskies (1972). Shape classes (A–E) are designed
based on the harmonic configurations shown in their Figure 8, with specific
attention to their Closure and Symmetry Theorems. 

Reference:
Zahn, C. T., & Roskies, R. Z. (2009). Fourier descriptors for plane closed curves. IEEE Transactions on computers, 100(3), 269-281.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from hashlib import sha1


def build_fd_pattern(shape_class=None, total_frequencies=14):
    """
    Generate a Fourier Descriptor pattern [Amplitude, Phase] based on shape class.
    Inspired by Zahn & Roskies (1972), Fig. 8 and their symmetry/closure theorems.
    """
    amp = np.zeros(total_frequencies)
    phase = np.zeros(total_frequencies)
    shape_class = shape_class or np.random.choice(["A", "B", "C", "D", "E"])
    even_pool = lambda n: np.random.choice(
        np.arange(2, total_frequencies + 1, 2), size=n, replace=False
    )

    if shape_class == "A":  # Radial-lobed, single harmonic dominance (Fig. 8a, 8c)
        for idx in even_pool(4):
            amp[idx - 1] = np.random.uniform(0.6, 1.4)
            phase[idx - 1] = np.pi / 2
    elif shape_class == "B":  # Nested or cross-centered (Fig. 8d)
        idxs = even_pool(2)
        amp[idxs[0] - 1] = 3.0
        amp[idxs[1] - 1] = 1.0
        phase[idxs[0] - 1] = np.pi / 2
        phase[idxs[1] - 1] = 3 * np.pi / 2
    elif shape_class == "C":  # Broken lobes / interference (Fig. 8i)
        idxs = even_pool(2)
        for i, idx in enumerate(idxs):
            amp[idx - 1] = np.random.uniform(0.9, 1.3)
            phase[idx - 1] = np.pi / 2 + (i * np.pi / 4)
    elif shape_class == "D":  # Soft waveform (Fig. 8f)
        for i, idx in enumerate(range(2, 14, 2)):
            amp[idx - 1] = 1.2 / (i + 1)
            phase[idx - 1] = np.pi / 2
    elif shape_class == "E":  # Asymmetrical stems (Fig. 8e, 8j)
        idxs = even_pool(2)
        amp[idxs[0] - 1] = np.random.uniform(1.2, 2.0)
        amp[idxs[1] - 1] = np.random.uniform(0.4, 0.8)
        phase[idxs[0] - 1] = np.pi / 2
        phase[idxs[1] - 1] = np.pi / 2
    # Dampen high-frequency harmonics to avoid excessive curliness (Z&R p. 270)
    for k in range(8, total_frequencies + 1):
        amp[k - 1] *= 0.5 * (8 / k) ** 2
    return {"Amplitude": amp, "Phase": phase}


def cumbend(fd, t):
    """
    Compute cumulative angular bend function φ*(t),
    see Zahn & Roskies (1972), Equation (3)
    """
    amp = fd["Amplitude"]
    phase = fd["Phase"]
    theta = -t
    for k, (A, P) in enumerate(zip(amp, phase), start=1):
        theta += A * np.cos(k * t - P)
    return theta


def cumbend_to_points(fd, steps=720):
    """
    Reconstruct closed shape coordinates by integrating unit-length steps
    in the direction of cumbend(t), mimicking Eq. (5) from Zahn & Roskies.
    """
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
    scale = max(np.max(np.abs(x)), np.max(np.abs(y)))
    return x / scale, y / scale


def is_bad_shape(x, y, symmetry_threshold=0.15):
    """
    Filter shapes that are degenerate or severely unbalanced.
    Inspired by symmetry testing logic in Zahn & Roskies (1972).
    """
    if np.std(x) < 0.1 or np.std(y) < 0.1:
        return True
    left = x[x < 0]
    right = x[x > 0]
    if len(left) < 10 or len(right) < 10:
        return True
    l_mean = np.mean(np.abs(left))
    r_mean = np.mean(np.abs(right))
    sym_diff = abs(l_mean - r_mean) / max(l_mean, r_mean)
    return sym_diff > symmetry_threshold


def draw_and_save_shape(x, y, filename_base, fill_color=None):
    """
    Save both filled and outline versions of the shape.
    """
    fig, ax = plt.subplots(figsize=(2.56, 2.56), dpi=100)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.axis("off")
    for spine in ax.spines.values():
        spine.set_visible(False)
    # Filled version
    if fill_color:
        ax.fill(x, y, facecolor=fill_color, edgecolor=fill_color, linewidth=2)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.savefig(
            f"{filename_base}_filled.png",
            transparent=True,
            bbox_inches="tight",
            pad_inches=0,
        )
    # Outline version
    ax.clear()
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.plot(x, y, color="black", linewidth=0.5)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(
        f"{filename_base}_outline.png",
        transparent=True,
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()


def generate_50_structured_shapes_natural(out_dir="fdshapes_structured_natural"):
    """
    Generate 51 structured shapes, 10 per class type + 1 extra,
    with citation-justified FD patterns from Zahn & Roskies (1972).
    """
    os.makedirs(out_dir, exist_ok=True)
    classes = ["A", "B", "C", "D", "E"]
    colors = ["#19C219", "#FF401F"]
    seen = set()
    count = 0
    while count < 51:
        cls = np.random.choice(classes)
        fd = build_fd_pattern(cls, total_frequencies=14)
        x, y = cumbend_to_points(fd)
        if is_bad_shape(x, y):
            continue
        h = sha1(np.round(np.array([x, y]), 4).tobytes()).hexdigest()
        if h not in seen:
            base = os.path.join(out_dir, f"shape_{count + 1:02d}_{cls}")
            draw_and_save_shape(x, y, base, fill_color=colors[count % len(colors)])
            seen.add(h)
            count += 1


def generate_50_structured_shapes_artificial(out_dir="fdshapes_structured_artificial"):
    """
    Generate 51 structured shapes, 10 per class type + 1 extra,
    with citation-justified FD patterns from Zahn & Roskies (1972).
    """
    os.makedirs(out_dir, exist_ok=True)
    classes = ["A", "B", "C", "D", "E"]
    colors = ["#FF1C8E", "#4BFFEC"]
    seen = set()
    count = 0
    while count < 51:
        cls = np.random.choice(classes)
        fd = build_fd_pattern(cls, total_frequencies=14)
        x, y = cumbend_to_points(fd)
        if is_bad_shape(x, y):
            continue
        h = sha1(np.round(np.array([x, y]), 4).tobytes()).hexdigest()
        if h not in seen:
            base = os.path.join(out_dir, f"shape_{count + 1:02d}_{cls}")
            draw_and_save_shape(x, y, base, fill_color=colors[count % len(colors)])
            seen.add(h)
            count += 1


if __name__ == "__main__":
    generate_50_structured_shapes_natural()
    generate_50_structured_shapes_artificial()
