from matplotlib.colors import LinearSegmentedColormap, to_rgba
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def _get_colors(n_labels, cmap=None):
    """
    Generate distinct colors for any number of labels.

    Args:
        n_labels: Number of unique labels

    Returns:
        Array of RGB colors, shape (n_labels, 4) with RGBA values
    """

    if n_labels <= 10:
        colors = sns.color_palette("deep", n_labels)
    elif n_labels <= 20:
        # TODO: real 20 colors
        custom_hex_list = [
            "#3182bd",
            "#6baed6",
            "#9ecae1",
            "#e6550d",
            "#fd8d3c",
            "#fdae6b",
            "#fdd0a2",
            "#e377c2",
            "#f7b6d2",
            "#31a354",
            "#74c476",
            "#a1d99b",
            "#756bb1",
            "#9e9ac8",
            "#bcbddc",
            "#dadaeb",
        ]
        rgb_list = [to_rgba(c) for c in custom_hex_list[: n_labels]]
        colors = np.array(rgb_list)
    else:
        colors = plt.cm.hsv(np.linspace(0, 1, n_labels))

    return colors


def get_faded_cmap(name: str = "turbo", gamma: float = 3.0, step: int = 256) -> LinearSegmentedColormap:
    """
    Create custom colormap.

    Returns a colormap that blends from white at low values to turbo colors
    at high values, suitable for density visualization.

    Args:
        `name` (str): Name of the base colormap (default: "coolwarm").
        `gamma` (float): Power for alpha blending (default: 1.0).
        `step` (int): Number of steps in the colormap (default: 256).

    Returns:
        `cmap` (LinearSegmentedColormap): Custom colormap instance.
    """
    x = np.linspace(0, 1, step)
    base_cmap = plt.get_cmap(name)
    base_colors = base_cmap(x)

    # Apply alpha blending to create white-to-turbo gradient
    alpha = x ** (1 / gamma)
    blended_colors = np.ones((step, 4))
    blended_colors[:, : 3] = base_colors[:, : 3] * alpha[:, np.newaxis] + (1 - alpha[:, np.newaxis])
    blended_colors[:, 3] = 1.0

    blended_colors[: 4] = [1.0, 1.0, 1.0, 1.0]

    return LinearSegmentedColormap.from_list(f"faded_{name}", blended_colors)
