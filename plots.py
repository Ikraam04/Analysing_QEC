import numpy as np
import matplotlib.pyplot as plt


# Number of trials
trials = 2500

# Given probability and error count data
p_values = np.arange(0.0001, 0.2, 0.005)

p_values_2 = np.linspace(0.001, 0.5, 40)





def plot_qber_with_degeneracy(p_values, qber1, qber2, degeneracy_ratio,
                              legend1, legend2, title, color1, color2):
    """
    Plot two QBER curves and a degeneracy ratio on a secondary y-axis.

    Parameters:
        p_values (list): Depolarizing probabilities.
        qber1 (list): QBER values for the nondegen curve.
        qber2 (list): QBER values for the second curve.
        degeneracy_ratio (list): Degeneracy ratio values.
        legend1 (str): Label for the first QBER curve.
        legend2 (str): Label for the second QBER curve.
        title (str): Plot title.
        color1 (str): Color for the first QBER curve.
        color2 (str): Color for the second QBER curve.
    """
    fig, ax1 = plt.subplots(figsize=(6.5, 4.8))

    # Plot QBER curves
    ax1.plot(p_values, qber1, label=legend1, color=color1, marker='o', ms= "3")
    ax1.plot(p_values, qber2, label=legend2, color=color2, marker='s', ms = "3", ls = "--")
    ax1.set_xlabel('Depolarizing Probability (p)')
    ax1.set_ylabel('QBER')
    ax1.grid(True)

    # Secondary y-axis for degeneracy ratio
    ax2 = ax1.twinx()
    ax2.plot(p_values, degeneracy_ratio, label='Degeneracy Ratio',
             color='gray', linestyle='-', marker='x')
    ax2.set_ylabel('Degeneracy Ratio')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right')

    plt.title(title)
    plt.tight_layout()
    plt.show()


def multi_plot(*graphs, name=None, p_val=None, legend_names=None):
    """
    Parameters
    ----------
    *graphs : array-like
        One or more sequences of y-values to plot.

    name : str, optional
        Title of the plot. If None, no title is displayed.

    p_val : array-like
        Sequence of x-values (e.g., depolarizing probabilities). Required.

    legend_names : list of str, optional
        Labels for the plotted graphs. Must match the number of graphs.
        Defaults to "Graph 1", "Graph 2", etc., if not provided.

    Returns
    -------
    None
        Displays the plot using matplotlib.
    """
    if p_val is None:
        raise ValueError("p_val (x-axis values) must be provided.")

    if legend_names is None:
        legend_names = [f"Graph {i+1}" for i in range(len(graphs))]
    elif len(legend_names) != len(graphs):
        raise ValueError("Number of legend names must match number of graphs.")

    colors = plt.get_cmap('tab10').colors
    colour_list = [colors[0], colors[1], colors[2]]  # Red, Green, Blue

    plt.figure(figsize=(6.5,4.8))


    for idx, graph in enumerate(graphs):
        color = colour_list[idx % len(colour_list)]  # Loop colors if more than 3  # Reduce graph points
        plt.plot(p_val, graph, marker='o', ms=3, color=color, label=legend_names[idx])

    plt.xscale('linear')
    plt.yscale('log')
    plt.xlabel("Depolarizing Probability (p)")
    plt.ylabel("QBER (log scale)")
    plt.grid(True, ls="--", lw=0.5)
    plt.legend()
    if name is not None:
        plt.title(name)
    plt.show()



surface = np.load("surface.npy")
rotated = np.load("rotated.npy")
color = np.load("color.npy")

#multi_plot(surface,rotated,color, name="QBER For Demo", p_val= p_values, legend_names=["Surfac","Rotated","Color"])

surface_nondegen = np.load("surface_nondegen_comp.npy")
surface_degen = np.load("surface_degen_comp.npy")
surface_ratio = np.load("degen_ratios_surface_2.npy")

rotated_nondegen = np.load("rotated_nondegen_comp.npy")
rotated_degen = np.load("rotated_degen_comp.npy")
rotated_ratio = np.load("degen_ratios_rotated.npy")

color_nondegen = np.load("color_nondegen_comp.npy")
color_degen = np.load("color_degen_comp.npy")
color_ratio = np.load("degen_ratios_color.npy")

cmap = plt.get_cmap("tab20").colors


# plot_qber_with_degeneracy(p_values_2,surface_nondegen, surface_degen,surface_ratio, "non-deg", "deg","lala",cmap[2],cmap[3])

# plot_qber_with_degeneracy(p_values_2,rotated_nondegen, rotated_degen, rotated_ratio, "non-deg", "deg", "lala",cmap[0],cmap[1])

# plot_qber_with_degeneracy(p_values_2,color_nondegen, color_degen, color_ratio, "le","la","lela", cmap[4],cmap[5])