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
        qber1 (list): QBER values for the first curve.
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

    # If legend names are not given, default to "Graph 1", "Graph 2", etc.
    if legend_names is None:
        legend_names = [f"Graph {i+1}" for i in range(len(graphs))]
    elif len(legend_names) != len(graphs):
        raise ValueError("Number of legend names must match number of graphs.")

    colors = plt.get_cmap('tab10').colors
    colour_list = [colors[0], colors[1], colors[2]]  # Red, Green, Blue

    plt.figure(figsize=(6.5,4.8))

    # Only take every other point: slice with step 2
    p_val_reduced = p_val[::2]

    for idx, graph in enumerate(graphs):
        color = colour_list[idx % len(colour_list)]  # Loop colors if more than 3
        graph_reduced = graph[::2]  # Reduce graph points
        plt.plot(p_val_reduced, graph_reduced, marker='o', ms=3, color=color, label=legend_names[idx])

    plt.xscale('linear')
    plt.yscale('log')
    plt.xlabel("Depolarizing Probability (p)")
    plt.ylabel("QBER (log scale)")
    plt.grid(True, ls="--", lw=0.5)
    plt.legend()
    if name is not None:
        plt.title(name)
    plt.show()



