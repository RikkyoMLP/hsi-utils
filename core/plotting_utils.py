import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Union, Optional


@dataclass
class PlotInput:
    data: Union[List[float], np.ndarray]
    identifier: Optional[str] = None  # legend
    show_max: Optional[bool] = False  # should max value be marked
    show_min: Optional[bool] = False  # should min value be marked
    line_color: Optional[str] = None  # line color, use default color if None
    line_style: Optional[str] = None  # line style, use default style if None
    line_width: Optional[float] = None  # line width, use default width if None
    fill_color: Optional[str] = None  # fill color, do not fill if None


def _plot_on_axis(ax, plots: List[PlotInput]):
    if not plots:
        return

    for plot in plots:
        if not plot.data:
            print(f"Warning: Skipping plot '{plot.identifier}' due to empty data.")
            continue

        data = np.asarray(plot.data)
        x_axis = np.arange(len(data))

        plot_kwargs = {
            "label": plot.identifier,
            "color": plot.line_color,
            "linestyle": plot.line_style,
            "linewidth": plot.line_width,
        }
        # filter out all None values, matplotlib does not like them
        plot_kwargs = {k: v for k, v in plot_kwargs.items() if v is not None}

        # ensure we at least have a default style
        if "linestyle" not in plot_kwargs:
            plot_kwargs["linestyle"] = "-"

        ax.plot(x_axis, data, **plot_kwargs)

        # fill
        if plot.fill_color:
            ax.fill_between(x_axis, data, color=plot.fill_color, alpha=0.2)

        if plot.show_max:
            best_idx = np.argmax(data)
            best_val = data[best_idx]

            ax.plot(
                best_idx,
                best_val,
                marker="*",
                markersize=12,
                color="green",
                linestyle="None",
            )
            ax.annotate(
                f"{best_val:.4f}",
                (best_idx, best_val),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                color="green",
            )

        if plot.show_min:
            worst_idx = np.argmin(data)
            worst_val = data[worst_idx]
            ax.plot(
                worst_idx,
                worst_val,
                marker="v",
                markersize=10,
                color="red",
                linestyle="None",
            )
            ax.annotate(
                f"{worst_val:.4f}",
                (worst_idx, worst_val),
                textcoords="offset points",
                xytext=(0, -20),
                ha="center",
                color="red",
            )


def draw_plot(
    left_axis_plots: List[PlotInput],
    left_axis_label: str,
    right_axis_plots: List[PlotInput],
    right_axis_label: str,
    x_axis_label: str,
    title: str,
    output_path: str,
):
    """
    Generic dual Y-axis plotting function
    Accepts two sets of PlotInput, one for each Y-axis
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Always assume ax1 exists
    ax1.set_xlabel(x_axis_label)
    ax1.set_ylabel(left_axis_label, color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Plot on left axis
    _plot_on_axis(ax1, left_axis_plots)

    ax2 = None
    if right_axis_plots:
        ax2 = ax1.twinx()
        ax2.set_ylabel(right_axis_label, color="tab:blue")
        ax2.tick_params(axis="y", labelcolor="tab:blue")
        # Plot on right axis
        _plot_on_axis(ax2, right_axis_plots)

    # Title and legend
    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # Unified legend
    lines, labels = ax1.get_legend_handles_labels()
    if ax2:
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines += lines2
        labels += labels2

    if labels:  # Only show legend if there are labels
        ax1.legend(lines, labels, loc="lower right")

    try:
        plt.savefig(output_path)
        plt.close(fig)  # Explicitly close, release memory
        print(f"Plot saved successfully to '{output_path}'")
    except Exception as e:
        print(f"Error saving plot to '{output_path}': {e}")
