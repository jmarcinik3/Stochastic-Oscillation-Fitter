import io
from tkinter import Image
from matplotlib import gridspec, pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
from Fitter import Trace
from PerceptualUniform import Luminance
from Scalebar import plotScalebar


def hideAxis(ax: Axes):
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xticks(())
    ax.set_yticks(())


def imageFromFigure(fig: Figure):
    buffer = io.BytesIO()
    fig.canvas.print_png(buffer)
    buffer.seek(0)
    image = Image.open(buffer)
    return image


class CrossingPlotter:
    horizontal_padding = 0.04

    def __init__(
        self,
        data: Trace,
        tlim: tuple[float, float] = (None, None),
    ):
        self.data = data
        self.tlim = tlim

        crossing_pairs = data.toCrossingPairs()
        (
            self.crossing_distribution_2d,
            self.crossing_distributions,
        ) = crossing_pairs.toDistribution2D(t_amp=data.dt)

        self.__max_density = np.max(
            [distr._pdf_matrix.max() for distr in self.crossing_distributions]
        )

    @property
    def trace_span(self):
        trace = self.data
        ymin = np.min(trace.x)
        ymax = np.max(trace.x)
        return (ymin, ymax)

    @property
    def trace_period(self):
        psd = self.data.toPsd()
        peak_frequency = psd.peak_frequency
        trace_period = 1 / peak_frequency
        return trace_period

    def __len__(self):
        return len(self.crossing_distributions)

    def getCrossingLine(self, index: int):
        x_edges = self.crossing_distribution_2d.x_bin_edges[index : index + 2]
        return np.mean(x_edges)

    def plotCrossings(self, ax: Axes, index: int, **kwargs):
        crossing_distribution = self.crossing_distributions[index]
        x_crossing = self.getCrossingLine(index)
        t = self.data.t
        crossing_distribution.plotScatter(
            ax,
            t,
            x_crossing,
            **kwargs,
        )

    def plotCrossingBins(
        self,
        ax: Axes,
        index: int,
        s: float = 100,
        enter_color: str = "green",
        exit_color: str = "red",
        **kwargs,
    ):
        crossing_distribution = self.crossing_distributions[index]
        data = self.data
        t = data.t
        x = data.x

        crossing_distribution.plotDownEnterScatter(
            ax,
            t,
            x,
            color=enter_color,
            marker="_",
            s=s,
            **kwargs,
        )
        crossing_distribution.plotDownExitScatter(
            ax,
            t,
            x,
            color=exit_color,
            marker="_",
            s=s,
            **kwargs,
        )
        crossing_distribution.plotUpEnterScatter(
            ax,
            t,
            x,
            color=enter_color,
            marker="+",
            s=s,
            **kwargs,
        )
        crossing_distribution.plotUpExitScatter(
            ax,
            t,
            x,
            color=exit_color,
            marker="+",
            s=s,
            **kwargs,
        )

    def plotTrace(self, ax: Axes, **kwargs):
        ymin, ymax = self.trace_span
        trace_period = self.trace_period
        trace = self.data.betweenTimes(0, 12 * trace_period)

        trace.plotTrace(ax, **kwargs)
        ax.set_xlim(0, 12 * trace_period)
        ax.set_ylim(ymin, ymax)
        hideAxis(ax)

        t_scale = 2 * trace_period
        y_scale = 0.5 * (ymax - ymin)
        plotScalebar(
            ax,
            (0, 0),
            t_scale,
            y_scale,
            labels=(f"{t_scale:.0f}ms", f"{y_scale:.0f}nm"),
            horizontal_padding=CrossingPlotter.horizontal_padding,
            linewidth=2,
        )

        # ax.set_title(r"$x(t)$")

    def plotCrossingDistribution1D(
        self,
        ax: Axes,
        index: int,
        color: str = "black",
        include_label: bool = True,
        **kwargs,
    ):
        if include_label:
            x_crossing = self.getCrossingLine(index)
            label = f"$\\gamma={x_crossing:.0f}$nm"
            ax.annotate(
                label,
                (0, 0),
                xycoords="axes fraction",
                annotation_clip=False,
                color=color,
            )

        crossing_distribution = self.crossing_distributions[index]
        crossing_distribution.plotStairsPdf(
            ax,
            label=label,
            color=color,
            **kwargs,
        )

        trace_period = self.trace_period
        ax.set_xlim(0, trace_period)
        ax.set_ylim(None, self.__max_density)
        hideAxis(ax)

        ax.set_xlabel(r"$\Delta{t}$")
        ax.set_ylabel(r"$\Delta\Theta_\gamma\{x\}$")
        # ax.set_title(r"$\Delta\Theta_\gamma\{x\}(\Delta{t})$")

    def plotCrossingLine(self, ax: Axes, index: int, **kwargs):
        x_crossing = self.getCrossingLine(index)
        ax.axhline(x_crossing, **kwargs)

    def plotCrossingDistribution2D(self, ax: Axes, **kwargs):
        crossing_distribution_2d = self.crossing_distribution_2d
        heatmap = crossing_distribution_2d.swapAxes().plotPdfHeatmap(ax, **kwargs)

        ymin, ymax = self.trace_span
        trace_period = self.trace_period
        ax.set_xlim(0, trace_period)
        ax.set_ylim(ymin, ymax)
        hideAxis(ax)

        y_scale = 0.5 * (ymax - ymin)
        plotScalebar(
            ax,
            (0, 0),
            0.5 * trace_period,
            y_scale,
            labels=(f"{0.5*trace_period:.0f}ms", f"{y_scale:.0f}nm"),
            horizontal_padding=CrossingPlotter.horizontal_padding,
        )

        fig = ax.get_figure()
        cbar = fig.colorbar(heatmap, ax=ax)
        cbar.set_ticks(())
        cbar.set_label(
            r"$\vartheta \{x\} \left( \gamma, \Delta{t} \right)$",
            rotation=270,
            labelpad=12,
        )

        ax.set_xlabel(r"$\Delta{t}$")
        ax.xaxis.get_label().set_position((0.75, None))
        ax.set_ylabel(r"$\gamma$")
        ax.yaxis.get_label().set_position((None, 0.75))


def generateCrossingAnimation(
    plotter: CrossingPlotter,
    filepath: str,
    linewidth: float = 2,
    **kwargs,
):
    def generateCrossingFrame(index: int):
        fig, axs = plt.subplots(1, 3, **kwargs)
        axs: list[Axes]
        ax_trace, ax_distribution_1d, ax_distribution_2d = axs

        plotter.plotTrace(
            ax_trace,
            color="black",
            zorder=0,
        )
        plotter.plotCrossingBins(
            ax_trace,
            index,
            zorder=1,
        )

        plotter.plotCrossingDistribution2D(
            ax_distribution_2d,
            cmap="Greys",
        )
        plotter.plotCrossingLine(
            ax_distribution_2d,
            index,
            color="black",
            linestyle="dashed",
            linewidth=linewidth,
        )

        plotter.plotCrossingDistribution1D(
            ax_distribution_1d,
            index,
            color="black",
            fill=True,
        )

        ax_trace.set_xlim(plotter.tlim)
        ax_trace.set_xticks(plotter.tlim)
        return fig

    figs = list(map(generateCrossingFrame, range(len(plotter))))
    images = list(map(imageFromFigure, figs))
    images[0].save(
        filepath,
        save_all=True,
        append_images=images[1:],
        duration=250,
        loop=0,
    )


def generateCrossingFigure(
    plotter: CrossingPlotter,
    include_crossings: bool = True,
    include_1d: bool = True,
    include_2d=True,
    trace_linewidth: float = 1,
    crossing_linewidth: float = 1,
    **kwargs,
):
    distr_count = len(plotter)
    crossing_colors = (
        Luminance.addUntilLuminance(0.333, "blue", "green"),
        Luminance.addUntilLuminance(0.666, "red", "green"),
    )
    crossing_percentiles = np.array([0.25, 0.5])
    crossing_indices = np.int64(np.round(crossing_percentiles * (distr_count + 1))) - 1

    fig = plt.figure(**kwargs)
    gs = gridspec.GridSpec(
        nrows=2,
        ncols=2,
        figure=fig,
        hspace=0.0,
        wspace=0.0,
    )

    ##### Plot trace #####
    ax_trace = fig.add_subplot(gs[0, :])
    plotter.plotTrace(
        ax_trace,
        color="black",
        linewidth=trace_linewidth,
        zorder=-1,
    )

    if include_crossings:
        for index, crossing_index in enumerate(crossing_indices):
            crossing_color = crossing_colors[index]
            plotter.plotCrossingLine(
                ax_trace,
                crossing_index,
                color=crossing_color,
                linestyle="dashed",
                linewidth=crossing_linewidth,
                zorder=0,
            )
            plotter.plotCrossings(
                ax_trace,
                crossing_index,
                color=crossing_color,
                marker="o",
                s=(2 * crossing_linewidth) ** 2,
                zorder=0,
            )

    ##### Plot 1D distributions #####
    if include_1d:
        ax_distrs_1d = fig.add_subplot(gs[1, 0])
        for index, crossing_index in enumerate(crossing_indices):
            crossing_color = crossing_colors[index]
            alpha = 1.0 if index == 0 else 0.95
            plotter.plotCrossingDistribution1D(
                ax_distrs_1d,
                crossing_index,
                fill=True,
                color=crossing_color,
                alpha=alpha,
            )

    ##### Plot 2D distribution #####
    if include_2d:
        assert include_1d
        ax_distr_2d = fig.add_subplot(
            gs[1, 1],
            sharex=ax_distrs_1d,
            sharey=ax_trace,
        )

        plotter.plotCrossingDistribution2D(
            ax_distr_2d,
            cmap="Greys",
            zorder=0,
        )
        for index, crossing_index in enumerate(crossing_indices):
            if include_crossings:
                crossing_color = crossing_colors[index]
                plotter.plotCrossingLine(
                    ax_distr_2d,
                    crossing_index,
                    color=crossing_color,
                    linestyle="dashed",
                    linewidth=crossing_linewidth,
                    zorder=0,
                )

    return fig


if __name__ == "__main__":
    data = Trace.fromCsv("traces_sac/cell0.csv")
    crossing_plotter = CrossingPlotter(data.rescale(t_amp=1e3))
    fig = generateCrossingFigure(
        crossing_plotter,
        include_crossings=True,
        include_1d=True,
        include_2d=True,
        trace_linewidth=2,
        crossing_linewidth=2,
        figsize=(3.375, 0.666 * 3.375),
        layout="constrained",
    )

    # % start: automatic generated code from pylustrator
    crossing_texts = fig.axes[1].texts
    crossing_texts[0].set(position=(0.0847, 0.8433))
    crossing_texts[1].set(position=(0.4781, 0.6035))
    # % end: automatic generated code from pylustrator

    plt.show()

    # generateCrossingAnimation(
    #     crossing_plotter,
    #     "output.gif",
    #     figsize=(7, 3),
    #     dpi=300,
    #     layout="constrained",
    # )
