from numbers import Number
from typing import Iterable
from matplotlib import cm, colors, gridspec, pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from numpy import ndarray
from scipy import signal
from Fitter import Trace
from PerceptualUniform import Luminance
from Scalebar import plotScalebar
from plot_cc_matrix import CcMatrices, CcMatrix


def hideAxis(ax: Axes):
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_xticks((), minor=True)
    ax.set_yticks((), minor=True)

    ax.set_facecolor("none")
    ax.set_xlabel("", labelpad=0)
    ax.set_ylabel("", labelpad=0)


class ComparePlotter:
    data_color = "black"
    model_color = 0.833 * np.ones(3)

    def __init__(self, *args):
        self.__traces: list[Trace] = args

    @property
    def data(self):
        return self.__traces[0]

    @property
    def model(self):
        return self.__traces[1]

    ##### Functions to plot data or model properties onto given Axes #####
    def getTrace(self, index: int = 0):
        return self.__traces[index]

    def plotAnalyticDistribution2D(
        self,
        ax: Axes,
        index: int = 0,
        **kwargs,
    ):
        trace = self.getTrace(index)
        trace_as = trace.toAnalyticDistribution2D().normalizeByMax()
        ax.set_aspect("equal")
        return trace_as.plotPdfHeatmap(ax, **kwargs)

    def plotCrossingDistribution2D(
        self,
        ax: Axes,
        index: int = 0,
        **kwargs,
    ):
        trace = self.getTrace(index)
        crossing = trace.toCrossingDistribution2D(denoise=True)
        crossing = crossing.normalizeByMax()
        return crossing.plotPdfHeatmap(ax, **kwargs)

    def plotPsd(
        self,
        ax: Axes,
        index: int = 0,
        **kwargs,
    ):
        trace = self.getTrace(index)
        psd = trace.toPsd()
        ax.set_xscale("log")
        ax.set_yscale("log")
        return psd.plot(ax, **kwargs)

    def plotPsdPeak(
        self,
        ax: Axes,
        index: int = 0,
        **kwargs,
    ):
        trace = self.getTrace(index)
        psd = trace.toPsd()
        peak_frequency = psd.peak_frequency
        return ax.axvline(peak_frequency, **kwargs)

    def plotDataAnalyticDistribution2D(
        self,
        ax: Axes,
        **kwargs,
    ):
        return self.plotAnalyticDistribution2D(ax, 0, **kwargs)

    def plotDataCrossingDistribution2D(
        self,
        ax: Axes,
        **kwargs,
    ):
        return self.plotCrossingDistribution2D(ax, 0, **kwargs)

    def plotDataPsd(
        self,
        ax: Axes,
        **kwargs,
    ):
        return self.plotPsd(ax, 0, **kwargs)

    def plotModelAnalyticDistribution2D(
        self,
        ax: Axes,
        **kwargs,
    ):
        return self.plotAnalyticDistribution2D(ax, 1, **kwargs)

    def plotModelCrossingDistribution2D(
        self,
        ax: Axes,
        **kwargs,
    ):
        return self.plotCrossingDistribution2D(ax, 1, **kwargs)

    def plotModelPsd(self, ax: Axes, **kwargs):
        return self.plotPsd(ax, 1, **kwargs)

    ##### Functions to directly generate a figure comparing data and model #####
    def generateTraceFigure(self, t_span: tuple[float, float] = (None, None)):
        data = self.data.betweenTimes(*t_span)
        model = self.model.betweenTimes(*t_span)

        fig, ax = plt.subplots(1)
        data.plotTrace(
            ax,
            color=self.data_color,
            alpha=1,
            label="Data",
        )
        model.plotTrace(
            ax,
            color=self.model_color,
            alpha=0.83,
            label="Model",
        )

        ax.legend()
        ax.set_title("Trace")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Position [nm]")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        return fig

    def generateAnalytic2dFigure(self):
        data_as = self.data.toAnalyticDistribution2D()
        model_as = self.model.toAnalyticDistribution2D()

        fig, axs = plt.subplots(
            nrows=1,
            ncols=2,
            sharex=True,
            sharey=True,
        )
        axs: list[Axes]
        ax_data, ax_model = axs

        data_as.swapAxes().plotPdfHeatmap(
            ax_data,
            cmap="Greys",
        )
        ax_data.set_title("Data")
        ax_data.set_xlabel("Hilbert Transform")
        ax_data.set_ylabel("Position")
        ax_data.spines["top"].set_visible(False)
        ax_data.spines["right"].set_visible(False)
        ax_data.set_aspect("equal")

        model_as.swapAxes().plotPdfHeatmap(
            ax_model,
            cmap="Greys",
        )
        ax_model.set_title("Model")
        hideAxis(ax_model)
        ax_model.set_aspect("equal")

        return fig

    def generateVelocityFieldsFigure(self, bins: int = 24):
        data_vf = self.data.toAnalyticVelocity2D(
            bins=bins,
            weighted=False,
        )
        model_vf = self.model.toAnalyticVelocity2D(
            bins=bins,
            weighted=False,
        )

        arrow_scale = None
        fig, axs = plt.subplots(
            nrows=1,
            ncols=2,
            sharex=True,
            sharey=True,
        )
        axs: list[Axes]
        ax_data, ax_model = axs

        data_vf.swapAxes().plotPdfVectorField(
            ax_data,
            scale=arrow_scale,
            color=self.data_color,
        )
        ax_data.set_title("Data")
        ax_data.set_xlabel("Hilbert Transform")
        ax_data.set_ylabel("Position")
        ax_data.spines["top"].set_visible(False)
        ax_data.spines["right"].set_visible(False)
        ax_data.set_aspect("equal")

        model_vf.swapAxes().plotPdfVectorField(
            ax_model,
            scale=arrow_scale,
            color=self.data_color,
        )
        ax_model.set_title("Model")
        hideAxis(ax_model)
        ax_model.set_aspect("equal")

        return fig

    def generateCrossing2dFigure(self):
        data_distr = self.data.toCrossingDistribution2D(denoise=True)
        model_distr = self.model.toCrossingDistribution2D(denoise=True)

        fig, axs = plt.subplots(
            nrows=1,
            ncols=2,
            sharex=True,
            sharey=True,
        )
        axs: list[Axes]
        ax_data, ax_model = axs

        data_distr.swapAxes().plotPdfHeatmap(
            ax_data,
            cmap="Greys",
        )
        ax_data.set_xlim([data_distr.y_min, 0.5 * data_distr.y_max])
        ax_data.set_title("Data")
        ax_data.set_xlabel("Crossing Time Difference")
        ax_data.set_ylabel("Position")
        ax_data.spines["top"].set_visible(False)
        ax_data.spines["right"].set_visible(False)

        model_distr.swapAxes().plotPdfHeatmap(
            ax_model,
            cmap="Greys",
        )
        ax_model.set_title("Model")
        hideAxis(ax_model)

        return fig

    def generatePsdFigure(self):
        data_psd = self.data.toPsd(normalized=True)
        model_psd = self.model.toPsd(normalized=True)

        fig, ax = plt.subplots(1)

        data_psd.plot(
            ax,
            color=self.data_color,
            alpha=1,
            label="Data",
        )
        model_psd.plot(
            ax,
            color=self.model_color,
            alpha=1,
            label="Model",
        )

        ax.legend()
        ax.set_title("Power Spectral Density")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Power [s]")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        return fig


class TraceCostPlotter:
    @classmethod
    def bracketTracePair(cls, axss: list[tuple[Axes]], label: str = "Pair"):
        ax = axss[-1][-1]
        fig = ax.get_figure()

        ax.annotate(
            r"$]$",
            fontsize=128,
            xy=(1.02, 0.05),
            xycoords="axes fraction",
            ha="center",
            va="bottom",
        )

        norm = colors.Normalize(vmin=0, vmax=1)
        cmap = colors.LinearSegmentedColormap.from_list("", ["black", "black"])
        mappable = cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = fig.colorbar(
            mappable,
            ax=np.array(axss),
            fraction=0.001,
            pad=0,
            aspect=1,
        )
        cbar.set_ticks(())
        cbar.set_label(
            label,
            rotation=270,
            labelpad=12,
        )

    @classmethod
    def __getDensitySpan(cls, traces: list[Trace]):
        psds = [trace.toPsd() for trace in traces]
        densities = np.concatenate([psd.densities for psd in psds])
        return (densities.min(), densities.max())

    @classmethod
    def __getFrequencySpan(cls, traces: list[Trace]):
        psds = [trace.toPsd() for trace in traces]
        frequencies = np.concatenate([psd.frequencies for psd in psds])
        freq_min = frequencies[frequencies > 0].min()
        freq_max = frequencies.max()
        return (freq_min, freq_max)

    @classmethod
    def __getScalebarPositionScale(
        cls,
        traces: list[Trace],
        first: bool = True,
    ):
        traces_as = [trace.toAnalyticSignal() for trace in traces]
        trace_maxs = [2 * np.mean(np.abs(trace_as)) for trace_as in traces_as]

        trace_max = np.max(trace_maxs)
        if first:
            trace_max = trace_maxs[0]

        return trace_max

    @classmethod
    def __getScalebarTimescale(
        cls,
        traces: list[Trace],
        first: bool = True,
    ):
        psds = [trace.toPsd() for trace in traces]
        trace_periods = [psd.peak_period for psd in psds]

        trace_period = np.max(trace_periods)
        if first:
            trace_period = trace_periods[0]

        return trace_period

    def __init__(self, traces: list[Trace]):
        self.traces = traces

    def plotOnGridSpec(
        self,
        gs: gridspec.GridSpec,
        traces: list[Trace] = None,
        include_titles: list[bool] = True,
        include_scalebars: list[bool] = True,
        t_starts: float = 0,
        linecolors: list[str] = "black",
        cmaps: list[str] = "Greys",
        **kwargs,
    ):
        if traces is None:
            traces = self.traces
        trace_count = len(traces)

        if isinstance(include_scalebars, bool):
            include_scalebars = [include_scalebars for _ in range(trace_count)]
        if isinstance(include_titles, bool):
            include_titles = [include_titles for _ in range(trace_count)]
        if isinstance(linecolors, str) or not isinstance(linecolors, Iterable):
            linecolors = [linecolors for _ in range(trace_count)]
        if isinstance(cmaps, str) or not isinstance(cmaps, Iterable):
            cmaps = [cmaps for _ in range(trace_count)]
        if isinstance(t_starts, Number):
            t_starts = t_starts * np.ones(trace_count)

        axss: list[tuple[Axes]] = []
        fig = gs.figure
        for index in range(trace_count):
            trace = traces[index]
            include_scalebar = include_scalebars[index]
            include_title = include_titles[index]
            linecolor = linecolors[index]
            cmap = cmaps[index]
            t_start = t_starts[index]

            ax_trace = fig.add_subplot(gs[index, 0])
            ax_psd = fig.add_subplot(gs[index, 1])
            ax_as = fig.add_subplot(gs[index, 2])
            ax_crossing = fig.add_subplot(gs[index, 3])

            axs = (
                ax_trace,
                ax_psd,
                ax_as,
                ax_crossing,
            )
            axss.append(axs)
            self.plotOnAxes(
                axs,
                trace,
                include_titles=include_title,
                include_scalebars=include_scalebar,
                linecolor=linecolor,
                cmap=cmap,
                t_start=t_start,
                **kwargs,
            )

        return axss

    def plotOnAxes(
        self,
        axs: list[Axes],
        trace: Trace,
        cycle_count: float = 2,
        include_titles: bool = True,
        include_scalebars: bool = True,
        frequency_span: tuple[float, float] = None,
        psd_span: tuple[float, float] = None,
        trace_max: float = None,
        trace_period: float = None,
        t_start: float = 0,
        **kwargs,
    ):
        traces = self.traces
        if trace_max is None:
            trace_max = self.__getScalebarPositionScale(traces)
        if trace_period is None:
            trace_period = self.__getScalebarTimescale(traces)

        trace.toCrossingDistribution2D(denoise=True)

        plotWaveTrace(
            axs,
            trace,
            **kwargs,
        )
        self.__formatAxes(
            axs,
            trace_max=trace_max,
            trace_period=trace_period,
            cycle_count=cycle_count,
            t_start=t_start,
            frequency_span=frequency_span,
            psd_span=psd_span,
        )
        self.__setSharedAxes(axs)

        if include_scalebars:
            self.__plotScalebars(
                axs,
                trace_max=trace_max,
                trace_period=trace_period,
            )

        if include_titles:
            self.__setTitles(axs)

    @classmethod
    def __plotScalebars(
        cls,
        axs: list[Axes],
        trace_max: float,
        trace_period: float,
    ):
        ax_trace, _, ax_as, ax_crossing = axs
        plotScalebar(
            ax_trace,
            (-0.01, 0),
            0.5 * trace_period,
            trace_max,
            (f"{trace_period:.0f}ms", f"{trace_max:.0f}nm"),
            horizontal_padding=0.04,
            vertical_padding=0.002,
        )
        plotScalebar(
            ax_as,
            (0, 0),
            trace_max,
            trace_max,
            ("", ""),
        )
        plotScalebar(
            ax_crossing,
            (0, 0),
            0.5 * trace_period,
            trace_max,
            ("", ""),
        )

    @classmethod
    def __setSharedAxes(self, axs: list[Axes]):
        ax_trace, _, ax_as, ax_crossing = axs
        ax_as.sharey(ax_trace)  # analytic distribution shares y-axis of trace
        ax_crossing.sharey(ax_trace)  # crossing distribution shares y-axis of trace

    @classmethod
    def __setTitles(cls, axs: list[Axes]):
        ax_trace, ax_psd, ax_as, ax_crossing = axs
        ax_trace.set_title(r"$x(t)$")
        ax_psd.set_title(r"$\widetilde{S}_{xx}\{x\}(f)$")
        ax_as.set_title(
            r"$\rho_a (x, \mathcal{H}\{x\})$",
            y=1.035,
        )
        ax_crossing.set_title(r"$\vartheta\{x\} (\gamma, \Delta{t})$")
        axs[0].get_figure().align_titles(axs)

    def __formatAxes(
        self,
        axs: list[Axes],
        trace_max: float,
        trace_period: float,
        cycle_count: float = 3,
        t_start: float = 0,
        frequency_span: tuple = None,
        psd_span: tuple = None,
    ):
        traces = self.traces
        if frequency_span is None:
            frequency_span = self.__getFrequencySpan(traces)
        if psd_span is None:
            psd_span = self.__getDensitySpan(traces)

        ax_trace, ax_psd, ax_as, ax_crossing = axs

        ax_trace.set_facecolor("none")
        ax_trace.set_xlim(t_start, t_start + cycle_count * trace_period)
        ax_trace.set_ylim(-trace_max, trace_max)

        ax_psd.set_facecolor("none")
        ax_psd.set_xlim(*frequency_span)
        ax_psd.set_ylim(*psd_span)

        ax_as.set_facecolor("none")
        ax_as.set_xlim(-trace_max, trace_max)
        ax_as.set_ylim(-trace_max, trace_max)

        ax_crossing.set_facecolor("none")
        ax_crossing.set_xlim(0, 1.33 * trace_period)
        ax_crossing.set_ylim(-trace_max, trace_max)


def plotWaveTrace(
    axs: list[Axes],
    trace: Trace,
    include_labels: bool = False,
    include_peak_frequency: bool = True,
    cmap: str = "Greys",
    linecolor: str = "black",
    psd_linewidth: float = 0.25,
    psd_peak_linewidth: float = 0.5,
    trace_linewidth: float = 0.25,
    title_fontsize: str = None,
):
    plotter = ComparePlotter(trace)
    ax_trace = axs[0]
    ax_psd = axs[1]
    ax_as = axs[2]
    ax_crossing = axs[3]

    ##### Plot trace #####
    trace.plotTrace(
        ax_trace,
        color=linecolor,
        linewidth=trace_linewidth,
    )

    ##### Plot power spectral density #####
    plotter.plotPsd(
        ax_psd,
        color=linecolor,
        linewidth=psd_linewidth,
    )
    if include_peak_frequency:
        plotter.plotPsdPeak(
            ax_psd,
            color=linecolor,
            linestyle="dashed",
            linewidth=psd_peak_linewidth,
        )

    ##### Plot distributions of analytic signal #####
    plotter.plotAnalyticDistribution2D(
        ax_as,
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        swap_axes=True,
        clip_on=True,
    )
    ax_as.scatter(
        0,
        0,
        color=linecolor,
        marker="+",
    )

    ##### Plot distributions of position crossing #####
    plotter.plotCrossingDistribution2D(
        ax_crossing,
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        swap_axes=True,
    )

    hideAxis(ax_trace)
    hideAxis(ax_psd)
    hideAxis(ax_as)
    hideAxis(ax_crossing)

    if include_labels:
        handles, _ = ax_psd.get_legend_handles_labels()
        if len(handles) >= 1:
            ax_psd.legend()

        ax_psd.set_title(
            r"$\widetilde{S}_{xx}(f)$",
            fontsize=title_fontsize,
        )
        ax_as.set_title(
            r"$\rho_a (x, \mathcal{H}\{x\})$",
            fontsize=title_fontsize,
        )
        ax_crossing.set_title(
            r"$\vartheta(\gamma, \Delta{t})$",
            fontsize=title_fontsize,
        )


def generateWaveFigure(
    t: ndarray,
    eta_std: float,
    width_ratios: tuple = (2, 1, 1, 1),
    title_fontsize: str = None,
    figsize: tuple[float, float] = None,
):
    def setAxisLimits(
        axs: list[Axes],
        ymax: float = 1.0,
    ):
        ax_trace = axs[0]
        ax_psd = axs[1]
        ax_as = axs[2]
        ax_crossing = axs[3]

        ax_trace.set_xlim(0, 2)
        ax_trace.set_ylim(-ymax, ymax)

        ax_psd.set_xlim(10**-2.5, 10**2.5)
        ax_psd.set_ylim(10**-4.5, 10**2.5)

        ax_as.set_xlim(-ymax, ymax)
        ax_as.set_ylim(-ymax, ymax)

        ax_crossing.set_xlim(0, 1)
        ax_crossing.set_ylim(-ymax, ymax)

    eta = np.random.normal(0, eta_std, size=t.shape)
    ymax = 1 + 4 * eta_std
    trace_names = (
        "Sine",
        r"Square (50\% Duty)",
        r"Triangle (75\% Width)",
    )

    x_sine = np.sin(2 * np.pi * t) + eta
    x_square = signal.square(2 * np.pi * t) + eta
    x_triangle = signal.sawtooth(2 * np.pi * t + 0.5 * np.pi, 0.75) + eta

    trace_sine = Trace(t, x_sine)
    trace_square = Trace(t, x_square)
    trace_triangle = Trace(t, x_triangle)
    traces = [trace_sine, trace_square, trace_triangle]
    wave_count = len(traces)

    ncols = len(width_ratios)
    fig, axss = plt.subplots(
        nrows=wave_count,
        ncols=ncols,
        width_ratios=width_ratios,
        figsize=figsize,
    )
    axss: list[list[Axes]]

    ##### Plot waves (common) #####
    for wave_index in range(wave_count):
        axs = axss[wave_index]
        trace = traces[wave_index]
        title = trace_names[wave_index]
        include_labels = wave_index == 0

        axs[0].set_title(title, fontsize=title_fontsize)
        plotWaveTrace(
            axs,
            trace,
            include_peak_frequency=False,
            include_labels=include_labels,
            title_fontsize=title_fontsize,
        )
        setAxisLimits(axs, ymax=ymax)

    ##### Plot scalebar with first wave #####
    axs_sine = axss[0]
    ax_trace_sine = axs_sine[0]
    ax_psd_sine = axs_sine[1]
    ax_as_sine = axs_sine[2]
    ax_crossing_sine = axs_sine[3]

    # ax_psd_sine.spines["bottom"].set_visible(True)
    # ax_psd_sine.set_xticks(
    #     np.logspace(-2, 2, 5),
    #     labels=(r"$10^{-2}$", "", r"$10^0$", "", r"$10^2$"),
    # )

    scalebar_kwargs = {
        "fontsize": title_fontsize,
        "horizontal_padding": 0.04,
        "linewidth": 1,
    }
    plotScalebar(
        ax_trace_sine,
        (0, 0),
        1,
        1,
        (r"$T$", r"$A$"),
        **scalebar_kwargs,
    )
    plotScalebar(
        ax_psd_sine,
        (0, 0),
        1,
        0,
        (r"$1/T$", ""),
        fontsize=title_fontsize,
        linewidth=1,
        horizontal_padding=0.04,
        ha="right",
    )
    plotScalebar(
        ax_as_sine,
        (0, 0),
        1,
        1,
        (r"$A$", r"$A$"),
        **scalebar_kwargs,
    )
    plotScalebar(
        ax_crossing_sine,
        (0, 0),
        0.5,
        1,
        (r"$T/2$", r"$A$"),
        **scalebar_kwargs,
    )

    ##### Set shared axes #####
    for col_index in [0, 2, 3]:
        for wave_index in range(1, wave_count):
            ax = axss[0][col_index]
            axss[wave_index][col_index].sharex(ax)
            axss[wave_index][col_index].sharex(ax)

    axs_sine[2].sharey(axs_sine[0])  # analytic distribution shares y-axis of trace
    axs_sine[3].sharey(axs_sine[0])  # crossing distribution shares y-axis of trace

    return fig


if __name__ == "__main__":
    plt.rcParams["text.usetex"] = True
    plt.rcParams["text.latex.preamble"] = r"\usepackage{lmodern}"
    plt.rcParams["font.family"] = "lmodern"

    if True:
        fig = generateWaveFigure(
            t=np.linspace(0, 1000, int(1e6)),
            eta_std=0.25,
            title_fontsize="medium",
            figsize=(3.375, 0.75 * 3),
        )
        fig.tight_layout(pad=0)
        fig.subplots_adjust(
            top=0.90,
            right=0.99,
        )
        plt.show()

    if False:
        cell_indices = [0, 1, 2]  # , 6
        cycle_count = 12
        cc_matrices: list[CcMatrix] = CcMatrices.fromHdf5(
            f"cc_sac/dm-{cycle_count:d}psd.hdf5",
            is_diag=True,
        )

        cell_count = len(cell_indices)
        fig = plt.figure(
            figsize=(7, 2 * cell_count),
            layout="tight",
        )
        gs = fig.add_gridspec(
            nrows=2 * cell_count,
            hspace=0,
            left=0.045,
            right=0.97,
            bottom=0,
            top=0.95,
        )

        include_scalebars = [True, False]
        blue_color = Luminance.addUntilLuminance(0.25, "blue", "green")

        width_ratios = (4, 1, 1, 1)
        linecolors = ["black", blue_color]
        cmap_model = colors.LinearSegmentedColormap.from_list("", ["white", blue_color])
        cmaps = ["Greys", cmap_model]

        for index, cell_index in enumerate(cell_indices):
            cc_matrix = cc_matrices[cell_index]
            argmax = cc_matrix.argpercentile(1.0)
            t_lag = cc_matrix.lags[*argmax]
            data_start_time = cc_matrix.getDataChunkTimeStart(argmax[0])
            model_start_time = cc_matrix.getModelChunkTimeStart(argmax[1])

            data = cc_matrix.data
            model = cc_matrix.model
            data = data.rescale(t_amp=1e3)
            model = model.rescale(t_amp=1e3)
            traces = [data, model]

            plotter = TraceCostPlotter(traces)
            gs_set = gridspec.GridSpecFromSubplotSpec(
                nrows=len(traces),
                ncols=len(width_ratios),
                subplot_spec=gs[2 * index : 2 * index + 2],
                width_ratios=width_ratios,
                hspace=0,
                wspace=0.05,
            )

            include_titles = [index == 0, False]
            t_starts = 1e3 * np.array([data_start_time + t_lag, model_start_time])

            axss = plotter.plotOnGridSpec(
                gs_set,
                cycle_count=cycle_count,
                psd_linewidth=0.5,
                psd_peak_linewidth=1,
                trace_linewidth=1.5,
                include_scalebars=include_scalebars,
                include_titles=include_titles,
                linecolors=linecolors,
                cmaps=cmaps,
                t_starts=t_starts,
            )
            TraceCostPlotter.bracketTracePair(axss, label=f"Cell {cell_index+1:d}")

            if index == 0:
                ax_trace_data = axss[0][0]
                ax_trace_model = axss[1][0]

                ax_trace_data.set_ylabel(
                    "Measurement",
                    color=linecolors[0],
                    labelpad=12,
                )
                ax_trace_model.set_ylabel(
                    "Simulation",
                    color=linecolors[1],
                    labelpad=12,
                )

        plt.show()
