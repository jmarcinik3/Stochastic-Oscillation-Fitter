import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.axes import Axes
import numpy as np
from numpy import ndarray
from tqdm import tqdm
from Fitter import DifferentialEvolutionParser, Distribution2D, Trace, TraceFitter
import HairBundleModel


def hideAxis(ax: Axes):
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xticks(())
    ax.set_yticks(())


class CompareSuperimposedPlotter:
    data_color = "black"
    model_color = "red"

    @staticmethod
    def __alphaFromDistribution(
        distribution: Distribution2D,
        alpha_min: float = 0.0,
        alpha_max: float = 1.0,
        n: float = 1.0,
    ):
        pdf = distribution._pdf_matrix.T
        dpdf = pdf - np.min(pdf)
        dpdf = dpdf / np.max(dpdf)
        alpha = alpha_min + (alpha_max - alpha_min) * dpdf**n
        return alpha

    @classmethod
    def __plotTraces(
        cls,
        ax: Axes,
        data: Trace,
        model: Trace,
        t_amp: float = 1.0,
        x_amp: float = 1.0,
        x_shift: float = 0.0,
    ):
        model = model.rescale(
            t_amp=t_amp,
            x_amp=x_amp,
            x_shift=x_shift,
        )

        if model.sample_rate >= data.sample_rate:
            model = model.downsample(data)
        else:
            data = data.downsample(model)

        data.plotTrace(
            ax,
            color=cls.data_color,
            linewidth=0.5,
            alpha=0.67,
        )
        model.plotTrace(
            ax,
            color=cls.model_color,
            linewidth=0.5,
            alpha=0.67,
        )

        t_max = min(np.max(data.t), np.max(model.t))
        ax.set_xlabel("time")
        ax.set_ylabel("bundle position")
        ax.set_xlim([0, t_max])

    @classmethod
    def __plotAnalyticDistributions2D(
        cls,
        ax: Axes,
        data: Trace,
        model: Trace,
        x_amp: float = 1.0,
        x_shift: float = 0.0,
    ):
        data_as = data.toAnalyticSignal()
        data_distr = Distribution2D.fromSamples(data_as)

        model_as = model.toAnalyticSignal()
        model_distr = Distribution2D.fromSamples(model_as)
        model_distr = model_distr.rescale(x_amp, (x_shift, 0))

        data_distr.swapAxes().plotPdfHeatmap(
            ax,
            cmap="Greys",
            alpha=1,
        )
        model_distr.swapAxes().plotPdfHeatmap(
            ax,
            cmap="Reds",
            alpha=0.5,
        )

        ax.set_aspect("equal")
        ax.set_xlim([data_distr.y_min, data_distr.y_max])
        ax.set_xlabel("Hilbert")

    @classmethod
    def __plotCrossingDistributions2D(
        cls,
        ax: Axes,
        data: Trace,
        model: Trace,
        t_amp: float = 1.0,
        x_amp: float = 1.0,
        x_shift: float = 0.0,
    ):
        data_distr = data.toCrossingDistribution2D(denoise=True)
        model_distr = model.toCrossingDistribution2D(denoise=True)
        model_distr = model_distr.rescale(t_amp, x_amp, x_shift)

        data_distr.swapAxes().plotPdfHeatmap(
            ax,
            cmap="Greys",
            alpha=1,
        )
        model_distr.swapAxes().plotPdfHeatmap(
            ax,
            cmap="Reds",
            alpha=0.5,
        )

        ax.set_xlim([data_distr.y_min, 0.5 * data_distr.y_max])
        ax.set_xlabel("Crossing Time Difference")

    @classmethod
    def __plotPSDs(
        cls,
        ax: Axes,
        data: Trace,
        model: Trace,
        t_amp: float = 1.0,
        x_amp: float = 1.0,
    ):
        data_psd = data.toPsd().normalizeByArea()
        model_psd = model.toPsd().rescale(x_amp, t_amp).normalizeByArea()

        data_psd.plot(
            ax,
            color=cls.data_color,
            linewidth=0.3,
            alpha=0.67,
        )
        model_psd.plot(
            ax,
            color=cls.model_color,
            linewidth=0.3,
            alpha=0.67,
        )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Amplitude (Normalized)")

    @classmethod
    def __plotVelocityFields(
        cls,
        ax: Axes,
        data: Trace,
        model: Trace,
        x_amp: float = 1.0,
        x_shift: float = 0.0,
        bins: int = 24,
    ):
        data_vf = data.toAnalyticVelocity2D(
            bins=bins,
            density=True,
            weighted=False,
        )
        model_vf = model.toAnalyticVelocity2D(
            bins=bins,
            density=True,
            weighted=False,
        )
        model_vf = model_vf.rescale(x_amp, (x_shift, 0)).normalizeByArea()

        arrow_scale = None
        data_alpha = cls.__alphaFromDistribution(
            data_vf.toMagnitudeDistribution(),
            alpha_min=0.2,
            n=0.5,
        )
        model_alpha = cls.__alphaFromDistribution(
            model_vf.toMagnitudeDistribution(),
            alpha_min=0.2,
            n=0.5,
        )

        model_vf.swapAxes().plotPdfVectorField(
            ax,
            scale=arrow_scale,
            color=cls.model_color,
            alpha=model_alpha,
        )
        data_vf.swapAxes().plotPdfVectorField(
            ax,
            scale=arrow_scale,
            color=cls.data_color,
            alpha=data_alpha,
        )

        ax.set_aspect("equal")
        ax.set_xlabel("Hilbert")
        ax.set_ylabel("Position")

    def __init__(
        self,
        data: Trace,
        model: Trace,
        t_amp: float = 1.0,
        x_amp: float = 1.0,
        x_shift: float = 0.0,
    ):
        self.data = data
        self.model = model
        self.t_amp = t_amp
        self.x_amp = x_amp
        self.x_shift = x_shift

    def plotTraces(self, ax: Axes):
        model = self.model
        self.__plotTraces(
            ax,
            self.data,
            model.rescale(t_shift=model.t[0]),
            t_amp=self.t_amp,
            x_amp=self.x_amp,
            x_shift=self.x_shift,
        )

    def plotAnalyticDistributions2D(self, ax: Axes):
        self.__plotAnalyticDistributions2D(
            ax,
            self.data,
            self.model,
            x_amp=self.x_amp,
            x_shift=self.x_shift,
        )

    def plotCrossingDistributions2D(self, ax: Axes):
        self.__plotCrossingDistributions2D(
            ax,
            self.data,
            self.model,
            t_amp=self.t_amp,
            x_amp=self.x_amp,
            x_shift=self.x_shift,
        )

    def plotPSDs(self, ax: Axes):
        self.__plotPSDs(
            ax,
            self.data,
            self.model,
            t_amp=self.t_amp,
            x_amp=self.x_amp,
        )

    def plotVelocityFields(self, ax: Axes):
        self.__plotVelocityFields(
            ax,
            self.data,
            self.model,
            x_amp=self.x_amp,
            x_shift=self.x_shift,
        )

    def generateFigure(
        self,
        tlim: tuple[float, float] = (None, None),
        **kwargs,
    ):
        fig = plt.figure(**kwargs)
        gs = fig.add_gridspec(
            nrows=2,
            ncols=3,
            width_ratios=[3, 1, 1],
        )

        ax_trace = fig.add_subplot(gs[0, 0])
        ax_crossing = fig.add_subplot(
            gs[0, 2],
            sharey=ax_trace,
        )
        ax_fourier = fig.add_subplot(gs[1, :])

        gs_distr = gridspec.GridSpecFromSubplotSpec(
            nrows=2,
            ncols=1,
            subplot_spec=gs[0, 1],
        )
        ax_analytic = fig.add_subplot(
            gs_distr[0],
            sharey=ax_trace,
        )
        ax_velocity = fig.add_subplot(
            gs_distr[1],
            sharex=ax_analytic,
            sharey=ax_analytic,
        )

        self.plotTraces(ax_trace)
        ax_trace.set_xlim(tlim)

        self.plotAnalyticDistributions2D(ax_analytic)
        self.plotCrossingDistributions2D(ax_crossing)
        self.plotPSDs(ax_fourier)
        self.plotVelocityFields(ax_velocity)
        return fig


class TracePlotter:
    trace_color = "black"
    hilbert_color = 0.833 * np.ones(3)

    def __init__(self, data: Trace):
        self.data = data

    def plotAnalyticSignal(self, ax: Axes):
        data_as = self.data.toAnalyticDistribution2D()
        data_as.swapAxes().plotPdfHeatmap(
            ax,
            cmap="Greys",
        )

        ymin = min(data_as.x_min, data_as.y_min)
        ymax = max(data_as.x_max, data_as.y_max)
        ax.set_ylim((ymin, ymax))

        ax.set_xlabel("Hilbert Transform")
        ax.set_ylabel("Position")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_aspect("equal")

    def plotAnalyticDistribution2D(
        self,
        ax: Axes,
        include_hilbert: bool = True,
    ):
        data = self.data
        data_as = data.toAnalyticSignal()
        as_real: ndarray = np.real(data_as)
        as_imag: ndarray = np.imag(data_as)

        ax.plot(
            data.t,
            as_real,
            color=self.trace_color,
            alpha=1,
            label="Signal",
        )

        if include_hilbert:
            ax.plot(
                data.t,
                as_imag,
                color=self.hilbert_color,
                alpha=1,
                label="Hilbert",
            )
            ax.legend(loc="upper left")

        ymin = min(as_real.min(), as_imag.min())
        ymax = max(as_real.max(), as_imag.max())
        ax.set_ylim((ymin, ymax))

        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Position [nm]")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_aspect("equal")


if __name__ == "__main__":
    plt.rcParams["text.usetex"] = True
    plt.rcParams["text.latex.preamble"] = r"\usepackage{lmodern}"
    plt.rcParams["font.family"] = "lmodern"
    np.set_printoptions(precision=2, suppress=True)

    """Demo for SdeFitter using HairBundleModel.Nondimensional"""
    if False:
        model_sym = HairBundleModel.Nondimensional(
            Cmin=1,  # constant climbing rate
            tauT0=0,  # equilibrium transduction-channel gating
            taugs0=1,  # fast gating-spring time constant
            Cm=1,  # moderate calcium-feedback strength on motor
            xc=0,  # null offset on gating-spring force
            chihb=1,  # moderate gating-spring coupling to bundle position
            Smin=0,  # maximum change in slipping rate, relative to calcium bound to motor
            # Cgs=1000,  # strong calcium-feedback strength on gating-spring
            # tauhb0=1,  # moderate time constant for bundle position
            # Smax=0.5,  # comparable effect from slipping and climbing
            # Ugsmax=10,  # moderate elastic potential energy for gating spring
            # chia=1,  # moderate gating-spring coupling to motor position
            # kgsmin=1,  # constant gating-spring stiffness
            # taum0=10,  # moderate time constant for Ca-feedback at motor
            # dE0=1,  # weak free-energy for channel opening,
        )
        model_handler = model_sym.generateHandler(
            t=np.linspace(0, int(1000 * 32), int(500000 * 32)),
            x0=np.array([-1, -1, 0.5, 0.5]),
            t_start=50,
            generator=np.random.default_rng(),
            noise_count=2,
        )

        data_indices = [135, 137, 157, 162, 192, 203, 211, 218, 229, 233, 235]
        for i in tqdm(data_indices):
            data = Trace.fromCsv(f"traces_ap/cell{i:d}.csv")
            fitter = TraceFitter(data, weights=[0.5, 0.1, 0, 0.4])
            de_parser = DifferentialEvolutionParser.fromHdf5(
                f"traces_ap/cell{i:d}-2000.hdf5"
            )

            parameter_values = de_parser.population(-1, 0, None)
            model = model_handler.generateModel(parameter_values, inds=0)
            fit = fitter.fit(model)
            print(parameter_values)
            print(fit.fun, fit.x)
            t_amp, x_amp, x_shift = fit.x
            model = model.rescale(
                t_amp=t_amp,
                x_amp=x_amp,
                x_shift=x_shift,
            )
            model.toCsv(f"traces_ap/fit{i:d}-2000.csv")

    """Demo for CompareSuperimposedPlotter"""
    if False:
        i = 0
        data = Trace.fromCsv(f"traces/cell{i:d}.csv")
        model = Trace.fromCsv(f"traces/fit{i:d}-2000.csv")
        compare_plotter = CompareSuperimposedPlotter(
            data,
            model,
            # t_amp=t_amp,
            # x_amp=x_amp,
            # x_shift=x_shift,
        )
        fig = compare_plotter.generateFigure(figsize=(14, 7))
        fig.tight_layout()
        plt.show()
