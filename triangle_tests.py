import copy
from typing import Callable, Generator
import h5py
from matplotlib import gridspec, pyplot as plt, ticker
from matplotlib.axes import Axes
import numpy as np
from numpy import ndarray
from scipy import optimize, signal
from tqdm import tqdm

from Fitter import (
    DifferentialEvolutionParser,
    Distribution2D,
    TraceFitter,
    Trace,
    TraceCoster,
)


def generateComparisonFigure(
    hdf5_filepath: str,
    point_color=(0.8, 0.8, 0.8),
    **kwargs,
):
    parameter_names = (
        r"$w$",
        r"$\sigma_\triangle$",
        r"$f$",
        r"$A$",
        r"$x_0$",
    )
    triangle_hdf5 = TriangleHdf5(hdf5_filepath)

    fig = plt.figure(**kwargs)
    gs_rows = fig.add_gridspec(
        1,
        2,
        width_ratios=(3, 2),
        hspace=0,
        wspace=0,
    )
    gs_top = gridspec.GridSpecFromSubplotSpec(
        2,
        1,
        subplot_spec=gs_rows[0],
        hspace=0,
        wspace=0,
    )
    gs_bottom = gridspec.GridSpecFromSubplotSpec(
        3,
        1,
        subplot_spec=gs_rows[1],
        hspace=0,
        wspace=0,
    )
    axs: list[Axes] = []

    for index in range(len(parameter_names)):
        if index <= 1:
            ax = fig.add_subplot(gs_top[index])
        elif index >= 2:
            ax = fig.add_subplot(gs_bottom[index - 2])
        axs.append(ax)

        name = parameter_names[index]
        triangle_hdf5.plot(
            ax,
            index,
            color=point_color,
            marker=".",
            clip_on=False,
        )
        ax.plot(
            [-10, 10],
            [-10, 10],
            color="black",
            linestyle="solid",
        )

        triangle_hdf5.formatAxis(ax, index)
        ax.set_title(
            name,
            y=1,
            pad=0,
            verticalalignment="top",
        )

    ax_top_left = axs[0]
    ax_top_left.set_xlabel("Actual")
    ax_top_left.set_ylabel("Fit")
    return fig


def generateConvergenceFigure(
    dataset: h5py.Dataset,
    parameter_names: str,
    microcolor: str = "black",
    microalpha: float = 0.2,
    macrocolor: str = "red",
    statistic: Callable[[ndarray], ndarray] = np.mean,
):
    de_parser = DifferentialEvolutionParser.fromHdf5(dataset)
    parameter_values = dataset.attrs["real_parameter"]

    fig, axs = plt.subplots(3, sharex="col")
    axs: list[Axes]
    ax_cost = axs[0]
    ax_parameters = axs[1:]

    de_parser.plotEnergies(
        ax_cost,
        color=microcolor,
        alpha=microalpha,
    )
    de_parser.plotEnergies(
        ax_cost,
        statistic,
        color=macrocolor,
    )

    ax_cost.set_ylabel("Cost")
    ax_cost.set_xscale("log")
    ax_cost.set_yscale("log")
    ax_cost.set_xlim(1, None)
    ax_cost.set_ylim(0.001, 1.0)

    for parameter_index in range(len(parameter_names)):
        parameter_name = parameter_names[parameter_index]
        ax_parameter = ax_parameters[parameter_index]

        de_parser.plotPopulation(
            ax_parameter,
            parameter_index,
            color=microcolor,
            alpha=microalpha,
            zorder=1,
        )
        de_parser.plotPopulation(
            ax_parameter,
            parameter_index,
            statistic,
            color=macrocolor,
            zorder=1,
        )

        parameter_value = parameter_values[parameter_index]
        ax_parameter.axhline(
            parameter_value,
            linestyle="dashed",
            color=microcolor,
            zorder=0,
        )

        if isinstance(parameter_name, str):
            ax_parameter.set_ylabel(parameter_name)
            ax_parameter.set_ylim(0, 1)

    for ax in axs:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    return fig


def appendRescalingParameters(hdf5_filepath: str, t: ndarray):
    data_generator = TriangleGenerator(t=t)
    model_generator = TriangleGenerator(t=t)

    def getRescaling(group: h5py.Group):
        actual_parameters = group.attrs["real_parameter"]
        weights = group.attrs["weights"]

        width, eta, t_amp, x_amp, x_shift = actual_parameters
        data = data_generator.generateTrace(np.array([width, eta]))
        data = data.rescale(
            t_amp=t_amp,
            x_amp=x_amp,
            x_shift=x_shift,
        )
        fitter = TraceFitter(data, weights=weights)

        fit_parameters = group["populations"]
        width, eta = fit_parameters[-1, 0, :]
        model = model_generator.generateTrace(np.array([width, eta]))
        fit = fitter.fit(model)
        return fit

    with h5py.File(hdf5_filepath, "a") as file:
        for group_name in tqdm(file.keys()):
            group = file[group_name]
            fit = getRescaling(group)
            t_amp, x_amp, x_shift = fit.x

            fit_parameters = group["populations"]
            width, eta = fit_parameters[-1, 0, :]
            fit_parameters = np.array([width, eta, t_amp, x_amp, x_shift])
            group.attrs["fit_parameter"] = fit_parameters


class TriangleGenerator:
    def __init__(self, t: ndarray, generator: Generator = None):
        if generator is None:
            generator = np.random.default_rng()

        self.t = t
        self.generator = copy.deepcopy(generator)
        self.x_noise: ndarray = generator.normal(0, 1, t.size)

    def __generatePosition(self, parameter_values: ndarray):
        assert parameter_values.size == 2
        width, noise_strength = parameter_values
        assert 0 <= width <= 1 and noise_strength >= 0
        t = self.t

        x_noise = noise_strength * self.x_noise
        x_wave = signal.sawtooth(2 * np.pi * t, width)
        return x_wave + x_noise

    def generateTrace(self, parameter_values: ndarray):
        t = self.t
        x = self.__generatePosition(parameter_values)
        return Trace(t, x)


class TriangleFitter:
    def __init__(
        self,
        t: ndarray,
        hdf5_filepath: str,
        population_size=16,
        sampling="sobol",
        strategy="rand1exp",
        recombination=0.7,
        mutation=(0.5, 1.0),
        width_bounds: tuple[float, float] = (0.0, 1.0),
        noise_bounds: tuple[float, float] = (0.0, 1.0),
        weights: ndarray = np.array([0.5, 0.1, 0, 0.4]),
    ):
        self.population_size = population_size
        self.sampling = sampling
        self.strategy = strategy
        self.recombination = recombination
        self.mutation = mutation

        self.time = t
        self.bounds = np.array([width_bounds, noise_bounds])
        self.weights = weights

        self.hdf5_filepath = hdf5_filepath
        self.trial_index = -1
        self.de_parser = DifferentialEvolutionParser()
        self.triangle_generator = TriangleGenerator(t=t)

        self.width: float = None
        self.eta: float = None
        self.t_amp: float = None
        self.x_amp: float = None
        self.x_shift: float = None

    @property
    def noise_bounds(self):
        return self.bounds[1, :]

    @property
    def width_bounds(self):
        return self.bounds[0, :]

    def __differentialEvolutionCallback(self, intermediate_result):
        group_name = f"{self.trial_index:d}"
        self.de_parser.appendResult(intermediate_result)
        if len(self.de_parser) % 10 != 0:
            return

        with h5py.File(self.hdf5_filepath, "a") as file:
            if group_name in file.keys():
                del file[group_name]
            group = file.create_group(group_name)
            self.de_parser.toHdf5(group)
            self.__metadataToHdf5(group)

    def __generateFakeDataCostFunction(self):
        data_generator = TriangleGenerator(t=self.time)
        data = data_generator.generateTrace(np.array([self.width, self.eta]))
        data = data.rescale(t_amp=self.t_amp, x_amp=self.x_amp, x_shift=self.x_shift)
        fitter = TraceFitter(data, weights=self.weights)
        fitter_handler = TraceCoster(
            self.triangle_generator.generateTrace,
            fitter.cost,
            include_parameters=False,
            include_trace=True,
        )
        return fitter_handler.cost

    def __instantiateParameters(self):
        self.width = np.random.uniform(*self.width_bounds)
        self.eta = np.random.uniform(*self.noise_bounds)
        self.t_amp = 10 ** np.random.uniform(-1, 1)
        self.x_amp = 10 ** np.random.uniform(-1, 1)
        self.x_shift = np.random.uniform(-10, 10)

    def __metadataToHdf5(self, group: h5py.Group):
        group.create_dataset("time", data=self.time, dtype=np.float64)
        group.attrs["parameter_name"] = ("width", "noise_strength")
        group.attrs["real_parameter"] = (
            self.width,
            self.eta,
            self.t_amp,
            self.x_amp,
            self.x_shift,
        )
        group.attrs["bounds"] = self.bounds
        group.attrs["weights"] = self.weights
        group.attrs["population_size"] = self.population_size
        group.attrs["mutation"] = self.mutation
        group.attrs["recombination"] = self.recombination
        group.attrs["sampling"] = self.sampling
        group.attrs["strategy"] = self.strategy

    def runFit(
        self,
        workers: int = None,
        maxiter: int = 500,
        atol: float = 0,
    ):
        self.trial_index += 1
        bounds = self.bounds
        population_size = self.population_size
        if workers is None:
            workers = min(population_size, 22)

        self.de_parser = DifferentialEvolutionParser()
        self.__instantiateParameters()

        actual_parameters = np.array(
            [
                self.width,
                self.eta,
                self.t_amp,
                self.x_amp,
                self.x_shift,
            ]
        )
        print("actual:", actual_parameters)

        soln = optimize.differential_evolution(
            self.__generateFakeDataCostFunction(),
            bounds=bounds,
            workers=workers,
            updating="deferred",
            popsize=np.floor(population_size / bounds.shape[0]),
            init=self.sampling,
            strategy=self.strategy,
            mutation=self.mutation,
            recombination=self.recombination,
            maxiter=maxiter,
            atol=atol,
            callback=self.__differentialEvolutionCallback,
            polish=False,
        )
        print(soln)


class TriangleHdf5:
    @classmethod
    def __groupNames(cls, filepath: str):
        with h5py.File(filepath, "r") as file:
            group_names = list(file.keys())
        return sorted(group_names)

    @classmethod
    def __actualParameters(
        cls,
        filepath: str,
        group_names: list[str],
    ):
        parameters = []
        with h5py.File(filepath, "r") as file:
            for group_name in group_names:
                group = file[group_name]
                parameter = group.attrs["real_parameter"]
                parameters.append(parameter)
        return np.array(parameters)

    @classmethod
    def __differentialEvolutionParsers(
        cls,
        filepath: str,
        group_names: list[str],
    ):
        de_parsers: list[DifferentialEvolutionParser] = []
        with h5py.File(filepath, "r") as file:
            for group_name in group_names:
                group = file[group_name]
                de_parser = DifferentialEvolutionParser.fromHdf5(group)
                de_parsers.append(de_parser)
        return de_parsers

    @classmethod
    def __fitParameters(
        cls,
        filepath: str,
        group_names: list[str],
    ):
        parameters = []
        with h5py.File(filepath, "r") as file:
            for group_name in group_names:
                group = file[group_name]
                parameter = group.attrs["fit_parameter"]
                parameters.append(parameter)
        return np.array(parameters)

    def __init__(self, filepath: str):
        group_names = self.__groupNames(filepath)
        self.actual_parameters = self.__actualParameters(filepath, group_names)
        self.fit_parameters = self.__fitParameters(filepath, group_names)
        self.de_parsers = self.__differentialEvolutionParsers(filepath, group_names)
        self.filepath = filepath

    def __len__(self):
        return len(self.de_parsers)

    @property
    def actual_widths(self):
        return self.actual_parameters[:, 0]

    @property
    def actual_noises(self):
        return self.actual_parameters[:, 1]

    @property
    def actual_timescales(self):
        return self.actual_parameters[:, 2]

    @property
    def actual_amplitudes(self):
        return self.actual_parameters[:, 3]

    @property
    def actual_shifts(self):
        return self.actual_parameters[:, 4]

    @property
    def fit_widths(self):
        widths = np.zeros(len(self))
        for index, de_parser in enumerate(self.de_parsers):
            width = de_parser.population(-1, 0, 0)[0]
            widths[index] = width
        return widths

    @property
    def fit_noises(self):
        noises = np.zeros(len(self))
        for index, de_parser in enumerate(self.de_parsers):
            noise = de_parser.population(-1, 0, 1)[0]
            noises[index] = noise
        return noises

    @property
    def fit_timescales(self):
        return self.fit_parameters[:, 2]

    @property
    def fit_amplitudes(self):
        return self.fit_parameters[:, 3]

    @property
    def fit_shifts(self):
        return self.fit_parameters[:, 4]

    def plotHeatmapActual(self, ax: Axes, **kwargs):
        xy = np.array([self.actual_widths, self.actual_noises])
        distr = Distribution2D.fromSamples(xy, density=False)
        distr.plotPdfHeatmap(ax, **kwargs)

    def plotHeatmapFit(self, ax: Axes, **kwargs):
        xy = np.array([self.fit_widths, self.fit_noises])
        distr = Distribution2D.fromSamples(xy, density=False)
        distr.plotPdfHeatmap(ax, **kwargs)

    def plot(self, ax: Axes, parameter_index: int, **kwargs):
        match parameter_index:
            case 0:
                return self.plotWidths(ax, **kwargs)
            case 1:
                return self.plotNoises(ax, **kwargs)
            case 2:
                return self.plotFrequencies(ax, **kwargs)
            case 3:
                return self.plotAmplitudes(ax, **kwargs)
            case 4:
                return self.plotShifts(ax, **kwargs)

    def plotAmplitudes(self, ax: Axes, **kwargs):
        actual_parameters = self.actual_amplitudes
        fit_parameters = self.fit_amplitudes
        return ax.scatter(actual_parameters, fit_parameters, **kwargs)

    def plotFrequencies(self, ax: Axes, **kwargs):
        actual_frequencies = 1 / self.actual_parameters[:, 2]
        fit_frequencies = 1 / self.fit_parameters[:, 2]
        return ax.scatter(actual_frequencies, fit_frequencies, **kwargs)

    def plotNoises(self, ax: Axes, **kwargs):
        actual_parameters = self.actual_noises
        fit_parameters = self.fit_noises
        return ax.scatter(actual_parameters, fit_parameters, **kwargs)

    def plotShifts(self, ax: Axes, **kwargs):
        actual_parameters = self.actual_shifts
        fit_parameters = self.fit_shifts
        return ax.scatter(actual_parameters, fit_parameters, **kwargs)

    def plotTimescales(self, ax: Axes, **kwargs):
        actual_parameters = self.actual_timescales
        fit_parameters = self.fit_timescales
        return ax.scatter(actual_parameters, fit_parameters, **kwargs)

    def plotWidths(self, ax: Axes, **kwargs):
        actual_parameters = self.actual_widths
        fit_parameters = self.fit_widths
        return ax.scatter(actual_parameters, fit_parameters, **kwargs)

    def formatAxis(self, ax: Axes, parameter_index: int):
        match parameter_index:
            case 0:
                self.formatWidthAxis(ax)
            case 1:
                self.formatNoiseAxis(ax)
            case 2:
                self.formatTimescaleAxis(ax)
            case 3:
                self.formatAmplitudeAxis(ax)
            case 4:
                self.formatShiftAxis(ax)

    def formatAmplitudeAxis(self, ax: Axes):
        lim = (0.1, 10)
        tick_locs = (0.1, 1, 10)
        tick_labels = (r"$10^{-1}$", "", r"$10^1$")

        ax.set_xscale("log")
        ax.set_xlim(lim)
        ax.set_xticks(
            tick_locs,
            labels=tick_labels,
        )

        ax.set_yscale("log")
        ax.set_ylim(lim)
        ax.set_yticks(
            tick_locs,
            labels=tick_labels,
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_aspect("equal")

    def formatNoiseAxis(self, ax: Axes):
        ax.set_xlim(0, 0.25)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.25))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))

        ax.set_ylim(0, 0.25)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_aspect("equal")

    def formatTimescaleAxis(self, ax: Axes):
        lim = (0.1, 10)
        tick_locs = (0.1, 1, 10)
        tick_labels = (r"$10^{-1}$", "", r"$10^1$")

        ax.set_xscale("log")
        ax.set_xlim(lim)
        ax.set_xticks(
            tick_locs,
            labels=tick_labels,
        )

        ax.set_yscale("log")
        ax.set_ylim(lim)
        ax.set_yticks(
            tick_locs,
            labels=tick_labels,
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_aspect("equal")

    def formatShiftAxis(self, ax: Axes):
        lim = (-10, 10)
        tick_locs_major = (-10, 0, 10)
        tick_labels = ("-10", "", "10")

        ax.set_xlim(lim)
        ax.set_xticks(
            tick_locs_major,
            labels=tick_labels,
        )
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))

        ax.set_ylim(lim)
        ax.set_yticks(
            tick_locs_major,
            labels=tick_labels,
        )
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_aspect("equal")

    def formatWidthAxis(self, ax: Axes):
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))

        ax.set_ylim(0, 1)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_aspect("equal")


if __name__ == "__main__":
    plt.rcParams["text.usetex"] = True
    plt.rcParams["text.latex.preamble"] = r"\usepackage{lmodern}"
    plt.rcParams["font.family"] = "lmodern"
    np.set_printoptions(precision=3)

    ##### Simulate triangle waves and fit using differential evolution #####
    if False:
        t = np.linspace(0, 20, 10000)
        trial_count = 3

        triangle_fitter = TriangleFitter(
            hdf5_filepath="triangle.hdf5",
            t=t,
            width_bounds=(0, 1),
            noise_bounds=(0, 0.25),
        )
        for index in tqdm(range(trial_count)):
            triangle_fitter.runFit()

        appendRescalingParameters("triangle.hdf5", t=t)

    ##### Plot fit vs. actual parameter values from fitted triangle wave simulations #####
    
    fig = generateComparisonFigure(
        "triangle.hdf5",
        figsize=(3.375, 3.375),
        layout="constrained",
    )
    plt.show()
    
