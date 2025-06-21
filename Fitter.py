from __future__ import annotations

import copy
from functools import partial
from numbers import Number
import pickle as pkl
from typing import Callable, Iterable, Union
import warnings

import h5py
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numba
import numpy as np
from numpy import ndarray
import pandas as pd
from scipy import integrate, interpolate, ndimage, optimize, signal, stats
from scipy.ndimage import filters
from scipy.stats import rv_continuous
from skimage import restoration
import sympy as sym
from tqdm import tqdm

warnings.simplefilter("error", RuntimeWarning)


def crossingSigns(x: ndarray, x_cross: float = 0.0):
    return np.ceil(0.5 * np.diff(np.sign(x - x_cross)))


def midpoints(x: ndarray):
    return 0.5 * (x[1:] + x[:-1])


def denoiseTrace(x: ndarray):
    return restoration.denoise_wavelet(
        x,
        method="VisuShrink",
        mode="soft",
        wavelet="sym4",
        rescale_sigma=True,
    )


def rescale(
    x: ndarray,
    x_amp: float = 1.0,
    x_shift: float = 0.0,
):
    return x_amp * (x - x_shift)


class Function2dPlotter:
    def __init__(self, z: ndarray, x_bin_edges: ndarray, y_bin_edges: ndarray):
        z_shape = z.shape
        assert (
            z_shape == (x_bin_edges.size - 1, y_bin_edges.size - 1)
            or x_bin_edges.size == 0
            or y_bin_edges.size == 0
        )

        self.x_bin_edges = x_bin_edges
        self.y_bin_edges = y_bin_edges
        self.x_bin_centers = Binner.edgeToCenter(x_bin_edges)
        self.y_bin_centers = Binner.edgeToCenter(y_bin_edges)

        self.x_min = x_bin_edges[0]
        self.x_max = x_bin_edges[-1]
        self.y_min = y_bin_edges[0]
        self.y_max = y_bin_edges[-1]
        self.min = np.min(z)
        self.max = np.max(z)

        self.z = z

    @property
    def xy_bin_centers(self):
        return Binner.toMeshgrid(
            self.x_bin_centers,
            self.y_bin_centers,
        )

    @property
    def xy_bin_edges(self):
        return Binner.toMeshgrid(
            self.x_bin_edges,
            self.y_bin_edges,
        )

    def swapAxes(self):
        z = self.z.T
        if np.any(np.iscomplex(z)):
            z = np.imag(z) + 1j * np.real(z)
        return Function2dPlotter(
            z,
            self.y_bin_edges,
            self.x_bin_edges,
        )

    def plotContour(
        self,
        ax: Axes,
        smoothing: float = 0,
        **kwargs,
    ):
        xy_bin_centers = self.xy_bin_centers
        z = self.z
        z_smooth = filters.gaussian_filter(z, smoothing)
        return ax.contour(
            xy_bin_centers[0, :, :],
            xy_bin_centers[1, :, :],
            z_smooth,
            **kwargs,
        )

    def plotHeatmap(self, ax: Axes, **kwargs):
        return ax.pcolormesh(
            self.x_bin_edges,
            self.y_bin_edges,
            self.z.T,
            **kwargs,
        )

    def plotScatter(
        self,
        ax: Axes,
        s: float = plt.rcParams["lines.markersize"] ** 2,
        **kwargs,
    ):
        x_bin_centers, y_bin_centers = self.x_bin_centers, self.y_bin_centers
        z = self.z.flatten()
        z /= z.max()
        return ax.scatter(
            x_bin_centers.flatten(),
            y_bin_centers.flatten(),
            s=s * z,
            **kwargs,
        )

    def plotSurface(self, ax: Axes3D, **kwargs):
        return ax.plot_surface(
            *self.xy_bin_centers,
            self.z,
            **kwargs,
        )

    def plotVectorField(self, ax: Axes, **kwargs):
        z = self.z.T
        return ax.quiver(
            self.x_bin_centers,
            self.y_bin_centers,
            np.real(z),
            np.imag(z),
            **kwargs,
        )


class Binner:
    @classmethod
    def calculateCdf(
        cls,
        pdf: ndarray,
        x_bin_edges: ndarray,
        y_bin_edges: ndarray,
    ):
        xy_bin_areas = cls.edgesToArea(x_bin_edges, y_bin_edges)
        cdf: ndarray = np.cumsum(np.cumsum(np.abs(pdf), axis=0), axis=1) * xy_bin_areas
        return cdf

    @classmethod
    def edgeToCenter(cls, bin_edges: ndarray, *args) -> ndarray:
        if len(args) == 0:
            return 0.5 * (bin_edges[1:] + bin_edges[:-1])
        bin_edges = (bin_edges, *args)
        return cls.toMeshgrid(*list(map(cls.edgeToCenter, bin_edges)))

    @classmethod
    def edgeToWidth(cls, bin_edges: ndarray, *args) -> ndarray:
        if len(args) == 0:
            return np.diff(bin_edges)
        bin_edges = (bin_edges, *args)
        return cls.toMeshgrid(*list(map(cls.edgeToWidth, bin_edges)))

    @classmethod
    def edgesToArea(cls, x_bin_edges: ndarray, y_bin_edges: ndarray) -> ndarray:
        xy_bin_widths = cls.edgeToWidth(x_bin_edges, y_bin_edges)
        xy_bin_areas = np.prod(xy_bin_widths, axis=0)
        return xy_bin_areas

    @classmethod
    def toMeshgrid(cls, *args) -> ndarray:
        return np.array(np.meshgrid(*args, indexing="ij"))


class Cdf2DInterpolator:
    def __init__(self, xy: tuple[ndarray, ndarray], cdf: ndarray):
        self.x_max: float = np.max(xy[0])
        self.y_max: float = np.max(xy[1])
        self.linear = interpolate.RegularGridInterpolator(
            xy,
            cdf,
            bounds_error=False,
            fill_value=0,
        )

    def normalizedX(self, x: ndarray):
        x_max = self.x_max
        x[x > x_max] = x_max
        return x

    def normalizedY(self, y: ndarray):
        y_max = self.y_max
        y[y > y_max] = y_max
        return y

    def __call__(self, xy):
        x = self.normalizedX(xy[..., 0])
        y = self.normalizedY(xy[..., 1])
        xy = np.transpose([x, y])
        return self.linear(xy)


class DifferentialEvolutionParser:
    ##### Methods to read information from optimize.IntermediateResults or HDF5 #####
    @classmethod
    def fromHdf5(cls, dataset: h5py.Group):
        if isinstance(dataset, str):
            with h5py.File(dataset, "r") as file:
                parser = DifferentialEvolutionParser.fromHdf5(file)
            return parser

        convergences = dataset["convergences"][:]
        nfevs = dataset["nfevs"][:]
        nits = dataset["nits"][:]
        populations = dataset["populations"][:, :, :]
        populations_energies = dataset["populations_energies"][:, :]
        return DifferentialEvolutionParser(
            convergences=convergences,
            nfevs=nfevs,
            nits=nits,
            populations=populations,
            populations_energies=populations_energies,
        )

    @classmethod
    def fromPickles(cls, filepaths: list[str]):
        intermediate_results = map(cls.__fromPickleToIntermediateResult, filepaths)
        intermediate_results = list(intermediate_results)
        return DifferentialEvolutionParser.fromIntermediateResults(intermediate_results)

    @classmethod
    def fromIntermediateResults(cls, intermediate_results: list):
        assert isinstance(intermediate_results, list)
        convergences = cls.__getConvergence(intermediate_results)
        nfevs = cls.__getFunctionEvaluationCount(intermediate_results)
        nits = cls.__getIterationCount(intermediate_results)
        populations = cls.__getPopulation(intermediate_results)
        populations_energies = cls.__getPopulationEnergies(intermediate_results)
        return DifferentialEvolutionParser(
            convergences=convergences,
            nfevs=nfevs,
            nits=nits,
            populations=populations,
            populations_energies=populations_energies,
        )

    @classmethod
    def __fromPickleToIntermediateResult(cls, filepath: str):
        with open(filepath, "rb") as file:
            intermediate_result = pkl.load(file)
        return intermediate_result

    @classmethod
    def __getConvergence(cls, intermediate_results):
        if isinstance(intermediate_results, list):
            return np.array(list(map(cls.__getConvergence, intermediate_results)))
        x: float = intermediate_results.convergence
        return x

    @classmethod
    def __getFunctionEvaluationCount(cls, intermediate_results):
        if isinstance(intermediate_results, list):
            return np.array(
                list(map(cls.__getFunctionEvaluationCount, intermediate_results))
            )
        x: ndarray = intermediate_results.nfev
        return x

    @classmethod
    def __getIterationCount(cls, intermediate_results):
        if isinstance(intermediate_results, list):
            return np.array(list(map(cls.__getIterationCount, intermediate_results)))
        x: ndarray = intermediate_results.nit
        return x

    @classmethod
    def __getPopulation(cls, intermediate_results):
        if isinstance(intermediate_results, list):
            return np.array(list(map(cls.__getPopulation, intermediate_results)))
        x: ndarray = intermediate_results.population
        return x

    @classmethod
    def __getPopulationBest(cls, intermediate_results):
        if isinstance(intermediate_results, list):
            return np.array(list(map(cls.__getPopulation, intermediate_results)))
        x: ndarray = intermediate_results.x
        return x

    @classmethod
    def __getPopulationEnergies(cls, intermediate_results):
        if isinstance(intermediate_results, list):
            return np.array(
                list(map(cls.__getPopulationEnergies, intermediate_results))
            )
        x: ndarray = intermediate_results.population_energies
        return x

    @classmethod
    def __toHdf5(
        cls,
        dataset: h5py.Group,
        convergences: ndarray,
        nfevs: ndarray,
        nits: ndarray,
        populations: ndarray,
        populations_energies: ndarray,
    ):
        dataset.create_dataset(
            "convergences",
            dtype=np.float64,
            data=convergences,
        )
        dataset.create_dataset(
            "nfevs",
            dtype=np.int64,
            data=nfevs,
        )
        dataset.create_dataset(
            "nits",
            dtype=np.int64,
            data=nits,
        )
        dataset.create_dataset(
            "populations",
            dtype=np.float64,
            data=populations,
        )
        dataset.create_dataset(
            "populations_energies",
            dtype=np.float64,
            data=populations_energies,
        )

    @classmethod
    def printResultSummary(cls, intermediate_result):
        convergence: float = cls.__getConvergence(intermediate_result)
        energies: ndarray = cls.__getPopulationEnergies(intermediate_result)
        nit: int = cls.__getIterationCount(intermediate_result)
        nfev = cls.__getFunctionEvaluationCount(intermediate_result)
        population_best: ndarray = cls.__getPopulationBest(intermediate_result)
        print(f"Best: {population_best} ({nit:d} iterations, {nfev:d} evaluations)")
        cost_mean = np.mean(energies)
        cost_std = np.std(energies)
        print(
            f"Energies: {cost_mean:.2e} +/- {cost_std:.2e} ({convergence:.2e} convergence)"
        )

    def __init__(
        self,
        convergences: ndarray = None,
        nfevs: ndarray = None,
        nits: ndarray = None,
        populations: ndarray = None,
        populations_energies: ndarray = None,
    ):
        """
        convergences (number of iterations,)
        nfevs (number of iterations,)
        nits (number of iterations,)
        populations (number of iterations, population size, number of parameters)
        populations_energies (number of iterations, population size)
        """
        x = [
            convergences,
            nfevs,
            nits,
            populations,
            populations_energies,
        ]
        assert all([xi is None for xi in x]) or len(set(list(map(len, x)))) == 1

        self.convergences = convergences
        self.nfevs = nfevs
        self.nits = nits
        self.populations = populations
        self.populations_energies = populations_energies
        self.population_size = populations.shape[-1]

    def __len__(self):
        nits = self.nits
        if nits is None:
            return 0
        if isinstance(nits, ndarray):
            return len(nits)

    ##### Methods to retrieve slices of information #####
    def convergence(self, iteration_index: int) -> float:
        return self.convergences[iteration_index]

    def nfev(self, iteration_index: int) -> float:
        return self.nfevs[iteration_index]

    def nit(self, iteration_index: int) -> float:
        return self.nits[iteration_index]

    def population(
        self,
        iteration_index: int = None,
        population_index: int = None,
        parameter_index: int = None,
    ):
        populations = self.populations
        if iteration_index is None:
            iteration_index = slice(populations.shape[0])
        if population_index is None:
            population_index = slice(populations.shape[1])
        if parameter_index is None:
            parameter_index = slice(populations.shape[2])
        return populations[iteration_index, population_index, parameter_index]

    def population_energies(
        self,
        iteration_index: int = None,
        population_index: int = None,
    ):
        return self.population_energies[iteration_index, population_index]

    ##### Methods to append or export results #####
    def __instantiateProperties(
        self,
        population_size: int,
        parameter_count: int,
    ):
        self.convergences = np.array([], dtype=np.float64)
        self.nfevs = np.array([], dtype=np.int64)
        self.nits = np.array([], dtype=np.int64)
        self.populations = np.empty(
            (0, population_size, parameter_count),
            dtype=np.float64,
        )
        self.populations_energies = np.empty(
            (0, population_size),
            dtype=np.float64,
        )

    def appendResult(self, intermediate_result):
        convergence: float = self.__getConvergence(intermediate_result)
        nfev: int = self.__getFunctionEvaluationCount(intermediate_result)
        nit: int = self.__getIterationCount(intermediate_result)
        population: ndarray = self.__getPopulation(intermediate_result)
        population_energies: ndarray = self.__getPopulationEnergies(intermediate_result)

        if len(self) == 0:
            population_size, parameter_count = population.shape
            self.__instantiateProperties(population_size, parameter_count)

        self.convergences = np.append(self.convergences, convergence)
        self.nfevs = np.append(self.nfevs, nfev)
        self.nits = np.append(self.nits, nit)
        self.populations = np.concatenate(
            (self.populations, [population]),
            axis=0,
        )
        self.populations_energies = np.concatenate(
            (self.populations_energies, [population_energies]),
            axis=0,
        )

    def toHdf5(self, dataset: h5py.Group):
        convergences = self.convergences
        nfevs = self.nfevs
        nits = self.nits
        populations = self.populations
        populations_energies = self.populations_energies
        self.__toHdf5(
            dataset,
            convergences=convergences,
            nfevs=nfevs,
            nits=nits,
            populations=populations,
            populations_energies=populations_energies,
        )

    ##### Methods to plot on given Axes #####
    def plotConvergence(self, ax: Axes, **kwargs):
        convergence = self.convergences
        ax.plot(
            self.nits,
            convergence,
            **kwargs,
        )

    def plotEnergies(
        self,
        ax: Axes,
        func: Callable[[ndarray], ndarray] = None,
        **kwargs,
    ):
        costs = self.populations_energies
        if isinstance(func, Callable):
            costs = np.apply_along_axis(func, 1, costs)
        ax.plot(
            self.nits,
            costs,
            **kwargs,
        )

    def plotPopulation(
        self,
        ax: Axes,
        index: int,
        func: Callable[[ndarray], ndarray] = None,
        **kwargs,
    ):
        population = self.population(parameter_index=index)
        if isinstance(func, Callable):
            population = np.apply_along_axis(func, 1, population)
        ax.plot(
            self.nits,
            population,
            **kwargs,
        )


class SdeSolver:
    @staticmethod
    def __itoEuler(
        f: Callable[[float, ndarray], ndarray],
        G: ndarray,
        t: ndarray,
        y0: ndarray,
        generator=None,
        nan_index: float = 10,
    ):
        if generator is None:
            generator = np.random.default_rng()
        t_size = t.size
        y_count = y0.shape[0]

        dt = (t[t_size - 1] - t[0]) / (t_size - 1)
        y = np.zeros((t_size, y_count), dtype=y0.dtype)
        eta = G * generator.normal(0, np.sqrt(dt), (t_size, y_count))

        y[0] = y0
        yn = y0

        for n in range(nan_index):
            yn = yn + f(t[n], yn) * dt + eta[n, :]
            y[n + 1, :] = yn
        if np.any(np.isnan(y[:nan_index, :])):
            y[nan_index:, :] = np.nan
            return y

        for n in range(nan_index, t_size - 1):
            yn = yn + f(t[n], yn) * dt + eta[n, :]
            y[n + 1, :] = yn

        return y.T

    @classmethod
    def solve(
        cls,
        dxdt: Callable[[ndarray], float],
        t: ndarray,
        x0: ndarray,
        noise_values: ndarray = (),
        generator=None,
    ):
        noise_values = np.array(noise_values)
        noise_count = noise_values.size
        if noise_count == 0:
            return cls.__solve_deterministic(
                dxdt,
                t,
                x0,
            )
        elif noise_count >= 1:
            return cls.__solve_stochastic(
                dxdt,
                t,
                x0,
                noise_values,
                generator=generator,
            )
        else:
            raise ValueError(f"Noise values must be numpy array with non-negative size")

    @classmethod
    def __solve_deterministic(
        cls,
        dxdt: Callable[[ndarray], float],
        t: ndarray,
        x0: ndarray,
    ):
        t_size = len(t)
        if t_size == 2:
            t_span = t
            t_eval = None
        elif t_size >= 3:
            t_span = (np.min(t), np.max(t))
            t_eval = t
        else:
            raise ValueError(f"time must have at least two elements {len(t):d}")

        soln = integrate.solve_ivp(
            dxdt,
            t_span,
            x0,
            t_eval=t_eval,
            dense_output=True,
            method="LSODA",
        )
        return (soln.t, soln.y)

    @classmethod
    def __solve_stochastic(
        cls,
        dxdt: Callable[[ndarray], float],
        t: ndarray,
        x0: ndarray,
        noise_values: ndarray,
        generator=None,
    ):
        y = cls.__itoEuler(
            dxdt,
            noise_values,
            t,
            x0,
            generator=generator,
        )
        return (t, y)


class Distribution(rv_continuous):
    EPSILON = np.finfo(float).eps

    @classmethod
    def fromHdf5(cls, group: h5py.Group):
        pdf = group["density"][:]
        bin_edges = group["bin_edges"][:]
        return Distribution(pdf, bin_edges)

    @classmethod
    def fromSamples(
        cls,
        x: ndarray,
        bins: int = None,
        **kwargs,
    ):
        if bins is None:
            bins = int(np.sqrt(x.size))
        return Distribution(*np.histogram(x, bins=bins, **kwargs))

    @classmethod
    def __boundingBox(
        cls,
        dist1: Distribution,
        dist2: Distribution,
        only_overlap: bool = False,
    ):
        if only_overlap:
            x_minimax = min(dist1.b, dist2.b)
            x_maximin = max(dist1.a, dist2.a)
            x_bounds = np.array([x_maximin, x_minimax])
            return x_bounds

        x_min = min(dist1.a, dist2.a)
        x_max = max(dist1.b, dist2.b)
        x_bounds = np.array([x_min, x_max])
        return x_bounds

    @classmethod
    def __bhattacharyyaCoefficient(
        cls,
        dist1: Distribution,
        dist2: Distribution,
    ):
        def bcIntegrand(x: ndarray):
            return np.sqrt(x)

        return cls.__fDivergence(
            bcIntegrand,
            dist1,
            dist2,
            only_overlap=True,
        )

    @classmethod
    def __bhattacharyyaDistance(
        cls,
        dist1: Distribution,
        dist2: Distribution,
    ):
        bc = cls.__bhattacharyyaCoefficient(dist1, dist2)
        return -np.log(bc)

    @classmethod
    def __pearsonDivergence(cls, dist1: Distribution, dist2: Distribution):
        def pearsonFunction(x: ndarray):
            return (x - 1) ** 2

        return cls.__fDivergence(pearsonFunction, dist1, dist2)

    @classmethod
    def __fDivergence(
        cls,
        func: Callable[[ndarray], ndarray],
        dist1: Distribution,
        dist2: Distribution,
        epsilon: float = None,
        only_overlap: bool = False,
    ):
        if epsilon is None:
            epsilon = cls.EPSILON

        def fIntegrand(x: ndarray):
            p = dist1._pdf(x)
            q = dist2._pdf(x)
            p[p == 0] = epsilon
            q[q == 0] = epsilon
            return p * func(q / p)

        x_min, x_max = cls.__boundingBox(dist1, dist2, only_overlap=only_overlap)
        x = np.linspace(x_min, x_max, 1000)
        x_integral: float = integrate.simpson(fIntegrand(x), x)
        return x_integral

    @classmethod
    def __hellingerDistance(cls, dist1: Distribution, dist2: Distribution):
        return 1 - cls.__bhattacharyyaCoefficient(dist1, dist2)

    @classmethod
    def __jsDivergence(cls, dist1: Distribution, dist2: Distribution):
        """Jensen-Shannon divergence"""

        def jsFunction(x: ndarray):
            return x * np.log2(x) - (x + 1) * np.log2((x + 1) / 2)

        return 0.5 * cls.__fDivergence(jsFunction, dist1, dist2)

    @classmethod
    def __klDivergence(cls, dist1: Distribution, dist2: Distribution):
        """Kullback-Leibler divergence"""

        def klFunction(x: ndarray):
            return -np.log2(x)

        return cls.__fDivergence(klFunction, dist1, dist2)

    @classmethod
    def __leCamDistance(cls, dist1: Distribution, dist2: Distribution):
        def leCamFunction(x: ndarray):
            return (1 - x) / (2 * x + 2)

        return cls.__fDivergence(leCamFunction, dist1, dist2)

    @classmethod
    def __shannonEntropy(cls, dist: Distribution):
        x = dist.bin_centers
        p = dist._pdf_matrix
        p[p == 0] = 1
        partial_entropy = p * np.log2(p)
        entropy: float = -integrate.simpson(partial_entropy, x)
        return entropy

    @classmethod
    def __tvDistance(
        cls,
        dist1: Distribution,
        dist2: Distribution,
    ):
        """Total variation distance"""
        if not dist1.overlaps(dist2):
            return 1.0

        cdf1 = dist1._cdf
        cdf2 = dist2._cdf
        x_min, x_max = cls.__boundingBox(
            dist1,
            dist2,
            only_overlap=True,
        )
        tvd_pre: float = cdf1(x_min) + cdf2(x_min)
        tvd_post: float = 2 - (cdf1(x_max) + cdf2(x_max))

        def tvdIntegrand(x: ndarray):
            return np.abs(x - 1)

        tvd_overlap = cls.__fDivergence(
            tvdIntegrand,
            dist1,
            dist2,
            only_overlap=True,
        )
        return 0.5 * (tvd_pre + tvd_overlap + tvd_post)

    def __init__(
        self,
        pdf: ndarray,
        bin_edges: ndarray,
        density: bool = False,
    ):
        bin_widths = bin_edges[1:] - bin_edges[:-1]
        bin_centers = Binner.edgeToCenter(bin_edges)
        cdf = np.cumsum(pdf) * bin_widths

        if density:
            pdf_area = cdf[-1]
            pdf /= pdf_area
            cdf /= pdf_area

        self._pdf = interpolate.interp1d(
            bin_centers,
            pdf,
            kind="linear",
            bounds_error=False,
            fill_value=0,
        )
        self._cdf = interpolate.interp1d(
            bin_edges[1:],
            cdf,
            kind="linear",
            bounds_error=False,
            fill_value=(0, 1),
        )

        if np.sum(pdf) != 0:
            self.__mean = np.average(bin_centers, weights=pdf)
            self.__variance = np.average(bin_centers**2, weights=pdf) - self.__mean**2
        else:
            self.__mean = np.mean(bin_centers)
            self.__variance = np.mean(bin_centers**2) - self.__mean**2

        rv_continuous.__init__(
            self,
            a=bin_edges[0],
            b=bin_edges[-1],
            shapes=",",
        )

        self._pdf_matrix = pdf
        self._cdf_matrix = cdf
        self.bin_centers = bin_centers
        self.bin_edges = bin_edges
        self.bin_widths = bin_widths
        self.size = pdf.size

    def __add__(self, other: Union[Distribution, float]):
        if isinstance(other, Number):
            return self.rescale(x_shift=-other)

        x_min, x_max = Distribution.__boundingBox(self, other)
        dx = (x_max - x_min) / (self.size + other.size)
        x1 = np.arange(self.a, self.b, dx)
        x2 = np.arange(other.a, other.b, dx)
        pdf1 = self._pdf(x1)
        pdf2 = other._pdf(x2)

        pdf_joint = signal.convolve(pdf1, pdf2, mode="full")
        x_joint: ndarray = np.linspace(x_min, x_max, pdf_joint.size + 1)
        return Distribution(
            pdf_joint,
            x_joint,
            density=True,
        )

    def __mul__(self, other: float):
        return self.jsDivergence(other)

    def __truediv__(self, other: float):
        assert isinstance(other, Number)
        return self * (1 / other)

    def __eq__(self, other: Distribution):
        tvd = self.tvDistance(other)
        epsilon = self.size * np.finfo(float).eps
        return tvd <= epsilon

    def __getitem__(self, inds: ndarray):
        if inds.size == 0:
            return Distribution(np.array([0.5, 0.5]), np.array([0, 1, 2]))

        bin_centers = self.bin_centers[inds]
        bin_midpoints = midpoints(bin_centers)
        bin_start = bin_centers[0] - 0.5 * (bin_centers[1] - bin_centers[0])
        bin_end = bin_centers[-1] + 0.5 * (bin_centers[-1] - bin_centers[-2])
        bin_edges = np.array([bin_start, *bin_midpoints, bin_end])
        return Distribution(self._pdf_matrix[inds], bin_edges)

    def _stats(self, *args, **kwargs):
        return self.__mean, self.__variance, None, None

    def between(self, bin_min: float, bin_max: float):
        bin_centers = self.bin_centers
        inds = np.logical_and(bin_min <= bin_centers, bin_centers <= bin_max)
        return self[inds]

    def rescale(self, x_amp: float = 1.0, x_shift: float = 0.0):
        return Distribution(
            self._pdf_matrix / x_amp,
            rescale(self.bin_edges, x_amp, x_shift),
        )

    def resample(self, num: int):
        pdf = self._pdf_matrix
        bin_edge_zoom = (num + 1) / (pdf.size + 1)
        bin_edges_resampled = ndimage.zoom(self.bin_edges, bin_edge_zoom, order=1)
        pdf_resampled = signal.resample(pdf, num)
        return Distribution(pdf_resampled, bin_edges_resampled)

    def zoom(self, zoom: float, order: int = 3):
        pdf = self._pdf_matrix
        bin_edge_zoom = (zoom * pdf.size + 1) / (pdf.size + 1)
        bin_edges_zoomed = ndimage.zoom(self.bin_edges, bin_edge_zoom, order=1)
        pdf_zoomed = ndimage.zoom(pdf, zoom, order=order)
        return Distribution(pdf_zoomed, bin_edges_zoomed)

    def toHdf5(self, group: h5py.Group):
        group.create_dataset(
            "density",
            data=self._pdf_matrix,
            dtype=np.float64,
            compression="gzip",
            compression_opts=9,
        )
        group.create_dataset(
            "bin_edges",
            data=self.bin_edges,
            dtype=np.float64,
            compression="gzip",
            compression_opts=9,
        )

    def cost(self, other: Distribution):
        return Distribution.__tvDistance(self, other)

    def bhattacharyyaCoefficient(self, other: Distribution):
        return Distribution.__bhattacharyyaCoefficient(self, other)

    def bhattacharyyaDistance(self, other: Distribution):
        return Distribution.__bhattacharyyaDistance(self, other)

    def pearsonDivergence(self, other: Distribution):
        return Distribution.__pearsonDivergence(self, other)

    def fDivergence(
        self,
        func: Callable[[ndarray], ndarray],
        other: Distribution,
    ):
        return Distribution.__fDivergence(func, self, other)

    def hellingerDistance(self, other: Distribution):
        return Distribution.__hellingerDistance(self, other)

    def jsDivergence(self, other: Distribution):
        return Distribution.__jsDivergence(self, other)

    def klDivergence(self, other: Distribution):
        return Distribution.__klDivergence(self, other)

    def leCamDistance(self, other: Distribution):
        return Distribution.__leCamDistance(self, other)

    def shannonEntropy(self):
        return Distribution.__shannonEntropy(self)

    def tvDistance(self, other: Distribution):
        return Distribution.__tvDistance(self, other)

    def overlaps(self, other: Distribution):
        return not (self.b <= other.a or other.b <= self.a)

    def plotStairsPdf(self, ax: Union[Axes, ndarray], **kwargs):
        if isinstance(ax, Axes):
            return ax.stairs(
                self._pdf_matrix,
                self.bin_edges,
                **kwargs,
            )

        assert isinstance(ax, ndarray)
        plot_distribution = partial(self.plotStairsPdf, **kwargs)
        return list(map(plot_distribution, ax.flatten()))

    def plotStairsCdf(self, ax: Axes, **kwargs):
        return ax.stairs(
            self._cdf_matrix,
            self.bin_edges,
            **kwargs,
        )

    def plotLineCdf(self, ax: Axes, **kwargs):
        return ax.plot(
            self.bin_centers,
            self._cdf_matrix,
            **kwargs,
        )

class Distributions(ndarray):
    @classmethod
    def fromHdf5(cls, file: h5py.Group):
        if isinstance(file, str):
            return cls.__fromHdf5_filepath(file)
        elif isinstance(file, (h5py.Group, h5py.File)):
            return cls.__fromHdf5_group(file)
        else:
            raise TypeError("file must be of type str of h5py.File")

    @classmethod
    def __fromHdf5_filepath(cls, filepath: str):
        with h5py.File(filepath, "r") as file:
            distributions = cls.__fromHdf5_group(file)
        return distributions

    @classmethod
    def __fromHdf5_group(cls, group: h5py.Group):
        shape = group.attrs["shape"]
        groups = [group[group_name] for group_name in group.keys()]
        distributions = list(map(Distribution.fromHdf5, groups))
        distributions = np.reshape(distributions, shape)
        return Distributions(distributions)

    def __new__(cls, distributions: ndarray):
        return np.asarray(distributions).view(cls)

    def toHdf5(self, group: h5py.Group):
        group.attrs["shape"] = self.shape
        for index, distribution in enumerate(self.flatten()):
            distribution: Distribution
            subgroup = group.create_group(f"-{index:d}-")
            distribution.toHdf5(subgroup)

    def plot(self, axs: ndarray, **kwargs):
        assert axs.shape == self.shape
        axs: list[Axes] = axs.flatten()
        distributions: list[Distribution] = self.flatten()

        for ax, distribution in zip(axs, distributions):
            distribution.plotStairsPdf(ax, **kwargs)


class Distribution2D(rv_continuous):
    __tvd_coefficients = np.array([-1, 1, 1, -1])

    @classmethod
    def fromSamples(
        cls,
        xy: ndarray,
        z: ndarray = None,
        bins: int = None,
        weighted: bool = False,
        density: bool = True,
        **kwargs,
    ):
        if bins is None:
            z_is_complex = np.iscomplex(z).any()
            if z is None or not z_is_complex:
                bins = int(np.cbrt(xy.shape[-1]))
            elif z_is_complex:
                bins = int(xy.shape[-1] ** (1 / 6))

        if xy.ndim == 1:
            x = np.real(xy)
            y = np.imag(xy)
        elif xy.ndim == 2:
            x, y = xy

        if z is None:
            z_binned, x_bin_edges, y_bin_edges = np.histogram2d(
                x,
                y,
                bins=bins,
                **kwargs,
            )
        elif isinstance(z, ndarray):
            z_binned, x_bin_edges, y_bin_edges, _ = stats.binned_statistic_2d(
                x,
                y,
                z,
                bins=bins,
                **kwargs,
            )
            z_binned[np.isnan(z_binned)] = 0
        else:
            raise TypeError(f"z must be of type None or ndarray (real or complex)")

        if weighted:
            distribution_2d = Distribution2D.fromSamples(z, bins=bins, **kwargs)
            z_binned = distribution_2d._pdf_matrix * z_binned

        return Distribution2D(
            z_binned,
            x_bin_edges,
            y_bin_edges,
            density=density,
        )

    @classmethod
    def tvDistance(
        cls,
        dist1: Distribution2D,
        dist2: Distribution2D,
    ):
        if not dist1.isOverlapping(dist2):
            return 1.0

        (
            x_maximin,
            x_minimax,
            y_maximin,
            y_minimax,
        ) = cls.__overlappingBoundingBox(dist1, dist2)
        xy_m = np.array(
            [
                [x_minimax, y_minimax],
                [x_minimax, y_maximin],
                [x_maximin, y_minimax],
                [x_maximin, y_maximin],
            ]
        )

        tvdc = cls.__tvd_coefficients
        tvd_nonoverlap = 2 + np.sum(tvdc * dist1._cdf(xy_m) + tvdc * dist2._cdf(xy_m))

        def tvdIntegrand(xy: ndarray):
            return np.abs(dist2._pdf(xy) - dist1._pdf(xy))

        tvd_overlap = cls.__overlappingIntegral(tvdIntegrand, dist1, dist2)
        return 0.5 * (tvd_nonoverlap + tvd_overlap)

    @classmethod
    def __overlappingBoundingBox(
        cls,
        dist1: Distribution2D,
        dist2: Distribution2D,
    ):
        x_maximin = max(dist1.x_min, dist2.x_min)
        x_minimax = min(dist1.x_max, dist2.x_max)
        y_maximin = max(dist1.y_min, dist2.y_min)
        y_minimax = min(dist1.y_max, dist2.y_max)
        xy_bounds = np.array([x_maximin, x_minimax, y_maximin, y_minimax])
        return xy_bounds

    @classmethod
    def __overlappingIntegral(
        cls,
        func: Callable[[ndarray], ndarray],
        dist1: Distribution2D,
        dist2: Distribution2D,
    ):
        (
            x_maximin,
            x_minimax,
            y_maximin,
            y_minimax,
        ) = cls.__overlappingBoundingBox(dist1, dist2)
        x_overlap = np.linspace(
            x_maximin,
            x_minimax,
            max(dist1.shape[0], dist2.shape[0]),
        )
        y_overlap = np.linspace(
            y_maximin,
            y_minimax,
            max(dist1.shape[1], dist2.shape[1]),
        )
        xy_overlap = Binner.toMeshgrid(x_overlap, y_overlap).T

        xy_integrand = func(xy_overlap)
        xy_integral: float = integrate.simpson(
            integrate.simpson(xy_integrand, x_overlap),
            y_overlap,
        )
        return xy_integral

    @staticmethod
    def __rescaleFloatToArray(xy: ndarray, n: int = 2):
        xy = np.array(xy)
        if xy.size == 1:
            xy = xy * np.ones(n)
        return xy

    def __init__(
        self,
        pdf: ndarray,
        x_bin_edges: ndarray,
        y_bin_edges: ndarray,
        density: bool = False,
    ):
        cdf = Binner.calculateCdf(pdf, x_bin_edges, y_bin_edges)
        if density:
            pdf_area = cdf[-1, -1]
            pdf /= pdf_area
            cdf /= pdf_area

        x_bin_centers = Binner.edgeToCenter(x_bin_edges)
        y_bin_centers = Binner.edgeToCenter(y_bin_edges)
        xy_bin_centers = Binner.toMeshgrid(x_bin_centers, y_bin_centers)

        x_min = x_bin_edges[0]
        x_max = x_bin_edges[-1]
        y_min = y_bin_edges[0]
        y_max = y_bin_edges[-1]

        if np.sum(pdf) != 0:
            self.__mean = np.average(
                xy_bin_centers,
                weights=pdf,
                axis=[1, 2],
            )
            m2 = np.average(
                xy_bin_centers**2,
                weights=pdf,
                axis=[1, 2],
            )
            self.__variance = m2 - self.__mean**2
        else:
            self.__mean = np.mean(
                xy_bin_centers,
                axis=[1, 2],
            )
            m2 = np.mean(
                xy_bin_centers**2,
                axis=[1, 2],
            )
            self.__variance = m2 - self.__mean**2

        self.__pdf = interpolate.RegularGridInterpolator(
            (x_bin_centers, y_bin_centers),
            pdf,
            bounds_error=False,
            fill_value=0,
        )
        self._cdf = Cdf2DInterpolator((x_bin_edges[1:], y_bin_edges[1:]), cdf)

        rv_continuous.__init__(
            self,
            a=min(x_min, y_min),
            b=max(x_max, y_max),
            shapes="",
        )

        self._pdf_matrix = pdf
        self._cdf_matrix = cdf
        self.x_bin_edges = x_bin_edges
        self.y_bin_edges = y_bin_edges
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.area = cdf[-1, -1]
        self.shape = pdf.shape

        self.__is_complex = np.iscomplex(pdf).any()

    @property
    def x_bin_centers(self):
        return Binner.edgeToCenter(self.x_bin_edges)

    @property
    def y_bin_centers(self):
        return Binner.edgeToCenter(self.y_bin_edges)

    @property
    def cdf_plotter(self):
        return Function2dPlotter(
            self._cdf_matrix,
            self.x_bin_edges,
            self.y_bin_edges,
        )

    @property
    def pdf_plotter(self):
        return Function2dPlotter(
            self._pdf_matrix,
            self.x_bin_edges,
            self.y_bin_edges,
        )

    def _stats(self, *args, **kwargs):
        return self.__mean, self.__variance, None, None

    def _pdf(self, xy: ndarray):
        return self.__pdf(xy)

    def swapAxes(self):
        pdf = self._pdf_matrix.T
        if self.__is_complex:
            pdf = np.imag(pdf) + 1j * np.real(pdf)

        return Distribution2D(
            pdf,
            self.y_bin_edges,
            self.x_bin_edges,
        )

    def rescale(
        self,
        xy_amp: ndarray = 1,
        xy_shift: ndarray = 0,
        rescale_edges: bool = True,
    ):
        x_amp, y_amp = self.__rescaleFloatToArray(xy_amp)
        xy_shift = self.__rescaleFloatToArray(xy_shift)

        pdf = self._pdf_matrix
        if self.__is_complex:
            pdf = np.real(pdf) * x_amp + 1j * np.imag(pdf) * y_amp

        xy_area = x_amp * y_amp
        if not rescale_edges:
            x_amp = 1
            y_amp = 1

        return Distribution2D(
            pdf / xy_area,
            rescale(self.x_bin_edges, x_amp, xy_shift[0]),
            rescale(self.y_bin_edges, y_amp, xy_shift[1]),
        )

    def normalizeByMax(self):
        pdf_max = self._pdf_matrix.max()
        return self.rescale(xy_amp=np.sqrt(pdf_max), rescale_edges=False)

    def normalizeByArea(self):
        return self.rescale(xy_amp=self.area, rescale_edges=False)

    def zoom(self, zoom: float, order: int = 3):
        pdf = self._pdf_matrix
        pdf_shape = np.array(pdf.shape)
        bin_edge_zooms = (zoom * pdf_shape + 1) / (pdf_shape + 1)
        x_bin_edges_zoomed = ndimage.zoom(
            self.x_bin_edges,
            bin_edge_zooms[0],
            order=1,
        )
        y_bin_edges_zoomed = ndimage.zoom(
            self.y_bin_edges,
            bin_edge_zooms[1],
            order=1,
        )
        pdf_zoomed = ndimage.zoom(pdf, zoom, order=order)
        return Distribution2D(
            pdf_zoomed,
            x_bin_edges_zoomed,
            y_bin_edges_zoomed,
        )

    def isOverlapping(self, other: Distribution2D):
        return not (
            self.x_max <= other.x_min
            or other.x_max <= self.x_min
            or self.y_max <= other.y_min
            or other.y_max <= self.y_min
        )

    def toMagnitudeDistribution(self):
        magnitude = np.abs(self._pdf_matrix)
        return Distribution2D(
            magnitude,
            self.x_bin_edges,
            self.y_bin_edges,
            density=False,
        )

    def cost(self, other: Distribution2D):
        return Distribution2D.tvDistance(self, other)

    def plotCdfHeatmap(self, ax: Axes, **kwargs):
        plotter = self.cdf_plotter
        return plotter.plotHeatmap(ax, **kwargs)

    def plotCdfSurface(self, ax: Axes3D, **kwargs):
        plotter = self.cdf_plotter
        return plotter.plotSurface(ax, **kwargs)

    def plotPdfContour(self, ax: Axes, **kwargs):
        plotter = self.pdf_plotter
        return plotter.plotContour(ax, **kwargs)

    def plotPdfHeatmap(
        self,
        ax: Axes,
        swap_axes: bool = False,
        **kwargs,
    ):
        plotter = self.pdf_plotter
        if swap_axes:
            plotter = plotter.swapAxes()
        return plotter.plotHeatmap(ax, **kwargs)

    def plotPdfScatter(self, ax: Axes, **kwargs):
        plotter = self.pdf_plotter
        return plotter.plotScatter(ax, **kwargs)

    def plotPdfSurface(self, ax: Axes3D, **kwargs):
        plotter = self.pdf_plotter
        return plotter.plotSurface(ax, **kwargs)

    def plotPdfVectorField(self, ax: Axes, **kwargs):
        plotter = Function2dPlotter(
            self._pdf_matrix,
            self.x_bin_edges,
            self.y_bin_edges,
        )
        return plotter.plotVectorField(ax, **kwargs)


class CrossingDistributionBin2D(Distribution2D):
    @classmethod
    def fromSamples(
        cls,
        x: ndarray,
        x_bin_edges: ndarray = None,
        dt_bin_edges: ndarray = None,
        **kwargs,
    ):
        crossing_pairs = CrossingPairsBins(x, x_bin_edges=x_bin_edges)
        crossing_distr2d = crossing_pairs.toDistribution2D(
            dt_bin_edges=dt_bin_edges,
            **kwargs,
        )[0]
        return crossing_distr2d

    @classmethod
    def fromList(
        cls,
        distributions: list[CrossingDistributionBin],
        x_bin_edges: ndarray,
    ):
        dt_bin_edges = distributions[0].bin_edges
        pdfs = np.float64([distr._pdf_matrix for distr in distributions])
        return CrossingDistributionBin2D(
            pdfs,
            x_bin_edges,
            dt_bin_edges,
            density=True,
        )

    def __init__(
        self,
        pdf: ndarray,
        x_bin_edges: ndarray,
        dt_bin_edges: ndarray,
        **kwargs,
    ):
        Distribution2D.__init__(
            self,
            pdf,
            x_bin_edges,
            dt_bin_edges,
            **kwargs,
        )

    def rescale(
        self,
        t_amp: float = 1.0,
        x_amp: float = 1.0,
        x_shift: float = 0.0,
        rescale_edges=True,
    ):
        tx_area = t_amp * x_amp
        if not rescale_edges:
            t_amp = 1
            x_amp = 1

        return CrossingDistributionBin2D(
            self._pdf_matrix / tx_area,
            rescale(self.x_bin_edges, x_amp, x_shift),
            rescale(self.y_bin_edges, t_amp),
        )

    def normalizeByMax(self):
        pdf_max = self._pdf_matrix.max()
        return self.rescale(pdf_max, 1, rescale_edges=False)


class CrossingPairsBin:
    @classmethod
    def fromPosition(
        cls,
        x: ndarray,
        bin_edges: ndarray,
        bottom_crossing_signs: ndarray = None,
        top_crossing_signs: ndarray = None,
    ):
        if bottom_crossing_signs is None:
            bottom_crossing_signs = crossingSigns(x, bin_edges[0])
        if top_crossing_signs is None:
            top_crossing_signs = crossingSigns(x, bin_edges[1])

        up_pairs = cls.__directionalCrossingPairs(
            bottom_crossing_signs,
            top_crossing_signs,
            x,
            bin_edges,
        )
        down_pairs = cls.__directionalCrossingPairs(
            top_crossing_signs,
            bottom_crossing_signs,
            x,
            np.flip(bin_edges),
        )
        return CrossingPairsBin(down_pairs, up_pairs)

    @classmethod
    def __directionalCrossingPairs(
        cls,
        pre_signs: ndarray,
        post_signs: ndarray,
        x: ndarray,
        bin_edges: ndarray,
    ):
        sign = 1 if bin_edges[0] < bin_edges[1] else -1
        pre_inds = np.where(pre_signs == sign)[0]
        post_inds = np.where(post_signs == sign)[0]
        inds_pairs_int = cls.__crossingPairs(pre_inds, post_inds)
        inds_pairs_mantissa = cls._partialIndex(inds_pairs_int, x, bin_edges)
        return inds_pairs_int + inds_pairs_mantissa

    @classmethod
    def __crossingPairs(
        cls,
        pre_inds: ndarray,
        post_inds: ndarray,
    ):
        crossing_pairs = np.zeros((pre_inds.size, 2))
        for ind, post_ind in enumerate(pre_inds):
            try:
                ind_argmax = np.argmax(post_inds >= post_ind)
            except ValueError:
                continue
            pre_ind = post_inds[ind_argmax]
            if pre_ind < post_ind:
                continue
            crossing_pairs[ind, :] = (post_ind, pre_ind)

        is_null_crossing_pair = crossing_pairs[:, 0] != 0
        crossing_pairs = crossing_pairs[is_null_crossing_pair, :]
        crossing_pairs = np.flip(crossing_pairs, axis=0)
        _, unique_inds = np.unique(
            crossing_pairs[:, 1],
            return_index=True,
        )
        crossing_pairs = np.int64(crossing_pairs[unique_inds, :])
        return crossing_pairs

    @classmethod
    def _partialIndex(
        cls,
        inds: ndarray,
        x: ndarray,
        bin_edges: ndarray,
    ):
        x_left = x[inds]
        x_right = x[inds + 1]
        inds_partial = (bin_edges - x_left) / (x_right - x_left)
        return inds_partial

    def __init__(
        self,
        down_pairs: ndarray,
        up_pairs: ndarray,
    ):
        self.up_pairs = up_pairs
        self.down_pairs = down_pairs

    def __interpolate(self, x: ndarray, inds: ndarray):
        if isinstance(x, ndarray):
            return np.interp(inds, np.arange(x.size), x)
        return x * np.ones(inds.shape)

    def toDistribution(self, **kwargs):
        return CrossingDistributionBin.fromPairs(
            self.down_pairs,
            self.up_pairs,
            **kwargs,
        )

    def plotScatter(
        self,
        ax: Axes,
        x: ndarray,
        y: ndarray,
        marker: str = "o",
        **kwargs,
    ):
        down_inds = np.mean(self.down_pairs, axis=1)
        up_inds = np.mean(self.up_pairs, axis=1)
        inds = np.append(down_inds, up_inds)
        inds = inds[inds <= x.size]
        
        x_crossings = self.__interpolate(x, inds)
        y_crossings = self.__interpolate(y, inds)
        
        if marker != "|":
            ax.scatter(
                x_crossings,
                y_crossings,
                marker=marker,
                **kwargs,
            )
            return
        
        for x_crossing in x_crossings:
            ax.axvline(x_crossing, **kwargs)

    def plotDownEnterScatter(
        self,
        ax: Axes,
        x: ndarray,
        y: ndarray,
        **kwargs,
    ):
        down_enter_inds = self.down_pairs[:, 0]
        return ax.scatter(
            self.__interpolate(x, down_enter_inds),
            self.__interpolate(y, down_enter_inds),
            **kwargs,
        )

    def plotDownExitScatter(
        self,
        ax: Axes,
        x: ndarray,
        y: ndarray,
        **kwargs,
    ):
        down_exit_inds = self.down_pairs[:, 1]
        return ax.scatter(
            self.__interpolate(x, down_exit_inds),
            self.__interpolate(y, down_exit_inds),
            **kwargs,
        )

    def plotUpEnterScatter(
        self,
        ax: Axes,
        x: ndarray,
        y: ndarray,
        **kwargs,
    ):
        up_enter_inds = self.up_pairs[:, 0]
        return ax.scatter(
            self.__interpolate(x, up_enter_inds),
            self.__interpolate(y, up_enter_inds),
            **kwargs,
        )

    def plotUpExitScatter(
        self,
        ax: Axes,
        x: ndarray,
        y: ndarray,
        **kwargs,
    ):
        up_exit_inds = self.up_pairs[:, 1]
        return ax.scatter(
            self.__interpolate(x, up_exit_inds),
            self.__interpolate(y, up_exit_inds),
            **kwargs,
        )


class CrossingDistributionBin(CrossingPairsBin, Distribution):
    @classmethod
    def fromSamples(
        cls,
        x: ndarray,
        bin_edges: ndarray,
        bottom_crossing_signs: ndarray = None,
        top_crossing_signs: ndarray = None,
        **kwargs,
    ):
        if bottom_crossing_signs is None:
            bottom_crossing_signs = crossingSigns(x, bin_edges[0])
        if top_crossing_signs is None:
            top_crossing_signs = crossingSigns(x, bin_edges[1])

        down_pairs = cls.__directionalCrossingPairs(
            top_crossing_signs,
            bottom_crossing_signs,
            x,
            np.flip(bin_edges),
        )
        up_pairs = cls.__directionalCrossingPairs(
            bottom_crossing_signs,
            top_crossing_signs,
            x,
            bin_edges,
        )
        return CrossingDistributionBin.fromPairs(
            down_pairs,
            up_pairs,
            **kwargs,
        )

    @classmethod
    def fromPairs(
        cls,
        down_pairs: ndarray,
        up_pairs: ndarray,
        dt_bin_edges: ndarray = None,
        t_amp: float = 1,
        **kwargs,
    ):
        if dt_bin_edges is None:
            dt_bin_edges = 10

        down_crossing_inds = np.round(np.mean(down_pairs, axis=1))
        up_crossing_inds = np.round(np.mean(up_pairs, axis=1))
        crossing_inds = np.sort(np.append(down_crossing_inds, up_crossing_inds))

        hist = np.histogram(
            np.diff(crossing_inds) * t_amp,
            bins=dt_bin_edges * t_amp,
            **kwargs,
        )
        return CrossingDistributionBin(*hist, down_pairs, up_pairs)

    def __init__(
        self,
        pdf: ndarray,
        dt_bin_edges: ndarray,
        down_pairs: ndarray,
        up_pairs: ndarray,
    ):
        CrossingPairsBin.__init__(self, down_pairs, up_pairs)
        Distribution.__init__(self, pdf, dt_bin_edges)

    def rescale(self, t_amp: float = 1.0):
        return CrossingDistributionBin(
            self._pdf_matrix / t_amp,
            rescale(self.bin_edges, t_amp),
            self.down_pairs,
            self.up_pairs,
        )


class CrossingPairsBins:
    def __init__(
        self,
        x: ndarray,
        x_bin_edges: ndarray = None,
    ):
        if x_bin_edges is None:
            x_bin_edges = 21

        if isinstance(x_bin_edges, int):
            x_bin_edges = np.linspace(x.min(), x.max(), x_bin_edges)

        bin_count = x_bin_edges.size - 1
        crossing_signs = [
            crossingSigns(x, x_bin_edges[bin_index])
            for bin_index in range(bin_count + 1)
        ]

        def crossingPairSingleBin(bin_index: int):
            bin_edges = x_bin_edges[bin_index : bin_index + 2]
            bottom_crossing_signs = crossing_signs[bin_index]
            top_crossing_signs = crossing_signs[bin_index + 1]
            return CrossingPairsBin.fromPosition(
                x,
                bin_edges,
                bottom_crossing_signs=bottom_crossing_signs,
                top_crossing_signs=top_crossing_signs,
            )

        self.crossing_signs = crossing_signs
        self.crossing_pairs = list(map(crossingPairSingleBin, range(bin_count)))
        self.x_bin_edges = x_bin_edges

    @property
    def up_pairs(self):
        return [crossing_pair.up_pairs for crossing_pair in self.crossing_pairs]

    @property
    def down_pairs(self):
        return [crossing_pair.down_pairs for crossing_pair in self.crossing_pairs]

    def toDistributions(self, **kwargs):
        return [
            crossing_pair.toDistribution(**kwargs)
            for crossing_pair in self.crossing_pairs
        ]

    def toDistribution2D(
        self,
        dt_bin_edges: ndarray = None,
        **kwargs,
    ):
        if dt_bin_edges is None:
            up_pairs = np.concatenate(self.up_pairs)
            down_pairs = np.concatenate(self.down_pairs)
            crossing_pair = CrossingPairsBin(down_pairs, up_pairs)

            distribution = crossing_pair.toDistribution()
            dt_bin_count = round((down_pairs.size * up_pairs.size) ** (1 / 6))
            x_bin_count = self.x_bin_edges.size
            dt_bin_end = np.sqrt(x_bin_count) * distribution.bin_edges[-1]
            dt_bin_edges = np.linspace(
                distribution.bin_edges[0],
                dt_bin_end,
                dt_bin_count,
            )

        distributions = self.toDistributions(
            dt_bin_edges=dt_bin_edges,
            **kwargs,
        )
        distribution_2d = CrossingDistributionBin2D.fromList(
            distributions,
            self.x_bin_edges,
        )
        return distribution_2d, distributions


class PowerSpectralDensity:
    @classmethod
    def fromPosition(cls, x: ndarray, dt: float, nsegments: int = 8):
        x_welch = signal.welch(
            x,
            1 / dt,
            nperseg=x.size // nsegments,
            scaling="spectrum",
            window="hann",
        )
        return PowerSpectralDensity(*x_welch)

    @staticmethod
    def _cost(x_psd: PowerSpectralDensity, y_psd: PowerSpectralDensity):
        y_psd_overlap = y_psd.between(0, x_psd.max_frequency)
        try:
            y_psd_resample = y_psd_overlap.resample(x_psd.size).normalizeByArea()
            cost = integrate.simpson(
                np.abs(y_psd_resample.densities - x_psd.densities),
                x_psd.frequencies,
            )
            return min(cost, 1.0)
        except (ValueError, ZeroDivisionError):
            return 1.0

    def __init__(self, x_freq: ndarray, x_psd: ndarray):
        self.densities = x_psd
        self.frequencies = x_freq
        self.size = x_psd.size

        self.min_frequency: float = x_freq[0]
        self.max_frequency: float = x_freq[-1]

    @property
    def area(self):
        return integrate.simpson(self.densities, self.frequencies)

    @property
    def peak_frequency(self):
        return self._peakFrequency_median()

    def _peakFrequency_max(self):
        x_freq = self.frequencies
        x_psd = self.densities
        psd_argmax = np.argmax(x_psd)
        return x_freq[psd_argmax]

    def _peakFrequency_median(self):
        x_freq = self.frequencies
        x_psd = self.densities
        psd_int = integrate.cumulative_simpson(x_psd, x=x_freq)
        psd_arg_median = np.argmin(np.abs(psd_int - 0.5))
        return x_freq[psd_arg_median]

    @property
    def peak_period(self):
        return 1 / self.peak_frequency

    def __getitem__(self, inds: ndarray):
        return PowerSpectralDensity(
            self.frequencies[inds],
            self.densities[inds],
        )

    def between(self, freq_min: float, freq_max: float):
        frequencies = self.frequencies
        inds = np.logical_and(
            frequencies >= freq_min,
            frequencies <= freq_max,
        )
        return self[inds]

    def rescale(self, x_amp: float = 1.0, t_amp: float = 1.0):
        return PowerSpectralDensity(
            self.frequencies / t_amp,
            self.densities * x_amp**2,
        )

    def resample(self, num: int):
        densities = self.densities
        freq_zoom = num / densities.size
        freq_resampled = ndimage.zoom(self.frequencies, freq_zoom, order=1)
        densities_resampled = signal.resample(densities, num)
        return PowerSpectralDensity(freq_resampled, densities_resampled)

    def zoom(self, zoom: float, order: int = 3):
        freq_zoomed = ndimage.zoom(self.frequencies, zoom, order=1)
        densities_zoomed = ndimage.zoom(self.densities, zoom, order=order)
        return PowerSpectralDensity(freq_zoomed, densities_zoomed)

    def normalizeByArea(self):
        return self.rescale(1 / np.sqrt(self.area), 1)

    def cost(self, other: PowerSpectralDensity):
        return self._cost(self, other)

    def plot(self, ax: Axes, **kwargs):
        return ax.plot(
            self.frequencies,
            self.densities,
            **kwargs,
        )


class Trace:
    @classmethod
    def fromCsv(cls, filepath: str):
        df_data = pd.read_csv(filepath)
        x = np.array(df_data["trace"])
        t = np.array(df_data["time"])
        return Trace(t, x)

    @classmethod
    def fromHdf5(cls, group: h5py.Group):
        t = group[0, :]
        x = group[1, :]
        return Trace(t, x)

    def toDataframe(self):
        t = self.t
        xi = self.x
        df = pd.DataFrame(
            np.transpose([t, xi]),
            columns=["time", "trace"],
        )
        return df

    def toCsv(self, filepath: str):
        self.toDataframe().to_csv(filepath)

    def __init__(
        self,
        t: ndarray,
        x: ndarray,
        metadata: dict = None,
    ):
        assert t.shape[-1] == x.shape[-1]
        assert x.ndim in [1, 2]

        self.x = x
        self.t = t
        self.metadata = metadata
        self.__memory = {
            "analytic_signal": None,
            "analytic_distribution": None,
            "analytic_velocity": None,
            "crossing_distribution": None,
            "psd": None,
            "distribution": None,
        }

    def __len__(self):
        return int(self.t.size)

    @property
    def dt(self):
        t = self.t
        return t[1] - t[0]

    @property
    def sample_rate(self):
        return 1 / self.dt

    def __getitem__(self, inds: ndarray):
        if isinstance(inds, (slice, Number)):
            return self.__getitem__slice(inds)
        elif isinstance(inds, ndarray):
            return self.__getitem__ndarray(inds)

    def __getitem__ndarray(self, inds: slice):
        x = self.x
        if x.ndim == 1:
            x = np.interp(inds, np.arange(x.size), x)
        elif x.ndim == 2:
            x = x[:, inds]

        t = self.t
        t = np.interp(inds, np.arange(t.size), t)

        return Trace(
            t,
            x,
            metadata=self.metadata,
        )

    def __getitem__slice(self, inds: slice):
        x = self.x
        if x.ndim == 1:
            x = x[inds]
        elif x.ndim == 2:
            x = x[:, inds]

        t = self.t[inds]

        return Trace(
            t,
            x,
            metadata=self.metadata,
        )

    def flip(self):
        return Trace(
            np.flip(self.t),
            np.flip(self.x, axis=-1),
            metadata=self.metadata,
        )

    def betweenTimes(
        self,
        t_start: float = None,
        t_end: float = None,
    ):
        t = self.t
        if t_start is None:
            t_start = t[0]
        if t_end is None:
            t_end = t[-1]

        assert t[0] <= t_start < t_end <= t[-1]
        start_index = np.where(crossingSigns(t - t_start) == 1)[0][0]
        end_index = np.where(crossingSigns(t - t_end) == 1)[0][0]
        return self[start_index:end_index]

    def rescale(
        self,
        t_amp: float = 1.0,
        x_amp: float = 1.0,
        x_shift: float = 0.0,
        t_shift: float = 0.0,
    ):
        return Trace(
            rescale(self.t, t_amp, t_shift),
            rescale(self.x, x_amp, x_shift),
        )

    def downsample(self, other: Trace):
        sample_rate = other.sample_rate / self.sample_rate
        if sample_rate >= 1:
            return self

        fs_self = 1 / sample_rate
        inds = np.arange(0, len(self), fs_self)
        return self[inds]

    def normalize(self):
        x = self.x
        return Trace(
            self.t,
            x / np.linalg.norm(x),
            metadata=self.metadata,
        )

    def toHdf5(
        self,
        dataset: h5py.Group,
        name: str = "trace",
    ):
        data = np.array([self.t, self.x])
        dataset.create_dataset(
            name,
            dtype=np.float64,
            data=data,
            compression="gzip",
            compression_opts=9,
        )

    def toTuple(self):
        t = self.t
        x = self.x
        metadata = self.metadata
        return (t, x, metadata)

    def toAnalyticSignal(self):
        analytic_signal = self.__memory["analytic_signal"]
        if analytic_signal is not None:
            return analytic_signal

        analytic_signal = signal.hilbert(self.x)
        self.__memory["analytic_signal"] = analytic_signal
        return analytic_signal

    def toAnalyticDistribution2D(self, **kwargs):
        analytic = self.__memory["analytic_distribution"]
        if analytic is not None:
            return analytic

        analytic_signal = self.toAnalyticSignal()
        distribution = Distribution2D.fromSamples(
            analytic_signal,
            density=True,
            **kwargs,
        )
        self.__memory["analytic_distribution"] = distribution
        return distribution

    def toAnalyticVelocity2D(
        self,
        density: bool = True,
        **kwargs,
    ):
        distribution = self.__memory["analytic_velocity"]
        if distribution is not None:
            return distribution

        analytic_signal = self.toAnalyticSignal()
        dt = np.diff(self.t)
        vfield = Distribution2D.fromSamples(
            midpoints(analytic_signal),
            np.diff(analytic_signal) / dt,
            density=density,
            **kwargs,
        )

        self.__memory["analytic_velocity"] = vfield
        return vfield

    def toPsd(
        self,
        normalized: bool = True,
        **kwargs,
    ):
        psd = self.__memory["psd"]
        if psd is not None:
            return psd

        psd = PowerSpectralDensity.fromPosition(
            self.x,
            self.dt,
            **kwargs,
        )
        if normalized:
            psd = psd.normalizeByArea()

        self.__memory["psd"] = psd
        return psd

    def toDistribution(self, **kwargs):
        distribution = self.__memory["distribution"]
        if distribution is not None:
            return distribution

        distribution = Distribution.fromSamples(self.x, **kwargs)
        self.__memory["distribution"] = distribution
        return distribution

    def toCrossingPairs(
        self,
        denoise: bool = False,
        **kwargs,
    ):
        x = self.x
        if denoise:
            x = denoiseTrace(x)
        return CrossingPairsBins(x, **kwargs)

    def toCrossingDistribution2D(
        self,
        dt_bin_edges: float = None,
        denoise: bool = False,
        **kwargs,
    ):
        distribution = self.__memory["crossing_distribution"]
        if distribution is not None:
            return distribution

        if dt_bin_edges is not None:
            dt_bin_edges = dt_bin_edges / self.dt

        x = self.x
        if denoise:
            x = denoiseTrace(self.x)

        distribution = CrossingDistributionBin2D.fromSamples(
            x,
            dt_bin_edges=dt_bin_edges,
            **kwargs,
        )
        distribution = distribution.rescale(self.dt)
        self.__memory["crossing_distribution"] = distribution
        return distribution

    def plotTrace(self, ax: Axes, **kwargs):
        return ax.plot(self.t, self.x, **kwargs)


class TraceChunker:
    @classmethod
    def __getChunk(
        cls,
        trace: Trace,
        index: int,
        size: int,
        overlap: float,
        max_size: int,
        count: int = None,
    ):
        start_index, end_index = cls.__getChunkIndices(
            index,
            size,
            overlap,
            max_size,
            count=count,
        )
        return trace[start_index:end_index]

    @classmethod
    def __getChunkIndices(
        cls,
        index: int,
        size: int,
        overlap: float,
        max_size: int,
        count: int = None,
    ):
        if count is not None:
            assert 0 <= index <= count - 1
        start_index = round(size * index * (1 - overlap))
        end_index = min(start_index + size, max_size)
        return start_index, end_index

    def __init__(
        self,
        trace: Trace,
        chunk_time: Number = None,
        chunk_overlap: Number = 0,
    ):
        assert 0 <= chunk_overlap <= 1
        assert isinstance(chunk_overlap, Number)

        trace_size = len(trace)
        if chunk_time is None:
            chunk_time = round(trace_size ** (2 / 3))
        assert isinstance(chunk_time, Number)

        chunk_size = int(chunk_time / trace.dt)
        chunk_count = int(trace_size / (chunk_size * (1 - chunk_overlap)))

        self.trace = trace
        self.count = chunk_count
        self.overlap = chunk_overlap
        self.size = chunk_size
        self.chunk_time = chunk_time

    def getChunk(self, index: int):
        return self.__getChunk(
            self.trace,
            index,
            size=self.size,
            overlap=self.overlap,
            max_size=len(self.trace),
            count=self.count,
        )

    def getChunkIndices(self, index: int):
        return self.__getChunkIndices(
            index,
            size=self.size,
            overlap=self.overlap,
            max_size=len(self.trace),
            count=self.count,
        )

    def getChunkTimeSpan(self, index: int):
        indices = np.array(self.getChunkIndices(index))
        t_span: tuple[float, float] = tuple(self.trace.t[indices])
        return t_span

    def getChunkTimeStart(self, index: int):
        return self.getChunkTimeSpan(index)[0]

    def getChunkTimeEnd(self, index: int):
        return self.getChunkTimeSpan(index)[1]


class SdeModelSymbolic:
    time = sym.Symbol("t")

    def _assertParametersExist(self, names: Iterable[str] = None):
        if names is None:
            return True

        names = list(map(str, names))
        self_names = set(self.parameter_names)
        for name in names:
            if name not in self_names:
                msg = f"Input parameters {name} do not match model parameters {self_names}"
                raise ValueError(msg)

        return True

    @staticmethod
    def _assertParametersType(parameters: Iterable):
        for parameter in parameters:
            if not isinstance(parameter, (Number, sym.Symbol)):
                parameter_type = type(parameter).__name__
                message = f"{parameter} ({parameter_type}) must be number or symbol"
                raise TypeError(message)

    def __init__(
        self,
        derivative_expressions: sym.Tuple,
        variable_symbols: list[sym.Symbol],
        parameter_symbols: list[sym.Symbol],
        noise_count: int = 0,
    ):
        self.derivatives = derivative_expressions
        self.variables = variable_symbols
        self.parameters = parameter_symbols
        self.noise_count = noise_count

    @property
    def variable_names(self):
        return list(map(str, self.variables))

    @property
    def variable_count(self):
        return len(self.variables)

    @property
    def parameter_names(self):
        return list(map(str, self.parameters))

    @property
    def parameter_count(self):
        return len(self.parameters) + self.noise_count

    def __removeQuantity(self, name: Union[str, sym.Symbol]):
        if self.isVariable(name):
            self.__removeVariable(name)
        elif self.isParameter(name):
            parameter_index = self.index(name)
            self.parameters.pop(parameter_index)
        else:
            message = f"{name} matches neither variable nor parameter name"
            raise ValueError(message)

    def __removeVariable(self, index: Union[int, str, sym.Symbol]):
        if isinstance(index, (str, sym.Symbol)):
            index = self.index(index)
        assert isinstance(index, int)

        self.variables.pop(index)

        derivatives = self.derivatives
        self.derivatives = sym.Tuple(
            *[
                *derivatives[:index],
                *derivatives[index + 1 :],
            ]
        )

    def index(self, name: Union[int, str, sym.Symbol]):
        if isinstance(name, int):
            assert name <= max(self.variable_count, self.parameter_count)
            return name
        assert isinstance(name, (str, sym.Symbol))

        variable_names = self.variable_names
        parameter_names = self.parameter_names
        if self.isVariable(name):
            return variable_names.index(name)
        elif self.isParameter(name):
            return parameter_names.index(name)

        message = f"{name} neither in variables {variable_names} nor parameters {parameter_names}"
        raise ValueError(message)

    def isParameter(self, name: Union[str, sym.Symbol]):
        if isinstance(name, sym.Symbol):
            name = str(name)
        return name in self.parameter_names

    def isVariable(self, name: Union[str, sym.Symbol]):
        if isinstance(name, sym.Symbol):
            name = str(name)
        return name in self.variable_names

    def subs(self, substitution: dict[str, float]):
        for quantity_name, value in substitution.items():
            assert isinstance(quantity_name, (str, sym.Symbol))
            assert isinstance(value, (Number, sym.Expr))

        self_copy = copy.deepcopy(self)
        for quantity_name in substitution.keys():
            self_copy.__removeQuantity(quantity_name)
        self_copy.derivatives = sym.simplify(self_copy.derivatives.subs(substitution))

        return self_copy

    def assumeEquilibrium(self, variable_name: str):
        variable_index = self.index(variable_name)
        variable_symbol = self.variables[variable_index]
        steady_state = sym.solve(
            self.derivatives.subs(sym.zoo, 1)[variable_index],
            variable_symbol,
        )[0]

        self.__removeVariable(variable_name)
        self.derivatives = sym.simplify(
            self.derivatives.subs(
                variable_symbol,
                steady_state,
            )
        )

    def addTermToDerivative(self, variable_name: str, term: sym.Expr):
        term_symbols = term.free_symbols
        variable_index = self.index(variable_name)
        remaining_symbols = list(
            set(term_symbols) - set(self.variables) - set([self.time])
        )

        derivatives = self.derivatives
        derivative_with_term = derivatives[variable_index] + term
        derivative_expressions = [
            *derivatives[:variable_index],
            derivative_with_term,
            *derivatives[variable_index + 1 :],
        ]
        self.derivatives = sym.Tuple(*derivative_expressions)
        self.parameters = [*self.parameters, *remaining_symbols]

    def reorderParameters(self, names: Iterable[str]):
        if names is None:
            return
        names = list(map(str, names))
        self._assertParametersExist(names)

        self_names = self.parameter_names
        indices = np.array(list(map(self_names.index, names)))

        for index, self_index in enumerate(np.sort(indices)):
            parameter_name = names[index]
            self.parameters[self_index] = sym.Symbol(parameter_name)

    def __generateSignature(
        self,
        signature: list[str] = None,
        include_time: bool = True,
        include_parameters: bool = False,
    ):
        if signature is None:
            signature = []
            if include_time:
                t_sym = self.time
                signature.append(t_sym)

            x_sym = self.variables
            signature.append(x_sym)

            if include_parameters:
                p_sym = self.parameters
                signature.append(p_sym)

        argument_names = []
        for argument_group in signature:
            if isinstance(argument_group, (str, sym.Symbol)):
                argument_names.append(argument_group)
                continue
            for argument in argument_group:
                assert isinstance(argument, (str, sym.Symbol))
                argument_names.append(argument)

        return signature

    def lambdifyDerivative(
        self,
        parameter_values: list[float] = None,
        signature: list[str] = None,
        **kwargs,
    ):
        signature = self.__generateSignature(
            signature,
            include_parameters=parameter_values is None,
        )
        dxdt_sym = self.derivatives
        if parameter_values is not None:
            p_sym = self.parameters
            assert self.parameter_count == len(parameter_values)
            p_name2value = dict(zip(p_sym, parameter_values))
            dxdt_sym = dxdt_sym.subs(p_name2value)

        dxdt = numba.jit(
            sym.lambdify(
                tuple(signature),
                dxdt_sym,
                cse=True,
                **kwargs,
            )
        )

        return dxdt

    def generateHandler(
        self,
        t: ndarray,
        x0: ndarray,
        **kwargs,
    ):
        return SdeModelHandler(
            self.derivatives,
            tuple(self.variables),
            tuple(self.parameters),
            t,
            x0,
            **kwargs,
        )


class SdeModelHandler(SdeModelSymbolic):
    def __init__(
        self,
        derivative_expressions: sym.Tuple,
        variable_symbols: list[sym.Symbol],
        parameter_symbols: list[sym.Symbol],
        t: ndarray,
        x0: ndarray,
        t_start: float = 0.0,
        noise_count: int = 0,
        generator=None,
    ):
        assert 0 <= noise_count <= x0.size and isinstance(noise_count, int)

        SdeModelSymbolic.__init__(
            self,
            derivative_expressions,
            variable_symbols,
            parameter_symbols,
            noise_count=noise_count,
        )
        self.x0 = x0
        self.t = t
        self.t_start = t_start
        self.generator = copy.deepcopy(generator)

    def toHdf5(self, group: h5py.Group):
        model_group = group.create_group("model")
        model_group.create_dataset(
            "time",
            dtype=np.float64,
            data=self.t,
            compression="gzip",
            compression_opts=9,
        )
        model_group.attrs["parameter_name"] = self.parameter_names
        model_group.attrs["variable_name"] = self.variable_names
        model_group.attrs["initial_variable"] = self.x0
        model_group.attrs["noise_count"] = self.noise_count
        model_group.attrs["initial_time"] = self.t_start
        model_group.attrs["derivative_expressions"] = str(self.derivatives)

    def generateModel(
        self,
        parameter_values: ndarray,
        inds: ndarray = None,
    ):
        if isinstance(parameter_values, Number):
            parameter_values = [parameter_values]

        x0 = self.x0
        if inds is None:
            inds = np.arange(x0.size)

        noise_count = self.noise_count
        extra_count = self.variable_count - noise_count
        noise_values = []
        has_noise = extra_count >= 1 and noise_count >= 1

        if has_noise:
            noise_values = [
                *parameter_values[-self.noise_count :],
                *np.zeros(extra_count),
            ]

        dxdt_func = self.lambdifyDerivative(
            parameter_values,
            modules=["math"],
        )

        def f(t: ndarray, x: ndarray) -> ndarray:
            return np.array(dxdt_func(t, x))

        t, x = SdeSolver.solve(
            f,
            self.t,
            x0,
            noise_values=noise_values,
            generator=self.generator,
        )
        metadata = {"parameters": parameter_values}
        trace = Trace(
            t,
            x[inds, :],
            metadata=metadata,
        )
        trace = trace.betweenTimes(t_start=self.t_start)
        return trace


class TraceFitter:
    weights = [0.25, 0.25, 0.25, 0.25]
    # 2D distribution of analytic signal
    # 1D power spectral density
    # 2D velocity field of analytic signal
    # 2D distribution of crossings

    @classmethod
    def __t_amp_guess(
        cls,
        data: Trace,
        model: Trace,
    ):
        model_peak_frequency = model.toPsd().peak_frequency
        data_peak_frequency = data.toPsd().peak_frequency
        t_amp_guess = model_peak_frequency / data_peak_frequency
        return t_amp_guess

    @classmethod
    def __x_amp_guess(
        cls,
        data: Trace,
        model: Trace,
    ):
        model_amp = model.toDistribution().std()
        data_amp = data.toDistribution().std()
        x_amp_guess = data_amp / model_amp if model_amp > 0 else 1
        return x_amp_guess

    @classmethod
    def __x_shift_guess(
        cls,
        data: Trace,
        model: Trace,
    ):
        data_mean = data.toDistribution().mean()
        model_mean = model.toDistribution().mean()
        x_amp_guess = cls.__x_amp_guess(data, model)
        x_shift_guess = model_mean - data_mean / x_amp_guess
        return x_shift_guess

    @classmethod
    def __byPsd(
        cls,
        data: Trace,
        model: Trace,
        t_amp: float = 1.0,
        x_amp: float = 1.0,
    ):
        data_psd = data.toPsd()
        model_psd = model.toPsd().normalizeByArea()
        model_psd = model_psd.rescale(x_amp, t_amp)
        model_psd = model_psd.normalizeByArea()
        cost = data_psd.cost(model_psd)
        return cost

    @classmethod
    def __byDistribution(
        cls,
        data: Trace,
        model: Trace,
        x_amp: float = 1.0,
        x_shift: float = 0.0,
    ):
        data_pdf = data.toDistribution()
        model_pdf = model.toDistribution().rescale(x_amp, x_shift)
        cost = data_pdf.cost(model_pdf)
        return cost

    @classmethod
    def __byAnalyticDistribution2D(
        cls,
        data: Trace,
        model: Trace,
        x_amp: float = 1.0,
        x_shift: float = 0.0,
    ):
        data_pdf = data.toAnalyticDistribution2D()
        model_pdf = model.toAnalyticDistribution2D().rescale(x_amp, (x_shift, 0))
        cost = data_pdf.cost(model_pdf)
        return cost

    @classmethod
    def __byCrossingDistribution2D(
        cls,
        data: Trace,
        model: Trace,
        t_amp: float = 1.0,
        x_amp: float = 1.0,
        x_shift: float = 0.0,
    ):
        data_pdf = data.toCrossingDistribution2D(denoise=True)
        model_pdf = model.toCrossingDistribution2D(denoise=True)
        model_pdf = model_pdf.rescale(t_amp, x_amp, x_shift)
        cost = data_pdf.cost(model_pdf)
        return cost

    @classmethod
    def __byAnalyticVelocity2D(
        cls,
        data: Trace,
        model: Trace,
        x_amp: float = 1.0,
        x_shift: float = 0.0,
    ):
        data_vfield = data.toAnalyticVelocity2D(weighted=True)
        model_vfield = model.toAnalyticVelocity2D(weighted=True)
        model_vfield = model_vfield.rescale(x_amp, (x_shift, 0)).normalizeByArea()
        cost = data_vfield.cost(model_vfield)
        return cost

    @classmethod
    def __cost(
        cls,
        data: Trace,
        model: Trace,
        t_amp: float = 1.0,
        x_amp: float = 1.0,
        x_shift: float = 0.0,
        weights: ndarray = None,
    ):
        if weights is None:
            weights = cls.weights
        if not isinstance(weights, ndarray):
            weights = np.array(weights)

        cost_analytic = 0.0
        cost_psd = 0.0
        cost_vfield = 0.0
        cost_crossing = 0.0
        weight_is_zero = weights == 0.0

        if not weight_is_zero[0]:
            cost_analytic = cls.__byAnalyticDistribution2D(
                data,
                model,
                x_amp,
                x_shift,
            )
        if not weight_is_zero[1]:
            cost_psd = cls.__byPsd(
                data,
                model,
                t_amp,
                x_amp,
            )
        if not weight_is_zero[2]:
            cost_vfield = cls.__byAnalyticVelocity2D(
                data,
                model,
                x_amp,
                x_shift,
            )
        if not weight_is_zero[3]:
            cost_crossing = cls.__byCrossingDistribution2D(
                data,
                model,
                t_amp,
                x_amp,
                x_shift,
            )

        costs = np.float64([cost_analytic, cost_psd, cost_vfield, cost_crossing])
        cost: float = np.sum(weights * costs)
        return cost

    @classmethod
    def __byCombined(
        cls,
        data: Trace,
        model: Trace,
        weights: ndarray,
        factor: tuple[float, float] = (0.5, 2),
    ):
        factor = np.array(factor)

        t_amp_guess = cls.__t_amp_guess(data, model)
        x_amp_guess = cls.__x_amp_guess(data, model)
        x_shift_guess = cls.__x_shift_guess(data, model)
        fit_guess = (t_amp_guess, x_amp_guess, x_shift_guess)
        fit_bounds = [
            factor * t_amp_guess,
            factor * x_amp_guess,
            (x_shift_guess, x_shift_guess),
        ]

        def costFunction(parameters: ndarray):
            return cls.__cost(
                data,
                model,
                *parameters,
                weights=weights,
            )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            fit = optimize.minimize(
                costFunction,
                fit_guess,
                bounds=fit_bounds,
                method="SLSQP",
                jac="2-point",
                tol=1e-3,
            )
            fit_guess: ndarray = fit.x
            fit_bounds = np.sort(
                fit_guess.reshape(3, 1) @ factor.reshape(1, 2),
                1,
            )
            fit = optimize.minimize(
                costFunction,
                fit_guess,
                bounds=fit_bounds,
                method="SLSQP",
                jac="3-point",
                tol=1e-9,
            )
        return fit

    def __init__(
        self,
        data: Trace,
        weights: ndarray = None,
    ):
        if weights is None:
            weights = np.array(TraceFitter.weights)

        self.data = data
        self.weights = weights

    def fit(self, model: Trace):
        return self.__byCombined(
            self.data,
            model,
            weights=self.weights,
        )

    def cost(self, model: Trace):
        try:
            fit = self.fit(model)
            cost = fit.fun
            if np.isnan(cost):
                return 1.0
        except (ValueError, IndexError, RuntimeWarning) as error:
            return 1.0

        return cost


class TraceCoster:
    def __init__(
        self,
        generate_trace: Callable[[ndarray], Trace],
        cost_function: Union[
            Callable[[Trace], float],
            Callable[[ndarray], float],
            Callable[[ndarray, Trace], float],
        ],
        include_parameters: bool = False,
        include_trace: bool = True,
    ):
        self.generate_trace = generate_trace
        self.cost_function = cost_function
        self.include_parameters = include_parameters
        self.include_trace = include_trace

    def cost(self, parameter_values: ndarray):
        args = []
        if self.include_parameters:
            args.append(parameter_values)
        if self.include_trace:
            trace = self.generate_trace(parameter_values)
            args.append(trace)
        return self.cost_function(*args)


class DetailedBalancer:
    @classmethod
    def fromPosition(
        cls,
        x: ndarray,
        bin_count: int = None,
    ):
        if bin_count is None:
            bin_count = int(np.cbrt(x.shape[-1]))

        x_2d = np.array([np.real(x), np.imag(x)])
        return DetailedBalancer(x_2d, x_bin_count=bin_count)

    @classmethod
    def __transitionStatePoints(
        cls,
        xy: ndarray,
        bin_count: int,
        include_still_states: bool = False,
        include_intermediate_states: bool = True,
    ):
        xy = np.transpose(xy)
        dxy_min = xy - np.min(xy, axis=0)
        xy_float = np.transpose(
            (1 - 10**-9) * bin_count * dxy_min / np.max(dxy_min, axis=0)
        )

        def bresenhamPoints(ind: int):
            return cls.__pointsOnBresenhamLine(*xy_float[:, ind - 1], *xy_float[:, ind])

        xy_int: ndarray = np.int64(xy_float).T
        xy_sad = cls.taxicabDistance(xy_int)

        xy_sad_near_inds = np.append(
            0,
            np.where(xy_sad <= 1 if include_still_states else xy_sad == 1)[0] + 1,
        )
        xy_sad_far_inds = np.where(xy_sad > 1)[0] + 1
        xy_sad_inds = np.sort([*xy_sad_near_inds, *xy_sad_far_inds])

        xy_states_near = [
            [xy_state_near] for xy_state_near in xy_int[xy_sad_near_inds, :]
        ]
        xy_states_far = (
            list(map(bresenhamPoints, xy_sad_far_inds))
            if include_intermediate_states
            else [[xy_state_far] for xy_state_far in xy_int[xy_sad_far_inds, :]]
        )

        xy_states = np.empty((xy.shape[0]), dtype=list)
        xy_states[xy_sad_near_inds] = xy_states_near
        xy_states[xy_sad_far_inds] = xy_states_far
        xy_states = np.concatenate(xy_states[xy_sad_inds], axis=0)
        return xy_states.T

    @classmethod
    def __transitionVectorField(
        cls,
        xy_states: ndarray,
        bin_count: int,
    ):
        xydim_count, state_count = xy_states.shape
        xy_vector = np.zeros((xydim_count, bin_count, bin_count))
        xy_steps = np.diff(xy_states, axis=1)

        state_inds = np.arange(state_count - 1)
        xy_indss = xy_states[:, [state_inds, state_inds + 1]]

        for state_ind in state_inds:
            for point_ind in range(xydim_count):
                xy_state = xy_indss[:, point_ind, state_ind]
                xy_vector[:, *xy_state] += xy_steps[:, state_ind]

        return xy_vector

    @classmethod
    def __pointsOnBresenhamLine(
        cls,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
    ):
        """
        Draws a line between two points using Bresenham's algorithm with floats.

        Args:
            x0, y0: Starting coordinates (float).
            x1, y1: Ending coordinates (float).

        Returns:
            A list of (x, y) tuples representing the points on the line (integers).
        """
        points = []
        x, y = x0, y0
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1

        if dx > dy:
            error = dx / 2.0
            while int(x) != int(x1):
                points.append((int(x), int(y)))
                x += sx
                error -= dy
                if error < 0:
                    y += sy
                    error += dx
        else:
            error = dy / 2.0
            while int(y) != int(y1):
                points.append((int(x), int(y)))
                y += sy
                error -= dx
                if error < 0:
                    x += sx
                    error += dy

        points.append((int(x1), int(y1)))  # Ensure the final point is included
        return points

    @staticmethod
    def taxicabDistance(xy: ndarray):
        return np.sum(np.abs(np.diff(xy, axis=0)), axis=1)

    def __init__(self, xy: ndarray, x_bin_count: int = None):
        if x_bin_count is None:
            state_count = xy.shape[1]
            x_bin_count = int(np.cbrt(state_count))

        self.states = self.__transitionStatePoints(xy, x_bin_count)
        self.vector_field = self.__transitionVectorField(self.states, x_bin_count)

        self.x_bin_edges = np.arange(x_bin_count + 1)
        self.y_bin_edges = np.arange(x_bin_count + 1)
        self.x_bin_count = x_bin_count
        self.bin_count = x_bin_count**2

    @property
    def vector_scale(self, normalization: float = 1.0):
        vector_field = self.vector_field
        vector_scale = np.sqrt(np.max(np.sum(vector_field**2, axis=0)))
        return vector_scale / normalization

    def toStateDistribution2D(self, bins: int = None, **kwargs):
        if bins is None:
            bins = (self.x_bin_edges, self.y_bin_edges)

        ax_hist2d = np.histogram2d(
            *self.states,
            bins=bins,
            density=True,
            **kwargs,
        )
        return Distribution2D(*ax_hist2d)

    def plotVectorField(self, ax: Axes, **kwargs):
        vector_field = self.vector_field
        vector_scale = self.vector_scale
        xy_mesh = Binner.edgeToCenter(self.x_bin_edges, self.y_bin_edges)
        return ax.quiver(
            *xy_mesh,
            *vector_field,
            scale=vector_scale,
            scale_units="xy",
            angles="xy",
            **kwargs,
        )
