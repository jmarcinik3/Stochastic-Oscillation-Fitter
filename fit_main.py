from functools import partial
import h5py
import numpy as np
from numpy import ndarray
import multiprocessing as mp
from scipy import optimize

from Fitter import (
    DifferentialEvolutionParser,
    TraceFitter,
    SdeModelHandler,
    Trace,
    TraceCoster,
)
import HairBundleModel


class SdeFitter:
    def __init__(
        self,
        data: Trace,
        model_handler: SdeModelHandler,
        hdf5_filepath: str,
        bounds: ndarray,
        weights: ndarray = np.array([0.5, 0.1, 0, 0.4]),
        population_size=64,
        sampling="sobol",
        strategy="rand1exp",
        recombination=0.7,
        mutation=(0.5, 1.0),
    ):
        self.population_size = population_size
        self.sampling = sampling
        self.strategy = strategy
        self.recombination = recombination
        self.mutation = mutation

        self.data = data
        self.model_handler = model_handler
        self.weights = weights
        self.bounds = bounds

        self.hdf5_filepath = hdf5_filepath
        self.de_parser = DifferentialEvolutionParser()

        fitter = TraceFitter(data, weights=weights)
        generate_model = partial(model_handler.generateModel, inds=0)
        fitter_handler = TraceCoster(generate_model, fitter.cost)
        self.cost_function = fitter_handler.cost

    def __differentialEvolutionCallback(self, intermediate_result):
        DifferentialEvolutionParser.printResultSummary(intermediate_result)
        self.de_parser.appendResult(intermediate_result)

        with h5py.File(self.hdf5_filepath, "w") as file:
            self.de_parser.toHdf5(file)
            self.__metadataToHdf5(file)

    def __metadataToHdf5(self, group: h5py.Group):
        self.data.toHdf5(group, name="data")
        self.model_handler.toHdf5(group)

        group.attrs["parameter_name"] = model_handler.parameter_names
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
        maxiter: int = 100,
        atol: float = 0,
    ):
        bounds = self.bounds
        population_size = self.population_size
        if workers is None:
            workers = min(population_size, mp.cpu_count() - 1)

        self.de_parser = DifferentialEvolutionParser()
        soln = optimize.differential_evolution(
            self.cost_function,
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


if __name__ == "__main__":
    np.set_printoptions(precision=2)
    model = HairBundleModel.Nondimensional(
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
        # dE0=1,  # weak free-energy for channel opening
    )
    model_handler = model.generateHandler(
        t=np.linspace(0, int(1000), int(500000)),
        x0=np.array([-1, -1, 0.5, 0.5]),
        t_start=50,
        generator=np.random.default_rng(),
        noise_count=2,
    )
    bounds = np.array(
        [
            (0, 100),  # Cgs
            (0, 32),  # tauhb0
            (0, 1),  # Smax
            (0, 32),  # Ugsmax
            (0, 10),  # chia
            (0, 1),  # kgsmin
            (0, 32),  # taum0
            (0, 3.2),  # dE0
            (0, 0.1),  # nhb
            (0, 0.32),  # na
        ]
    )
    weights = np.array([0.5, 0.1, 0, 0.4])

    for i in range(6, 9):
        hdf5_filepath = f"traces/cell{i:d}-2000.hdf5"
        data_filepath = f"traces/cell{i:d}.csv"
        data = Trace.fromCsv(data_filepath)

        de_saver = SdeFitter(
            data,
            model_handler,
            hdf5_filepath,
            bounds=bounds,
            population_size=64,
            weights=weights,
        )
        de_saver.runFit(
            maxiter=2000,
            atol=-10,
            workers=28,
        )
