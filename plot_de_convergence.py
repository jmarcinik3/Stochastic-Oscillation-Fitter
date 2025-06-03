import h5py
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np

from Fitter import DifferentialEvolutionParser
import HairBundleModel


def generateConvergenceFigure(
    de_parser: DifferentialEvolutionParser,
    parameter_names: str,
    microcolor: str = "black",
    microalpha: float = 0.2,
    macrocolor: str = "red",
):
    parameter_count = len(parameter_names)

    fig, axs = plt.subplots(
        parameter_count + 1,
        figsize=(6, parameter_count),
        sharex="col",
    )
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
        np.mean,
        color=macrocolor,
    )

    ax_cost.set_ylabel("Cost")
    ax_cost.set_xscale("log")
    ax_cost.set_yscale("log")
    ax_cost.set_xlim(1, None)

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
            np.mean,
            color=macrocolor,
            zorder=1,
        )

        if isinstance(parameter_name, str):
            ax_parameter.set_ylabel(parameter_name)

    for ax in axs:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    return fig


if __name__ == "__main__":
    model = HairBundleModel.Nondimensional(
        Cmin=1,  # constant climbing rate
        tauT0=0,  # equilibrium transduction-channel gating
        taugs0=1,  # equilibrium gating-spring stiffness
        Cm=1,  # arbitrary calcium-feedback strength on motor
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
    parameter_names = [*model.parameter_names, "nhb", "na"]

    with h5py.File(f"traces_sac/cell2-2000.hdf5", "r") as file:
        de_parser = DifferentialEvolutionParser.fromHdf5(file)

    fig = generateConvergenceFigure(de_parser, parameter_names)
    plt.show()
