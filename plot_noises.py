from matplotlib import pyplot as plt
import HairBundleModel
import numpy as np


if __name__ == "__main__":
    parameter_values = [
        0.789,
        39.804,
        0.531,
        0.589,
        4.225,
        0.009,
        0.026,
        0.1,
    ]

    x0 = np.array([-1, -1, 0.5, 0.5])
    t_span = np.linspace(0, 1000, 500000)  # (0, 1000)

    model_sym = HairBundleModel.Nondimensional(
        Cmin=1,  # constant climbing rate
        tauT0=0,  # equilibrium transduction-channel gating
        taugs0=1,  # equilibrium gating-spring stiffness
        Cm=1,  # arbitrary calcium-feedback strength on motor
        xc=0,  # null offset on gating-spring force
        chihb=1,  # moderate gating-spring coupling to bundle position
        Smin=0,  # maximum change in slipping rate, relative to calcium bound to motor
        Cgs=1000,  # strong calcium-feedback strength on gating-spring
        tauhb0=1,  # moderate time constant for bundle position
        # Smax=0.5,  # comparable effect from slipping and climbing
        # Ugsmax=10,  # moderate elastic potential energy for gating spring
        # chia=1,  # moderate gating-spring coupling to motor position
        # kgsmin=1,  # constant gating-spring stiffness
        # taum0=10,  # moderate time constant for Ca-feedback at motor
        # dE0=1,  # weak free-energy for channel opening,
    )
    stochastic_handler = model_sym.generateHandler(
        t_span,
        x0,
        t_start=100,
        noise_count=2,
        generator=np.random.default_rng(),
    )
    deterministic_handler = model_sym.generateHandler(
        (t_span.min(), t_span.max()),
        x0,
        t_start=100,
    )

    parameters_both = [*parameter_values[:-2], *parameter_values[-2:]]
    parameters_onlyhb = [*parameter_values[:-2], parameter_values[-2], 0]
    parameters_onlya = [*parameter_values[:-2], 0, parameter_values[-1]]
    parameters_neither = parameter_values[:-2]

    model_both = stochastic_handler.generateModel(parameters_both, inds=0)
    model_hb = stochastic_handler.generateModel(parameters_onlyhb, inds=0)
    model_a = stochastic_handler.generateModel(parameters_onlya, inds=0)
    model_neither = deterministic_handler.generateModel(parameters_neither, inds=0)

    fig, ax = plt.subplots()
    model_both[::10].plotTrace(
        ax,
        color="magenta",
        linewidth=1,
        alpha=0.8,
        label="Both",
    )
    model_hb[::10].plotTrace(
        ax,
        color="blue",
        linewidth=1,
        alpha=0.8,
        label="Bundle",
    )
    model_a[::10].plotTrace(
        ax,
        color="red",
        linewidth=1,
        alpha=0.8,
        label="Myosin",
    )
    model_neither.plotTrace(
        ax,
        color="black",
        linewidth=1,
        alpha=0.8,
        label="Neither",
    )
    ax.legend()
    plt.show()
