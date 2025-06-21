import io
from typing import Iterable, Union

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pypdf import PageObject, PdfReader, PdfWriter
from tqdm import tqdm
import HairBundleModel
import numpy as np
from numpy import ndarray
from ap_sac_compare import CellDataFrame


class NoiseComparisonPlotter:
    etaa_ind = -1
    taum_ind = -4
    kgsmin_ind = -5
    chia_ind = -6
    Ugsmax_ind = -7
    smax_ind = -8
    tauhb_ind = -9

    @classmethod
    def __exportPages(
        cls,
        pages: list[PageObject],
        filepath: str,
    ):
        pages = np.array(pages).flatten()
        pdf_writer = PdfWriter()
        for page in pages:
            pdf_writer.add_page(page)
        pdf_writer.write(filepath)
        pdf_writer.close()

    @classmethod
    def __pageFromFigure(cls, fig: Figure) -> Union[list[PageObject], PageObject]:
        if isinstance(fig, Iterable):
            figs = np.array(fig).flatten()
            return list(map(cls.__pageFromFigure, figs))

        if isinstance(fig, PageObject):
            return fig
        elif isinstance(fig, Figure):
            pdf_buffer = io.BytesIO()  # type: ignore
            fig.savefig(pdf_buffer, format="pdf")
            pdf_buffer.seek(0)
            pdf_reader = PdfReader(pdf_buffer)
            page = pdf_reader.pages[0]
            return page

        assert TypeError

    @classmethod
    def __removeSpines(cls, axs: list[Axes]):
        for ax in axs:
            ax.spines["left"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.spines["top"].set_visible(False)

            ax.set_xticks(())
            ax.set_yticks(())

    @classmethod
    def __generateModelHandlers(cls, x0: ndarray, t_span: ndarray):
        model_sym = HairBundleModel.Nondimensional(
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
            # dE0=1,  # weak free-energy for channel opening,
        )

        deterministic_handler = model_sym.generateHandler(
            (t_span.min(), t_span.max()),
            x0,
            t_start=25,
        )
        stochastic_handler = model_sym.generateHandler(
            t_span,
            x0,
            t_start=25,
            noise_count=2,
            generator=np.random.default_rng(),
        )

        return deterministic_handler, stochastic_handler

    def __init__(
        self,
        parameter_values: list[ndarray],
        x0: ndarray,
        t_span: ndarray,
        fig: Figure = None,
        **kwargs,
    ):
        fig, axs = plt.subplots(
            4,
            1,
            sharex=True,
            sharey=True,
            **kwargs,
        )
        axs: ndarray

        self.parameter_values = np.array(parameter_values)
        self.handers = self.__generateModelHandlers(x0, t_span)

        self.fig = fig
        self.axs = axs

    @property
    def iteration_count(self):
        return self.parameter_values.shape[0]

    @property
    def deterministic_handler(self):
        return self.handers[0]

    @property
    def stochastic_handler(self):
        return self.handers[1]

    def __generateDeterministic(self, parameter_value: ndarray):
        deterministic_handler = self.deterministic_handler
        parameters = parameter_value[:-2]
        model = deterministic_handler.generateModel(parameters, inds=0)
        return model

    def __generateBundle(self, parameter_value: ndarray):
        stochastic_handler = self.stochastic_handler
        parameters = [*parameter_value[:-2], parameter_value[-2], 0]
        model = stochastic_handler.generateModel(parameters, inds=0)
        return model

    def __generateMyosin(self, parameter_value: ndarray):
        stochastic_handler = self.stochastic_handler
        parameters = [*parameter_value[:-2], 0, parameter_value[-1]]
        model = stochastic_handler.generateModel(parameters, inds=0)
        return model

    def __generateStochastic(self, parameter_value: ndarray):
        stochastic_handler = self.stochastic_handler
        parameters = [*parameter_value[:-2], *parameter_value[-2:]]
        model = stochastic_handler.generateModel(parameters, inds=0)
        return model

    def __plotOnAxes(
        self,
        axs: ndarray,
        parameter_value: ndarray,
    ):
        eta_hb = parameter_value[-2]
        eta_a = parameter_value[self.etaa_ind]

        tau_m = parameter_value[self.taum_ind]
        kgs_min = parameter_value[self.kgsmin_ind]
        chi_a = parameter_value[self.chia_ind]
        Ugs_max = parameter_value[self.Ugsmax_ind]
        s_max = parameter_value[self.smax_ind]
        tau_hb = parameter_value[self.tauhb_ind]

        model_both = self.__generateStochastic(parameter_value)
        model_hb = self.__generateBundle(parameter_value)
        model_a = self.__generateMyosin(parameter_value)
        model_neither = self.__generateDeterministic(parameter_value)

        self.__removeSpines(axs)
        ax_neither: Axes = axs[0]
        ax_hb: Axes = axs[1]
        ax_a: Axes = axs[2]
        ax_both: Axes = axs[3]

        title = (
            "Both "
            # + r"$\chi_a=$"
            # + f"{chi_a:.1f}, "
            # + r"$S_{max}=$"
            # + f"{s_max:.1f}, "
            + r"$\tau_{hb}=$"
            + f"{tau_hb:.2f}, "
            + r"$\tau_m=$"
            + f"{tau_m:.2f}, "
            + r"$k_{gs,min}=$"
            + f"{kgs_min:.2f}, "
            + r"$U_{gs,max}=$"
            + f"{Ugs_max:.2f}"
        )

        model_both[::10].plotTrace(
            ax_both,
            color="magenta",
            linewidth=1,
            alpha=0.8,
        )
        ax_both.set_title(title)

        model_hb[::10].plotTrace(
            ax_hb,
            color="blue",
            linewidth=1,
            alpha=0.8,
        )
        ax_hb.set_title(r"Bundle $\eta_{hb}=$" + f"{eta_hb:.3f}")

        model_a[::10].plotTrace(
            ax_a,
            color="red",
            linewidth=1,
            alpha=0.8,
        )
        ax_a.set_title(r"Myosin $\eta_a=$" + f"{eta_a:.3f}")

        model_neither.plotTrace(
            ax_neither,
            color="black",
            linewidth=1,
            alpha=0.8,
        )
        ax_neither.set_title("Neither")

        ax_both.set_xlim(25, 100)

    def exportFigures(self, filepath: str, title_pre: str = None):
        fig = self.fig
        axs = self.axs

        pages: list[PageObject] = []
        for index in tqdm(range(self.iteration_count)):
            parameter_value = self.parameter_values[index, :]

            for ax in axs.flatten():
                ax: Axes
                ax.cla()
            self.__plotOnAxes(axs, parameter_value)

            if isinstance(title_pre, str):
                fig.suptitle(f"{title_pre:s}{index:d}")

            page = self.__pageFromFigure(fig)
            pages.append(page)

        self.__exportPages(pages, filepath)


if __name__ == "__main__":
    df, parameter_bounds = CellDataFrame.generate(
        population_indices=0,
        iteration_indices=-1,
    )
    df_sac = df[df["End Organ"] == "Sacculus"]  # [3, 4, 5]
    df_ap = df[df["End Organ"] == "AP"]  # [1, 6, 9]

    parameter_values = np.array(
        df_sac.iloc[:, :-1],
        dtype=np.float64,
    )

    plotter = NoiseComparisonPlotter(
        figsize=(11, 8.5),
        parameter_values=parameter_values,
        x0=np.array([-1, -1, 0.5, 0.5]),
        t_span=np.linspace(0, 150, 75000),
    )
    plotter.exportFigures("temp.pdf", title_pre="Sacculus ")
    plt.show()
