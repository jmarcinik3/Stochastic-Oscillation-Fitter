from matplotlib import cm, colors, pyplot as plt
from matplotlib.axes import Axes
import numpy as np

from plot_cc_matrix import DivergenceMatrix


def generateColorbar(
    cax: Axes,
    mappable: cm.ScalarMappable,
    label: str = None,
):
    cbar = fig.colorbar(
        mappable,
        cax=cax,
        location="right",
    )
    cbar.set_ticks(
        (0, 0.5, 1),
        labels=("0", "", "1"),
    )
    cbar.set_ticks(
        np.linspace(0, 1, 11),
        minor=True,
    )
    cbar.set_label(
        label,
        rotation=270,
        labelpad=4,
    )

    return cbar


def plotRow(
    axs: list[Axes],
    div_matrices: list[DivergenceMatrix],
    include_labels: bool = True,
    norm=None,
    cmap: str = "Greys",
    cbar_label: str = None,
):
    ax_dd, ax_dm, ax_mm, ax_cbar = axs
    div_dd, div_dm, div_mm = div_matrices

    div_dd.plotMatrix(
        ax_dd,
        norm=norm,
        cmap=cmap,
    )
    div_dm.plotMatrix(
        ax_dm,
        norm=norm,
        cmap=cmap,
    )
    mappable = div_mm.plotMatrix(
        ax_mm,
        norm=norm,
        cmap=cmap,
    )
    cbar = generateColorbar(
        ax_cbar,
        mappable,
        label=cbar_label,
    )

    if include_labels:
        fontsize = "x-small"
        x_labelpad = 3
        y_labelpad = 1.5

        ax_dd.set_xlabel(
            "Measurement",
            fontsize=fontsize,
            labelpad=x_labelpad,
        )
        ax_dd.set_ylabel(
            "Measurement",
            fontsize=fontsize,
            labelpad=y_labelpad,
        )
        ax_dm.set_xlabel(
            "Measurement",
            fontsize=fontsize,
            labelpad=x_labelpad,
        )
        ax_dm.set_ylabel(
            "Simulation",
            fontsize=fontsize,
            labelpad=y_labelpad,
        )
        ax_mm.set_xlabel(
            "Simulation",
            fontsize=fontsize,
            labelpad=x_labelpad,
        )
        ax_mm.set_ylabel(
            "Simulation",
            fontsize=fontsize,
            labelpad=y_labelpad,
        )


if __name__ == "__main__":
    np.set_printoptions(precision=3)
    plt.rcParams["text.usetex"] = True
    plt.rcParams["text.latex.preamble"] = (
        r"\usepackage{amsmath}" r"\usepackage{lmodern}"
    )
    plt.rcParams["font.family"] = "lmodern"

    filenames = (
        "hellinger_dd.npy",
        "hellinger_dm.npy",
        "hellinger_mm.npy",
        "tvd_dd.npy",
        "tvd_dm.npy",
        "tvd_mm.npy",
        "jsd_dd.npy",
        "jsd_dm.npy",
        "jsd_mm.npy",
    )
    filepaths = [f"divergence/{filename}" for filename in filenames]

    div_all = map(
        DivergenceMatrix,
        map(np.load, filepaths),
    )
    div_all = list(div_all)
    div_hell = div_all[0:3]
    div_tvd = div_all[3:6]
    div_jsd = div_all[6:9]

    norm = colors.Normalize(vmin=0, vmax=1)
    cmap = "Greys"

    fig, axs = plt.subplots(
        nrows=3,
        ncols=4,
        figsize=(3.375, 3.375 * 8 / 9),
        width_ratios=(1, 1, 1, 1 / 12),
    )

    plotRow(
        axs[0, :],
        div_jsd,
        cbar_label=r"$d_{JS}$",
        norm=norm,
        cmap=cmap,
        include_labels=True,
    )
    plotRow(
        axs[1, :],
        div_hell,
        cbar_label=r"$d_H$",
        norm=norm,
        cmap=cmap,
        include_labels=False,
    )
    plotRow(
        axs[2, :],
        div_tvd,
        cbar_label=r"$d_{TV}$",
        norm=norm,
        cmap=cmap,
        include_labels=False,
    )

    fig.tight_layout(
        h_pad=0.5,
        w_pad=0.5,
    )
    fig.subplots_adjust(
        left=0.01,
        right=0.93,
        bottom=0.02,
        top=0.98,
        wspace=0,
    )
    # fig.savefig("temp.pdf")

    plt.show()
