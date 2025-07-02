from __future__ import annotations
import pylustrator

import copy
import io
import itertools
from multiprocessing import Pool
from numbers import Number
import time
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing import Callable, Iterable, Union
import h5py

from matplotlib import colors, cm, pyplot as plt, ticker
import numpy as np
from numpy import ndarray
from pypdf import PageObject, PdfReader, PdfWriter
from scipy import optimize, signal, stats
from tqdm import tqdm
from Fitter import (
    Distribution,
    Distributions,
    Trace,
    TraceChunker,
    TraceFitter,
)
from PerceptualUniform import Luminance
from Scalebar import plotScalebar


def hideAxis(
    ax: Axes,
    hide_top: bool = True,
    hide_bottom: bool = True,
    hide_left: bool = True,
    hide_right: bool = True,
    hide_ticks: bool = True,
):
    if hide_top:
        ax.spines["top"].set_visible(False)
    if hide_bottom:
        ax.spines["bottom"].set_visible(False)
    if hide_left:
        ax.spines["left"].set_visible(False)
    if hide_right:
        ax.spines["right"].set_visible(False)

    if hide_ticks:
        ax.set_xticks(())
        ax.set_yticks(())


def plotComparisonGrid(
    axs: ndarray,
    cc_d: Distributions,
    cc_dm: Distributions,
    d_color: str = "black",
    dm_color: str = "gray",
    d_label: str = "Data",
    m_label: str = "Model",
    label_fontsize: str = None,
    legend_fontsize: str = None,
    tick_fontsize: str = None,
    alpha: float = 0.5,
):
    for row_index in range(axs.shape[0]):
        cc_d.plot(
            axs[row_index, :],
            fill=True,
            color=d_color,
            alpha=1,
            zorder=0,
        )

    cc_dm.T.plot(
        axs,
        fill=True,
        color=dm_color,
        alpha=alpha,
        zorder=1,
    )

    for ax in axs.flatten():
        ax: Axes
        ax.set_xlim((0, 1))
        hideAxis(ax, hide_bottom=False)
        ax.patch.set_alpha(0)

    ax: Axes = axs[0, 0]
    fig = ax.get_figure()

    ax_bottom_left: Axes = axs[-1, 0]
    ax_bottom_left.xaxis.set_minor_locator(ticker.MultipleLocator(0.25))
    ax_bottom_left.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax_bottom_left.spines["bottom"].set_visible(True)
    ax_bottom_left.spines["left"].set_visible(True)
    fig.text(
        0.1,
        0.1,
        r"$x$",
        ha="left",
        va="top",
        fontsize=legend_fontsize,
        color=d_color,
        clip_on=False,
    )
    fig.text(
        0.1,
        0.1,
        r"$y$",
        ha="left",
        va="top",
        fontsize=legend_fontsize,
        color=dm_color,
        clip_on=False,
    )
    ax_bottom_left.set_ylabel(
        r"$\chi\{x,\_\}$",
        fontsize=label_fontsize,
    )

    ax_bottom_left.set_xticks((0, 1))
    ax_bottom_left.set_xticks(
        (0, 0.5, 1),
        minor=True,
        labels=("", r"$\hat{C}$", ""),
        fontsize=label_fontsize,
    )
    ax_bottom_left.tick_params(labelsize=tick_fontsize)

    ax_middle_left: Axes = axs[cc_dm.shape[0] // 2, 0]
    ax_middle_left.set_ylabel(m_label + r" $y$")
    ax_bottom_center: Axes = axs[-1, cc_dm.shape[1] // 2]
    ax_bottom_center.set_xlabel(d_label + r" $x$")


def plotComparisonTrace(
    fig: Figure,
    cc_matrix: CcMatrix,
    percentiles: ndarray = (0.5, 1.0),
    trace_colors: tuple[str] = ("red", "blue"),
    include_cc_dots: bool = True,
    include_cc_lines: bool = True,
    include_colorbar: bool = False,
    trace_linewidth: float = None,
    cc_dot_size: float = None,
    cc_linewidth: float = 2,
    max_matrix_length: int = 100,
    equal_matrix_aspect: bool = True,
):
    trace_count = len(percentiles)
    assert len(trace_colors) == trace_count

    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=(trace_count, 1),
        hspace=0.1,
        wspace=0,
    )
    gs_trace = gs[0].subgridspec(
        trace_count,
        1,
        hspace=0,
    )
    gs_cc = gs[-1].subgridspec(
        1,
        2,
        width_ratios=(3, 2),
        wspace=0,
    )

    ax_traces = [fig.add_subplot(gs_trace[i]) for i in range(trace_count)]
    ax_corr = fig.add_subplot(gs_cc[0])
    ax_distr = fig.add_subplot(gs_cc[1])

    cc_args = np.array(list(map(cc_matrix.argpercentile, percentiles)))
    cc_values = cc_matrix.matrix[*cc_args.T]

    data = cc_matrix.data
    model = cc_matrix.model
    trace_ylim = (
        min(data.x.min(), model.x.min()),
        max(data.x.max(), model.x.max()),
    )

    ##### Plot cross-correlation matrix #####
    cc_matrix.plotMatrix(
        ax_corr,
        cmap="Greys",
        vmin=0,
        vmax=1,
        max_length=max_matrix_length,
    )
    hideAxis(ax_corr)
    if equal_matrix_aspect:
        ax_corr.set_aspect("equal")

    if include_colorbar:
        mappable = cm.ScalarMappable(
            norm=colors.Normalize(0, 1),
            cmap="Greys",
        )
        fig.colorbar(mappable, ax=ax_corr)

    ax_corr.invert_yaxis()
    ax_corr.set_xlabel(r"Model $m$ $\rightarrow$")
    ax_corr.set_ylabel(r"$\leftarrow$ Data $n$")

    ##### Plot cross-correlation distribution #####
    cc_distr = cc_matrix.toDistribution(bins=np.linspace(0, 1, 51))
    cc_distr.plotStairsPdf(
        ax_distr,
        fill=True,
        color="black",
    )
    hideAxis(ax_distr)
    ax_distr.spines["bottom"].set_visible(True)

    ax_distr.set_xlim((0, 1))
    ax_distr.set_xticks((0, 0.5, 1), labels=("0", r"$\hat{C}_{nm}$", "1"))
    ax_distr.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax_distr.set_ylabel(
        r"$\chi\{x,y\}$",
        ha="center",
        va="top",
    )
    ax_distr.set_facecolor("none")

    ##### Plot model traces #####
    data_kwargs = {
        "color": "black",
        "linewidth": trace_linewidth,
    }
    model_kwargs = {
        "color": None,
        "linewidth": trace_linewidth,
    }
    for trace_index in range(trace_count):
        ax_trace = ax_traces[trace_index]
        color = trace_colors[trace_index]
        cc_arg = cc_args[trace_index]
        percentile = percentiles[trace_index]

        model_kwargs["color"] = color

        cc_matrix.plotSection(
            ax_trace,
            *cc_arg,
            data_kwargs=data_kwargs,
            model_kwargs=model_kwargs,
        )
        hideAxis(ax_trace)

        plabel = f"{100*percentile:.0f}" + r"$^{\text{th}}$"
        ax_trace.annotate(
            plabel,
            xy=(0.005, 0.5),
            xycoords="axes fraction",
            color=color,
        )
        ax_trace.set_ylim(trace_ylim)
        ax_trace.set_facecolor("none")

        if include_cc_dots:
            ax_corr.scatter(
                *cc_arg,
                color=color,
                s=cc_dot_size,
                marker="o",
                clip_on=False,
            )

        if include_cc_lines:
            cc_value = cc_values[trace_index]
            ax_distr.axvline(
                cc_value,
                color=color,
                linestyle="dashed",
                linewidth=cc_linewidth,
                label=plabel,
            )

    ax_trace_top = ax_traces[0]
    tamp = 0.25 * cc_matrix.chunk_time[0]
    yamp = 0.5 * np.diff(trace_ylim)[0]
    plotScalebar(
        ax_trace_top,
        (0, 0.05),
        tamp,
        yamp,
        labels=(f"{1000*tamp:.0f}ms", f"{yamp:.0f}nm"),
        horizontal_padding=0.04,
        linewidth=2,
    )


def plotComparisonTraceColumn(
    fig: Figure,
    cc_matrix: CcMatrix,
    percentiles: ndarray = (0.5, 1.0),
    trace_colors: tuple[str] = ("red", "blue"),
    include_cc_dots: bool = True,
    include_cc_lines: bool = True,
    include_colorbar: bool = False,
    trace_linewidth: float = None,
    cc_dot_size: float = None,
    cc_linewidth: float = 2,
    max_matrix_length: int = 100,
    equal_matrix_aspect: bool = True,
):
    trace_count = len(percentiles)
    assert len(trace_colors) == trace_count

    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=(3, 2),
        hspace=0,
        wspace=0.08,
    )
    gs_left = gs[0].subgridspec(trace_count, 1, hspace=0)
    gs_right = gs[1].subgridspec(2, 1, hspace=0)

    ax_traces = [fig.add_subplot(gs_left[i]) for i in range(trace_count)]
    ax_corr = fig.add_subplot(gs_right[0])
    ax_distr = fig.add_subplot(gs_right[1])

    cc_args = np.array(list(map(cc_matrix.argpercentile, percentiles)))
    cc_values = cc_matrix.matrix[*cc_args.T]

    data = cc_matrix.data
    model = cc_matrix.model
    trace_ylim = (
        min(data.x.min(), model.x.min()),
        max(data.x.max(), model.x.max()),
    )

    ##### Plot cross-correlation matrix #####
    cc_matrix.plotMatrix(
        ax_corr,
        cmap="Greys",
        vmin=0,
        vmax=1,
        max_length=max_matrix_length,
    )
    hideAxis(ax_corr)
    if equal_matrix_aspect:
        ax_corr.set_aspect("equal")

    if include_colorbar:
        mappable = cm.ScalarMappable(
            norm=colors.Normalize(0, 1),
            cmap="Greys",
        )
        fig.colorbar(mappable, ax=ax_corr)

    ax_corr.invert_yaxis()
    ax_corr.set_xlabel(r"Model $m$ $\rightarrow$")
    ax_corr.set_ylabel(r"$\leftarrow$ Data $n$")

    ##### Plot cross-correlation distribution #####
    cc_distr = cc_matrix.toDistribution(bins=np.linspace(0, 1, 51))
    cc_distr.plotStairsPdf(
        ax_distr,
        fill=True,
        color="black",
    )
    hideAxis(ax_distr)
    ax_distr.spines["bottom"].set_visible(True)

    ax_distr.set_xlim((0, 1))
    ax_distr.set_xticks((0, 0.5, 1), labels=("0", "", "1"))
    ax_distr.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax_distr.set_xlabel(r"$\hat{C}_{nm}$")
    ax_distr.set_ylabel(r"$\chi\{x,y\}$")
    ax_distr.set_facecolor("none")

    ##### Plot model traces #####
    data_kwargs = {
        "color": "black",
        "linewidth": trace_linewidth,
    }
    model_kwargs = {
        "color": None,
        "linewidth": trace_linewidth,
    }
    for trace_index in range(trace_count):
        ax_trace = ax_traces[trace_index]
        color = trace_colors[trace_index]
        cc_arg = cc_args[trace_index]
        percentile = percentiles[trace_index]

        model_kwargs["color"] = color

        cc_matrix.plotSection(
            ax_trace,
            *cc_arg,
            data_kwargs=data_kwargs,
            model_kwargs=model_kwargs,
        )
        hideAxis(ax_trace)

        plabel = f"{100*percentile:.0f}" + r"$^{\text{th}}$"
        ax_trace.annotate(
            plabel,
            xy=(0.005, 0.5),
            xycoords="axes fraction",
            color=color,
        )
        ax_trace.set_ylim(trace_ylim)
        ax_trace.set_facecolor("none")

        if include_cc_dots:
            ax_corr.scatter(
                *cc_arg,
                color=color,
                s=cc_dot_size,
                marker="o",
                clip_on=False,
            )

        if include_cc_lines:
            cc_value = cc_values[trace_index]
            ax_distr.axvline(
                cc_value,
                color=color,
                linestyle="dashed",
                linewidth=cc_linewidth,
                label=plabel,
            )

    ax_trace_top = ax_traces[0]
    tamp = 0.25 * cc_matrix.chunk_time[0]
    yamp = 0.5 * np.diff(trace_ylim)[0]
    plotScalebar(
        ax_trace_top,
        (0, 0.05),
        tamp,
        yamp,
        labels=(f"{1000*tamp:.0f}ms", f"{yamp:.0f}nm"),
        horizontal_padding=0.04,
        linewidth=2,
    )


def plotDivergence(
    axs: list[Axes],
    cc_dd: Distributions,
    cc_dm: Distributions,
    cc_mm: Distributions = None,
    norm=None,
    cmap: str = "Greys",
    data_label: str = "Data",
    model_label: str = "Model",
    cbar_label: str = "Divergence",
):
    if norm is None:
        norm = colors.Normalize(vmin=0, vmax=1)
    include_mm = isinstance(cc_mm, Distributions)

    div_dd = DivergenceMatrix.fromDistributions(cc_dd, cc_dd)
    div_dm = DivergenceMatrix.fromDistributions(cc_dd, cc_dm)

    fig = axs[0].get_figure()
    axs: list[Axes]
    ax_dd = axs[0]
    ax_dm = axs[1]
    ax_cbar = axs[-1]

    div_dd.plotMatrix(
        ax_dd,
        norm=norm,
        cmap=cmap,
    )
    ax_dd.set_xlabel(data_label)
    ax_dd.set_ylabel(data_label)

    div_dm.plotMatrix(
        ax_dm,
        norm=norm,
        cmap=cmap,
    )
    ax_dm.set_xlabel(data_label)
    ax_dm.set_ylabel(model_label)

    if include_mm:
        div_mm = DivergenceMatrix.fromDistributions(cc_mm, cc_mm)
        ax_mm = axs[2]
        div_mm.plotMatrix(
            ax_mm,
            norm=norm,
            cmap=cmap,
        )
        ax_mm.set_xlabel(model_label)
        ax_mm.set_ylabel(model_label)

    mappable = cm.ScalarMappable(
        norm=norm,
        cmap=cmap,
    )
    cbar = fig.colorbar(
        mappable,
        cax=ax_cbar,
        location="right",
    )
    cbar.set_ticks(
        (0, 0.5, 1),
        labels=("0.0", "0.5", "1.0"),
    )
    cbar.set_ticks(np.linspace(0, 1, 11), minor=True)
    cbar.set_label(
        cbar_label,
        rotation=270,
        labelpad=12,
    )


class ComparisonTracePlotter:
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
    def __pageFromFigure(cls, fig: Figure):
        if isinstance(fig, Iterable):
            figs = np.array(fig).flatten()
            return list(map(cls.__pageFromFigure, figs))

        if isinstance(fig, PageObject):
            return fig
        elif isinstance(fig, Figure):
            pdf_buffer = io.BytesIO()
            fig.savefig(pdf_buffer, format="pdf")
            pdf_buffer.seek(0)
            pdf_reader = PdfReader(pdf_buffer)
            page = pdf_reader.pages[0]
            return page

        assert TypeError

    def __init__(self, matrices: CcMatrices):
        if isinstance(matrices, CcMatrix):
            matrices = np.reshape(CcMatrices(matrices), (1, 1))

        self.matrices = matrices

    def exportFigures(
        self,
        fig: Figure,
        filepath: str,
        **kwargs,
    ):
        matrices = self.matrices
        progress_bar = tqdm(total=matrices.size)
        data_count, model_count = matrices.shape
        index_pairs = itertools.product(
            range(data_count),
            range(model_count),
        )
        pages = np.empty(matrices.shape, dtype=PageObject)

        for data_index, model_index in tqdm(index_pairs):
            fig.clear()
            cc_matrix: CcMatrix = self.matrices[data_index, model_index]

            plotComparisonTrace(
                fig,
                cc_matrix,
                **kwargs,
            )
            fig.suptitle(f"(Data {data_index:d}, Model {model_index:d})")

            page = self.__pageFromFigure(fig)
            pages[data_index, model_index] = page
            progress_bar.update(1)

        self.__exportPages(pages, filepath)


class CcMatrixGeneratorContainer:
    @classmethod
    def __generateGroup(
        cls,
        data_name: Union[str, int],
        model_name: Union[str, int],
        file: h5py.File,
    ):
        data_name = str(data_name)
        model_name = str(model_name)
        group_name = f"{data_name}{model_name}"
        group = file.create_group(group_name)
        return group

    @classmethod
    def __generateIndexPairs(
        cls,
        data_count: int,
        model_count: int,
        mode: str,
    ):
        if mode == "pairwise":
            index_pairs = list(itertools.product(range(data_count), range(model_count)))
        elif mode == "elementwise":
            assert data_count == model_count
            index_pairs = [(data_index, data_index) for data_index in range(data_count)]
        else:
            raise ValueError(f"mode must be 'pairwise' or 'elementwise'")

        return index_pairs

    @classmethod
    def __isSymmetric(
        cls,
        data_index: int,
        model_index: int,
        chunk_overlap: float,
        trace_types: list[str],
    ):
        is_same_type = len(set(trace_types)) == 1
        is_same_index = data_index == model_index
        same_chunk_time = isinstance(chunk_overlap, Number)
        is_symmetric = is_same_type and is_same_index and same_chunk_time
        return is_symmetric

    def __init__(
        self,
        datas: list[Trace],
        models: list[Trace] = None,
        mode: str = "pairwise",  # "pairwise" or "elementwise"
        trace_types: tuple[str, str] = ("data", "model"),
    ):
        if models is None:
            models = datas
            trace_types = ("data", "data")

        self.__datas = datas
        self.__models = models
        self.__mode = mode
        self.__trace_types = trace_types

    @property
    def data_count(self):
        return len(self.__datas)

    @property
    def model_count(self):
        return len(self.__models)

    def generateIndexPairs(self):
        mode = self.__mode
        data_count = self.data_count
        model_count = self.model_count
        index_pairs = self.__generateIndexPairs(
            data_count,
            model_count,
            mode=mode,
        )
        return index_pairs

    def correlatePairs(
        self,
        hdf5_filepath: str,
        processes: int = 1,
        **kwargs,
    ):
        if processes == 1:
            results = self.__correlatePairs_single(**kwargs)
        elif processes >= 2:
            results = self.__correlationPairs_parallel(
                processes=processes,
                **kwargs,
            )

        with h5py.File(hdf5_filepath, "w") as file:
            self.__writeResultsToHdf5File(file, results)

    def __correlatePairs_single(self, **kwargs):
        index_pairs = self.generateIndexPairs()
        results = []
        progress_bar = tqdm(total=len(index_pairs))

        for data_index, model_index in index_pairs:
            result = self._correlatePair(
                data_index,
                model_index,
                **kwargs,
            )
            results.append(result)
            progress_bar.update(1)

        return results

    def __correlationPairs_parallel(
        self,
        processes: int = 1,
        **kwargs,
    ):
        index_pairs = self.generateIndexPairs()
        results = []
        progress_bar = tqdm(total=len(index_pairs))

        def correlatePairFinished(result: tuple[CcMatrix, int, int]):
            results.append(result)
            progress_bar.update(1)

        with Pool(processes=processes) as pool:
            for data_index, model_index in index_pairs:
                args = (data_index, model_index)
                pool.apply_async(
                    self._correlatePair,
                    args=args,
                    kwds=kwargs,
                    callback=correlatePairFinished,
                )
            pool.close()
            pool.join()

        return results

    def _correlatePair(
        self,
        data_index: int,
        model_index: int,
        chunk_cycle_count: float = 4.0,
        chunk_overlap: float = 0.0,
        rescale: bool = True,
    ):
        trace_types = self.__trace_types
        data, model = self.__preparePair(
            data_index,
            model_index,
            rescale=rescale,
            trace_types=trace_types,
        )

        chunk_time = self.__calculateChunkTime(
            data_index,
            chunk_cycle_count=chunk_cycle_count,
        )
        is_symmetric = self.__isSymmetric(
            data_index,
            model_index,
            chunk_overlap=chunk_overlap,
            trace_types=trace_types,
        )

        correlationer = CcMatrix.fromPosition(
            data,
            model,
            chunk_time=chunk_time,
            chunk_overlap=chunk_overlap,
            is_symmetric=is_symmetric,
            show_progress=False,
        )

        return (
            correlationer.toTuple(),
            data.toTuple(),
            model.toTuple(),
            data_index,
            model_index,
        )

    def __calculateChunkTime(
        self,
        data_index: int,
        chunk_cycle_count: float = 1.0,
    ):
        data = self.__datas[data_index]
        data_psd = data.toPsd(nsegments=int(np.cbrt(len(data))))
        chunk_time = chunk_cycle_count * data_psd.peak_period
        return chunk_time * np.ones(2)

    def __preparePair(
        self,
        data_index: int,
        model_index: int,
        rescale: bool = True,
        downsample: bool = True,
        trace_types: tuple = None,
    ):
        data = self.__datas[data_index]
        model = self.__models[model_index]

        is_same_type = trace_types is not None and len(set(trace_types)) == 1
        is_same_index = data_index == model_index
        is_same_trace = is_same_type and is_same_index
        if is_same_trace:
            return data, model

        if rescale:
            fitter = TraceFitter(data, weights=[0.5, 0.1, 0.0, 0.4])
            fit = fitter.fit(model)
            model = model.rescale(*fit.x, t_shift=model.t[0])

        if downsample:
            data_sr = data.sample_rate
            model_sr = model.sample_rate
            if model_sr > data_sr:
                model = model.downsample(data)
            elif data_sr > model_sr:
                data = data.downsample(model)

        return data, model

    def __writeResultsToHdf5File(self, file: h5py.File, results: list):
        file.attrs["mode"] = self.__mode
        file.attrs["data_count"] = self.data_count
        file.attrs["model_count"] = self.model_count

        for result in results:
            self.__writeResultToNewHdf5Group(file, result)

    def __writeResultToNewHdf5Group(self, file: h5py.File, result: tuple):
        trace_types = self.__trace_types
        correlationer = CcMatrix(*result[0])
        data = Trace(*result[1])
        model = Trace(*result[2])
        data_index: int = result[3]
        model_index: int = result[4]

        data_name = f"{data_index:d}"
        model_name = f"{model_index:d}"
        group = self.__generateGroup(
            data_name,
            model_name,
            file=file,
        )

        group.attrs["data_name"] = (data_name, trace_types[0])
        group.attrs["model_name"] = (model_name, trace_types[1])

        data.toHdf5(group, name="data")
        model.toHdf5(group, name="model")
        correlationer.toHdf5(group)


class CcMatrixGenerator:
    @classmethod
    def __calculateMaxCcfromTrace(
        cls,
        data: Trace,
        model: Trace,
        method: str = "fft",
        mode: str = "full",
        normalize: bool = True,
        downsample: bool = True,
    ):
        if downsample:
            if data.sample_rate > model.sample_rate:
                data = data.downsample(model)
            elif model.sample_rate > data.sample_rate:
                model = model.downsample(data)

        if normalize:
            data = data.normalize()
            model = model.normalize()

        cross_corr = signal.correlate(
            data.x,
            model.x,
            method=method,
            mode=mode,
        )
        t_lags = signal.correlation_lags(
            data.x.size,
            model.x.size,
            mode=mode,
        )

        ind = np.argmax(cross_corr)
        max_corr = cross_corr[ind]
        t_lag = t_lags[ind] * data.dt
        return max_corr, t_lag

    @classmethod
    def __generateChunks(cls, chunker: TraceChunker):
        chunk_count = chunker.count
        chunks = [
            copy.deepcopy(chunker.getChunk(index)).normalize()
            for index in range(chunk_count)
        ]
        return chunks

    def __init__(self, data: Trace, model: Trace):
        self.data = data
        self.model = model
        self.matrix: ndarray = None
        self.lags: ndarray = None

    def calculateMatrix(
        self,
        chunk_overlap: tuple[float, float] = (0.0, 0.0),
        chunk_time: tuple[float, float] = (1.0, 1.0),
        is_symmetric: bool = False,
        show_progress: bool = True,
    ):
        assert np.all(0 <= chunk_overlap) and np.all(chunk_overlap <= 1)
        data = self.data
        model = self.model

        data_chunker = TraceChunker(
            data,
            chunk_time=chunk_time[0],
            chunk_overlap=chunk_overlap[0],
        )
        model_chunker = TraceChunker(
            model,
            chunk_time=chunk_time[1],
            chunk_overlap=chunk_overlap[1],
        )
        data_chunk_count = data_chunker.count
        model_chunk_count = model_chunker.count
        chunk_count = (data_chunk_count, model_chunk_count)

        data_chunks = self.__generateChunks(data_chunker)
        model_chunks = self.__generateChunks(model_chunker)

        self.matrix = np.zeros(chunk_count, dtype=np.float64)
        self.lags = np.zeros(chunk_count, dtype=np.float64)

        if show_progress:
            progress_bar = tqdm(
                total=np.prod(chunk_count),
                desc="Cross_Correlation",
                leave=False,
            )

        for data_index in range(data_chunk_count):
            data_chunk = data_chunks[data_index]
            for model_index in range(model_chunk_count):
                if show_progress:
                    progress_bar.update(1)
                if self.matrix[data_index, model_index] != 0.0:
                    continue
                if is_symmetric and data_index == model_index:
                    self.__setSymmetricDiagonal(data_index, model_index)
                    continue

                model_chunk = model_chunks[model_index]
                max_corr, t_lag = self.__calculateMaxCcfromTrace(
                    data_chunk,
                    model_chunk,
                    normalize=False,
                    downsample=False,
                )
                self.__setElement(
                    data_index,
                    model_index,
                    correlation=max_corr,
                    lag=t_lag,
                    is_symmetric=is_symmetric,
                )

    def __setElement(
        self,
        data_index: int,
        model_index: int,
        correlation: float,
        lag: float,
        is_symmetric: bool,
    ):
        self.matrix[data_index, model_index] = correlation
        self.lags[data_index, model_index] = lag
        if is_symmetric:
            self.matrix[model_index, data_index] = correlation
            self.lags[model_index, data_index] = -lag

    def __setSymmetricDiagonal(
        self,
        data_index: int,
        model_index: int,
    ):
        self.matrix[data_index, model_index] = 1.0
        self.matrix[model_index, data_index] = 1.0
        self.lags[data_index, model_index] = 0.0
        self.lags[model_index, data_index] = 0.0


class CcMatrix:
    @classmethod
    def fromHdf5(cls, group: h5py.Group):
        data = cls.__importData(group)
        model = cls.__importModel(group)

        chunk_overlap = cls.__importChunkOverlap(group)
        chunk_time = cls.__importChunkTime(group)
        is_symmetric = cls.__importIsSymmetric(group)
        lags = cls.__importLag(group)
        matrix = cls.__importMatrix(group)

        return CcMatrix(
            data,
            model,
            matrix,
            lags,
            chunk_overlap=chunk_overlap,
            chunk_time=chunk_time,
            is_symmetric=is_symmetric,
        )

    @classmethod
    def fromPosition(
        cls,
        data: Trace,
        model: Trace,
        chunk_time: Union[Number, tuple[Number, Number]],
        chunk_overlap: Union[Number, tuple[Number, Number]] = 0,
        is_symmetric: bool = False,
        show_progress: bool = True,
    ):
        if isinstance(chunk_time, Number):
            chunk_time = chunk_time * np.ones(2)
        if isinstance(chunk_overlap, Number):
            chunk_overlap = chunk_overlap * np.ones(2)

        matrix_generator = CcMatrixGenerator(data, model)
        matrix_generator.calculateMatrix(
            chunk_time=chunk_time,
            chunk_overlap=chunk_overlap,
            is_symmetric=is_symmetric,
            show_progress=show_progress,
        )
        cc_matrix = matrix_generator.matrix
        t_lags = matrix_generator.lags

        return CcMatrix(
            data,
            model,
            cc_matrix,
            t_lags,
            chunk_time=chunk_time,
            chunk_overlap=chunk_overlap,
            is_symmetric=is_symmetric,
        )

    @classmethod
    def __importChunkOverlap(cls, group: h5py.Group):
        chunk_overlap: tuple[float, float] = group.attrs["chunk_overlap"]
        return chunk_overlap

    @classmethod
    def __importChunkTime(cls, group: h5py.Group):
        chunk_time: tuple[float, float] = group.attrs["chunk_time"]
        return chunk_time

    @classmethod
    def __importData(cls, group: h5py.Group):
        ds = group["data"]
        return Trace.fromHdf5(ds)

    @classmethod
    def __importIsSymmetric(cls, group: h5py.Group):
        is_symmetric: bool = group.attrs["is_symmetric"]
        return is_symmetric

    @classmethod
    def __importLag(cls, group: h5py.Group):
        t_lags: ndarray = group["time_lag"][:, :]
        return t_lags

    @classmethod
    def __importMatrix(cls, group: h5py.Group):
        cc_matrix: ndarray = group["cross_correlation"][:, :]
        return cc_matrix

    @classmethod
    def __importModel(cls, group: h5py.Group):
        ds = group["model"]
        return Trace.fromHdf5(ds)

    def __init__(
        self,
        data: Trace,
        model: Trace,
        cc_matrix: ndarray,
        t_lags: ndarray,
        chunk_time: tuple[float, float] = (1.0, 1.0),
        chunk_overlap: tuple[float, float] = (0.0, 0.0),
        is_symmetric: bool = False,
    ):
        self.data_chunker = TraceChunker(
            data,
            chunk_time=chunk_time[0],
            chunk_overlap=chunk_overlap[0],
        )
        self.model_chunker = TraceChunker(
            model,
            chunk_time=chunk_time[1],
            chunk_overlap=chunk_overlap[1],
        )

        self.lags = t_lags
        self.matrix = cc_matrix

        self.chunk_overlap = chunk_overlap
        self.chunk_time = chunk_time
        self.is_symmetric = is_symmetric

        self.shape = cc_matrix.shape

    def __mul__(self, other: CcMatrix):
        return self.jsDivergence(other)

    def __getitem__(self, inds: slice):
        return CcMatrix(
            self.data,
            self.model,
            cc_matrix=self.matrix[inds],
            t_lags=self.lags[inds],
            chunk_overlap=self.chunk_overlap,
            chunk_time=self.chunk_time,
        )

    @property
    def argmax(self):
        matrix = self.matrix
        return np.unravel_index(np.argmax(matrix), matrix.shape)

    @property
    def data(self):
        return self.data_chunker.trace

    @property
    def model(self):
        return self.model_chunker.trace

    def argpercentile(self, q: float):
        assert 0 <= q <= 1
        if q >= 1:
            return self.argmax

        matrix = self.matrix
        arg = np.argsort(matrix.flatten())[np.int64(matrix.size * q)]
        return np.unravel_index(arg, matrix.shape)

    def getDataChunkTimeStart(self, index: int):
        return self.data_chunker.getChunkTimeStart(index)

    def getModelChunkTimeStart(self, index: int):
        return self.model_chunker.getChunkTimeStart(index)

    def toDistribution(self, **kwargs):
        return Distribution.fromSamples(
            self.matrix,
            density=True,
            **kwargs,
        )

    def toHdf5(self, dataset: h5py.Group):
        dataset.create_dataset(
            "cross_correlation",
            dtype=np.float64,
            data=self.matrix,
            compression="gzip",
            compression_opts=9,
        )
        dataset.create_dataset(
            "time_lag",
            dtype=np.float64,
            data=self.lags,
            compression="gzip",
            compression_opts=9,
        )

        dataset.attrs["chunk_time"] = self.chunk_time
        dataset.attrs["chunk_overlap"] = self.chunk_overlap
        dataset.attrs["is_symmetric"] = self.is_symmetric

    def toTuple(self):
        return (
            self.data,
            self.model,
            self.matrix,
            self.lags,
            self.chunk_time,
            self.chunk_overlap,
            self.is_symmetric,
        )

    def divergence(
        self,
        other: CcMatrix,
        func: Callable[[CcMatrix, CcMatrix], float],
        **kwargs,
    ):
        return func(
            self.toDistribution(**kwargs),
            other.toDistribution(**kwargs),
        )

    def hellingerDistance(self, other: CcMatrix, **kwargs):
        def cost(x: Distribution, y: Distribution):
            return x.hellingerDistance(y)

        return self.divergence(other, cost, **kwargs)

    def jsDivergence(self, other: CcMatrix, **kwargs):
        def cost(x: Distribution, y: Distribution):
            return x.jsDivergence(y)

        return self.divergence(other, cost, **kwargs)

    def tvDistance(self, other: CcMatrix, **kwargs):
        def cost(x: Distribution, y: Distribution):
            return x.tvDistance(y)

        return self.divergence(other, cost, **kwargs)

    def plotMatrix(
        self,
        ax: Axes,
        max_length: int = 100,
        **kwargs,
    ):
        assert isinstance(max_length, Number)

        cc_matrix = self.matrix.T
        cc_shape = cc_matrix.shape
        cc_aspect = cc_shape[1] / cc_shape[0]

        x_size = max_length
        y_size = max_length
        if cc_shape[0] < max_length and cc_shape[1] < max_length:
            xy_size = min(max_length, *cc_shape)
            x_size = xy_size
            y_size = xy_size

        if cc_aspect >= 1:
            x_size = int(max_length / cc_aspect)
        elif cc_aspect < 1:
            y_size = int(max_length / cc_aspect)

        x_inds = np.linspace(
            0,
            cc_shape[0],
            min(x_size, cc_shape[0]),
            endpoint=False,
            dtype=np.int64,
        )
        y_inds = np.linspace(
            0,
            cc_shape[1],
            min(y_size, cc_shape[1]),
            endpoint=False,
            dtype=np.int64,
        )
        x, y = np.meshgrid(x_inds, y_inds, indexing="ij")
        cc_matrix = cc_matrix[x, y]

        return ax.pcolormesh(
            y,
            x,
            cc_matrix,
            **kwargs,
        )

    def plotSection(
        self,
        ax: Axes,
        data_index: int,
        model_index: int,
        data_kwargs: dict = {},
        model_kwargs: dict = {},
    ):
        t_lag = self.lags[data_index, model_index]
        data = self.data_chunker.getChunk(data_index)
        model = self.model_chunker.getChunk(model_index)

        data = data.rescale(t_shift=data.t[0] + t_lag)
        model = model.rescale(t_shift=model.t[0])
        t0 = min(data.t[0], model.t[0])
        data = data.rescale(t_shift=t0)
        model = model.rescale(t_shift=t0)

        data.plotTrace(ax, **data_kwargs)
        model.plotTrace(ax, **model_kwargs)


class CcMatrices(ndarray):
    @classmethod
    def fromHdf5(
        cls,
        file: h5py.File,
        **kwargs,
    ):
        if isinstance(file, str):
            return cls.__fromHdf5_filepath(file, **kwargs)
        elif isinstance(file, h5py.File):
            return cls.__fromHdf5_file(file, **kwargs)
        else:
            raise TypeError("file must be of type str of h5py.File")

    @classmethod
    def __fromHdf5_filepath(
        cls,
        filepath: str,
        **kwargs,
    ):
        with h5py.File(filepath, "r") as file:
            cc_matrices = cls.__fromHdf5_file(file, **kwargs)
        return cc_matrices

    @classmethod
    def __fromHdf5_file(
        cls,
        file: h5py.File,
        is_diag: bool = False,
    ):
        data_count = file.attrs["data_count"]
        model_count = file.attrs["model_count"]

        cc_parsers = []
        group_names = np.array(list(file.keys()))
        group_names = np.reshape(group_names, (data_count, model_count))
        if is_diag:
            group_names = np.diag(group_names)

        for group_name in group_names.flatten():
            group = file[group_name]
            parser = CcMatrix.fromHdf5(group)
            cc_parsers.append(parser)

        cc_parsers = np.reshape(cc_parsers, group_names.shape)
        return CcMatrices(cc_parsers)

    def __new__(cls, matrices: ndarray):
        return np.asarray(matrices).view(cls)

    def toDistributions(self, **kwargs):
        self_flat: list[CcMatrix] = self.flatten()
        distributions = [parser.toDistribution(**kwargs) for parser in self_flat]
        distributions = np.reshape(distributions, self.shape)
        return Distributions(distributions)


class DistributionQuadrant:
    @classmethod
    def fromHdf5(cls, filepath: str):
        with h5py.File(filepath, "r") as file:
            cc_dd = cls.__importDataData(file)
            cc_dm = cls.__importDataModel(file)
            cc_md = cls.__importModelData(file)
            cc_mm = cls.__importModelModel(file)

        return DistributionQuadrant(
            cc_dd=cc_dd,
            cc_dm=cc_dm,
            cc_md=cc_md,
            cc_mm=cc_mm,
        )

    @classmethod
    def __importDistributions(cls, group: h5py.Group):
        if len(group) == 0:
            return None
        return Distributions.fromHdf5(group)

    @classmethod
    def __importDataData(cls, file: h5py.File):
        group = file["data_data"]
        return cls.__importDistributions(group)

    @classmethod
    def __importDataModel(cls, file: h5py.File):
        group = file["data_model"]
        return cls.__importDistributions(group)

    @classmethod
    def __importModelData(cls, file: h5py.File):
        group = file["model_data"]
        return cls.__importDistributions(group)

    @classmethod
    def __importModelModel(cls, file: h5py.File):
        group = file["model_model"]
        return cls.__importDistributions(group)

    @classmethod
    def fromMatrices(
        cls,
        cc_dd_matrix: CcMatrices = None,
        cc_dm_matrix: CcMatrices = None,
        cc_md_matrix: CcMatrices = None,
        cc_mm_matrix: CcMatrices = None,
        **kwargs,
    ):
        cc_dd = cls.__fromMatrix(cc_dd_matrix, **kwargs)
        cc_dm = cls.__fromMatrix(cc_dm_matrix, **kwargs)
        cc_md = cls.__fromMatrix(cc_md_matrix, **kwargs)
        cc_mm = cls.__fromMatrix(cc_mm_matrix, **kwargs)
        return DistributionQuadrant(
            cc_dd=cc_dd,
            cc_dm=cc_dm,
            cc_md=cc_md,
            cc_mm=cc_mm,
        )

    @classmethod
    def __fromMatrix(cls, matrix: CcMatrices, **kwargs):
        distributions = None
        if isinstance(matrix, CcMatrices):
            distributions = matrix.toDistributions(**kwargs)
        return distributions

    @classmethod
    def __divergence(
        cls,
        cc_reference: Distributions,
        cc_comparison: Distributions,
    ):
        return DivergenceMatrix.fromDistributions(
            cc_reference,
            cc_comparison,
        )

    def __init__(
        self,
        cc_dd: Distributions = None,
        cc_dm: Distributions = None,
        cc_md: Distributions = None,
        cc_mm: Distributions = None,
    ):
        ccs = [cc_dd, cc_dm, cc_md, cc_mm]
        shapes = [cc.shape for cc in ccs if isinstance(cc, Distributions)]
        assert len(set(shapes)) == 1

        self.data_data = cc_dd
        self.data_model = cc_dm
        self.model_data = cc_md
        self.model_model = cc_mm

        self.shape = shapes[0]

    def toHdf5(self, file: h5py.File):
        dd_group = file.create_group("data_data")
        dm_group = file.create_group("data_model")
        md_group = file.create_group("model_data")
        mm_group = file.create_group("model_model")

        cc_dd = self.data_data
        if isinstance(cc_dd, Distributions):
            cc_dd.toHdf5(dd_group)

        cc_dm = self.data_model
        if isinstance(cc_dm, Distributions):
            cc_dm.toHdf5(dm_group)

        cc_md = self.model_data
        if isinstance(cc_md, Distributions):
            cc_md.toHdf5(md_group)

        cc_mm = self.model_model
        if isinstance(cc_mm, Distributions):
            cc_mm.toHdf5(mm_group)

    def divergenceDataData(self):
        return self.__divergence(
            self.data_data,
            self.data_data,
        )

    def divergenceDataModel(self):
        return self.__divergence(
            self.data_data,
            self.data_model,
        )

    def divergenceModelData(self):
        return self.__divergence(
            self.model_model,
            self.model_data,
        )

    def divergenceModelModel(self):
        return self.__divergence(
            self.model_model,
            self.model_model,
        )


class DivergenceMatrix:
    @classmethod
    def fromDistributions(
        cls,
        cc_reference: Distributions,
        cc_comparison: Distributions,
    ):
        cc_ref_diag = np.diag(cc_reference)[:, np.newaxis]
        div_matrix = np.asarray(
            cc_comparison * cc_ref_diag,
            dtype=np.float64,
        )
        return DivergenceMatrix(div_matrix)

    @classmethod
    def fromMatrices(
        cls,
        cc_reference_matrix: CcMatrices,
        cc_comparison_matrix: CcMatrices,
    ):
        cc_container = DistributionQuadrant.fromMatrices(
            cc_dd_matrix=cc_reference_matrix,
            cc_dm_matrix=cc_comparison_matrix,
        )
        cc_reference = cc_container.data_data
        cc_comparison = cc_container.data_model
        return cls.fromDistributions(
            cc_reference=cc_reference,
            cc_comparison=cc_comparison,
        )

    def __init__(self, div_matrix: ndarray):
        self.matrix = div_matrix

    def __formatAxis(self, ax: Axes):
        ax.set_aspect("equal")
        ax.set_xticks(())
        ax.set_yticks(())

    def plotMatrix(self, ax: Axes, norm=None, **kwargs):
        if norm is None:
            norm = colors.Normalize(vmin=0, vmax=1)

        matrix = np.flip(self.matrix, axis=1).T
        p = ax.pcolormesh(
            matrix,
            norm=norm,
            **kwargs,
        )

        self.__formatAxis(ax)
        return p


class DivergenceRegressioner:
    def __generateDivergenceMatrices(
        cls,
        quadrants: list[DistributionQuadrant],
    ):
        quadrant_shape = quadrants[0].shape
        quadrant_count = len(quadrants)
        div_matrices = np.zeros((quadrant_count, *quadrant_shape))

        for index, quadrant in enumerate(quadrants):
            div_matrix = cls.__generateDivergenceMatrix_quadrant(quadrant)
            div_matrices[index, :, :] = div_matrix

        return div_matrices

    @classmethod
    def __generateDivergenceMatrix_quadrant(
        cls,
        quadrant: DistributionQuadrant,
    ):
        cc_dd = quadrant.data_data
        cc_dm = quadrant.data_model
        div_matrix = DivergenceMatrix.fromDistributions(cc_dd, cc_dm).matrix
        return div_matrix

    def __init__(self, quadrants: list[DistributionQuadrant]):
        shapes: list[tuple] = [quadrant.shape for quadrant in quadrants]
        assert len(set(shapes)) == 1
        shape = shapes[0]
        assert shape[0] == shape[1]

        self.matrices = self.__generateDivergenceMatrices(quadrants)
        self.quadrants = quadrants

        self.divergence_count = len(quadrants)
        self.quadrant_shape = shape
        self.quadrant_size = np.prod(shape)

    def flatten(self, x: ndarray):
        return np.reshape(x, (self.divergence_count, -1))

    def calculatePageTrend(self, **kwargs):
        matrices = self.flatten(self.matrices).T
        return stats.page_trend_test(
            matrices,
            **kwargs,
        )

    def calculateRsquare(self):
        matrices = self.flatten(self.matrices)
        return np.corrcoef(matrices) ** 2

    def generateTimeMatrix(self, time: ndarray):
        times = np.tile(time, (*self.quadrant_shape, 1)).T
        return times

    def pairplot(self, **kwargs):
        div_count = self.divergence_count
        fig, axs = plt.subplots(
            div_count,
            div_count,
            layout="constrained",
        )
        axs: list[Axes]
        matrices = self.flatten(self.matrices)
        inds = list(range(self.divergence_count))

        for i, j in itertools.product(inds, inds):
            ax: Axes = axs[i, j]
            ax.scatter(
                matrices[i, :],
                matrices[j, :],
                **kwargs,
            )

        self.__pairplot_formatAxes(axs)
        return fig, axs

    def __pairplot_formatAxes(self, axs: ndarray):
        for i in range(axs.shape[0]):
            for j in range(axs.shape[1]):
                ax: Axes = axs[i, j]

                hideAxis(
                    ax,
                    hide_bottom=False,
                    hide_left=False,
                )

                ax.set_xlim((0, 1))
                ax.set_ylim((0, 1))
                ax.set_aspect("equal")

                ax.set_title(f"({i:d}, {j:d})")

        ax_bottom_left: Axes = axs[-1, 0]
        ax_bottom_left.xaxis.set_minor_locator(ticker.MultipleLocator(0.25))
        ax_bottom_left.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax_bottom_left.spines["left"].set_visible(True)


if __name__ == "__main__":
    data_indices = list(range(9))
    # [135, 137, 157, 162, 192, 203, 211, 218, 229, 233, 235]
    data_names = [f"{i:d}" for i in data_indices]
    model_names = data_names

    np.set_printoptions(precision=3)
    plt.rcParams["text.usetex"] = True
    plt.rcParams["text.latex.preamble"] = (
        r"\usepackage{amsmath}" r"\usepackage{lmodern}"
    )
    plt.rcParams["font.family"] = "lmodern"

    # Generate matrix of Jensen-Shannon divergences
    if False:
        data_filepaths = [f"traces_ap/cell{name:s}.csv" for name in data_names]
        datas = list(map(Trace.fromCsv, data_filepaths))

        model_filepaths = [f"traces_ap/fit{name:s}-2000.csv" for name in model_names]
        models = list(map(Trace.fromCsv, model_filepaths))
        models = [model[:4000000] for model in models]

        cycle_counts = [4, 8, 12, 16, 20]
        trace_typess = [
            ("data", "data"),
            ("data", "model"),
            ("model", "model"),
            # ("model", "data"),
        ]

        total_count = len(cycle_counts) * len(trace_typess)
        pbar = tqdm(total=total_count)

        for trace_types in trace_typess:
            t1, t2 = trace_types
            for cycle_count in cycle_counts:
                hdf5_filepath = f"cc_ap/{t1[0]:s}{t2[0]:s}-{cycle_count:d}psd.hdf5"

                datas_iter = datas if t1 == "data" else models
                models_iter = datas if t2 == "data" else models
                if t1 == t2:
                    models_iter = None

                cc_generator = CcMatrixGeneratorContainer(
                    datas_iter,
                    models=models_iter,
                    mode="pairwise",
                    trace_types=trace_types,
                )
                cc_generator.correlatePairs(
                    hdf5_filepath,
                    processes=12,
                    chunk_cycle_count=cycle_count,
                    chunk_overlap=0.25,
                    rescale=True,
                )

                pbar.update(1)

    # Generate plot to...
    # 1) compare data and model traces
    # 2) show cross-correlation matrix
    # 3) show distribution of cross-correlations
    if True:
        # pylustrator.start()
        cc_matrices = CcMatrices.fromHdf5("cc_sac/dm-4psd.hdf5")

        percentiles = (0.5, 0.841345, 0.99)
        trace_colors = (
            Luminance.addUntilLuminance(0.25, "blue", "green"),
            Luminance.addUntilLuminance(0.5, "red", "green"),
            Luminance.addUntilLuminance(0.75, (0, 1, 0), "magenta"),
        )
        fig = plt.figure(figsize=(3.375, 3.375))

        plotComparisonTrace(
            fig,
            cc_matrices[0, 0],
            percentiles=percentiles,
            trace_colors=trace_colors,
            trace_linewidth=2,
            cc_linewidth=2,
            cc_dot_size=4,
            max_matrix_length=200,
        )
        fig.tight_layout()
        fig.subplots_adjust(
            bottom=0.1,
            top=1,
            left=0.04,
            right=0.98,
        )

        # % start: automatic generated code from pylustrator
        texts = [fig.axes[i].texts[0] for i in range(len(percentiles))]
        texts[0].set(position=(0.0264, 0.3156))
        texts[1].set(position=(0.2206, 0.3155))
        texts[2].set(position=(0.1023, 0.4437))
        # % end: automatic generated code from pylustrator
        plt.show()

        """
        plotter = ComparisonTracePlotter(cc_matrices)
        plotter.exportFigures(
            fig,
            "figs/cc_traces.pdf",
            percentiles=percentiles,
            trace_colors=trace_colors,
        )
        """

    # Generate and export grid of cross-correlation distributions
    if False:
        cell_type = "sac"
        cycle_count = 4
        bins = np.linspace(0, 1, 201)
        bin_count = len(bins) - 1

        for cycle_count in tqdm([4, 8, 12, 16, 20]):
            cc_dd = CcMatrices.fromHdf5(f"cc_{cell_type}/dd-{cycle_count:d}psd.hdf5")
            cc_dm = CcMatrices.fromHdf5(f"cc_{cell_type}/dm-{cycle_count:d}psd.hdf5")
            # cc_md = CcMatrices.fromHdf5("cc_sac/md-4psd.hdf5")
            cc_mm = CcMatrices.fromHdf5(f"cc_{cell_type}/mm-{cycle_count:d}psd.hdf5")

            bins1 = np.linspace(0, 1, 101)
            bins2 = np.linspace(0, 1, 201)
            bins3 = np.linspace(0, 1, 401)

            for bins in [bins1, bins2, bins3]:
                bin_count = len(bins) - 1

                cc_distr_container = DistributionQuadrant(
                    cc_dd=cc_dd.toDistributions(bins=bins),
                    cc_dm=cc_dm.toDistributions(bins=bins),
                    # cc_md=cc_md.toDistributions(bins=bins),
                    cc_mm=cc_mm.toDistributions(bins=bins),
                )

                filepath = f"dist-{cycle_count:d}T_{bin_count:d}bins.hdf5"
                with h5py.File(filepath, "w") as file:
                    cc_distr_container.toHdf5(file)

    # Generate grid of plots to compare cross-correlation distributions
    if False:
        # pylustrator.start()
        cc_distr_container = DistributionQuadrant.fromHdf5("cc_sac/dist-4T_50bins.hdf5")
        cc_dd = cc_distr_container.data_data
        cc_dm = cc_distr_container.data_model
        cc_d = np.diag(cc_dd)

        fig, axs = plt.subplots(
            *cc_dm.shape,
            figsize=(3.375, 3.375),
        )

        plotComparisonGrid(
            axs,
            cc_d,
            cc_dm,
            dm_color=Luminance.addUntilLuminance(0.833, "black", "white"),
            alpha=0.833,
            d_label="Measurement",
            m_label="Simulation",
            legend_fontsize="x-small",
            label_fontsize="x-small",
            tick_fontsize="x-small",
        )
        fig.tight_layout(
            pad=0.12,
            h_pad=0,
            w_pad=0,
        )

        # % start: automatic generated code from pylustrator
        texts = fig.texts
        texts[0].set(position=(0.1167, 0.1267))
        texts[1].set(position=(0.1026, 0.1536))
        # % end: automatic generated code from pylustrator

        plt.show()

    # Generate plot to show JSD matrices
    if False:
        cc_distr_container = DistributionQuadrant.fromHdf5("cc_sac/dist-4T_50bins.hdf5")
        cc_dd = cc_distr_container.data_data
        cc_dm = cc_distr_container.data_model
        cc_mm = cc_distr_container.model_model

        fig, axs = plt.subplots(
            1,
            3,
            figsize=(3.375, 1.4375),
            layout="constrained",
            width_ratios=(1, 1, 1 / 12),
        )
        plotDivergence(
            axs,
            cc_dd,
            cc_dm,
            cc_mm=cc_mm,
            data_label="Measurement",
            model_label="Simulation",
            cbar_label=r"$D_{JS}$",
        )
        plt.show()

    # Perform Page's trend test and linear regression
    if False:
        cycle_counts = np.array([4, 8, 12, 16, 20])

        filepaths = [
            f"cc_sac/dist-{cycle_count:d}T_400bins.hdf5" for cycle_count in cycle_counts
        ]
        cc_quadrants = list(map(DistributionQuadrant.fromHdf5, filepaths))
        regressioner = DivergenceRegressioner(cc_quadrants)

        div_matrices = regressioner.flatten(regressioner.matrices)
        time = regressioner.generateTimeMatrix(cycle_counts)
        time = regressioner.flatten(time)

        """
        fig, axs = regressioner.pairplot(
            color="red",
            marker=".",
            s=1,
        )
        plt.show()
        """

        def func(y: tuple[ndarray, ndarray], a: float, b: float):
            t, x = y
            return a * (1 + b * t) * x

        x0_dm = np.tile(div_matrices[0, :], (regressioner.divergence_count, 1))
        fit_parameters, fit_cov = optimize.curve_fit(
            func,
            (time.flatten(), x0_dm.flatten()),
            div_matrices.flatten(),
            maxfev=10000,
        )
        fit_err = np.sqrt(np.diag(fit_cov))
        print(fit_parameters, fit_err)

        r2 = regressioner.calculateRsquare()
        page_trend = regressioner.calculatePageTrend(method="exact").pvalue
        print(r2.mean(), r2.std())

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        f_mesh = func((time, x0_dm), *fit_parameters)
        ax.scatter(
            time,
            x0_dm,
            f_mesh,
            color="red",
            alpha=0.25,
            label="Fit",
        )
        ax.scatter(
            time,
            x0_dm,
            div_matrices,
            color="black",
            alpha=0.25,
            label="Data",
        )

        ax.legend()
        ax.set_xticks(cycle_counts)

        ax.set_xlabel("Cycle count")
        ax.set_ylabel("JSD for 4 cycles")
        ax.set_zlabel("JSD for x cycles")

        plt.show()
