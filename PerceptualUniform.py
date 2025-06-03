from functools import partial
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.typing import ColorType
import numpy as np
from numpy import ndarray
from scipy import optimize


def formatAsRgb(color: ColorType) -> ndarray:
    if isinstance(color, ndarray):
        return color[:3]
    return np.array(to_rgb(color))


class LuminanceBounds:
    def minimumColor(luminance: float):
        if luminance <= np.max(Luminance.weights):
            return np.zeros(3)

        red_min = LuminanceBounds.minimumChannel(luminance, 0)
        green_min = LuminanceBounds.minimumChannel(luminance, 1)
        blue_min = LuminanceBounds.minimumChannel(luminance, 2)
        color_min = np.array([red_min, green_min, blue_min])
        return color_min

    def minimumChannel(luminance: float, channel: str):
        red_weight, green_weight, blue_weight = Luminance.weights

        if channel == 0:  # red
            channel_min = (luminance - (green_weight + blue_weight)) / red_weight
        elif channel == 1:  # green
            channel_min = (luminance - (red_weight + blue_weight)) / green_weight
        elif channel == 2:  # blue
            channel_min = (luminance - (red_weight + green_weight)) / blue_weight

        return max(0, channel_min)

    def maximumColor(luminance: float):
        weights = Luminance.weights
        if luminance >= np.min(weights):
            return np.ones(3)

        color_min = luminance / weights
        return np.minimum(1, color_min)


class Luminance:
    weights = np.array([0.2126, 0.7152, 0.0722])  # weights from ITU BT.709
    tolerance = 0.004

    def byColor(color: ColorType):
        color = formatAsRgb(color)
        weights = Luminance.weights
        luminance: ndarray = np.sum(weights * color)
        return luminance

    def generateColor(target_luminance: float):
        if target_luminance == 1:
            return np.ones(3)
        elif target_luminance == 0:
            return np.zeros(3)

        def luminanceError(color_guess: ndarray):
            luminance_guess = Luminance.byColor(color_guess)
            error = np.abs(luminance_guess - target_luminance)
            return error

        low_color = LuminanceBounds.minimumColor(target_luminance)
        high_color = LuminanceBounds.maximumColor(target_luminance)

        initial_color = np.random.uniform(low_color, high_color, 3)
        bounds = list(zip(low_color, high_color))
        color_root = optimize.minimize(
            luminanceError, initial_color, bounds=bounds, method="SLSQP"
        )
        color = color_root.x
        return color

    def interpolateTwoColors(
        x_luminance: float,
        start_color: ColorType,
        end_color: ColorType,
    ):
        start_color = formatAsRgb(start_color)
        end_color = formatAsRgb(end_color)
        color_difference = end_color - start_color

        start_luminance = Luminance.byColor(start_color)
        end_luminance = Luminance.byColor(end_color)
        remaining_luminance = x_luminance * (end_luminance - start_luminance)
        new_luminance = start_luminance + remaining_luminance

        new_color = Luminance.addUntilLuminance(
            new_luminance, start_color, color_difference
        )
        return new_color

    def interpolateTwoColorsSmoothly(
        start_color: ColorType,
        end_color: ColorType,
        N: int = 256,
    ):
        color_interp = partial(
            Luminance.interpolateTwoColors,
            start_color=start_color,
            end_color=end_color,
        )

        x_luminance = np.arange(N) / (N - 1)
        colors = list(map(color_interp, x_luminance))
        return colors

    def addUntilLuminance(
        target_luminance: float,
        base_color: ColorType,
        add_color: ColorType,
    ):
        base_color = formatAsRgb(base_color)
        add_color = formatAsRgb(add_color)

        base_luminance = Luminance.byColor(base_color)
        add_luminance = Luminance.byColor(add_color)
        remaining_luminance: ndarray = target_luminance - base_luminance

        remaining_color: ndarray = remaining_luminance / add_luminance * add_color
        new_color: ndarray = base_color + remaining_color
        return new_color

    def multiplyUntilLuminance(target_luminance: float, base_color: ColorType):
        base_color = formatAsRgb(base_color)
        base_luminance = Luminance.byColor(base_color)
        new_color = target_luminance / base_luminance * base_color
        return new_color


class PerceptualColormap:
    def interpolateTwoColors(
        x_color: float,
        start_color: ColorType,
        end_color: ColorType,
    ):
        start_color = formatAsRgb(start_color)
        end_color = formatAsRgb(end_color)
        new_color = start_color + x_color * (end_color - start_color)
        return new_color

    def generateRandom(
        color_count: int = 3,
        min_luminance: float = 0,
        max_luminance: float = 1,
    ):
        luminances_to_match = np.linspace(min_luminance, max_luminance, color_count)
        colors = list(map(Luminance.generateColor, luminances_to_match))
        colormap = PerceptualColormap.from_list("", colors)
        return colormap

    def from_list(
        name: str,
        colors: list[ColorType],
        N: int = 256,
    ):
        color_count = len(colors)
        colors_per_interpolation = int(N / color_count) + 1
        interpolate_by_luminance = partial(
            Luminance.interpolateTwoColorsSmoothly,
            N=colors_per_interpolation,
        )

        initial_color = formatAsRgb(colors[0])
        colors_colormap = [initial_color]
        for index in range(color_count - 1):
            start_color = colors[index]
            end_color = colors[index + 1]
            new_colors = interpolate_by_luminance(start_color, end_color)
            colors_colormap.extend(new_colors[1:])

        colors_colormap = np.array(colors_colormap)
        colors_colormap = np.minimum(1, colors_colormap)
        colors_colormap = np.maximum(0, colors_colormap)

        colormap = LinearSegmentedColormap.from_list(name, colors_colormap, N=N)
        return colormap
