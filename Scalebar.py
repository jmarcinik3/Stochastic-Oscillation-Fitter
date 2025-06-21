from matplotlib import patches
from matplotlib.axes import Axes
from matplotlib.path import Path


def plotScalebar(
    ax: Axes,
    position: tuple[float, float],
    horizontal_scale: float,
    vertical_scale: float,
    labels: tuple[str, str],
    linewidth: float = 2,
    horizontal_padding: float = 0,
    vertical_padding: float = 0,
    **kwargs,
):
    time_label, position_label = labels
    time_relative, position_relative = position

    xmin, xmax = ax.get_xlim()
    xrange = xmax - xmin
    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin

    time_position = xmin + time_relative * xrange
    position_position = ymin + position_relative * yrange

    scalebar_2d = Scalebar2D(
        (time_position, position_position),
        horizontal_scale,
        vertical_scale,
        ax,
    )
    scalebar_2d.addHorizontalLabel(
        time_label,
        padding=horizontal_padding,
        **kwargs,
    )
    scalebar_2d.addVerticalLabel(
        position_label,
        padding=vertical_padding,
        **kwargs,
    )
    scalebar_2d.addPatch(
        clip_on=False,
        linewidth=linewidth,
    )


class Anchor:
    def __init__(
        self, xy: tuple[float, float], width: float, height: float, anchor: str
    ):
        self.x, self.y = xy
        self.width = width
        self.height = height
        self.anchor = anchor

    def isUpper(self):
        return "upper" in self.anchor

    def isLower(self):
        return "lower" in self.anchor

    def isLeft(self):
        return "left" in self.anchor

    def isRight(self):
        return "right" in self.anchor

    def getVerticies(self):
        x = self.x
        y = self.y
        width = self.width
        height = self.height
        anchor = self.anchor

        if anchor == "lower left":
            lower_left = (x, y)
            upper_left = (x, y + height)
            lower_right = (x + width, y)
            verticies = [upper_left, lower_left, lower_right]
        elif anchor == "upper left":
            upper_left = (x, y)
            lower_left = (x, y - height)
            upper_right = (x + width, y)
            verticies = [upper_right, upper_left, lower_left]
        elif anchor == "lower right":
            lower_right = (x, y)
            upper_right = (x, y + height)
            lower_left = (x - width, y)
            verticies = [lower_left, lower_right, upper_right]
        elif anchor == "upper right":
            upper_right = (x, y)
            lower_right = (x, y - height)
            upper_left = (x - width, y)
            verticies = [lower_right, upper_right, upper_left]

        return verticies


class Scalebar2D:
    def __init__(
        self,
        xy: tuple[float, float],
        width: float,
        height: float,
        axis: Axes,
        anchor: str = "lower left",
        **kwargs,
    ):
        self.axis = axis
        self.anchor = Anchor(xy, width, height, anchor)

        path = self._generatePath(self.anchor, **kwargs)
        self.path = path
        self._setPoints(path)

    def _setPoints(self, path: Path):
        lower_left, upper_right = path.get_extents().get_points()
        x_left, y_lower = lower_left
        x_right, y_upper = upper_right

        self.left = x_left
        self.right = x_right
        self.lower = y_lower
        self.upper = y_upper

    def _generatePath(
        self,
        anchor: Anchor,
        **kwargs,
    ):
        verticies = anchor.getVerticies()
        path = Path(verticies, **kwargs)
        return path

    def addPatch(
        self,
        facecolor: str = "none",
        **kwargs,
    ):
        ax = self.axis
        path = self.path
        patch = patches.PathPatch(
            path,
            facecolor=facecolor,
            **kwargs,
        )
        ax.add_patch(patch)
        return patch

    def addVerticalLabel(
        self,
        label: str,
        location: float = 0.5,
        padding: float = 0.0,
        **kwargs,
    ):
        ax = self.axis
        anchor = self.anchor

        x_left = self.left
        x_right = self.right
        y_lower = self.lower
        y_upper = self.upper

        height = y_upper - y_lower
        y_location = y_lower + location * height

        x_min, x_max = ax.get_xlim()
        x_range = x_max - x_min
        x_padding = padding * x_range

        if anchor.isLeft():
            x_location = x_left - x_padding
            horizontal_alignment = "right"
            rotation = 90
        elif anchor.isRight():
            x_location = x_right + x_padding
            horizontal_alignment = "left"
            rotation = 270

        text = ax.text(
            x_location,
            y_location,
            label,
            ha=horizontal_alignment,
            va="center",
            rotation=rotation,
            **kwargs,
        )
        return text

    def addHorizontalLabel(
        self,
        label: str,
        location: float = 0.5,
        padding: float = 0.0,
        **kwargs,
    ):
        ax = self.axis
        anchor = self.anchor

        x_left = self.left
        x_right = self.right
        y_lower = self.lower
        y_upper = self.upper

        width = x_right - x_left
        x_location = x_left + location * width

        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        y_padding = padding * y_range

        if anchor.isLower():
            y_location = y_lower - y_padding
            vertical_alignment = "top"
        elif anchor.isUpper():
            y_location = y_upper + y_padding
            vertical_alignment = "bottom"

        text = ax.text(
            x_location,
            y_location,
            label,
            ha="center",
            va=vertical_alignment,
            **kwargs,
        )
        return text
