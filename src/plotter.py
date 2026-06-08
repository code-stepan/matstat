import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PySide6.QtGui import QImage, QPainter, QPixmap
from PySide6.QtWidgets import QWidget
from scipy import stats


class MplCanvas(QWidget):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        super().__init__(parent)
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self._pixmap = QPixmap()
        self.setMinimumSize(int(width * dpi), int(height * dpi))

    def draw(self):
        canvas = FigureCanvasAgg(self.fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        w, h = buf.shape[1], buf.shape[0]
        image = QImage(buf.tobytes(), w, h, QImage.Format_RGBA8888)
        self._pixmap = QPixmap.fromImage(image)
        self.update()

    def paintEvent(self, event):
        if self._pixmap.isNull():
            return
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self._pixmap)
        painter.end()


def plot_histogram(canvas, data, bounds):
    canvas.fig.clear()
    ax = canvas.fig.add_subplot(111)
    ax.hist(
        data,
        bins=bounds,
        edgecolor="black",
        linewidth=1.2,
        color="skyblue",
    )
    for b in bounds[1:-1]:
        ax.axvline(b, color="red", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Значение")
    ax.set_ylabel("Частота")
    ax.set_title("Гистограмма выборки")
    ax.grid(axis="y", alpha=0.4)
    canvas.draw()


def plot_cdf(canvas, data, x_min, x_max, var_corrected):
    canvas.fig.clear()
    ax = canvas.fig.add_subplot(111)
    sorted_d = np.sort(data)
    y = np.arange(1, len(sorted_d) + 1) / len(sorted_d)
    ax.step(
        sorted_d, y, where="post", linewidth=2, color="navy", label="Эмпирическая F(x)"
    )
    ax.plot(sorted_d, y, "o", markersize=2, alpha=0.5)

    if var_corrected > 0:
        x_th = np.linspace(x_min - 0.5, x_max + 0.5, 200)
        y_th = stats.norm.cdf(
            x_th, loc=np.mean(data), scale=np.std(data, ddof=1)
        )
        ax.plot(x_th, y_th, "r--", label="Нормальное распределение (аппроксимация)")
        ax.legend()

    ax.set_xlabel("Значение")
    ax.set_ylabel("Вероятность")
    ax.set_title("Функция распределения")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    canvas.draw()
