import sys

import matplotlib

matplotlib.use("Agg")

from PySide6.QtWidgets import QApplication

from src.gui import MainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    font = app.font()
    font.setPointSize(9)
    app.setFont(font)

    win = MainWindow()
    win.showMaximized()
    sys.exit(app.exec())
