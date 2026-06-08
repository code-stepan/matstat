import numpy as np
from PySide6.QtWidgets import (
    QHeaderView,
    QLabel,
    QMainWindow,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.calculator import StatisticsCalculator
from src.data_parser import RAW_DATA, parse_data
from src.plotter import MplCanvas, plot_cdf, plot_histogram


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Математическая статистика")
        self.resize(1250, 850)

        self.data = parse_data(RAW_DATA, 100)
        self.calc = StatisticsCalculator(self.data)

        self._init_ui()
        self._populate()

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        tabs = QTabWidget()

        tab1 = QWidget()
        lay1 = QVBoxLayout(tab1)
        self.txt_summary = QTextEdit()
        self.txt_summary.setReadOnly(True)
        lay1.addWidget(QLabel("<b>Базовые характеристики (Шаги 2–5)</b>"))
        lay1.addWidget(self.txt_summary)
        tabs.addTab(tab1, "Сводка")

        tab2 = QWidget()
        lay2 = QVBoxLayout(tab2)
        self.tbl = QTableWidget()
        self.tbl.setColumnCount(5)
        self.tbl.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tbl.setHorizontalHeaderLabels(
            ["Интервал", "Середина xᵢ*", "nᵢ", "pᵢ", "Накоп. Nᵢ"]
        )
        self.tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        lay2.addWidget(QLabel("<b>Статистическая таблица (Шаг 7)</b>"))
        lay2.addWidget(self.tbl)
        tabs.addTab(tab2, "Таблица")

        tab3 = QWidget()
        lay3 = QVBoxLayout(tab3)
        self.cvs_hist = MplCanvas(self, width=6, height=4)
        lay3.addWidget(QLabel("<b>Гистограмма (Шаг 6)</b>"))
        lay3.addWidget(self.cvs_hist)
        tabs.addTab(tab3, "Гистограмма")

        tab4 = QWidget()
        lay4 = QVBoxLayout(tab4)
        self.cvs_cdf = MplCanvas(self, width=6, height=4)
        lay4.addWidget(QLabel("<b>Эмпирическая функция распределения (Шаг 6)</b>"))
        lay4.addWidget(self.cvs_cdf)
        tabs.addTab(tab4, "F(x)")

        tab5 = QWidget()
        lay5 = QVBoxLayout(tab5)
        self.txt_tests = QTextEdit()
        self.txt_tests.setReadOnly(True)
        lay5.addWidget(QLabel("<b>Проверка гипотез (α = 0.05)</b>"))
        lay5.addWidget(self.txt_tests)
        tabs.addTab(tab5, "Гипотезы")

        main_layout.addWidget(tabs)
        self.statusBar().showMessage(
            f"Выборка: n={self.calc.n} | Интервалов: k={self.calc.k} | Шаг: h={self.calc.h}"
        )

    def _populate(self):
        raw_mean = np.mean(self.data)
        self.txt_summary.setText(
            f"""
n = {self.calc.n}
x_min = {self.calc.x_min:.4f} | x_max = {self.calc.x_max:.4f}
R = x_max - x_min = {self.calc.R:.4f}
k = ⌊1 + 3.32·lg({self.calc.n})⌋ = {self.calc.k}
h = R/k ≈ {self.calc.h} (округлено до 0.01)

Границы интервалов αᵢ: {[f"{b:.2f}" for b in self.calc.bounds]}

Среднее (сгрупп.): x̄ = {self.calc.mean:.4f}
Испр. дисперсия: s² = {self.calc.var_corrected:.4f}
Испр. СКО: s = {self.calc.std_corrected:.4f}

(Сырые данные для сравнения: x̄_raw = {raw_mean:.4f}, s²_raw = {np.var(self.data, ddof=1):.4f})
        """.strip()
        )

        self.tbl.setRowCount(len(self.calc.table))
        for i, r in enumerate(self.calc.table):
            self.tbl.setItem(i, 0, QTableWidgetItem(r["interval"]))
            self.tbl.setItem(i, 1, QTableWidgetItem(str(r["mid"])))
            self.tbl.setItem(i, 2, QTableWidgetItem(str(r["abs_freq"])))
            self.tbl.setItem(i, 3, QTableWidgetItem(str(r["rel_freq"])))
            self.tbl.setItem(i, 4, QTableWidgetItem(str(r["cum_freq"])))

        plot_histogram(self.cvs_hist, self.data, self.calc.bounds)
        plot_cdf(
            self.cvs_cdf, self.data,
            self.calc.x_min, self.calc.x_max, self.calc.var_corrected,
        )

        t_norm = self.calc.test_normality()
        t_var = self.calc.test_variance()
        t_mean = self.calc.test_mean()
        self.txt_tests.setText(
            f"""
1️⃣ Нормальность (Shapiro-Wilk)
   Статистика: {t_norm["stat"]} | p-value: {t_norm["p"]}
   ✅ {t_norm["text"]}

2️⃣ Дисперсия σ² = 1 (χ²-критерий)
   χ² = {t_var["chi2"]} | Крит. область: {t_var["crit"]}
   ✅ {t_var["text"]}

3️⃣ Мат. ожидание μ = 0 (t-критерий)
   t = {t_mean["t"]} | Крит. область: {t_mean["crit"]}
   ✅ {t_mean["text"]}
        """.strip()
        )
