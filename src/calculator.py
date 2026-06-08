import math

import numpy as np
from scipy import stats


class StatisticsCalculator:
    def __init__(self, data: np.ndarray):
        self.data = data
        self.n = len(data)

        # Шаг 2
        self.x_min = np.min(data)
        self.x_max = np.max(data)
        self.R = self.x_max - self.x_min

        # Шаг 3
        self.k = int(1 + 3.32 * math.log10(self.n))

        # Шаг 4
        self.h = round(self.R / self.k, 2)
        if self.h == 0:
            self.h = 0.01

        # Шаг 5
        self.bounds = [round(self.x_min + i * self.h, 2) for i in range(self.k + 1)]
        if self.bounds[-1] <= self.x_max:
            self.bounds[-1] = round(self.x_max + self.h, 2)

        self._build_frequency_table()
        self._calc_grouped_stats()

    # Шаг 6-7
    def _build_frequency_table(self):
        self.table = []
        for i in range(len(self.bounds) - 1):
            left, right = self.bounds[i], self.bounds[i + 1]
            if i == len(self.bounds) - 2:
                mask = (self.data >= left) & (self.data <= right)
            else:
                mask = (self.data >= left) & (self.data < right)

            ni = int(np.sum(mask))
            pi = ni / self.n
            bracket_r = "]" if i == len(self.bounds) - 2 else ")"
            self.table.append(
                {
                    "interval": f"[{left}; {right}{bracket_r}",
                    "mid": round((left + right) / 2, 3),
                    "abs_freq": ni,
                    "rel_freq": round(pi, 4),
                    "cum_freq": None,
                }
            )

        cum = 0
        for row in self.table:
            cum += row["abs_freq"]
            row["cum_freq"] = cum

    # Шаг 8
    def _calc_grouped_stats(self):
        self.mean = sum(r["mid"] * r["abs_freq"] for r in self.table) / self.n
        var_num = sum(r["abs_freq"] * (r["mid"] - self.mean) ** 2 for r in self.table)
        self.var_corrected = var_num / (self.n - 1)
        self.std_corrected = math.sqrt(self.var_corrected)

    # Шаг 9
    def test_normality(self, alpha=0.05):
        stat, p = stats.shapiro(self.data)
        return {
            "stat": f"{stat:.4f}",
            "p": f"{p:.4f}",
            "reject": p < alpha,
            "text": "H₀ отвергается (распределение НЕ нормальное)"
            if p < alpha
            else "H₀ не отвергается (распределение МОЖЕТ быть нормальным)",
        }

    # Шаг 10
    def test_variance(self, sigma2_0=1.0, alpha=0.05):
        chi2 = (self.n - 1) * self.var_corrected / sigma2_0
        crit_low = stats.chi2.ppf(alpha / 2, self.n - 1)
        crit_high = stats.chi2.ppf(1 - alpha / 2, self.n - 1)
        reject = chi2 < crit_low or chi2 > crit_high
        return {
            "chi2": f"{chi2:.4f}",
            "crit": f"[{crit_low:.4f}, {crit_high:.4f}]",
            "reject": reject,
            "text": "H₀ отвергается (σ² ≠ 1)"
            if reject
            else "H₀ не отвергается (σ² = 1)",
        }

    # Шаг 11
    def test_mean(self, mu_0=0.0, alpha=0.05):
        t = (np.mean(self.data) - mu_0) / (
            np.std(self.data, ddof=1) / math.sqrt(self.n)
        )
        crit = stats.t.ppf(1 - alpha / 2, self.n - 1)
        reject = abs(t) > crit
        return {
            "t": f"{t:.4f}",
            "crit": f"±{crit:.4f}",
            "reject": reject,
            "text": "H₀ отвергается (μ ≠ 0)" if reject else "H₀ не отвергается (μ = 0)",
        }
