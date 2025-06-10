# visualizer.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


class Visualizer:
    """
    Produces and saves plots to the 'results/' folder.
    """

    def __init__(self, results_dir: str = "results"):
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)

    def plot_scatter_log(
        self,
        df: pd.DataFrame,
        highlight_countries: dict[str, str],
        filename: str = "scatter_2020.png"
    ) -> None:
        """
        Scatter: life_expectancy vs. gdp_per_capita (log scale on x) for df (year=2020).
        highlight_countries: dict mapping country→color (if empty, all points are 'other countries').
        Saves to results_dir/filename.
        """
        # “Others” = all rows not in highlight_countries
        others = df[~df["country"].isin(highlight_countries.keys())]
        plt.figure(figsize=(8, 6))

        # Plot “others” in light gray
        plt.scatter(
            others["gdp_per_capita"],
            others["life_expectancy"],
            alpha=0.7,
            color="lightgray",
            edgecolor="none",
            label="Countries (2020)"
        )

        # If highlight_countries is empty, this loop does nothing.
        for country, color in highlight_countries.items():
            subset = df[df["country"] == country]
            if not subset.empty:
                plt.scatter(
                    subset["gdp_per_capita"],
                    subset["life_expectancy"],
                    alpha=0.9,
                    color=color,
                    edgecolor="k",
                    s=80,
                    label=country
                )

        plt.xscale("log")
        plt.xlabel("GDP per Capita (USD, 2020) [log₁₀ scale]")
        plt.ylabel("Life Expectancy (years, 2020)")
        plt.title("GDP per Capita vs. Life Expectancy (2020)")
        plt.grid(True, which="major", linestyle="--", linewidth=0.5)
        plt.legend(loc="lower right", fontsize="small")
        plt.tight_layout()

        save_path = os.path.join(self.results_dir, filename)
        plt.savefig(save_path, dpi=300)
        plt.close()

    def plot_regression_log(
        self,
        df: pd.DataFrame,
        model: sm.regression.linear_model.RegressionResultsWrapper,
        highlight_countries: dict[str, str],
        filename: str = "regression_2020.png"
    ) -> None:
        """
        Scatter + regression line for life_expectancy ~ log₁₀(gdp_per_capita) in df (year=2020).
        highlight_countries: dict mapping country→color (if empty, all points are 'other countries').
        Saves to results_dir/filename.
        """
        others = df[~df["country"].isin(highlight_countries.keys())]
        plt.figure(figsize=(8, 6))

        # “Other” countries in light gray
        plt.scatter(
            others["gdp_per_capita"],
            others["life_expectancy"],
            alpha=0.7,
            color="lightgray",
            edgecolor="none",
            label="Countries (2020)"
        )

        # If highlight_countries is empty, this loop does nothing.
        for country, color in highlight_countries.items():
            subset = df[df["country"] == country]
            if not subset.empty:
                plt.scatter(
                    subset["gdp_per_capita"],
                    subset["life_expectancy"],
                    alpha=0.9,
                    color=color,
                    edgecolor="k",
                    s=80,
                    label=country
                )

        # Regression line (fit on log₁₀(GDP))
        gdp_vals = np.logspace(
            np.log10(df["gdp_per_capita"].min()),
            np.log10(df["gdp_per_capita"].max()),
            200
        )
        log_vals = np.log10(gdp_vals)
        predicted = model.predict(sm.add_constant(log_vals))

        plt.plot(
            gdp_vals,
            predicted,
            color="red",
            linewidth=2,
            label="Fit: LifeExp ~ log₁₀(GDP)"
        )

        plt.xscale("log")
        plt.xlabel("GDP per Capita (USD, 2020) [log₁₀ scale]")
        plt.ylabel("Life Expectancy (years, 2020)")
        plt.title("Regression: Life Expectancy vs. log₁₀(GDP) (2020)")
        plt.grid(True, which="major", linestyle="--", linewidth=0.5)
        plt.legend(loc="lower right", fontsize="small")
        plt.tight_layout()

        save_path = os.path.join(self.results_dir, filename)
        plt.savefig(save_path, dpi=300)
        plt.close()

    def plot_czech_trend(
        self,
        cz_df: pd.DataFrame,
        model: sm.regression.linear_model.RegressionResultsWrapper,
        filename: str = "lex_trend_czech.png"
    ) -> None:
        """
        Plot Czech Republic’s life expectancy vs. year (2000–2020) plus trend line.
        cz_df: DataFrame filtered to Czech Republic & years 2000–2020.
        model: fitted regression (life ~ year).
        Saves to results_dir/filename.
        """
        plt.figure(figsize=(8, 6))
        plt.scatter(
            cz_df["year"],
            cz_df["life_expectancy"],
            color="blue",
            edgecolor="k",
            s=50,
            label="Czech Republic"
        )

        # Trend line: predict for years [2000, 2020]
        years_range = np.array([2000, 2020])
        pred = model.predict(sm.add_constant(years_range))

        plt.plot(
            years_range,
            pred,
            color="green",
            linewidth=2,
            label="Trend line"
        )

        plt.xlabel("Year")
        plt.ylabel("Life Expectancy (Czech Republic)")
        plt.title("Life Expectancy Trend (2000–2020) for Czech Republic")
        plt.grid(True, linestyle="--", linewidth=0.5)
        plt.legend(loc="lower right")
        plt.tight_layout()

        save_path = os.path.join(self.results_dir, filename)
        plt.savefig(save_path, dpi=300)
        plt.close()
