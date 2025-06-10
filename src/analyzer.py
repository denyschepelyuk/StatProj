# analyzer.py

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


class Analyzer:
    """
    Performs statistical analyses: correlation, regression, trend tests.
    """

    @staticmethod
    def filter_2020_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Given a merged DataFrame with at least columns [year, country, life_expectancy, gdp_per_capita],
        returns a filtered DataFrame for year==2020, dropping any invalid/outlier rows:
          - Keep only gdp_per_capita > 0 and life_expectancy > 0
          - Drop extremely low life_expectancy (e.g. ≤10) or extremely high GDP (>1e6)

        The returned DataFrame has columns: [country, year, life_expectancy, gdp_per_capita].
        """
        df_2020 = df[df["year"] == 2020].copy()
        df_2020 = df_2020[(df_2020["gdp_per_capita"] > 0) & (df_2020["life_expectancy"] > 0)]
        df_2020 = df_2020[df_2020["life_expectancy"] > 10]
        df_2020 = df_2020[df_2020["gdp_per_capita"] < 1e6]
        return df_2020

    @staticmethod
    def compute_pearson(df: pd.DataFrame) -> tuple[float, float]:
        """
        Return (r, p) = Pearson correlation between gdp_per_capita and life_expectancy
        on the provided DataFrame (un‐logged).
        """
        r, p = stats.pearsonr(df["gdp_per_capita"], df["life_expectancy"])
        return r, p

    @staticmethod
    def regression_log_gdp(df: pd.DataFrame) -> sm.regression.linear_model.RegressionResultsWrapper:
        """
        Fit OLS: life_expectancy ~ log10(gdp_per_capita) on df.
        Returns the fitted model object.
        """
        df["log_gdp"] = np.log10(df["gdp_per_capita"])
        X = sm.add_constant(df["log_gdp"])
        y = df["life_expectancy"]
        model = sm.OLS(y, X).fit()
        return model

    @staticmethod
    def trend_czech(df: pd.DataFrame) -> tuple[pd.DataFrame, sm.regression.linear_model.RegressionResultsWrapper]:
        """
        Filter df to (country == "Czech Republic" AND year in [2000..2020]),
        fit OLS: life_expectancy ~ year.
        Returns the filtered Czech time‐series DataFrame and the fitted model.
        """
        cz_df = df[(df["country"] == "Czech Republic") & df["year"].between(2000, 2020)].copy()
        X = sm.add_constant(cz_df["year"])
        y = cz_df["life_expectancy"]
        model = sm.OLS(y, X).fit()
        return cz_df, model

    @staticmethod
    def paired_ttest_czech(df_cz: pd.DataFrame) -> tuple[float, float]:
        """
        Given a DataFrame cz_df sorted by year with a "life_expectancy" column,
        compute year-over-year differences and run a one-sample t-test against μ=0.
        Returns (t_stat, p_value).
        """
        cz_sorted = df_cz.sort_values(by="year")
        le_vals = cz_sorted["life_expectancy"].values
        diffs = le_vals[1:] - le_vals[:-1]
        t_stat, p_value = stats.ttest_1samp(diffs, 0)
        return t_stat, p_value
