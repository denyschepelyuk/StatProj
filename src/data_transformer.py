# data_transformer.py

import pandas as pd
import numpy as np


class DataTransformer:
    """
    Cleans, reshapes, and merges life expectancy and GDP per capita data.
    """

    @staticmethod
    def melt_life_expectancy(life_df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert life expectancy from wide to long format:

        Input wide‐format columns: [country, 1800, 1801, ..., 2100]
        Output long‐format columns: [country, year (as int), life_expectancy]

        Drops any missing life_expectancy.
        """
        life_long = life_df.melt(
            id_vars="country",
            var_name="year",
            value_name="life_expectancy"
        )
        # Convert 'year' → int, drop NaN life_expectancy
        life_long["year"] = life_long["year"].astype(int)
        life_long = life_long.dropna(subset=["life_expectancy"])
        return life_long

    @staticmethod
    def clean_gdp_cell(val) -> float:
        """
        Converts a single raw GDP‐string cell (e.g. "27.7k", "81.5k", "5380", or NaN)
        into a float (absolute USD). Returns np.nan if invalid.
        """
        if pd.isna(val):
            return np.nan
        s = str(val).strip()
        # If ends with 'k' (or 'K'), treat as thousands
        if s.endswith(("k", "K")):
            num_str = s[:-1].replace(",", "").strip()
            try:
                return float(num_str) * 1000.0
            except ValueError:
                return np.nan
        # Otherwise remove commas
        num_str = s.replace(",", "").strip()
        try:
            return float(num_str)
        except ValueError:
            return np.nan

    @classmethod
    def melt_gdp_per_capita(cls, gdp_df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert GDP per capita from wide to long format:

        Input wide‐format columns: [country, 1800, 1801, ..., 2100]
        Output columns: [country, year (int), gdp_per_capita_raw, gdp_per_capita (float)]

        It uses clean_gdp_cell to parse "27.7k" → 27700. Drops NaN in gdp_per_capita.
        """
        gdp_melted = gdp_df.melt(
            id_vars="country",
            var_name="year",
            value_name="gdp_per_capita_raw"
        )
        # Convert 'year' → int
        gdp_melted["year"] = gdp_melted["year"].astype(int)

        # Apply cleaning
        gdp_melted["gdp_per_capita"] = gdp_melted["gdp_per_capita_raw"].apply(cls.clean_gdp_cell)

        # Drop raw column and rows where gdp_per_capita is NaN
        gdp_melted = gdp_melted.drop(columns=["gdp_per_capita_raw"]).dropna(subset=["gdp_per_capita"])
        return gdp_melted

    @staticmethod
    def merge_datasets(life_long: pd.DataFrame, gdp_long: pd.DataFrame) -> pd.DataFrame:
        """
        Inner‐join life_long and gdp_long on ["country", "year"]:

        Returns a merged DataFrame with columns:
        [country, year, life_expectancy, gdp_per_capita]
        """
        merged = pd.merge(
            life_long,
            gdp_long,
            on=["country", "year"],
            how="inner"
        )
        return merged
