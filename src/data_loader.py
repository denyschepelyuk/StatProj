# data_loader.py

import pandas as pd


class DataLoader:
    """
    Loads raw CSV files from disk.
    """

    def __init__(self, life_path: str, gdp_path: str):
        """
        life_path: path to the life expectancy wide‐format CSV
        gdp_path: path to the GDP per capita wide‐format CSV
        """
        self.life_path = life_path
        self.gdp_path = gdp_path

    def load_life_expectancy(self) -> pd.DataFrame:
        """
        Returns a DataFrame of the raw (wide‐format) life expectancy data.
        """
        return pd.read_csv(self.life_path)

    def load_gdp_per_capita(self) -> pd.DataFrame:
        """
        Returns a DataFrame of the raw (wide‐format) GDP per capita data.
        """
        return pd.read_csv(self.gdp_path)
