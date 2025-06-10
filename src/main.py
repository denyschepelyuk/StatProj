# main.py

import os
from data_loader import DataLoader
from data_transformer import DataTransformer
from analyzer import Analyzer
from visualizer import Visualizer


def main():
    # -----------------------------------------------------------------------------
    # 0. Ensure the 'results/' folder exists
    # -----------------------------------------------------------------------------
    os.makedirs("results", exist_ok=True)

    # -----------------------------------------------------------------------------
    # 1. Load raw CSVs
    # -----------------------------------------------------------------------------
    loader = DataLoader(
        life_path="data/lex.csv",
        gdp_path="data/gdp_pcap.csv"
    )
    life_raw = loader.load_life_expectancy()
    gdp_raw = loader.load_gdp_per_capita()

    # -----------------------------------------------------------------------------
    # 2. Transform data (melt + clean + merge)
    # -----------------------------------------------------------------------------
    transformer = DataTransformer()
    life_long = transformer.melt_life_expectancy(life_raw)
    gdp_long = transformer.melt_gdp_per_capita(gdp_raw)
    merged_df = transformer.merge_datasets(life_long, gdp_long)

    # (Optional) Print merged DataFrame head for verification
    print("\nMerged DataFrame (first 5 rows):")
    print(merged_df.head())

    # -----------------------------------------------------------------------------
    # 3. Analyze year 2020
    # -----------------------------------------------------------------------------
    analyzer = Analyzer()
    df_2020_filtered = analyzer.filter_2020_data(merged_df)

    # Pearson correlation
    pearson_r, pearson_p = analyzer.compute_pearson(df_2020_filtered)
    print(f"\nPearson correlation (Year 2020, filtered): r = {pearson_r:.3f}, p-value = {pearson_p:.4f}")

    # Regression: life_expectancy ~ log10(gdp_per_capita)
    model_log = analyzer.regression_log_gdp(df_2020_filtered)

    # -----------------------------------------------------------------------------
    # 4. Plot year 2020 scatter & regression WITHOUT any highlighting
    # -----------------------------------------------------------------------------
    viz = Visualizer(results_dir="results")
    highlights = {}  # Empty dict → no country is highlighted

    # Scatter-only plot with log-scale
    viz.plot_scatter_log(
        df=df_2020_filtered,
        highlight_countries=highlights,
        filename="scatter_2020.png"
    )

    # Scatter + regression line
    viz.plot_regression_log(
        df=df_2020_filtered,
        model=model_log,
        highlight_countries=highlights,
        filename="regression_2020.png"
    )

    # -----------------------------------------------------------------------------
    # 5. Czech Republic trend (2000–2020)
    # -----------------------------------------------------------------------------
    cz_df, trend_model = analyzer.trend_czech(merged_df)
    print("\nCzech Republic Trend Regression (2000–2020):")
    print(trend_model.summary())

    # Plot Czech trend (no changes needed here)
    viz.plot_czech_trend(
        cz_df=cz_df,
        model=trend_model,
        filename="trend_czech.png"
    )

    # -----------------------------------------------------------------------------
    # 6. (Optional) Paired t-test for year-over-year Czech differences
    # -----------------------------------------------------------------------------
    t_stat, p_val = analyzer.paired_ttest_czech(cz_df)
    print(f"\nPaired t-test on Czech year‐over‐year differences (2000–2020):")
    print(f"  t-statistic = {t_stat:.3f}, p-value = {p_val:.4f}")


if __name__ == "__main__":
    main()
