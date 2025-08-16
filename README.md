# Statistical Project: GDP per Capita vs Life Expectancy

## Goal
I investigate the relationship between **GDP per capita** and **life expectancy at birth** across countries, and I examine the time trend for the Czech Republic.

Specifically, I analyze:
- Correlation between GDP per capita and life expectancy in **2020**
- Linear regression: `Life expectancy = β0 + β1 · log10(GDP per capita)`
- Czech Republic trend, **2000–2020**
- Paired t-test of year-over-year Czech changes (**2000–2020**)

## Data
- `data/lex.csv` — life expectancy (wide format by year)
- `data/gdp_pcap.csv` — GDP per capita (wide format by year; values like “24.5k” are parsed to numbers)

Both files are included in this repository.

## Methods
- **Pearson correlation** (2020) on filtered data (valid, non-extreme values)
- **Linear regression:** `Life expectancy = β0 + β1 · log10(GDP per capita)`
- **Trend regression (Czech Republic, 2000–2020):** `Life expectancy = α0 + α1 · Year`
- **Paired t-test (CZ)** on year-over-year life-expectancy differences against μ = 0

Plots are generated with matplotlib and saved in `results/`.

## Results
*(Computed by running `python src/main.py`.)*

### Cross-section, 2020 (n = 194 countries)
- **Pearson r (GDP per capita vs life expectancy):** **0.624**, *p* ≈ **2.64×10⁻²²**
- **Regression on log₁₀(GDP):**
  - β₁ = **10.876** (95% CI **[9.790, 11.962]**)
  - β₀ = **28.226** (95% CI **[23.809, 32.644]**)
  - **R²** = **0.670** (Adj. **R²** = **0.669**)
  - Interpretation: a **10×** increase in GDP per capita is associated with about **+10.9 years** higher life expectancy (diminishing returns captured by the log term).

### Czech Republic, 2000–2020 (n = 21 years)
- **Trend slope:** **0.224** years per calendar year (≈ **+2.24 years per decade**)
- **R² = 0.947**
- **Paired t-test (year-over-year Δ):** *t* = **2.996**, *p* = **0.0074**; mean Δ ≈ **+0.18 years/year**

## Figures (written to `results/`)
- `scatter_2020.png` — 2020 scatter of life expectancy vs GDP per capita
- `regression_2020.png` — regression fit using log₁₀(GDP)
- `trend_czech.png` — Czech Republic 2000–2020 trend line

## How to Run

### 1) Set up the environment
Python **3.9+** recommended.

```bash
# from the project root (this folder)
python -m venv .venv

# activate:
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
