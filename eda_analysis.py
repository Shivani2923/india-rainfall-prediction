from pathlib import Path
import pandas as pd

# ============================================================
# INDIA RAINFALL PROJECT - EXPLORATORY DATA ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("INDIA RAINFALL - EXPLORATORY DATA ANALYSIS")
print("=" * 70)

# ------------------------------------------------------------
# 1. PROJECT AND DATASET PATHS
# ------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "datasets"

ds1_path = DATA_DIR / "rainfall in india 1901-2015.csv"
ds2_path = DATA_DIR / "district_wise_rainfall_normal[1].csv"

print("\nProject folder:")
print(PROJECT_DIR)

print("\nDataset 1:")
print(ds1_path)

print("\nDataset 2:")
print(ds2_path)

# Check that files exist
if not ds1_path.exists():
    raise FileNotFoundError(f"Dataset 1 not found:\n{ds1_path}")

if not ds2_path.exists():
    raise FileNotFoundError(f"Dataset 2 not found:\n{ds2_path}")

print("\nBoth datasets found successfully!")


# ------------------------------------------------------------
# 2. LOAD DATASETS
# ------------------------------------------------------------

ds1 = pd.read_csv(ds1_path)
ds2 = pd.read_csv(ds2_path)

print("\n" + "=" * 70)
print("DATASETS LOADED")
print("=" * 70)

print(f"\nDataset 1 shape: {ds1.shape}")
print(f"Dataset 2 shape: {ds2.shape}")


# ------------------------------------------------------------
# 3. BASIC DATASET INFORMATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DATASET 1 - BASIC INFORMATION")
print("=" * 70)

print("\nColumns:")
print(ds1.columns.tolist())

print("\nFirst 5 rows:")
print(ds1.head())

print("\nData types:")
print(ds1.dtypes)

print("\nMissing values:")
print(ds1.isnull().sum())

print("\nDuplicate rows:")
print(ds1.duplicated().sum())


print("\n" + "=" * 70)
print("DATASET 2 - BASIC INFORMATION")
print("=" * 70)

print("\nColumns:")
print(ds2.columns.tolist())

print("\nFirst 5 rows:")
print(ds2.head())

print("\nData types:")
print(ds2.dtypes)

print("\nMissing values:")
print(ds2.isnull().sum())

print("\nDuplicate rows:")
print(ds2.duplicated().sum())


# ------------------------------------------------------------
# 4. CONVERT NUMERIC COLUMNS
# ------------------------------------------------------------

monthly_cols = [
    "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
    "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"
]

season_cols = [
    "Jan-Feb", "Mar-May", "Jun-Sep", "Oct-Dec"
]

numeric_cols = monthly_cols + ["ANNUAL"] + season_cols

for col in numeric_cols:
    if col in ds1.columns:
        ds1[col] = pd.to_numeric(ds1[col], errors="coerce")

    if col in ds2.columns:
        ds2[col] = pd.to_numeric(ds2[col], errors="coerce")


# ------------------------------------------------------------
# 5. DATASET 1 - DESCRIPTIVE STATISTICS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DATASET 1 - DESCRIPTIVE STATISTICS")
print("=" * 70)

print(ds1[numeric_cols].describe().round(2))


# ------------------------------------------------------------
# 6. DATASET 2 - DESCRIPTIVE STATISTICS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DATASET 2 - DESCRIPTIVE STATISTICS")
print("=" * 70)

print(ds2[numeric_cols].describe().round(2))


# ------------------------------------------------------------
# 7. DATASET 1 - ANNUAL RAINFALL ANALYSIS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DATASET 1 - ANNUAL RAINFALL ANALYSIS")
print("=" * 70)

annual_ds1 = ds1.dropna(subset=["ANNUAL"])

print(f"\nMinimum annual rainfall: {annual_ds1['ANNUAL'].min():.2f} mm")
print(f"Maximum annual rainfall: {annual_ds1['ANNUAL'].max():.2f} mm")
print(f"Mean annual rainfall: {annual_ds1['ANNUAL'].mean():.2f} mm")
print(f"Median annual rainfall: {annual_ds1['ANNUAL'].median():.2f} mm")

min_row = annual_ds1.loc[annual_ds1["ANNUAL"].idxmin()]
max_row = annual_ds1.loc[annual_ds1["ANNUAL"].idxmax()]

print("\nLowest annual rainfall:")
print(min_row[["SUBDIVISION", "YEAR", "ANNUAL"]])

print("\nHighest annual rainfall:")
print(max_row[["SUBDIVISION", "YEAR", "ANNUAL"]])


# ------------------------------------------------------------
# 8. DATASET 1 - MONTHLY RAINFALL
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DATASET 1 - MONTHLY RAINFALL")
print("=" * 70)

monthly_means_ds1 = ds1[monthly_cols].mean().sort_values(ascending=False)

print("\nAverage rainfall by month:")
print(monthly_means_ds1.round(2))

print(f"\nWettest month: {monthly_means_ds1.idxmax()}")
print(f"Driest month: {monthly_means_ds1.idxmin()}")


# ------------------------------------------------------------
# 9. DATASET 1 - SUBDIVISION ANALYSIS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DATASET 1 - SUBDIVISION ANALYSIS")
print("=" * 70)

subdivision_mean = (
    ds1.groupby("SUBDIVISION")["ANNUAL"]
    .mean()
    .sort_values(ascending=False)
)

print("\nTop 10 subdivisions by average annual rainfall:")
print(subdivision_mean.head(10).round(2))

print("\nBottom 10 subdivisions by average annual rainfall:")
print(subdivision_mean.tail(10).round(2))


# ------------------------------------------------------------
# 10. DATASET 1 - YEARLY TREND
# ------------------------------------------------------------

yearly_mean = (
    ds1.groupby("YEAR")["ANNUAL"]
    .mean()
    .sort_index()
)

print("\n" + "=" * 70)
print("DATASET 1 - YEARLY RAINFALL TREND")
print("=" * 70)

print("\nFirst 10 yearly averages:")
print(yearly_mean.head(10).round(2))

print("\nLast 10 yearly averages:")
print(yearly_mean.tail(10).round(2))


# ------------------------------------------------------------
# 11. DATASET 2 - ANNUAL RAINFALL ANALYSIS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DATASET 2 - ANNUAL RAINFALL ANALYSIS")
print("=" * 70)

print(f"\nMinimum annual rainfall: {ds2['ANNUAL'].min():.2f} mm")
print(f"Maximum annual rainfall: {ds2['ANNUAL'].max():.2f} mm")
print(f"Mean annual rainfall: {ds2['ANNUAL'].mean():.2f} mm")
print(f"Median annual rainfall: {ds2['ANNUAL'].median():.2f} mm")

min_district = ds2.loc[ds2["ANNUAL"].idxmin()]
max_district = ds2.loc[ds2["ANNUAL"].idxmax()]

print("\nLowest annual rainfall district:")
print(min_district[["STATE_UT_NAME", "DISTRICT", "ANNUAL"]])

print("\nHighest annual rainfall district:")
print(max_district[["STATE_UT_NAME", "DISTRICT", "ANNUAL"]])


# ------------------------------------------------------------
# 12. DATASET 2 - MONTHLY ANALYSIS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DATASET 2 - MONTHLY RAINFALL")
print("=" * 70)

monthly_means_ds2 = ds2[monthly_cols].mean().sort_values(ascending=False)

print("\nAverage rainfall by month:")
print(monthly_means_ds2.round(2))

print(f"\nWettest month: {monthly_means_ds2.idxmax()}")
print(f"Driest month: {monthly_means_ds2.idxmin()}")


# ------------------------------------------------------------
# 13. DATASET 2 - STATE ANALYSIS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DATASET 2 - STATE/UT ANALYSIS")
print("=" * 70)

state_mean = (
    ds2.groupby("STATE_UT_NAME")["ANNUAL"]
    .mean()
    .sort_values(ascending=False)
)

print("\nTop 10 states/UTs by average annual rainfall:")
print(state_mean.head(10).round(2))

print("\nBottom 10 states/UTs by average annual rainfall:")
print(state_mean.tail(10).round(2))


# ------------------------------------------------------------
# 14. DATASET 2 - DISTRICT ANALYSIS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DATASET 2 - DISTRICT ANALYSIS")
print("=" * 70)

top_districts = ds2.nlargest(10, "ANNUAL")[
    ["STATE_UT_NAME", "DISTRICT", "ANNUAL"]
]

print("\nTop 10 districts by annual rainfall:")
print(top_districts.to_string(index=False))


# ------------------------------------------------------------
# 15. ZERO RAINFALL ANALYSIS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("ZERO RAINFALL ANALYSIS")
print("=" * 70)

zero_counts_ds1 = (ds1[monthly_cols] == 0).sum()
zero_counts_ds2 = (ds2[monthly_cols] == 0).sum()

print("\nDataset 1 zero values by month:")
print(zero_counts_ds1[zero_counts_ds1 > 0])

print("\nDataset 2 zero values by month:")
print(zero_counts_ds2[zero_counts_ds2 > 0])


# ------------------------------------------------------------
# 16. DATA QUALITY SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DATA QUALITY SUMMARY")
print("=" * 70)

print("\nDataset 1:")
print(f"Rows: {len(ds1)}")
print(f"Columns: {len(ds1.columns)}")
print(f"Missing cells: {ds1.isnull().sum().sum()}")
print(f"Duplicate rows: {ds1.duplicated().sum()}")
print(f"Negative numeric values: {(ds1[numeric_cols] < 0).sum().sum()}")

print("\nDataset 2:")
print(f"Rows: {len(ds2)}")
print(f"Columns: {len(ds2.columns)}")
print(f"Missing cells: {ds2.isnull().sum().sum()}")
print(f"Duplicate rows: {ds2.duplicated().sum()}")
print(f"Negative numeric values: {(ds2[numeric_cols] < 0).sum().sum()}")


# ------------------------------------------------------------
# 17. FINAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("EDA COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nKey findings:")
print(
    f"- Dataset 1 contains {len(ds1):,} rows "
    f"covering {ds1['YEAR'].min()}-{ds1['YEAR'].max()}."
)

print(
    f"- Dataset 2 contains {len(ds2):,} district-level records "
    f"across {ds2['STATE_UT_NAME'].nunique()} states/UTs."
)

print(
    f"- Dataset 1 mean annual rainfall: "
    f"{ds1['ANNUAL'].mean():.2f} mm"
)

print(
    f"- Dataset 2 mean annual rainfall: "
    f"{ds2['ANNUAL'].mean():.2f} mm"
)

print("\n" + "=" * 70)