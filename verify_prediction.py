import sys
from pathlib import Path

# Add project root to sys.path so rainfall_model can be imported from any working directory
PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from rainfall_model import RainfallPredictor
import pandas as pd

ds1_path = PROJECT_DIR / "data" / "datasets" / "rainfall in india 1901-2015.csv"
ds2_path = PROJECT_DIR / "data" / "datasets" / "district_wise_rainfall_normal[1].csv"

p = RainfallPredictor(ds1_path, ds2_path)
print(f"Model ready. Metrics: MAE={p.mae:.2f}, RMSE={p.rmse:.2f}, R2={p.r2:.4f}")

# Test diverse districts across India
test_cases = [
    ("MAHARASHTRA", "PUNE", 2015),
    ("MAHARASHTRA", "PUNE", 2026),
    ("KERALA", "WAYANAD", 2010),
    ("RAJASTHAN", "JAISALMER", 2024),
    ("ARUNACHAL PRADESH", "LOHIT", 2012),
    ("TAMIL NADU", "CHENNAI", 2030),
    ("JAMMU AND KASHMIR", "LADAKH (LEH)", 2025),
]

for state, dist, yr in test_cases:
    res = p.predict(state, dist, yr)
    act_str = f"{res['actual_annual']} mm" if res['actual_annual'] is not None else "N/A (Future)"
    print(f"[{res['state']} - {res['district']} ({res['year']})] Pred: {res['predicted_annual']} mm | Act: {act_str} | Normal: {res['district_normal']} mm | {res['category']} ({res['pct_deviation']:+.1f}%)")
    assert res['predicted_annual'] > 0, "Predicted rainfall must be positive"
    assert res['category'] in ["Low Rainfall", "Normal Rainfall", "High Rainfall"], "Category invalid"
    assert "mae" in res['metrics'] and "rmse" in res['metrics'] and "r2" in res['metrics'], "Metrics missing"
    assert len(res['plot_df']) > 0, "Plot data missing"

print("\nAll automated verification test cases PASSED successfully!")
