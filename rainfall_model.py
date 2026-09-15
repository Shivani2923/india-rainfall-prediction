"""
India Rainfall Prediction Module
Machine learning model for annual rainfall prediction and historical validation.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


class RainfallPredictor:
    """
    Trained Random Forest Regressor for predicting annual rainfall across India.
    Combines 115-year historical subdivision time-series with district-level climatological normals.
    """

    def __init__(self, ds1_path: Path, ds2_path: Path):
        self.ds1_path = ds1_path
        self.ds2_path = ds2_path
        self._load_and_preprocess()
        self._train_model()

    def _load_and_preprocess(self):
        self.df1 = pd.read_csv(self.ds1_path)
        self.df2 = pd.read_csv(self.ds2_path)

        # Ensure numeric conversion
        numeric_cols = [
            "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
            "JUL", "AUG", "SEP", "OCT", "NOV", "DEC",
            "ANNUAL", "Jan-Feb", "Mar-May", "Jun-Sep", "Oct-Dec"
        ]
        for col in numeric_cols:
            if col in self.df1.columns:
                self.df1[col] = pd.to_numeric(self.df1[col], errors="coerce")
            if col in self.df2.columns:
                self.df2[col] = pd.to_numeric(self.df2[col], errors="coerce")

        # Clean historical records
        self.df1_clean = (
            self.df1.dropna(subset=["SUBDIVISION", "YEAR", "ANNUAL"])
            .sort_values(["SUBDIVISION", "YEAR"])
            .reset_index(drop=True)
        )
        self.df1_clean["YEAR"] = self.df1_clean["YEAR"].astype(int)

        # Compute subdivision climatological baselines
        self.sub_means = self.df1_clean.groupby("SUBDIVISION")["ANNUAL"].mean().to_dict()
        self.sub_stds = self.df1_clean.groupby("SUBDIVISION")["ANNUAL"].std().fillna(0).to_dict()

        self.df1_clean["SUB_MEAN"] = self.df1_clean["SUBDIVISION"].map(self.sub_means)
        self.df1_clean["SUB_STD"] = self.df1_clean["SUBDIVISION"].map(self.sub_stds)

        # Build autoregressive and rolling features per subdivision
        records = []
        for _, grp in self.df1_clean.groupby("SUBDIVISION"):
            grp = grp.sort_values("YEAR").copy()
            grp["LAG_1"] = grp["ANNUAL"].shift(1)
            grp["LAG_2"] = grp["ANNUAL"].shift(2)
            grp["ROLL_MEAN_3"] = grp["ANNUAL"].shift(1).rolling(3, min_periods=1).mean()
            grp["ROLL_MEAN_5"] = grp["ANNUAL"].shift(1).rolling(5, min_periods=1).mean()
            records.append(grp)

        self.feature_df = pd.concat(records).dropna().reset_index(drop=True)

        # Mapping dictionary: Administrative State/UT to Meteorological Subdivision(s)
        self.state_to_subdiv = {
            "ANDAMAN And NICOBAR ISLANDS": ["ANDAMAN & NICOBAR ISLANDS"],
            "ARUNACHAL PRADESH": ["ARUNACHAL PRADESH"],
            "ASSAM": ["ASSAM & MEGHALAYA"],
            "MEGHALAYA": ["ASSAM & MEGHALAYA"],
            "BIHAR": ["BIHAR"],
            "CHANDIGARH": ["HARYANA DELHI & CHANDIGARH"],
            "DELHI": ["HARYANA DELHI & CHANDIGARH"],
            "HARYANA": ["HARYANA DELHI & CHANDIGARH"],
            "CHATISGARH": ["CHHATTISGARH"],
            "DADAR NAGAR HAVELI": ["GUJARAT REGION"],
            "DAMAN AND DUI": ["GUJARAT REGION", "SAURASHTRA & KUTCH"],
            "GOA": ["KONKAN & GOA"],
            "HIMACHAL": ["HIMACHAL PRADESH"],
            "JAMMU AND KASHMIR": ["JAMMU & KASHMIR"],
            "JHARKHAND": ["JHARKHAND"],
            "KERALA": ["KERALA"],
            "LAKSHADWEEP": ["LAKSHADWEEP"],
            "MANIPUR": ["NAGA MANI MIZO TRIPURA"],
            "MIZORAM": ["NAGA MANI MIZO TRIPURA"],
            "NAGALAND": ["NAGA MANI MIZO TRIPURA"],
            "TRIPURA": ["NAGA MANI MIZO TRIPURA"],
            "ORISSA": ["ORISSA"],
            "PONDICHERRY": ["TAMIL NADU"],
            "PUNJAB": ["PUNJAB"],
            "SIKKIM": ["SUB HIMALAYAN WEST BENGAL & SIKKIM"],
            "TAMIL NADU": ["TAMIL NADU"],
            "UTTARANCHAL": ["UTTARAKHAND"],
            "ANDHRA PRADESH": ["COASTAL ANDHRA PRADESH", "RAYALSEEMA", "TELANGANA"],
            "GUJARAT": ["GUJARAT REGION", "SAURASHTRA & KUTCH"],
            "KARNATAKA": ["COASTAL KARNATAKA", "NORTH INTERIOR KARNATAKA", "SOUTH INTERIOR KARNATAKA"],
            "MADHYA PRADESH": ["EAST MADHYA PRADESH", "WEST MADHYA PRADESH"],
            "MAHARASHTRA": ["KONKAN & GOA", "MADHYA MAHARASHTRA", "MATATHWADA", "VIDARBHA"],
            "RAJASTHAN": ["EAST RAJASTHAN", "WEST RAJASTHAN"],
            "UTTAR PRADESH": ["EAST UTTAR PRADESH", "WEST UTTAR PRADESH"],
            "WEST BENGAL": ["GANGETIC WEST BENGAL", "SUB HIMALAYAN WEST BENGAL & SIKKIM"],
        }

    def _train_model(self):
        self.feature_cols = [
            "YEAR", "SUB_MEAN", "SUB_STD",
            "LAG_1", "LAG_2", "ROLL_MEAN_3", "ROLL_MEAN_5"
        ]
        X = self.feature_df[self.feature_cols]
        y = self.feature_df["ANNUAL"]

        # Train/Test Split (80/20) for objective evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)

        # Performance evaluation metrics
        test_preds = self.model.predict(X_test)
        self.mae = float(mean_absolute_error(y_test, test_preds))
        self.rmse = float(root_mean_squared_error(y_test, test_preds))
        self.r2 = float(r2_score(y_test, test_preds))

        # Train on full dataset for maximum historical context during inference
        self.full_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        self.full_model.fit(X, y)

    def get_subdivision_for_district(self, state: str, district: str) -> str:
        """Find corresponding IMD meteorological subdivision for a given state & district."""
        row = self.df2[
            (self.df2["STATE_UT_NAME"] == state) & (self.df2["DISTRICT"] == district)
        ]
        if len(row) == 0:
            return "KERALA"

        dist_annual = float(row["ANNUAL"].values[0])
        possible_subs = self.state_to_subdiv.get(state, ["KERALA"])

        if len(possible_subs) == 1:
            return possible_subs[0]

        # Choose the subdivision whose climatological mean is closest to the district normal
        return min(
            possible_subs,
            key=lambda s: abs(self.sub_means.get(s, 1200.0) - dist_annual)
        )

    def predict(self, state: str, district: str, target_year: int) -> dict:
        """
        Predicts annual rainfall for a selected state, district, and year.
        Returns prediction value, category, metrics, and historical comparison series.
        """
        subdiv = self.get_subdivision_for_district(state, district)
        dist_row = self.df2[
            (self.df2["STATE_UT_NAME"] == state) & (self.df2["DISTRICT"] == district)
        ].iloc[0]

        dist_normal = float(dist_row["ANNUAL"])
        sub_mean = float(self.sub_means.get(subdiv, dist_normal))
        sub_std = float(self.sub_stds.get(subdiv, 200.0))

        # Extract historical records for this subdivision
        sub_hist = (
            self.feature_df[self.feature_df["SUBDIVISION"] == subdiv]
            .sort_values("YEAR")
            .copy()
        )

        # In-sample historical predictions
        sub_hist["PREDICTED_SUB"] = self.full_model.predict(sub_hist[self.feature_cols])

        # Downscale from subdivision to district using the climatological normal ratio
        scaling_ratio = dist_normal / sub_mean if sub_mean > 0 else 1.0
        sub_hist["PREDICTED_ANNUAL"] = sub_hist["PREDICTED_SUB"] * scaling_ratio
        sub_hist["ACTUAL_ANNUAL"] = sub_hist["ANNUAL"] * scaling_ratio

        if target_year in sub_hist["YEAR"].values:
            target_row = sub_hist[sub_hist["YEAR"] == target_year].iloc[0]
            pred_annual = float(target_row["PREDICTED_ANNUAL"])
            actual_annual = float(target_row["ACTUAL_ANNUAL"])
        else:
            # Future year iterative forecasting
            curr_hist = sub_hist[["YEAR", "ANNUAL"]].copy()
            last_known_year = int(curr_hist["YEAR"].max())

            for y in range(last_known_year + 1, target_year + 1):
                ann_series = curr_hist["ANNUAL"]
                lag1 = float(ann_series.iloc[-1])
                lag2 = float(ann_series.iloc[-2]) if len(ann_series) > 1 else lag1
                roll3 = float(ann_series.iloc[-3:].mean())
                roll5 = float(ann_series.iloc[-5:].mean())

                row_in = pd.DataFrame([{
                    "YEAR": y,
                    "SUB_MEAN": sub_mean,
                    "SUB_STD": sub_std,
                    "LAG_1": lag1,
                    "LAG_2": lag2,
                    "ROLL_MEAN_3": roll3,
                    "ROLL_MEAN_5": roll5,
                }])[self.feature_cols]

                p_sub = float(self.full_model.predict(row_in)[0])
                curr_hist = pd.concat(
                    [curr_hist, pd.DataFrame([{"YEAR": y, "ANNUAL": p_sub}])],
                    ignore_index=True
                )

            pred_annual = float(p_sub * scaling_ratio)
            actual_annual = None

        # Categorize based on IMD meteorological criteria
        # Deficient (Low): < -19% of normal
        # Normal: between -19% and +19% of normal
        # Excess (High): > +19% of normal
        pct_deviation = ((pred_annual - dist_normal) / dist_normal) * 100.0 if dist_normal > 0 else 0.0

        if pct_deviation < -19.0:
            category = "Low Rainfall"
            category_desc = "Deficient Rainfall (more than 19% below normal)"
            category_color = "#E74C3C"  # Red
        elif pct_deviation > 19.0:
            category = "High Rainfall"
            category_desc = "Excess Rainfall (more than 19% above normal)"
            category_color = "#2E86C1"  # Blue
        else:
            category = "Normal Rainfall"
            category_desc = "Normal Rainfall (within ±19% of normal)"
            category_color = "#27AE60"  # Green

        # Prepare comparison chart data
        plot_df = sub_hist[["YEAR", "ACTUAL_ANNUAL", "PREDICTED_ANNUAL"]].copy()
        if actual_annual is None:
            # Append future forecast row
            future_row = pd.DataFrame([{
                "YEAR": target_year,
                "ACTUAL_ANNUAL": np.nan,
                "PREDICTED_ANNUAL": pred_annual
            }])
            plot_df = pd.concat([plot_df, future_row], ignore_index=True)

        return {
            "state": state,
            "district": district,
            "subdivision": subdiv,
            "year": target_year,
            "predicted_annual": round(pred_annual, 1),
            "actual_annual": round(actual_annual, 1) if actual_annual is not None else None,
            "district_normal": round(dist_normal, 1),
            "pct_deviation": round(pct_deviation, 1),
            "category": category,
            "category_desc": category_desc,
            "category_color": category_color,
            "metrics": {
                "mae": round(self.mae, 2),
                "rmse": round(self.rmse, 2),
                "r2": round(self.r2, 4),
            },
            "plot_df": plot_df,
        }
