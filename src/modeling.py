import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error, r2_score

class InsurancePricingModeler:
    def __init__(self, dataframe):
        self.df = dataframe.copy()
        
    def engineer_features(self):
        """Prepares categorical variables via numeric encoding for modeling."""
        cols_to_encode = ['Province', 'Gender', 'VehicleType']
        X = self.df[cols_to_encode + ['CustomValueEstimate']].copy()
        X = pd.get_dummies(X, columns=cols_to_encode, drop_first=True)
        return X

    def train_and_evaluate(self):
        """Trains and compares Random Forest and Gradient Boosting pricing models."""
        X = self.engineer_features()
        y = np.where(self.df['TotalClaims'] > 0, self.df['TotalClaims'], 0.0)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 1. Random Forest Regressor Optimization
        rf = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
        rf.fit(X_train, y_train)
        rf_preds = rf.predict(X_test)
        
        # 2. Gradient Boosting (XGBoost Operational Proxy)
        gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        gb.fit(X_train, y_train)
        gb_preds = gb.predict(X_test)
        
        print("\n🏆 MACHINE LEARNING MODEL PRICING BENCHMARKS:")
        print(f"🌲 Random Forest RMSE: {root_mean_squared_error(y_test, rf_preds):.2f} | R2: {r2_score(y_test, rf_preds):.4f}")
        print(f"⚡ Gradient Boosting RMSE: {root_mean_squared_error(y_test, gb_preds):.2f} | R2: {r2_score(y_test, gb_preds):.4f}")
        
        return X.columns, gb.feature_importances_