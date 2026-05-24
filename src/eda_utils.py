import pandas as pd
import numpy as np

class InsuranceEDAAnalyzer:
    def __init__(self, dataframe):
        self.df = dataframe

    def get_summary_statistics(self):
        """Generates descriptive statistics for key financial metrics."""
        cols = ['TotalPremium', 'TotalClaims', 'loss_ratio', 'margin', 'CustomValueEstimate']
        valid_cols = [c for c in cols if c in self.df.columns]
        return self.df[valid_cols].describe().T

    def check_missing_data(self):
        """Tallies missing data counts and percentage distributions."""
        counts = self.df.isnull().sum()
        percentages = (counts / len(self.df)) * 100
        report = pd.DataFrame({'Missing_Count': counts, 'Percentage': percentages})
        return report[report['Missing_Count'] > 0]

    def compute_iqr_outlier_fences(self, column_name):
        """Computes statistical fences using Tukey's method for outlier tracking."""
        if column_name not in self.df.columns:
            return 0.0, 0.0
        q1 = self.df[column_name].quantile(0.25)
        q3 = self.df[column_name].quantile(0.75)
        iqr = q3 - q1
        lower_fence = max(0.0, q1 - (1.5 * iqr))
        upper_fence = q3 + (1.5 * iqr)
        return lower_fence, upper_fence