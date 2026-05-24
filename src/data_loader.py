import pandas as pd
import numpy as np
import os

class InsuranceDataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.raw_data = None
        self.processed_data = None

    def load_data(self):
        """Ingests raw South African car insurance data safely."""
        print(f"🔄 Ingesting dataset matrix from: {self.file_path}")
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"❌ Critical Error: Data asset missing at {self.file_path}")
        self.raw_data = pd.read_csv(self.file_path, low_memory=False)
        return self.raw_data

    def engineer_base_risk_metrics(self):
        """Calculates derived metrics: Loss Ratio and Net Underwriting Margin."""
        if self.raw_data is None:
            self.load_data()
            
        df = self.raw_data.copy()
        
        # Format key financial vectors to numeric representations safely
        df['TotalPremium'] = pd.to_numeric(df['TotalPremium'], errors='coerce').fillna(0.0)
        df['TotalClaims'] = pd.to_numeric(df['TotalClaims'], errors='coerce').fillna(0.0)
        df['CustomValueEstimate'] = pd.to_numeric(df['CustomValueEstimate'], errors='coerce').fillna(0.0)
        
        # Loss Ratio = TotalClaims / TotalPremium
        df['loss_ratio'] = np.where(df['TotalPremium'] > 0, df['TotalClaims'] / df['TotalPremium'], 0.0)
        
        # Margin = TotalPremium - TotalClaims
        df['margin'] = df['TotalPremium'] - df['TotalClaims']
        
        self.processed_data = df
        return self.processed_data