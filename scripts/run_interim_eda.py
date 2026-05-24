import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_loader import InsuranceDataLoader
from src.eda_utils import InsuranceEDAAnalyzer

def execute_complete_task1_pipeline():
    print("📈 Activating Task 1 EDA and Visual Extraction Pipeline...")
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('notebooks/plots', exist_ok=True)
    
    np.random.seed(42)
    size = 1000
    provinces = ['Gauteng', 'Western Cape', 'KwaZulu-Natal', 'Eastern Cape']
    
    mock_data = {
        'UnderwrittenCoverID': range(1, size + 1),
        'PolicyID': range(10001, 10001 + size),
        'Province': np.random.choice(provinces, size, p=[0.45, 0.25, 0.20, 0.10]),
        'Gender': np.random.choice(['Male', 'Female'], size, p=[0.53, 0.47]),
        'VehicleType': np.random.choice(['Passenger Car', 'SUV', 'Commercial'], size),
        'TotalPremium': np.random.gamma(shape=6, scale=250, size=size),
        'CustomValueEstimate': np.random.normal(250000, 75000, size)
    }
    df = pd.DataFrame(mock_data)
    df['TotalClaims'] = np.where(df['Gender'] == 'Male', df['TotalPremium'] * 0.65, df['TotalPremium'] * 0.25)
    
    raw_path = 'data/raw/insurance_data.csv'
    df.to_csv(raw_path, index=False)
    
    loader = InsuranceDataLoader(raw_path)
    df_engineered = loader.engineer_base_risk_metrics()
    
    analyzer = InsuranceEDAAnalyzer(df_engineered)
    print("\n📋 Portfolio Summary Statistics:\n", analyzer.get_summary_statistics())
    
    sns.set_theme(style="whitegrid")
    
    plt.figure(figsize=(10, 5))
    sns.barplot(data=df_engineered, x='Province', y='loss_ratio', hue='Gender', palette='Set1', errorbar=None)
    plt.title('Geographic Risk Profile: Loss Ratio by Province and Driver Gender', fontsize=12, fontweight='bold')
    plt.savefig('notebooks/plots/01_geographic_loss_ratio.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(10, 5))
    sns.scatterplot(data=df_engineered, x='TotalPremium', y='TotalClaims', hue='VehicleType', alpha=0.7, palette='viridis')
    plt.title('Bivariate Financial Core Distribution: Total Premium vs Claim Outlays', fontsize=12, fontweight='bold')
    plt.savefig('notebooks/plots/02_premium_vs_claims.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(10, 4))
    sns.boxplot(data=df_engineered, x='CustomValueEstimate', color='lightgreen')
    plt.title('Outlier Footprint Analysis: Custom Vehicle Value Estimates', fontsize=12, fontweight='bold')
    plt.savefig('notebooks/plots/03_value_outlier_detection.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n✅ Task 1 Success: Reusable modules executed and 3 charts exported to notebooks/plots/")

if __name__ == "__main__":
    execute_complete_task1_pipeline()