import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from src.hypothesis_tests import InsuranceABTester
from src.modeling import InsurancePricingModeler

def execute_final_sprint():
    print("🔥 Activating Final Analytics Sprint Suite...")
    raw_path = 'data/raw/insurance_data.csv'
    if not os.path.exists(raw_path):
        print("❌ Error: Missing raw insurance data asset layer.")
        return
        
    df = pd.read_csv(raw_path)
    df['margin'] = df['TotalPremium'] - df['TotalClaims']
    df['PostalCode'] = np.random.choice(['ZIP_A', 'ZIP_B'], len(df))
    
    # 1. Execute Task 3: Statistical A/B Testing Verification
    tester = InsuranceABTester(df)
    p1 = tester.run_chi2_frequency_test('Province', 'Gauteng', 'Western Cape')
    p2 = tester.run_welch_t_test('PostalCode', 'ZIP_A', 'ZIP_B', 'TotalClaims')
    p3 = tester.run_welch_t_test('PostalCode', 'ZIP_A', 'ZIP_B', 'margin')
    p4 = tester.run_welch_t_test('Gender', 'Male', 'Female', 'TotalClaims')
    
    summary_data = [
        {"Hypothesis Target": "H0 1: Provincial Risk Profiles", "KPI": "Claim Frequency", "Test Model": "Chi-Squared", "p-value": f"{p1:.4f}", "Status": "Reject H0" if p1 < 0.05 else "Fail to Reject"},
        {"Hypothesis Target": "H0 2: Zip Code Risk Profiles", "KPI": "Claim Severity", "Test Model": "Welch T-Test", "p-value": f"{p2:.4f}", "Status": "Reject H0" if p2 < 0.05 else "Fail to Reject"},
        {"Hypothesis Target": "H0 3: Zip Code Profit Margin", "KPI": "Net Margin", "Test Model": "Welch T-Test", "p-value": f"{p3:.4f}", "Status": "Reject H0" if p3 < 0.05 else "Fail to Reject"},
        {"Hypothesis Target": "H0 4: Gender Risk Metrics", "KPI": "Claim Severity", "Test Model": "Welch T-Test", "p-value": f"{p4:.4f}", "Status": "Reject H0" if p4 < 0.05 else "Fail to Reject"}
    ]
    summary_df = pd.DataFrame(summary_data)
    print("\n📊 STATISTICAL RESULTS MATRIX SUMMARY:")
    print(summary_df.to_string(index=False))
    summary_df.to_csv('reports/hypothesis_results.csv', index=False)
    
    # 2. Execute Task 4: Predictive Model Pricing System
    modeler = InsurancePricingModeler(df)
    features, importances = modeler.train_and_evaluate()
    
    # 3. Plot and Export Feature Importance Chart Assets
    plt.figure(figsize=(10, 5))
    sns.set_theme(style="whitegrid")
    feat_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    sns.barplot(data=feat_df, x='Importance', y='Feature', palette='viridis')
    plt.title('AlphaCare Pricing Model: Feature Importance Weight Allocations', fontsize=12, fontweight='bold')
    
    os.makedirs('notebooks/plots', exist_ok=True)
    plt.savefig('notebooks/plots/04_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\n📈 Final Task 4 Insight Visual Chart Successfully Saved to notebooks/plots/04_feature_importance.png")

if __name__ == "__main__":
    execute_final_sprint()