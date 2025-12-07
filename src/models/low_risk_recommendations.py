"""Generate low-risk group recommendations with financial impact simulation."""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import INTERIM_REPORTS_DIR, FINAL_REPORTS_DIR


def load_low_risk_groups() -> pd.DataFrame:
    """Load identified low-risk groups from EDA."""
    low_risk_path = INTERIM_REPORTS_DIR / "low_risk_groups.csv"
    
    if not low_risk_path.exists():
        print(f"Warning: {low_risk_path} not found")
        return pd.DataFrame()
    
    return pd.read_csv(low_risk_path)


def calculate_confidence_interval(loss_ratio: float, sample_size: int, confidence: float = 0.95) -> tuple:
    """Calculate confidence interval for loss ratio using normal approximation."""
    if sample_size == 0:
        return (loss_ratio, loss_ratio)
    
    # Standard error for proportion (simplified)
    se = np.sqrt(loss_ratio * (1 - loss_ratio) / sample_size) if loss_ratio < 1 else loss_ratio / np.sqrt(sample_size)
    
    # Z-score for confidence level
    z_score = 1.96 if confidence == 0.95 else 2.576  # 95% or 99%
    
    lower = max(0, loss_ratio - z_score * se)
    upper = loss_ratio + z_score * se
    
    return (lower, upper)


def simulate_premium_adjustment(
    current_premium: float,
    current_loss_ratio: float,
    premium_reduction_pct: float,
    target_loss_ratio: float = 0.7,
    conversion_increase_pct: float = None
) -> dict:
    """Simulate financial impact of premium adjustment."""
    
    # Default conversion increase based on premium reduction (elasticity assumption)
    if conversion_increase_pct is None:
        # Assume 0.5% conversion increase per 1% premium reduction
        conversion_increase_pct = premium_reduction_pct * 0.5
    
    new_premium = current_premium * (1 - premium_reduction_pct / 100)
    
    # Projected metrics
    projected_policies = 1 + (conversion_increase_pct / 100)  # Relative to current
    projected_premium_volume = projected_policies * new_premium
    current_premium_volume = current_premium
    
    # Assume claims frequency stays same, but total claims increase with more policies
    projected_total_claims = current_loss_ratio * current_premium_volume * projected_policies
    
    # Projected loss ratio
    if projected_premium_volume > 0:
        projected_loss_ratio = projected_total_claims / projected_premium_volume
    else:
        projected_loss_ratio = current_loss_ratio
    
    # Profit calculation (simplified: Premium - Claims - Expenses)
    # Assume 20% expense ratio
    expense_ratio = 0.20
    current_profit = current_premium_volume * (1 - current_loss_ratio - expense_ratio)
    projected_profit = projected_premium_volume * (1 - projected_loss_ratio - expense_ratio)
    profit_change = projected_profit - current_profit
    profit_change_pct = (profit_change / current_profit * 100) if current_profit > 0 else 0
    
    return {
        'current_premium': float(current_premium),
        'new_premium': float(new_premium),
        'premium_reduction_pct': float(premium_reduction_pct),
        'conversion_increase_pct': float(conversion_increase_pct),
        'current_loss_ratio': float(current_loss_ratio),
        'projected_loss_ratio': float(projected_loss_ratio),
        'meets_target': projected_loss_ratio <= target_loss_ratio,
        'current_profit': float(current_profit),
        'projected_profit': float(projected_profit),
        'profit_change': float(profit_change),
        'profit_change_pct': float(profit_change_pct)
    }


def generate_recommendations(low_risk_groups: pd.DataFrame, target_loss_ratio: float = 0.7) -> pd.DataFrame:
    """Generate recommendations for low-risk groups."""
    recommendations = []
    
    for idx, row in low_risk_groups.iterrows():
        group_def = row['Group']
        loss_ratio = row['LossRatio']
        sample_size = row['SampleSize']
        total_premium = row['TotalPremium']
        total_claims = row['TotalClaims']
        
        # Calculate confidence interval
        ci_lower, ci_upper = calculate_confidence_interval(loss_ratio, sample_size)
        
        # Average premium per policy
        avg_premium = total_premium / sample_size if sample_size > 0 else 0
        
        # Test different premium reduction scenarios
        scenarios = [5, 10, 15, 20]  # Percentage reductions
        
        best_scenario = None
        best_profit_change = -np.inf
        
        for reduction_pct in scenarios:
            simulation = simulate_premium_adjustment(
                current_premium=avg_premium,
                current_loss_ratio=loss_ratio,
                premium_reduction_pct=reduction_pct,
                target_loss_ratio=target_loss_ratio
            )
            
            if simulation['meets_target'] and simulation['profit_change'] > best_profit_change:
                best_profit_change = simulation['profit_change']
                best_scenario = {
                    'premium_reduction_pct': reduction_pct,
                    **simulation
                }
        
        if best_scenario:
            recommendations.append({
                'Group': group_def,
                'HistoricalLossRatio': loss_ratio,
                'LossRatio_CI_Lower': ci_lower,
                'LossRatio_CI_Upper': ci_upper,
                'SampleSize': sample_size,
                'TotalPolicies': sample_size,
                'CurrentAvgPremium': avg_premium,
                'RecommendedPremiumReduction': best_scenario['premium_reduction_pct'],
                'ProjectedLossRatio': best_scenario['projected_loss_ratio'],
                'MeetsTarget': best_scenario['meets_target'],
                'ProjectedConversionIncrease': best_scenario['conversion_increase_pct'],
                'ProjectedProfitChange': best_scenario['profit_change'],
                'ProjectedProfitChangePct': best_scenario['profit_change_pct']
            })
    
    return pd.DataFrame(recommendations)


def main():
    """Main recommendation pipeline."""
    print("Generating low-risk group recommendations...")
    
    # Load low-risk groups
    low_risk_groups = load_low_risk_groups()
    
    if low_risk_groups.empty:
        print("No low-risk groups found. Run EDA first.")
        return
    
    print(f"Found {len(low_risk_groups)} low-risk groups")
    
    # Generate recommendations
    recommendations = generate_recommendations(low_risk_groups, target_loss_ratio=0.7)
    
    if recommendations.empty:
        print("No recommendations generated")
        return
    
    # Save recommendations
    recommendations_path = FINAL_REPORTS_DIR / "low_risk_recommendations.csv"
    recommendations.to_csv(recommendations_path, index=False)
    print(f"Recommendations saved to {recommendations_path}")
    
    # Generate summary report
    summary = {
        'total_groups_analyzed': len(low_risk_groups),
        'recommendations_generated': len(recommendations),
        'groups_meeting_target': int(recommendations['MeetsTarget'].sum()),
        'total_projected_profit_change': float(recommendations['ProjectedProfitChange'].sum()),
        'average_premium_reduction': float(recommendations['RecommendedPremiumReduction'].mean()),
        'top_5_recommendations': recommendations.nlargest(5, 'ProjectedProfitChange')[
            ['Group', 'RecommendedPremiumReduction', 'ProjectedProfitChange']
        ].to_dict('records')
    }
    
    summary_path = FINAL_REPORTS_DIR / "recommendations_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nRecommendations Summary:")
    print(f"  Total groups analyzed: {summary['total_groups_analyzed']}")
    print(f"  Recommendations generated: {summary['recommendations_generated']}")
    print(f"  Groups meeting target: {summary['groups_meeting_target']}")
    print(f"  Total projected profit change: ${summary['total_projected_profit_change']:,.2f}")
    print(f"\nTop 5 Recommendations:")
    for i, rec in enumerate(summary['top_5_recommendations'], 1):
        print(f"  {i}. {rec['Group']}")
        print(f"     Premium reduction: {rec['RecommendedPremiumReduction']}%")
        print(f"     Projected profit change: ${rec['ProjectedProfitChange']:,.2f}")


if __name__ == "__main__":
    main()

