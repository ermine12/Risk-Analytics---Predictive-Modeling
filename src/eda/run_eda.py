"""Run exploratory data analysis."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import pickle

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import PROCESSED_DATA_DIR, INTERIM_REPORTS_DIR, PLOT_STYLE, FIGURE_SIZE


def load_processed_data() -> pd.DataFrame:
    """Load processed insurance data."""
    filepath = PROCESSED_DATA_DIR / "insurance_cleaned.csv"
    
    if not filepath.exists():
        print(f"Warning: {filepath} not found.")
        return pd.DataFrame()
    
    return pd.read_csv(filepath)


def generate_summary_statistics(df: pd.DataFrame) -> dict:
    """Generate summary statistics."""
    stats = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'numerical_summary': df.describe().to_dict() if len(df) > 0 else {},
        'categorical_summary': {}
    }
    
    # Categorical summary
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        stats['categorical_summary'][col] = df[col].value_counts().to_dict()
    
    return stats


def create_visualizations(df: pd.DataFrame, output_dir: Path):
    """Create EDA visualizations."""
    if df.empty:
        print("No data available for visualization")
        return
    
    plt.style.use('seaborn-v0_8')
    
    # 1. Distribution of charges (target variable)
    if 'charges' in df.columns:
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        df['charges'].hist(bins=50, ax=ax, edgecolor='black')
        ax.set_title('Distribution of Insurance Charges', fontsize=14, fontweight='bold')
        ax.set_xlabel('Charges ($)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        plt.tight_layout()
        plt.savefig(output_dir / 'charges_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Correlation heatmap
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 1:
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        corr_matrix = df[numerical_cols].corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, ax=ax, square=True)
        ax.set_title('Correlation Heatmap of Numerical Features', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Box plots for categorical vs numerical
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0 and 'charges' in df.columns:
        n_cats = len(categorical_cols)
        fig, axes = plt.subplots(1, min(n_cats, 3), figsize=(15, 5))
        if n_cats == 1:
            axes = [axes]
        
        for idx, col in enumerate(categorical_cols[:3]):
            df.boxplot(column='charges', by=col, ax=axes[idx] if n_cats > 1 else axes[0])
            if n_cats > 1:
                axes[idx].set_title(f'Charges by {col}')
            else:
                axes[0].set_title(f'Charges by {col}')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'categorical_boxplots.png', dpi=300, bbox_inches='tight')
        plt.close()


def perform_hypothesis_tests(df: pd.DataFrame) -> dict:
    """Perform hypothesis tests to identify low-risk groups."""
    results = {}
    
    if df.empty:
        return results
    
    from scipy import stats
    
    # Example: Test if smokers have significantly higher charges
    if 'smoker' in df.columns and 'charges' in df.columns:
        smoker_charges = df[df['smoker'] == 'yes']['charges'] if 'yes' in df['smoker'].values else pd.Series()
        non_smoker_charges = df[df['smoker'] == 'no']['charges'] if 'no' in df['smoker'].values else pd.Series()
        
        if len(smoker_charges) > 0 and len(non_smoker_charges) > 0:
            t_stat, p_value = stats.ttest_ind(smoker_charges, non_smoker_charges)
            results['smoker_vs_non_smoker'] = {
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significant': p_value < 0.05,
                'smoker_mean': float(smoker_charges.mean()),
                'non_smoker_mean': float(non_smoker_charges.mean())
            }
    
    return results


def generate_report(df: pd.DataFrame, stats: dict, hypothesis_results: dict, 
                   output_dir: Path):
    """Generate HTML EDA report."""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>EDA Report - Insurance Risk Analytics</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #2c3e50; }}
            h2 {{ color: #34495e; margin-top: 30px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #3498db; color: white; }}
            .stat {{ background-color: #ecf0f1; padding: 15px; margin: 10px 0; }}
        </style>
    </head>
    <body>
        <h1>Exploratory Data Analysis Report</h1>
        
        <h2>Dataset Overview</h2>
        <div class="stat">
            <p><strong>Shape:</strong> {stats['shape']}</p>
            <p><strong>Columns:</strong> {', '.join(stats['columns'])}</p>
        </div>
        
        <h2>Missing Values</h2>
        <table>
            <tr><th>Column</th><th>Missing Count</th></tr>
    """
    
    for col, count in stats['missing_values'].items():
        html_content += f"<tr><td>{col}</td><td>{count}</td></tr>"
    
    html_content += """
        </table>
        
        <h2>Hypothesis Test Results</h2>
    """
    
    for test_name, result in hypothesis_results.items():
        html_content += f"""
        <div class="stat">
            <h3>{test_name.replace('_', ' ').title()}</h3>
            <p><strong>P-value:</strong> {result['p_value']:.4f}</p>
            <p><strong>Significant:</strong> {result['significant']}</p>
        </div>
        """
    
    html_content += """
    </body>
    </html>
    """
    
    output_path = output_dir / "eda_report.html"
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"EDA report saved to {output_path}")


def main():
    """Main EDA pipeline."""
    print("Starting EDA...")
    
    # Load data
    df = load_processed_data()
    
    if df.empty:
        print("No data available for EDA")
        # Create empty report
        generate_report(pd.DataFrame(), {}, {}, INTERIM_REPORTS_DIR)
        return
    
    # Generate statistics
    stats = generate_summary_statistics(df)
    print(f"Dataset shape: {stats['shape']}")
    
    # Create visualizations
    create_visualizations(df, INTERIM_REPORTS_DIR)
    print("Visualizations created")
    
    # Perform hypothesis tests
    hypothesis_results = perform_hypothesis_tests(df)
    print(f"Performed {len(hypothesis_results)} hypothesis tests")
    
    # Save features for modeling
    features_path = PROCESSED_DATA_DIR / "eda_features.pkl"
    with open(features_path, 'wb') as f:
        pickle.dump({'stats': stats, 'hypothesis_results': hypothesis_results}, f)
    print(f"Features saved to {features_path}")
    
    # Generate report
    generate_report(df, stats, hypothesis_results, INTERIM_REPORTS_DIR)
    print("EDA complete!")


if __name__ == "__main__":
    main()

