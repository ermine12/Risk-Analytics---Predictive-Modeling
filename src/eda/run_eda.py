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
from utils.logger import logger


def load_processed_data() -> pd.DataFrame:
    """
    Load processed/cleaned insurance data.
    
    Returns:
        DataFrame with cleaned insurance data. Empty DataFrame if file not found.
        
    Assumptions:
        - Processed data file exists at expected location
        - File is valid CSV format
    """
    filepath = PROCESSED_DATA_DIR / "insurance_cleaned.csv"
    
    if not filepath.exists():
        logger.warning(f"{filepath} not found.")
        return pd.DataFrame()
    
    logger.info(f"Loading processed data from {filepath}")
    return pd.read_csv(filepath)


def generate_summary_statistics(df: pd.DataFrame) -> dict:
    """
    Generate comprehensive summary statistics for the dataset.
    
    Args:
        df: Input DataFrame with insurance data.
        
    Returns:
        Dictionary containing:
        - 'shape': Tuple of (rows, columns)
        - 'columns': List of column names
        - 'dtypes': Dictionary of column -> data type
        - 'missing_values': Dictionary of column -> missing count
        - 'numerical_summary': Dictionary of numerical column statistics
        - 'categorical_summary': Dictionary of categorical value counts
    """
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
    """
    Create comprehensive EDA visualizations.
    
    Generates:
    - Bar charts for top 5 categorical variables
    - Box plots for TotalPremium and TotalClaims (outlier detection)
    - Box plots by Province for claims
    - 3 creative plots: Loss ratio by province, claims distribution, heatmaps
    
    Args:
        df: DataFrame with insurance data.
        output_dir: Path object for saving visualization files.
        
    Assumptions:
        - Output directory exists and is writable
        - Required columns exist in df (handles missing columns gracefully)
    """
    if df.empty:
        logger.warning("No data available for visualization")
        return
    
    plt.style.use('seaborn-v0_8')
    
    # BAR CHARTS for key categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols[:5]:  # Top 5 categorical variables
        if col in df.columns:
            value_counts = df[col].value_counts().head(10)  # Top 10 categories
            if len(value_counts) > 0:
                fig, ax = plt.subplots(figsize=(12, 6))
                bars = ax.bar(range(len(value_counts)), value_counts.values, 
                             color=plt.cm.Set3(np.linspace(0, 1, len(value_counts))))
                ax.set_xlabel(col, fontsize=12, fontweight='bold')
                ax.set_ylabel('Count', fontsize=12, fontweight='bold')
                ax.set_title(f'Distribution of {col}', fontsize=14, fontweight='bold')
                ax.set_xticks(range(len(value_counts)))
                ax.set_xticklabels(value_counts.index, rotation=45, ha='right')
                ax.grid(axis='y', alpha=0.3, linestyle='--')
                
                # Add value labels on bars
                for i, v in enumerate(value_counts.values):
                    ax.text(i, v, str(v), ha='center', va='bottom', fontweight='bold')
                
                plt.tight_layout()
        plt.savefig(output_dir / f'bar_chart_{col}.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Created: Bar chart for {col}")
    
    # BOX PLOTS for critical numeric features to detect outliers
    critical_numeric = ['TotalPremium', 'TotalClaims']
    for col in critical_numeric:
        if col in df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            bp = ax.boxplot(df[col].dropna(), vert=True, patch_artist=True,
                           showmeans=True, meanline=True)
            
            # Color the boxes
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
                patch.set_alpha(0.7)
            
            ax.set_ylabel(f'{col} ($)', fontsize=12, fontweight='bold')
            ax.set_title(f'Box Plot: {col} (Outlier Detection)', 
                        fontsize=14, fontweight='bold')
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            
            # Add statistics text
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            outliers = df[(df[col] < q1 - 1.5*iqr) | (df[col] > q3 + 1.5*iqr)][col]
            
            stats_text = f"Q1: ${q1:,.0f}\nQ3: ${q3:,.0f}\nIQR: ${iqr:,.0f}\nOutliers: {len(outliers)}"
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   fontsize=10)
            
            plt.tight_layout()
            plt.savefig(output_dir / f'boxplot_{col}.png', dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Created: Box plot for {col} (found {len(outliers)} outliers)")
    
    # Box plots by categorical variable (if available)
    if 'Province' in df.columns and 'TotalClaims' in df.columns:
        fig, ax = plt.subplots(figsize=(14, 8))
        provinces = df['Province'].value_counts().head(10).index
        data_to_plot = [df[df['Province'] == prov]['TotalClaims'].dropna() for prov in provinces]
        bp = ax.boxplot(data_to_plot, labels=provinces, patch_artist=True, showmeans=True)
        
        for patch in bp['boxes']:
            patch.set_facecolor('lightcoral')
            patch.set_alpha(0.7)
        
        ax.set_xlabel('Province', fontsize=12, fontweight='bold')
        ax.set_ylabel('Total Claims ($)', fontsize=12, fontweight='bold')
        ax.set_title('Box Plot: Total Claims by Province', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(output_dir / 'boxplot_claims_by_province.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Created: Box plot for Total Claims by Province")
    
    # 1. Loss Ratio by Province (Creative Plot 1)
    if 'Province' in df.columns and 'loss_ratio' in df.columns:
        prov = df.groupby('Province').agg({
            'TotalClaims': 'sum' if 'TotalClaims' in df.columns else 'count',
            'TotalPremium': 'sum' if 'TotalPremium' in df.columns else 'count'
        })
        if 'TotalPremium' in prov.columns and 'TotalClaims' in prov.columns:
            prov['loss_ratio'] = prov['TotalClaims'] / prov['TotalPremium'].replace(0, np.nan)
            prov_sorted = prov.sort_values('loss_ratio')
            
            fig, ax = plt.subplots(figsize=(14, 8))
            colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(prov_sorted)))
            bars = ax.barh(prov_sorted.index, prov_sorted['loss_ratio'], color=colors)
            ax.set_xlabel('Loss Ratio', fontsize=12, fontweight='bold')
            ax.set_ylabel('Province', fontsize=12, fontweight='bold')
            ax.set_title('Loss Ratio by Province (Lower is Better)', 
                        fontsize=16, fontweight='bold', pad=20)
            ax.axvline(x=prov_sorted['loss_ratio'].median(), color='red', 
                      linestyle='--', linewidth=2, label='Median')
            ax.legend()
            plt.tight_layout()
            plt.savefig(output_dir / 'loss_ratio_by_province.png', dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("Created: Loss Ratio by Province plot")
    
    # 2. Distribution of Total Claims with Vehicle Age Overlay (Creative Plot 2)
    if 'TotalClaims' in df.columns:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Left: Distribution of claims
        ax1.hist(df['TotalClaims'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        ax1.set_xlabel('Total Claims ($)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax1.set_title('Distribution of Total Claims', fontsize=14, fontweight='bold')
        ax1.axvline(df['TotalClaims'].mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: ${df["TotalClaims"].mean():,.0f}')
        ax1.legend()
        
        # Right: Claims by Vehicle Age (if available)
        if 'VehicleAge' in df.columns:
            age_bins = pd.cut(df['VehicleAge'], bins=10)
            age_claims = df.groupby(age_bins)['TotalClaims'].mean()
            ax2.bar(range(len(age_claims)), age_claims.values, 
                   color='coral', edgecolor='black')
            ax2.set_xlabel('Vehicle Age (years)', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Average Total Claims ($)', fontsize=12, fontweight='bold')
            ax2.set_title('Average Claims by Vehicle Age', fontsize=14, fontweight='bold')
            ax2.set_xticks(range(len(age_claims)))
            ax2.set_xticklabels([f"{int(interval.left)}-{int(interval.right)}" 
                                for interval in age_claims.index], rotation=45, ha='right')
        else:
            ax2.text(0.5, 0.5, 'Vehicle Age data not available', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Vehicle Age Analysis', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'claims_distribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Created: Claims Distribution Analysis plot")
    
    # 3. Loss Ratio Heatmap by Vehicle Type and Province (Creative Plot 3)
    if 'VehicleType' in df.columns and 'Province' in df.columns and 'loss_ratio' in df.columns:
        pivot_data = df.groupby(['Province', 'VehicleType']).agg({
            'TotalClaims': 'sum' if 'TotalClaims' in df.columns else 'count',
            'TotalPremium': 'sum' if 'TotalPremium' in df.columns else 'count'
        }).reset_index()
        
        if 'TotalPremium' in pivot_data.columns and 'TotalClaims' in pivot_data.columns:
            pivot_data['loss_ratio'] = pivot_data['TotalClaims'] / pivot_data['TotalPremium'].replace(0, np.nan)
            pivot_table = pivot_data.pivot(index='Province', columns='VehicleType', values='loss_ratio')
            
            fig, ax = plt.subplots(figsize=(14, 10))
            sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='RdYlGn_r', 
                       center=pivot_table.values[~np.isnan(pivot_table.values)].mean(),
                       ax=ax, cbar_kws={'label': 'Loss Ratio'}, linewidths=0.5)
            ax.set_title('Loss Ratio Heatmap: Province × Vehicle Type', 
                        fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('Vehicle Type', fontsize=12, fontweight='bold')
            ax.set_ylabel('Province', fontsize=12, fontweight='bold')
            plt.tight_layout()
            plt.savefig(output_dir / 'loss_ratio_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("Created: Loss Ratio Heatmap plot")
    
    # 4. Correlation heatmap (bonus)
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 1:
        fig, ax = plt.subplots(figsize=(12, 10))
        corr_matrix = df[numerical_cols].corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, ax=ax, square=True, linewidths=0.5)
        ax.set_title('Correlation Heatmap of Numerical Features', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()


def perform_hypothesis_tests(df: pd.DataFrame) -> dict:
    """
    Perform statistical hypothesis tests to identify low-risk groups.
    
    Tests include loss ratio comparisons by Province and VehicleType,
    and vehicle age impact analysis.
    
    Args:
        df: DataFrame with insurance data including loss_ratio column.
        
    Returns:
        Dictionary of test_name -> {test_statistic, p_value, significant, ...}
        
    Assumptions:
        - Loss ratio column exists (calculated as TotalClaims/TotalPremium)
        - Sufficient sample sizes for statistical tests
    """
    results = {}
    
    if df.empty:
        return results
    
    from scipy import stats
    
    # Test 1: Loss ratio by Province
    if 'Province' in df.columns and 'loss_ratio' in df.columns:
        prov = df.groupby('Province').agg({
            'TotalClaims': 'sum' if 'TotalClaims' in df.columns else 'count',
            'TotalPremium': 'sum' if 'TotalPremium' in df.columns else 'count'
        })
        if 'TotalPremium' in prov.columns and 'TotalClaims' in prov.columns:
            prov['loss_ratio'] = prov['TotalClaims'] / prov['TotalPremium'].replace(0, np.nan)
            prov_sorted = prov.sort_values('loss_ratio')
            
            # Test if lowest province has significantly lower loss ratio
            if len(prov_sorted) >= 2:
                lowest_prov = prov_sorted.index[0]
                highest_prov = prov_sorted.index[-1]
                
                lowest_data = df[df['Province'] == lowest_prov]['loss_ratio'].dropna()
                highest_data = df[df['Province'] == highest_prov]['loss_ratio'].dropna()
                
                if len(lowest_data) > 0 and len(highest_data) > 0:
                    t_stat, p_value = stats.ttest_ind(lowest_data, highest_data)
                    results['province_loss_ratio_comparison'] = {
                        'lowest_province': lowest_prov,
                        'highest_province': highest_prov,
                        't_statistic': float(t_stat),
                        'p_value': float(p_value),
                        'significant': p_value < 0.05,
                        'lowest_mean_lr': float(lowest_data.mean()),
                        'highest_mean_lr': float(highest_data.mean())
                    }
    
    # Test 2: Loss ratio by Vehicle Type
    if 'VehicleType' in df.columns and 'loss_ratio' in df.columns:
        vehicle_groups = df.groupby('VehicleType')['loss_ratio'].agg(['mean', 'std', 'count'])
        vehicle_groups = vehicle_groups.sort_values('mean')
        
        if len(vehicle_groups) >= 2:
            lowest_vehicle = vehicle_groups.index[0]
            highest_vehicle = vehicle_groups.index[-1]
            
            lowest_data = df[df['VehicleType'] == lowest_vehicle]['loss_ratio'].dropna()
            highest_data = df[df['VehicleType'] == highest_vehicle]['loss_ratio'].dropna()
            
            if len(lowest_data) > 0 and len(highest_data) > 0:
                t_stat, p_value = stats.ttest_ind(lowest_data, highest_data)
                results['vehicle_type_loss_ratio_comparison'] = {
                    'lowest_vehicle': lowest_vehicle,
                    'highest_vehicle': highest_vehicle,
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05,
                    'lowest_mean_lr': float(lowest_data.mean()),
                    'highest_mean_lr': float(highest_data.mean())
                }
    
    # Test 3: Vehicle Age impact on claims
    if 'VehicleAge' in df.columns and 'TotalClaims' in df.columns:
        # Split into old vs new vehicles
        median_age = df['VehicleAge'].median()
        old_vehicles = df[df['VehicleAge'] > median_age]['TotalClaims']
        new_vehicles = df[df['VehicleAge'] <= median_age]['TotalClaims']
        
        if len(old_vehicles) > 0 and len(new_vehicles) > 0:
            t_stat, p_value = stats.ttest_ind(old_vehicles, new_vehicles)
            results['vehicle_age_impact'] = {
                'median_age': float(median_age),
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significant': p_value < 0.05,
                'old_vehicles_mean': float(old_vehicles.mean()),
                'new_vehicles_mean': float(new_vehicles.mean())
            }
    
    return results


def identify_low_risk_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify low-risk customer groups based on loss ratio analysis.
    
    Groups policies by Province and Province+VehicleType combinations,
    calculates loss ratios, and returns top 20 lowest-risk groups.
    
    Args:
        df: DataFrame with insurance data including loss_ratio column.
        
    Returns:
        DataFrame with columns: Group, LossRatio, SampleSize, TotalClaims, TotalPremium
        Sorted by LossRatio (ascending - lowest risk first).
        
    Assumptions:
        - Loss ratio column exists
        - Province and VehicleType columns exist (if grouping by them)
    """
    if df.empty or 'loss_ratio' not in df.columns:
        logger.warning("Cannot identify low-risk groups: missing data or loss_ratio column")
        return pd.DataFrame()
    
    low_risk_groups = []
    
    # Group by Province
    if 'Province' in df.columns:
        prov = df.groupby('Province').agg({
            'TotalClaims': 'sum' if 'TotalClaims' in df.columns else 'count',
            'TotalPremium': 'sum' if 'TotalPremium' in df.columns else 'count',
            'PolicyID': 'nunique' if 'PolicyID' in df.columns else 'count'
        })
        if 'TotalPremium' in prov.columns and 'TotalClaims' in prov.columns:
            prov['loss_ratio'] = prov['TotalClaims'] / prov['TotalPremium'].replace(0, np.nan)
            prov['sample_size'] = prov['PolicyID'] if 'PolicyID' in prov.columns else prov.index.map(lambda x: len(df[df['Province'] == x]))
            
            for province in prov.index:
                if not pd.isna(prov.loc[province, 'loss_ratio']):
                    low_risk_groups.append({
                        'Group': f"Province: {province}",
                        'LossRatio': prov.loc[province, 'loss_ratio'],
                        'SampleSize': int(prov.loc[province, 'sample_size']),
                        'TotalClaims': float(prov.loc[province, 'TotalClaims']),
                        'TotalPremium': float(prov.loc[province, 'TotalPremium'])
                    })
    
    # Group by Province + VehicleType
    if 'Province' in df.columns and 'VehicleType' in df.columns:
        combo = df.groupby(['Province', 'VehicleType']).agg({
            'TotalClaims': 'sum' if 'TotalClaims' in df.columns else 'count',
            'TotalPremium': 'sum' if 'TotalPremium' in df.columns else 'count',
            'PolicyID': 'nunique' if 'PolicyID' in df.columns else 'count'
        })
        if 'TotalPremium' in combo.columns and 'TotalClaims' in combo.columns:
            combo['loss_ratio'] = combo['TotalClaims'] / combo['TotalPremium'].replace(0, np.nan)
            combo['sample_size'] = combo['PolicyID'] if 'PolicyID' in combo.columns else combo.index.map(lambda x: len(df[(df['Province'] == x[0]) & (df['VehicleType'] == x[1])]))
            
            for (province, vehicle_type) in combo.index:
                if not pd.isna(combo.loc[(province, vehicle_type), 'loss_ratio']):
                    low_risk_groups.append({
                        'Group': f"Province: {province}, VehicleType: {vehicle_type}",
                        'LossRatio': combo.loc[(province, vehicle_type), 'loss_ratio'],
                        'SampleSize': int(combo.loc[(province, vehicle_type), 'sample_size']),
                        'TotalClaims': float(combo.loc[(province, vehicle_type), 'TotalClaims']),
                        'TotalPremium': float(combo.loc[(province, vehicle_type), 'TotalPremium'])
                    })
    
    low_risk_df = pd.DataFrame(low_risk_groups)
    if not low_risk_df.empty:
        low_risk_df = low_risk_df.sort_values('LossRatio').head(20)  # Top 20 low-risk groups
    
    return low_risk_df


def generate_report(df: pd.DataFrame, stats: dict, hypothesis_results: dict, 
                   output_dir: Path):
    """
    Generate HTML EDA report with statistics and hypothesis test results.
    
    Args:
        df: DataFrame with insurance data.
        stats: Dictionary of summary statistics.
        hypothesis_results: Dictionary of hypothesis test results.
        output_dir: Path object for saving report file.
        
    Returns:
        None. Saves HTML report to output_dir/eda_report.html.
        
    Assumptions:
        - Output directory exists and is writable
        - Stats and hypothesis_results dictionaries are properly formatted
    """
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
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"EDA report saved to {output_path}")


def main():
    """
    Main EDA pipeline execution.
    
    Orchestrates data loading, statistics generation, visualization creation,
    hypothesis testing, and report generation.
    """
    logger.info("=" * 60)
    logger.info("Starting EDA Pipeline")
    logger.info("=" * 60)
    
    # Load data
    df = load_processed_data()
    
    if df.empty:
        logger.warning("No data available for EDA")
        # Create empty report
        generate_report(pd.DataFrame(), {}, {}, INTERIM_REPORTS_DIR)
        return
    
    # Generate statistics
    stats = generate_summary_statistics(df)
    logger.info(f"Dataset shape: {stats['shape']}")
    
    # Create visualizations
    create_visualizations(df, INTERIM_REPORTS_DIR)
    logger.info("Visualizations created")
    
    # Perform hypothesis tests
    hypothesis_results = perform_hypothesis_tests(df)
    logger.info(f"Performed {len(hypothesis_results)} hypothesis tests")
    
    # Identify low-risk groups
    low_risk_groups = identify_low_risk_groups(df)
    if not low_risk_groups.empty:
        logger.info(f"Identified {len(low_risk_groups)} low-risk groups")
        low_risk_path = INTERIM_REPORTS_DIR / "low_risk_groups.csv"
        low_risk_groups.to_csv(low_risk_path, index=False)
        logger.info(f"Low-risk groups saved to {low_risk_path}")
    
    # Save features for modeling
    features_path = PROCESSED_DATA_DIR / "eda_features.pkl"
    with open(features_path, 'wb') as f:
        pickle.dump({
            'stats': stats, 
            'hypothesis_results': hypothesis_results,
            'low_risk_groups': low_risk_groups.to_dict('records') if not low_risk_groups.empty else []
        }, f)
    logger.info(f"Features saved to {features_path}")
    
    # Generate report
    generate_report(df, stats, hypothesis_results, INTERIM_REPORTS_DIR)
    logger.info("EDA complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

