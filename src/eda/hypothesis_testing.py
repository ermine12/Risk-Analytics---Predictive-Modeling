"""A/B Hypothesis Testing for Risk Drivers.

Tests key hypotheses about risk drivers to form segmentation strategy.
Metrics: Claim Frequency and Claim Severity
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import sys
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import PROCESSED_DATA_DIR, INTERIM_REPORTS_DIR
from utils.logger import logger


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to handle different data formats."""
    df = df.copy()
    column_mapping = {
        'region': 'Province',
        'Region': 'Province',
        'charges': 'TotalClaims',
        'Charges': 'TotalClaims',
        'sex': 'Gender',
        'Sex': 'Gender',
        'postalcode': 'PostalCode',
        'PostalCode': 'PostalCode'
    }
    
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns and new_col not in df.columns:
            df[new_col] = df[old_col]
    
    # For standard insurance dataset: treat charges as claims, estimate premium
    if 'TotalClaims' in df.columns and 'TotalPremium' not in df.columns:
        # Estimate premium as charges * 1.2 (assuming 20% margin)
        df['TotalPremium'] = df['TotalClaims'] * 1.2
    
    return df


def calculate_claim_frequency(df: pd.DataFrame) -> pd.Series:
    """Calculate claim frequency: proportion of policies with at least one claim."""
    # For standard dataset, all policies have charges, so we use a threshold
    if 'TotalClaims' in df.columns:
        # Use median as threshold to create binary classification
        threshold = df['TotalClaims'].median()
        return (df['TotalClaims'] > threshold).astype(int)
    return pd.Series()


def calculate_claim_severity(df: pd.DataFrame) -> pd.Series:
    """Calculate claim severity: average amount of a claim, given a claim occurred."""
    if 'TotalClaims' in df.columns:
        # For standard dataset, use median threshold
        threshold = df['TotalClaims'].median()
        return df[df['TotalClaims'] > threshold]['TotalClaims']
    return pd.Series()


def calculate_margin(df: pd.DataFrame) -> pd.Series:
    """Calculate margin: TotalPremium - TotalClaims."""
    if 'TotalPremium' in df.columns and 'TotalClaims' in df.columns:
        return df['TotalPremium'] - df['TotalClaims']
    return pd.Series()


def test_province_risk_differences(df: pd.DataFrame) -> dict:
    """
    H₀: There are no risk differences across provinces/regions.
    
    Tests:
    - Claim Frequency (chi-squared test)
    - Claim Severity (ANOVA + pairwise t-tests)
    """
    logger.info("Testing H0: No risk differences across provinces/regions")
    results = {}
    
    # Normalize column names
    df = normalize_column_names(df)
    
    if 'Province' not in df.columns:
        logger.warning("Province/Region column not found")
        return results
    
    # Calculate metrics
    df['has_claim'] = calculate_claim_frequency(df)
    
    # Test 1: Claim Frequency (chi-squared test)
    contingency_table = pd.crosstab(df['Province'], df['has_claim'])
    if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
        chi2, p_value_freq, dof, expected = stats.chi2_contingency(contingency_table)
        results['claim_frequency'] = {
            'test': 'chi-squared',
            'chi2_statistic': float(chi2),
            'p_value': float(p_value_freq),
            'degrees_of_freedom': int(dof),
            'reject_null': bool(p_value_freq < 0.05),
            'interpretation': 'Reject H0' if p_value_freq < 0.05 else 'Fail to reject H0'
        }
        logger.info(f"Claim Frequency chi-squared test: p={p_value_freq:.4f}, reject={p_value_freq < 0.05}")
    
    # Test 2: Claim Severity (ANOVA)
    severity_by_province = []
    province_names = []
    for province in df['Province'].unique():
        province_data = df[(df['Province'] == province) & (df['TotalClaims'] > 0)]['TotalClaims']
        if len(province_data) > 0:
            severity_by_province.append(province_data)
            province_names.append(province)
    
    if len(severity_by_province) >= 2:
        f_stat, p_value_sev = stats.f_oneway(*severity_by_province)
        results['claim_severity'] = {
            'test': 'ANOVA',
            'f_statistic': float(f_stat),
            'p_value': float(p_value_sev),
            'reject_null': bool(p_value_sev < 0.05),
            'interpretation': 'Reject H0' if p_value_sev < 0.05 else 'Fail to reject H0'
        }
        logger.info(f"Claim Severity ANOVA: p={p_value_sev:.4f}, reject={p_value_sev < 0.05}")
        
        # Pairwise comparisons
        if p_value_sev < 0.05 and len(severity_by_province) >= 2:
            pairwise_results = []
            province_means = {prov: data.mean() for prov, data in zip(province_names, severity_by_province)}
            sorted_provinces = sorted(province_means.items(), key=lambda x: x[1])
            
            lowest_prov, lowest_mean = sorted_provinces[0]
            highest_prov, highest_mean = sorted_provinces[-1]
            
            lowest_data = df[(df['Province'] == lowest_prov) & (df['TotalClaims'] > 0)]['TotalClaims']
            highest_data = df[(df['Province'] == highest_prov) & (df['TotalClaims'] > 0)]['TotalClaims']
            
            t_stat, p_pairwise = stats.ttest_ind(lowest_data, highest_data)
            pairwise_results.append({
                'comparison': f"{lowest_prov} vs {highest_prov}",
                'lowest_mean': float(lowest_mean),
                'highest_mean': float(highest_mean),
                'difference_pct': float((highest_mean - lowest_mean) / lowest_mean * 100),
                't_statistic': float(t_stat),
                'p_value': float(p_pairwise),
                'significant': bool(p_pairwise < 0.05)
            })
            
            results['claim_severity']['pairwise_comparisons'] = pairwise_results
    
    return results


def test_zipcode_risk_differences(df: pd.DataFrame) -> dict:
    """
    H₀: There are no risk differences between zip codes.
    
    Tests:
    - Claim Frequency (chi-squared test on top zip codes)
    - Claim Severity (ANOVA)
    """
    logger.info("Testing H0: No risk differences between zip codes")
    results = {}
    
    # Normalize column names
    df = normalize_column_names(df)
    
    if 'PostalCode' not in df.columns:
        logger.warning("PostalCode column not found - skipping zipcode tests")
        results['note'] = "PostalCode column not available in dataset"
        return results
    
    # Focus on zip codes with sufficient sample size (at least 10 policies)
    zipcode_counts = df['PostalCode'].value_counts()
    valid_zips = zipcode_counts[zipcode_counts >= 10].index[:20]  # Top 20 by sample size
    df_filtered = df[df['PostalCode'].isin(valid_zips)]
    
    if len(valid_zips) < 2:
        logger.warning("Insufficient zip codes with adequate sample size")
        return results
    
    df_filtered['has_claim'] = calculate_claim_frequency(df_filtered)
    
    # Test 1: Claim Frequency
    contingency_table = pd.crosstab(df_filtered['PostalCode'], df_filtered['has_claim'])
    if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
        chi2, p_value_freq, dof, expected = stats.chi2_contingency(contingency_table)
        results['claim_frequency'] = {
            'test': 'chi-squared',
            'chi2_statistic': float(chi2),
            'p_value': float(p_value_freq),
            'degrees_of_freedom': int(dof),
            'reject_null': bool(p_value_freq < 0.05),
            'interpretation': 'Reject H0' if p_value_freq < 0.05 else 'Fail to reject H0',
            'zipcodes_tested': len(valid_zips)
        }
        logger.info(f"Zipcode Claim Frequency test: p={p_value_freq:.4f}, reject={p_value_freq < 0.05}")
    
    # Test 2: Claim Severity
    severity_by_zip = []
    zip_names = []
    for zipcode in valid_zips:
        zip_data = df_filtered[(df_filtered['PostalCode'] == zipcode) & (df_filtered['TotalClaims'] > 0)]['TotalClaims']
        if len(zip_data) > 0:
            severity_by_zip.append(zip_data)
            zip_names.append(zipcode)
    
    if len(severity_by_zip) >= 2:
        f_stat, p_value_sev = stats.f_oneway(*severity_by_zip)
        results['claim_severity'] = {
            'test': 'ANOVA',
            'f_statistic': float(f_stat),
            'p_value': float(p_value_sev),
            'reject_null': bool(p_value_sev < 0.05),
            'interpretation': 'Reject H0' if p_value_sev < 0.05 else 'Fail to reject H0'
        }
        logger.info(f"Zipcode Claim Severity ANOVA: p={p_value_sev:.4f}, reject={p_value_sev < 0.05}")
    
    return results


def test_zipcode_margin_differences(df: pd.DataFrame) -> dict:
    """
    H₀: There is no significant margin (profit) difference between zip codes.
    
    Test: Margin (TotalPremium - TotalClaims) by zip code (ANOVA)
    """
    logger.info("Testing H0: No margin differences between zip codes")
    results = {}
    
    # Normalize column names
    df = normalize_column_names(df)
    
    if 'PostalCode' not in df.columns or 'TotalPremium' not in df.columns or 'TotalClaims' not in df.columns:
        logger.warning("Required columns not found for margin test - skipping")
        results['note'] = "PostalCode or premium/claims columns not available"
        return results
    
    df['margin'] = calculate_margin(df)
    
    # Focus on zip codes with sufficient sample size
    zipcode_counts = df['PostalCode'].value_counts()
    valid_zips = zipcode_counts[zipcode_counts >= 10].index[:20]
    df_filtered = df[df['PostalCode'].isin(valid_zips)]
    
    if len(valid_zips) < 2:
        logger.warning("Insufficient zip codes with adequate sample size")
        return results
    
    # ANOVA test
    margin_by_zip = [df_filtered[df_filtered['PostalCode'] == zipcode]['margin'] for zipcode in valid_zips]
    margin_by_zip = [m for m in margin_by_zip if len(m) > 0]
    
    if len(margin_by_zip) >= 2:
        f_stat, p_value = stats.f_oneway(*margin_by_zip)
        
        # Calculate means for interpretation
        zip_means = {zipcode: df_filtered[df_filtered['PostalCode'] == zipcode]['margin'].mean() 
                    for zipcode in valid_zips}
        sorted_zips = sorted(zip_means.items(), key=lambda x: x[1])
        
        results = {
            'test': 'ANOVA',
            'f_statistic': float(f_stat),
            'p_value': float(p_value),
            'reject_null': bool(p_value < 0.05),
            'interpretation': 'Reject H0' if p_value < 0.05 else 'Fail to reject H0',
            'zipcodes_tested': len(valid_zips),
            'lowest_margin_zip': sorted_zips[0][0] if sorted_zips else None,
            'highest_margin_zip': sorted_zips[-1][0] if sorted_zips else None,
            'lowest_margin': float(sorted_zips[0][1]) if sorted_zips else None,
            'highest_margin': float(sorted_zips[-1][1]) if sorted_zips else None
        }
        
        logger.info(f"Zipcode Margin ANOVA: p={p_value:.4f}, reject={p_value < 0.05}")
    
    return results


def test_gender_risk_differences(df: pd.DataFrame) -> dict:
    """
    H₀: There is no significant risk difference between Women and Men.
    
    Tests:
    - Claim Frequency (chi-squared test)
    - Claim Severity (t-test)
    """
    logger.info("Testing H0: No risk differences between Women and Men")
    results = {}
    
    # Normalize column names
    df = normalize_column_names(df)
    
    gender_col = None
    for col in ['Gender', 'Sex', 'gender', 'sex']:
        if col in df.columns:
            gender_col = col
            break
    
    if gender_col is None:
        logger.warning("Gender/Sex column not found")
        return results
    
    # Normalize gender values
    df['gender_normalized'] = df[gender_col].str.lower().str.strip()
    df['gender_normalized'] = df['gender_normalized'].replace({
        'female': 'female', 'f': 'female', 'woman': 'female', 'women': 'female',
        'male': 'male', 'm': 'male', 'man': 'male', 'men': 'male'
    })
    
    valid_genders = df['gender_normalized'].value_counts()
    if 'female' not in valid_genders.index or 'male' not in valid_genders.index:
        logger.warning("Both female and male categories not found")
        return results
    
    df['has_claim'] = calculate_claim_frequency(df)
    
    # Test 1: Claim Frequency (chi-squared)
    contingency_table = pd.crosstab(df['gender_normalized'], df['has_claim'])
    if contingency_table.shape[0] >= 2 and contingency_table.shape[1] >= 2:
        chi2, p_value_freq, dof, expected = stats.chi2_contingency(contingency_table)
        results['claim_frequency'] = {
            'test': 'chi-squared',
            'chi2_statistic': float(chi2),
            'p_value': float(p_value_freq),
            'reject_null': bool(p_value_freq < 0.05),
            'interpretation': 'Reject H0' if p_value_freq < 0.05 else 'Fail to reject H0'
        }
        logger.info(f"Gender Claim Frequency test: p={p_value_freq:.4f}, reject={p_value_freq < 0.05}")
    
    # Test 2: Claim Severity (t-test)
    female_severity = df[(df['gender_normalized'] == 'female') & (df['TotalClaims'] > 0)]['TotalClaims']
    male_severity = df[(df['gender_normalized'] == 'male') & (df['TotalClaims'] > 0)]['TotalClaims']
    
    if len(female_severity) > 0 and len(male_severity) > 0:
        t_stat, p_value_sev = stats.ttest_ind(female_severity, male_severity)
        results['claim_severity'] = {
            'test': 't-test',
            't_statistic': float(t_stat),
            'p_value': float(p_value_sev),
            'reject_null': bool(p_value_sev < 0.05),
            'interpretation': 'Reject H0' if p_value_sev < 0.05 else 'Fail to reject H0',
            'female_mean': float(female_severity.mean()),
            'male_mean': float(male_severity.mean()),
            'difference_pct': float((male_severity.mean() - female_severity.mean()) / female_severity.mean() * 100) if female_severity.mean() > 0 else 0
        }
        logger.info(f"Gender Claim Severity t-test: p={p_value_sev:.4f}, reject={p_value_sev < 0.05}")
    
    return results


def run_all_hypothesis_tests(df: pd.DataFrame) -> dict:
    """Run all hypothesis tests and generate report."""
    logger.info("=" * 60)
    logger.info("Starting A/B Hypothesis Testing")
    logger.info("=" * 60)
    
    # Normalize data first
    df_normalized = normalize_column_names(df)
    logger.info(f"Data shape: {df_normalized.shape}")
    logger.info(f"Available columns: {list(df_normalized.columns)}")
    
    all_results = {
        'province_risk': test_province_risk_differences(df_normalized),
        'zipcode_risk': test_zipcode_risk_differences(df_normalized),
        'zipcode_margin': test_zipcode_margin_differences(df_normalized),
        'gender_risk': test_gender_risk_differences(df_normalized)
    }
    
    # Save results
    results_path = INTERIM_REPORTS_DIR / "hypothesis_test_results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Hypothesis test results saved to {results_path}")
    
    return all_results


def generate_business_recommendations(results: dict) -> str:
    """Generate business recommendations based on hypothesis test results."""
    recommendations = []
    
    # Province recommendations
    if 'province_risk' in results and results['province_risk']:
        prov_results = results['province_risk']
        if 'claim_severity' in prov_results and prov_results['claim_severity']['reject_null']:
            if 'pairwise_comparisons' in prov_results['claim_severity']:
                comp = prov_results['claim_severity']['pairwise_comparisons'][0]
                recommendations.append(
                    f"We reject the null hypothesis for provinces (p < 0.05). "
                    f"Specifically, {comp['comparison']} shows a {comp['difference_pct']:.1f}% difference in claim severity, "
                    f"suggesting regional risk adjustment to premiums may be warranted."
                )
    
    # Zipcode recommendations
    if 'zipcode_risk' in results and results['zipcode_risk']:
        zip_results = results['zipcode_risk']
        if 'claim_frequency' in zip_results and zip_results['claim_frequency']['reject_null']:
            recommendations.append(
                f"We reject the null hypothesis for zip codes (p={zip_results['claim_frequency']['p_value']:.4f}). "
                f"Risk differences exist between postal codes, suggesting zipcode-based pricing segmentation."
            )
    
    # Margin recommendations
    if 'zipcode_margin' in results and results['zipcode_margin']:
        margin_results = results['zipcode_margin']
        if margin_results.get('reject_null'):
            recommendations.append(
                f"We reject the null hypothesis for margin differences (p={margin_results['p_value']:.4f}). "
                f"Profitability varies significantly by zip code, with {margin_results.get('highest_margin_zip')} "
                f"showing highest margins. Consider premium adjustments."
            )
    
    # Gender recommendations
    if 'gender_risk' in results and results['gender_risk']:
        gender_results = results['gender_risk']
        if 'claim_severity' in gender_results and gender_results['claim_severity']['reject_null']:
            diff_pct = gender_results['claim_severity'].get('difference_pct', 0)
            recommendations.append(
                f"We reject the null hypothesis for gender (p={gender_results['claim_severity']['p_value']:.4f}). "
                f"Gender shows {abs(diff_pct):.1f}% difference in claim severity, suggesting gender-based risk factors."
            )
    
    return "\n\n".join(recommendations) if recommendations else "No significant differences found to warrant pricing adjustments."


if __name__ == "__main__":
    # Try to load processed data first, then raw data
    data_path = PROCESSED_DATA_DIR / "insurance_cleaned.csv"
    if not data_path.exists():
        # Fall back to raw data
        from utils.config import RAW_DATA_DIR
        data_path = RAW_DATA_DIR / "insurance.csv"
        logger.info(f"Processed data not found, using raw data: {data_path}")
    
    if data_path.exists():
        df = pd.read_csv(data_path)
        logger.info(f"Loaded data: {df.shape}")
        results = run_all_hypothesis_tests(df)
        recommendations = generate_business_recommendations(results)
        print("\n" + "=" * 60)
        print("BUSINESS RECOMMENDATIONS")
        print("=" * 60)
        print(recommendations)
        
        # Generate detailed report
        report_path = INTERIM_REPORTS_DIR / "hypothesis_test_report.txt"
        with open(report_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("HYPOTHESIS TESTING REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(json.dumps(results, indent=2))
            f.write("\n\n" + "=" * 60 + "\n")
            f.write("BUSINESS RECOMMENDATIONS\n")
            f.write("=" * 60 + "\n\n")
            f.write(recommendations)
        logger.info(f"Detailed report saved to {report_path}")
    else:
        logger.error(f"Data not found at {data_path}")

