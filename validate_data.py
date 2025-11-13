"""
Validate that synthetic data follows realistic patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import config

def load_data():
    """Load generated data"""
    try:
        df = pd.read_csv(f"{config.RAW_DATA_DIR}/realistic_flood_data.csv")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        print(f"✅ Loaded {len(df):,} records")
        return df
    except FileNotFoundError:
        print("❌ Data not found. Run data_collection.py first!")
        return None

def validate_seasonality(df):
    """Check monsoon vs non-monsoon patterns"""
    print("\n" + "="*60)
    print("1. SEASONALITY CHECK")
    print("="*60)
    
    monthly = df.groupby('month').agg({
        'flood_occurred': 'mean',
        'rainfall_mm_hr': 'mean'
    }).round(4)
    
    print("\nMonthly Statistics:")
    print(monthly)
    
    monsoon_flood_rate = df[df['is_monsoon']==1]['flood_occurred'].mean()
    dry_flood_rate = df[df['is_monsoon']==0]['flood_occurred'].mean()
    
    print(f"\n✓ Monsoon flood rate: {monsoon_flood_rate:.2%}")
    print(f"✓ Dry season flood rate: {dry_flood_rate:.2%}")
    print(f"✓ Ratio: {monsoon_flood_rate/dry_flood_rate:.1f}x higher in monsoon")
    
    if monsoon_flood_rate > 2 * dry_flood_rate:
        print("✅ PASS: Monsoon shows significantly higher flood rate")
    else:
        print("⚠️  WARNING: Seasonality effect may be weak")

def validate_persistence(df):
    """Check flood persistence (floods last multiple hours)"""
    print("\n" + "="*60)
    print("2. PERSISTENCE CHECK")
    print("="*60)
    
    df = df.sort_values(['segment_id', 'timestamp'])
    df['prev_flood'] = df.groupby('segment_id')['flood_occurred'].shift(1)
    df['next_flood'] = df.groupby('segment_id')['flood_occurred'].shift(-1)
    
    # If flooded, what % of time is it still flooded next hour?
    persistence_rate = df[df['flood_occurred']==1]['next_flood'].mean()
    
    # How long do floods last on average?
    df['flood_run'] = (df.groupby('segment_id')['flood_occurred']
                       .transform(lambda x: x.ne(x.shift()).cumsum()))
    
    flood_durations = df[df['flood_occurred']==1].groupby(['segment_id', 'flood_run']).size()
    
    print(f"\n✓ Persistence rate: {persistence_rate:.2%}")
    print(f"✓ Average flood duration: {flood_durations.mean():.1f} hours")
    print(f"✓ Median flood duration: {flood_durations.median():.0f} hours")
    print(f"✓ Max flood duration: {flood_durations.max():.0f} hours")
    
    if persistence_rate > 0.5 and flood_durations.mean() > 6:
        print("✅ PASS: Floods persist realistically (6-48h)")
    else:
        print("⚠️  WARNING: Floods may not persist long enough")

def validate_correlations(df):
    """Check physics-based correlations"""
    print("\n" + "="*60)
    print("3. CORRELATION CHECK")
    print("="*60)
    
    corr_vars = ['R12h', 'R24h', 'river_anomaly', 'elevation_m', 
                 'drainage_capacity', 'lc_flood_susceptibility', 'flood_occurred']
    
    corr_matrix = df[corr_vars].corr()['flood_occurred'].sort_values(ascending=False)
    
    print("\nCorrelations with flood occurrence:")
    for var, corr in corr_matrix.items():
        if var != 'flood_occurred':
            direction = "↑" if corr > 0 else "↓"
            print(f"  {direction} {var:30s}: {corr:+.3f}")
    
    # Check expected relationships
    checks = {
        'R12h > 0.3': corr_matrix['R12h'] > 0.3,
        'river_anomaly > 0.2': corr_matrix['river_anomaly'] > 0.2,
        'elevation_m < -0.15': corr_matrix['elevation_m'] < -0.15,
        'drainage_capacity < -0.1': corr_matrix['drainage_capacity'] < -0.1,
    }
    
    print("\nExpected relationships:")
    for check, passed in checks.items():
        status = "✅" if passed else "⚠️ "
        print(f"{status} {check}")
    
    if all(checks.values()):
        print("\n✅ PASS: All expected correlations present")
    else:
        print("\n⚠️  WARNING: Some expected correlations weak")

def validate_spatial_variation(df):
    """Check segment-level variation"""
    print("\n" + "="*60)
    print("4. SPATIAL VARIATION CHECK")
    print("="*60)
    
    segment_stats = df.groupby('segment_id').agg({
        'flood_occurred': 'mean',
        'elevation_m': 'first',
        'known_hotspot': 'first'
    })
    
    print(f"\n✓ Segments with >10% flood rate: {(segment_stats['flood_occurred'] > 0.1).sum()}")
    print(f"✓ Segments with <2% flood rate: {(segment_stats['flood_occurred'] < 0.02).sum()}")
    
    # Check if known hotspots actually flood more
    hotspot_rate = segment_stats[segment_stats['known_hotspot']==1]['flood_occurred'].mean()
    normal_rate = segment_stats[segment_stats['known_hotspot']==0]['flood_occurred'].mean()
    
    print(f"\n✓ Known hotspot flood rate: {hotspot_rate:.2%}")
    print(f"✓ Normal segment flood rate: {normal_rate:.2%}")
    print(f"✓ Hotspot multiplier: {hotspot_rate/normal_rate:.1f}x")
    
    if hotspot_rate > 1.5 * normal_rate:
        print("\n✅ PASS: Spatial variation realistic")
    else:
        print("\n⚠️  WARNING: Weak spatial variation")

def validate_missing_data(df):
    """Check for missing values and outliers"""
    print("\n" + "="*60)
    print("5. DATA QUALITY CHECK")
    print("="*60)
    
    missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
    
    print("\nMissing data percentages:")
    for col, pct in missing_pct[missing_pct > 0].items():
        print(f"  {col}: {pct}%")
    
    total_missing = missing_pct.sum()
    if 0.5 < total_missing < 3:
        print(f"\n✅ PASS: Realistic missing data ({total_missing:.1f}% total)")
    else:
        print(f"\n⚠️  WARNING: Missing data may be unrealistic ({total_missing:.1f}%)")
    
    # Check for outliers
    rainfall_outliers = (df['rainfall_mm_hr'] > df['rainfall_mm_hr'].quantile(0.99)).sum()
    outlier_pct = rainfall_outliers / len(df) * 100
    
    print(f"\n✓ Rainfall outliers (>99th percentile): {outlier_pct:.1f}%")

def validate_class_balance(df):
    """Check flood occurrence rate"""
    print("\n" + "="*60)
    print("6. CLASS BALANCE CHECK")
    print("="*60)
    
    overall_rate = df['flood_occurred'].mean()
    
    print(f"\n✓ Overall flood rate: {overall_rate:.2%}")
    print(f"✓ Total flooded hours: {df['flood_occurred'].sum():,}")
    print(f"✓ Total normal hours: {(df['flood_occurred']==0).sum():,}")
    
    # Target: 3-10% annually
    if 0.03 <= overall_rate <= 0.10:
        print("\n✅ PASS: Flood rate in realistic range (3-10%)")
    else:
        print(f"\n⚠️  WARNING: Flood rate outside target range")

def plot_validation(df):
    """Create validation plots"""
    print("\n" + "="*60)
    print("7. GENERATING VALIDATION PLOTS")
    print("="*60)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Monthly flood rate
    monthly_flood = df.groupby('month')['flood_occurred'].mean() * 100
    axes[0, 0].bar(monthly_flood.index, monthly_flood.values, color='steelblue')
    axes[0, 0].axvline(x=6.5, color='red', linestyle='--', label='Monsoon Start')
    axes[0, 0].axvline(x=9.5, color='red', linestyle='--', label='Monsoon End')
    axes[0, 0].set_xlabel('Month')
    axes[0, 0].set_ylabel('Flood Rate (%)')
    axes[0, 0].set_title('Seasonality: Flood Rate by Month')
    axes[0, 0].legend()
    
    # Plot 2: Rainfall vs River Level
    sample = df.sample(min(5000, len(df)))
    axes[0, 1].scatter(sample['R12h'], sample['river_level_m'], 
                       c=sample['flood_occurred'], cmap='RdYlGn_r', alpha=0.5)
    axes[0, 1].set_xlabel('12-hour Rainfall (mm)')
    axes[0, 1].set_ylabel('River Level (m)')
    axes[0, 1].set_title('Rainfall vs River Level (colored by flood)')
    
    # Plot 3: Flood duration distribution
    df_sorted = df.sort_values(['segment_id', 'timestamp'])
    df_sorted['flood_run'] = (df_sorted.groupby('segment_id')['flood_occurred']
                               .transform(lambda x: x.ne(x.shift()).cumsum()))
    durations = df_sorted[df_sorted['flood_occurred']==1].groupby(['segment_id', 'flood_run']).size()
    axes[1, 0].hist(durations, bins=30, color='coral', edgecolor='black')
    axes[1, 0].set_xlabel('Flood Duration (hours)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Flood Persistence: Duration Distribution')
    axes[1, 0].axvline(x=durations.mean(), color='red', linestyle='--', label=f'Mean: {durations.mean():.1f}h')
    axes[1, 0].legend()
    
    # Plot 4: Elevation vs Flood Rate
    bins = pd.cut(df['elevation_m'], bins=10)
    elev_flood = df.groupby(bins)['flood_occurred'].mean() * 100
    axes[1, 1].bar(range(len(elev_flood)), elev_flood.values, color='forestgreen')
    axes[1, 1].set_xlabel('Elevation Bin')
    axes[1, 1].set_ylabel('Flood Rate (%)')
    axes[1, 1].set_title('Elevation vs Flood Rate (lower = more floods)')
    
    plt.tight_layout()
    plt.savefig(f"{config.RAW_DATA_DIR}/validation_plots.png", dpi=150)
    print(f"✅ Saved validation plots to {config.RAW_DATA_DIR}/validation_plots.png")

def main():
    """Run all validation checks"""
    print("\n🔍 FloodSafe Data Validation")
    print("="*60)
    
    df = load_data()
    if df is None:
        return
    
    # Run all checks
    validate_seasonality(df)
    validate_persistence(df)
    validate_correlations(df)
    validate_spatial_variation(df)
    validate_missing_data(df)
    validate_class_balance(df)
    
    # Generate plots
    plot_validation(df)
    
    print("\n" + "="*60)
    print("✅ VALIDATION COMPLETE!")
    print("="*60)
    print("\nIf all checks pass, your data is ready for ML training!")

if __name__ == "__main__":
    main()