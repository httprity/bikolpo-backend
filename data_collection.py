"""
Enhanced Data Collection Pipeline for FloodSafe
Generates physics-informed realistic synthetic data for training
Includes: Seasonality, event clustering, spatial correlation, persistence
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import config

# Initialize directories
Path(config.RAW_DATA_DIR).mkdir(parents=True, exist_ok=True)
Path(config.PROCESSED_DATA_DIR).mkdir(parents=True, exist_ok=True)


class RealisticDataGenerator:
    def __init__(self, seed=42):
        """Initialize with domain-informed parameters"""
        np.random.seed(seed)
        
        # Dhaka-specific characteristics
        self.monsoon_months = [6, 7, 8, 9]  # June-September
        self.peak_month = 7  # July
        
        # Physics parameters
        self.base_rainfall_mm_hr = 2.0  # Base rainfall outside monsoon
        self.monsoon_multiplier = 4.5  # 4.5x higher in monsoon
        self.river_response_lag = 6  # hours
        self.flood_persistence_hours = 12  # Floods last 6-48h
        
    def generate_temporal_dataset(self, start_date, end_date, num_segments=100):
        """
        Generate hourly time-series data with realistic patterns
        """
        print("🕐 Generating temporal dataset...")
        
        # Create hourly timeline
        dates = pd.date_range(start=start_date, end=end_date, freq='H')
        n_hours = len(dates)
        
        # Generate segment IDs (road segments in Dhaka)
        segment_ids = [f"seg_{i:04d}" for i in range(num_segments)]
        
        # Create base dataframe (all segments × all hours)
        data = []
        
        for seg_id in segment_ids:
            # Assign static segment attributes
            segment_attrs = self._generate_segment_attributes(seg_id)
            
            # Generate time series for this segment
            segment_data = self._generate_segment_timeseries(
                dates, seg_id, segment_attrs
            )
            
            data.extend(segment_data)
        
        df = pd.DataFrame(data)
        print(f"✅ Generated {len(df):,} hourly records across {num_segments} segments")
        
        return df
    
    def _generate_segment_attributes(self, seg_id):
        """Generate realistic static attributes for a road segment"""
        
        # Parse segment number for spatial correlation
        seg_num = int(seg_id.split('_')[1])
        
        # Elevation (Dhaka: mostly 0-15m above sea level)
        # Some areas are lower (flood-prone)
        base_elev = 5.0 + 8.0 * np.sin(seg_num * 0.1)
        elevation_m = max(0, base_elev + np.random.normal(0, 2))
        
        # Slope (mostly flat, some drainage gradient)
        slope = abs(np.random.normal(0.5, 0.3))
        
        # Distance to drainage (closer = better, but can overflow)
        distance_to_drain_m = np.random.lognormal(4.5, 1.2)  # Mean ~100m
        
        # Drainage density (inverse of distance, plus random variation)
        drain_density = 1.0 / (1.0 + distance_to_drain_m / 50) + np.random.uniform(-0.1, 0.1)
        drain_density = np.clip(drain_density, 0.1, 1.0)
        
        # Land cover type
        landcover_types = ['residential', 'commercial', 'industrial', 'park']
        landcover_probs = [0.5, 0.25, 0.15, 0.1]
        landcover = np.random.choice(landcover_types, p=landcover_probs)
        
        # Land cover flood susceptibility
        lc_susceptibility = {
            'residential': 0.6,
            'commercial': 0.4,  # Better drainage
            'industrial': 0.5,
            'park': 0.7  # Low-lying green areas
        }[landcover]
        
        # Known hotspot (some segments historically flood more)
        known_hotspot = int(np.random.random() < 0.15)  # 15% are hotspots
        
        # Road surface
        surfaces = ['asphalt', 'concrete', 'unpaved']
        surface_probs = [0.6, 0.3, 0.1]
        surface = np.random.choice(surfaces, p=surface_probs)
        
        # Drainage capacity based on surface
        drainage_capacity = {
            'asphalt': 0.7,
            'concrete': 0.8,
            'unpaved': 0.3
        }[surface]
        
        # Segment-specific flood proneness (random effect)
        segment_flood_bias = np.random.normal(0, 0.5)
        
        return {
            'segment_id': seg_id,
            'elevation_m': round(elevation_m, 2),
            'slope': round(slope, 3),
            'distance_to_drain_m': round(distance_to_drain_m, 1),
            'drain_density': round(drain_density, 3),
            'landcover': landcover,
            'lc_flood_susceptibility': lc_susceptibility,
            'known_hotspot': known_hotspot,
            'surface': surface,
            'drainage_capacity': drainage_capacity,
            'segment_flood_bias': round(segment_flood_bias, 3)
        }
    
    def _generate_segment_timeseries(self, dates, seg_id, seg_attrs):
        """Generate hourly time series for one segment with physics"""
        
        records = []
        n_hours = len(dates)
        
        # Initialize state variables
        river_level_m = 2.0  # Base river level
        last_flood_state = 0  # For persistence
        hours_since_flood = 100  # Large number
        
        # Generate storm events (Poisson process)
        storm_events = self._generate_storm_events(dates)
        
        for i, timestamp in enumerate(dates):
            month = timestamp.month
            hour = timestamp.hour
            
            # 1. RAINFALL with seasonality and storm events
            is_monsoon = month in self.monsoon_months
            base_rate = self.base_rainfall_mm_hr
            
            if is_monsoon:
                # Monsoon amplification (peaked in July)
                month_factor = 1.0 + (self.monsoon_multiplier - 1.0) * \
                              np.exp(-((month - self.peak_month)**2) / 4.0)
            else:
                month_factor = 1.0
            
            # Check if in a storm event
            in_storm = any(start <= i < end for start, end in storm_events)
            storm_multiplier = np.random.gamma(3, 2) if in_storm else 1.0
            
            # Rainfall rate (mm/hr)
            rainfall_mm_hr = base_rate * month_factor * storm_multiplier
            rainfall_mm_hr = max(0, rainfall_mm_hr + np.random.normal(0, 0.5))
            
            # 2. RIVER LEVEL (AR(1) + rainfall response)
            # Responds to recent rainfall with lag
            if i >= self.river_response_lag:
                recent_rain = np.mean([records[j]['rainfall_mm_hr'] 
                                      for j in range(max(0, i-24), i)])
            else:
                recent_rain = rainfall_mm_hr
            
            # AR(1) process + rain contribution
            river_level_m = 0.7 * river_level_m + 0.02 * recent_rain + np.random.normal(0, 0.1)
            river_level_m = max(1.0, min(8.0, river_level_m))  # Bounded
            
            # 3. TEMPERATURE (inversely correlated with rain)
            base_temp = 28.0 if is_monsoon else 25.0
            rain_cooling = -0.3 * min(rainfall_mm_hr, 10)
            temperature_c = base_temp + rain_cooling + np.random.normal(0, 1.5)
            
            # 4. HUMIDITY (correlated with rain)
            base_humidity = 75 if is_monsoon else 65
            rain_humidity_boost = min(15, rainfall_mm_hr * 1.5)
            humidity_pct = base_humidity + rain_humidity_boost + np.random.normal(0, 3)
            humidity_pct = np.clip(humidity_pct, 40, 98)
            
            # 5. ROLLING RAINFALL SUMS
            if i >= 6:
                R6h = sum(records[j]['rainfall_mm_hr'] for j in range(i-6, i))
            else:
                R6h = rainfall_mm_hr * (i + 1)
            
            if i >= 12:
                R12h = sum(records[j]['rainfall_mm_hr'] for j in range(i-12, i))
            else:
                R12h = R6h * 2
            
            if i >= 24:
                R24h = sum(records[j]['rainfall_mm_hr'] for j in range(i-24, i))
            else:
                R24h = R12h * 2
            
            # 6. RIVER ANOMALY (deviation from normal)
            river_anomaly = river_level_m - 2.5
            
            # 7. FLOOD PROBABILITY (logistic model)
            # β0 + β1*R12h + β2*river_anom + β3*low_elev + β4*poor_drain + β5*landcover + segment_bias
            low_elev = int(seg_attrs['elevation_m'] < 5.0)
            poor_drain = 1.0 - seg_attrs['drainage_capacity']
            
            logit = (
                -3.5  # β0: base (low probability)
                + 0.08 * R12h  # β1: recent rain
                + 0.6 * river_anomaly  # β2: river overflow
                + 1.2 * low_elev  # β3: elevation risk
                + 1.0 * poor_drain  # β4: drainage
                + 0.8 * seg_attrs['lc_flood_susceptibility']  # β5: land cover
                + 1.5 * seg_attrs['known_hotspot']  # Known hotspot boost
                + seg_attrs['segment_flood_bias']  # Random segment effect
            )
            
            base_prob = 1.0 / (1.0 + np.exp(-logit))
            
            # 8. PERSISTENCE: if flooded last hour, boost probability
            if last_flood_state == 1 and hours_since_flood < self.flood_persistence_hours:
                persistence_boost = 0.3 * (1.0 - hours_since_flood / self.flood_persistence_hours)
                flood_prob = min(0.98, base_prob + persistence_boost)
            else:
                flood_prob = base_prob
            
            # 9. SAMPLE FLOOD STATE
            flood_occurred = int(np.random.random() < flood_prob)
            
            # Update persistence tracking
            if flood_occurred:
                hours_since_flood = 0
            else:
                hours_since_flood += 1
            
            last_flood_state = flood_occurred
            
            # 10. ADD NOISE & OUTLIERS (1-3%)
            if np.random.random() < 0.02:  # 2% outliers
                rainfall_mm_hr *= np.random.uniform(1.5, 3.0)
            
            # 11. MISSING VALUES (1-3%)
            if np.random.random() < 0.015:  # 1.5% missing
                rainfall_mm_hr = np.nan
            
            # Store record
            record = {
                'timestamp': timestamp,
                'segment_id': seg_id,
                'rainfall_mm_hr': round(rainfall_mm_hr, 2),
                'R6h': round(R6h, 2),
                'R12h': round(R12h, 2),
                'R24h': round(R24h, 2),
                'river_level_m': round(river_level_m, 2),
                'river_anomaly': round(river_anomaly, 2),
                'temperature_c': round(temperature_c, 1),
                'humidity_pct': round(humidity_pct, 1),
                'flood_occurred': flood_occurred,
                'flood_probability': round(flood_prob, 4),
                'month': month,
                'hour': hour,
                'is_monsoon': int(is_monsoon),
                **seg_attrs  # Include all segment attributes
            }
            
            records.append(record)
        
        return records
    
    def _generate_storm_events(self, dates):
        """Generate storm events using Poisson process"""
        n_hours = len(dates)
        
        # Average ~2 storms per week during monsoon, ~0.5 during dry season
        events = []
        
        i = 0
        while i < n_hours:
            timestamp = dates[i]
            month = timestamp.month
            
            # Storm arrival rate (events per hour)
            if month in self.monsoon_months:
                lambda_rate = 1 / (7 * 24 / 2)  # 2 storms per week
            else:
                lambda_rate = 1 / (7 * 24 / 0.5)  # 0.5 storms per week
            
            # Time until next storm (exponential)
            hours_until_next = int(np.random.exponential(1 / lambda_rate))
            i += hours_until_next
            
            if i >= n_hours:
                break
            
            # Storm duration (geometric/lognormal: 6-48h)
            duration = int(np.random.lognormal(2.5, 0.8))  # Mean ~18h
            duration = np.clip(duration, 6, 48)
            
            events.append((i, i + duration))
            i += duration
        
        return events
    
    def save_dataset(self, df, filename="realistic_flood_data.csv"):
        """Save with proper train/test split"""
        
        # Sort by time
        df = df.sort_values(['timestamp', 'segment_id']).reset_index(drop=True)
        
        # Calculate flood rate by month
        monthly_stats = df.groupby('month')['flood_occurred'].agg(['mean', 'sum', 'count'])
        print("\n📊 Monthly Flood Statistics:")
        print(monthly_stats)
        
        # Overall statistics
        flood_rate = df['flood_occurred'].mean()
        print(f"\n📈 Overall flood rate: {flood_rate:.2%}")
        
        # Save full dataset
        output_path = f"{config.RAW_DATA_DIR}/{filename}"
        df.to_csv(output_path, index=False)
        print(f"💾 Saved to: {output_path}")
        
        # Create train/test split (temporal)
        # Train: First 80% of time, Test: Last 20%
        cutoff_idx = int(len(df['timestamp'].unique()) * 0.8)
        cutoff_date = sorted(df['timestamp'].unique())[cutoff_idx]
        
        train_df = df[df['timestamp'] < cutoff_date]
        test_df = df[df['timestamp'] >= cutoff_date]
        
        train_df.to_csv(f"{config.PROCESSED_DATA_DIR}/train_data.csv", index=False)
        test_df.to_csv(f"{config.PROCESSED_DATA_DIR}/test_data.csv", index=False)
        
        print(f"\n✅ Train set: {len(train_df):,} records (flood rate: {train_df['flood_occurred'].mean():.2%})")
        print(f"✅ Test set: {len(test_df):,} records (flood rate: {test_df['flood_occurred'].mean():.2%})")
        
        return df, train_df, test_df


def main():
    """Generate realistic synthetic dataset"""
    
    print("🚀 FloodSafe Realistic Data Generator")
    print("=" * 50)
    
    generator = RealisticDataGenerator(seed=42)
    
    # Generate 1 year of hourly data for 100 road segments
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    num_segments = 100
    
    print(f"\n📅 Generating data from {start_date} to {end_date}")
    print(f"🛣️  Number of road segments: {num_segments}")
    
    # Generate dataset
    df = generator.generate_temporal_dataset(start_date, end_date, num_segments)
    
    # Save with splits
    full_df, train_df, test_df = generator.save_dataset(df)
    
    # Validation checks
    print("\n✅ Validation Checks:")
    print(f"1. Monsoon has higher floods: {df[df['is_monsoon']==1]['flood_occurred'].mean():.2%} vs {df[df['is_monsoon']==0]['flood_occurred'].mean():.2%}")
    
    # Check persistence
    df['prev_flood'] = df.groupby('segment_id')['flood_occurred'].shift(1)
    persistence_rate = df[df['prev_flood']==1]['flood_occurred'].mean()
    base_rate = df[df['prev_flood']==0]['flood_occurred'].mean()
    print(f"2. Flood persistence: {persistence_rate:.2%} vs base {base_rate:.2%}")
    
    # Check correlation
    corr = df[['R12h', 'river_anomaly', 'elevation_m', 'flood_occurred']].corr()['flood_occurred']
    print(f"3. Correlations with flood:")
    print(f"   - R12h: {corr['R12h']:.3f}")
    print(f"   - River anomaly: {corr['river_anomaly']:.3f}")
    print(f"   - Elevation: {corr['elevation_m']:.3f}")
    
    print("\n🎉 Dataset generation complete!")


if __name__ == "__main__":
    main()