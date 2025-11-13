"""
Feature Engineering - Enhanced with proper data handling
"""

import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from pathlib import Path
import config
import os
import time


class FeatureEngineer:
    def __init__(self):
        self.data_dir = config.RAW_DATA_DIR
    
    def load_raw_data(self):
        """Load all raw datasets"""
        print("📂 Loading raw data...")
        
        # Check if raw data exists
        raw_files = {
            'rainfall': f"{self.data_dir}/rainfall.csv",
            'elevation': f"{self.data_dir}/elevation.csv",
            'landcover': f"{self.data_dir}/landcover.csv",
            'floods': f"{self.data_dir}/historic_floods.csv",
            'roads': f"{self.data_dir}/roads.csv"
        }
        
        data = {}
        for name, path in raw_files.items():
            if os.path.exists(path):
                data[name] = pd.read_csv(path)
                print(f"✅ Loaded {name}: {len(data[name])} records")
            else:
                print(f"⚠️  {name} not found, generating synthetic data...")
                data[name] = self._generate_synthetic_data(name)
        
        return data['rainfall'], data['elevation'], data['landcover'], data['floods'], data['roads']
    
    def _generate_synthetic_data(self, data_type):
        """Generate synthetic data if raw data doesn't exist"""
        np.random.seed(42)
        n_samples = 1000
        
        # Dhaka bounds
        lons = np.random.uniform(90.25, 90.52, n_samples)
        lats = np.random.uniform(23.65, 23.92, n_samples)
        
        if data_type == 'rainfall':
            return pd.DataFrame({
                'lon': lons,
                'lat': lats,
                'rainfall_mm': np.random.uniform(50, 200, n_samples)
            })
        
        elif data_type == 'elevation':
            elevation = 5 + 15 * np.sin(lats * 50) + np.random.normal(0, 3, n_samples)
            return pd.DataFrame({
                'lon': lons,
                'lat': lats,
                'elevation_m': np.clip(elevation, 0, 50),
                'slope': np.abs(np.random.normal(2, 1, n_samples))
            })
        
        elif data_type == 'landcover':
            return pd.DataFrame({
                'lon': lons,
                'lat': lats,
                'lc_flood_susceptibility': np.random.choice([0.2, 0.4, 0.6, 0.8], n_samples)
            })
        
        elif data_type == 'floods':
            return pd.DataFrame({
                'lon': lons,
                'lat': lats,
                'historic_flood_count': np.random.poisson(3, n_samples)
            })
        
        else:  # roads
            num_roads = 50
            roads = []
            for i in range(num_roads):
                start_lon = np.random.uniform(90.25, 90.52)
                start_lat = np.random.uniform(23.65, 23.92)
                coords = [(start_lon + j*0.005, start_lat + j*0.005) for j in range(5)]
                roads.append({
                    'id': i,
                    'highway': np.random.choice(['primary', 'secondary', 'residential']),
                    'surface': np.random.choice(['asphalt', 'unpaved']),
                    'geometry': str(coords)
                })
            return pd.DataFrame(roads)
    
    def create_road_segments(self, roads_df):
        """Convert road geometries to individual segments"""
        print("🔨 Creating road segments...")
        
        segments = []
        
        for idx, road in roads_df.iterrows():
            if isinstance(road['geometry'], str):
                import ast
                try:
                    coords = ast.literal_eval(road['geometry'])
                except:
                    continue
            else:
                coords = road['geometry']
            
            for i in range(len(coords) - 1):
                start = coords[i]
                end = coords[i + 1]
                
                center_lon = (start[0] + end[0]) / 2
                center_lat = (start[1] + end[1]) / 2
                
                length_km = self._haversine_distance(
                    start[1], start[0], end[1], end[0]
                )
                
                segments.append({
                    'segment_id': f"seg_{idx}_{i}",
                    'road_id': road['id'],
                    'highway_type': road.get('highway', 'unknown'),
                    'surface': road.get('surface', 'unknown'),
                    'center_lon': center_lon,
                    'center_lat': center_lat,
                    'start_lon': start[0],
                    'start_lat': start[1],
                    'end_lon': end[0],
                    'end_lat': end[1],
                    'length_km': length_km
                })
        
        segments_df = pd.DataFrame(segments)
        print(f"✅ Created {len(segments_df)} road segments")
        return segments_df
    
    def spatial_join_features(self, segments_df, rainfall_df, elevation_df, 
                             landcover_df, floods_df):
        """Join satellite data to road segments using nearest neighbor"""
        print("🔗 Performing spatial join...")
        
        def build_tree(df):
            return cKDTree(df[['lon', 'lat']].values)
        
        rainfall_tree = build_tree(rainfall_df)
        elevation_tree = build_tree(elevation_df)
        landcover_tree = build_tree(landcover_df)
        floods_tree = build_tree(floods_df)
        
        segment_coords = segments_df[['center_lon', 'center_lat']].values
        
        # Join features
        _, idx_rain = rainfall_tree.query(segment_coords, k=3)
        segments_df['rainfall_mm'] = rainfall_df.iloc[idx_rain[:, 0]]['rainfall_mm'].values
        segments_df['rainfall_mm_avg'] = np.mean([
            rainfall_df.iloc[idx_rain[:, i]]['rainfall_mm'].values 
            for i in range(3)
        ], axis=0)
        
        _, idx_elev = elevation_tree.query(segment_coords, k=1)
        segments_df['elevation_m'] = elevation_df.iloc[idx_elev]['elevation_m'].values
        segments_df['slope'] = elevation_df.iloc[idx_elev]['slope'].values
        
        _, idx_lc = landcover_tree.query(segment_coords, k=1)
        segments_df['lc_flood_susceptibility'] = landcover_df.iloc[idx_lc]['lc_flood_susceptibility'].values
        
        _, idx_flood = floods_tree.query(segment_coords, k=3)
        segments_df['historic_flood_count'] = floods_df.iloc[idx_flood[:, 0]]['historic_flood_count'].values
        segments_df['historic_flood_nearby'] = np.sum([
            floods_df.iloc[idx_flood[:, i]]['historic_flood_count'].values 
            for i in range(3)
        ], axis=0)
        
        print("✅ Spatial join complete")
        return segments_df
    
    def engineer_features(self, segments_df):
        """Create derived features for ML model"""
        print("⚙️  Engineering features...")
        
        # Road type risk
        road_type_risk = {
            'motorway': 0.1, 'trunk': 0.15, 'primary': 0.2,
            'secondary': 0.3, 'tertiary': 0.4, 'residential': 0.5,
            'unclassified': 0.6, 'unknown': 0.5
        }
        segments_df['road_type_risk'] = segments_df['highway_type'].map(
            lambda x: road_type_risk.get(x, 0.5)
        )
        
        # Surface drainage
        surface_drainage = {
            'asphalt': 0.7, 'concrete': 0.8, 'paved': 0.7,
            'unpaved': 0.3, 'gravel': 0.4, 'dirt': 0.2,
            'unknown': 0.5
        }
        segments_df['drainage_capacity'] = segments_df['surface'].map(
            lambda x: surface_drainage.get(x, 0.5)
        )
        
        # Derived features
        segments_df['low_elevation_risk'] = (segments_df['elevation_m'] < 10).astype(int)
        segments_df['high_slope_safety'] = (segments_df['slope'] > 5).astype(int)
        segments_df['heavy_rainfall'] = (segments_df['rainfall_mm'] > 100).astype(int)
        segments_df['water_proximity_risk'] = np.exp(-segments_df['elevation_m'] / 10)
        
        # Interactions
        segments_df['rainfall_x_susceptibility'] = (
            segments_df['rainfall_mm_avg'] * segments_df['lc_flood_susceptibility']
        )
        segments_df['elevation_x_rainfall'] = (
            segments_df['elevation_m'] * segments_df['rainfall_mm_avg']
        )
        
        # Normalize historic floods
        max_flood = segments_df['historic_flood_count'].max()
        if max_flood > 0:
            segments_df['historic_flood_norm'] = segments_df['historic_flood_count'] / max_flood
        else:
            segments_df['historic_flood_norm'] = 0
        
        print("✅ Feature engineering complete")
        return segments_df
    
    def create_labels(self, segments_df):
        """Create training labels based on historic flood pixels"""
        print("🏷️  Creating labels...")
        
        flood_threshold = segments_df['historic_flood_count'].quantile(0.7)
        
        segments_df['flood_label'] = (
            segments_df['historic_flood_count'] > flood_threshold
        ).astype(int)
        
        max_count = segments_df['historic_flood_count'].max()
        if max_count > 0:
            segments_df['flood_probability'] = np.clip(
                segments_df['historic_flood_count'] / max_count, 0, 1
            )
        else:
            segments_df['flood_probability'] = 0.5
        
        print(f"✅ Labels created - {segments_df['flood_label'].sum()} flooded segments")
        return segments_df
    
    def save_processed_data(self, segments_df):
        """Save final training dataset"""
        os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
        output_path = f"{config.PROCESSED_DATA_DIR}/training_data.csv"
        segments_df.to_csv(output_path, index=False)
        print(f"💾 Saved training data to {output_path}")
        return output_path
    
    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance between two coordinates in km"""
        R = 6371
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c
    
    def run_pipeline(self):
        """Execute full feature engineering pipeline"""
        print("\n🚀 Starting Feature Engineering Pipeline\n")
        
        rainfall, elevation, landcover, floods, roads = self.load_raw_data()
        segments = self.create_road_segments(roads)
        segments = self.spatial_join_features(segments, rainfall, elevation, landcover, floods)
        segments = self.engineer_features(segments)
        segments = self.create_labels(segments)
        output_path = self.save_processed_data(segments)
        
        print("\n✅ Feature engineering complete!")
        print(f"📊 Final dataset: {len(segments)} segments")
        print(f"📊 Features: {len(segments.columns)} columns")
        
        return segments


if __name__ == "__main__":
    start = time.time()
    engineer = FeatureEngineer()
    training_data = engineer.run_pipeline()
    
    print("\n📈 Feature Summary:")
    print(training_data[['rainfall_mm', 'elevation_m', 'slope', 
                          'lc_flood_susceptibility', 'flood_probability']].describe())
    
    print(f"\n⏱️ Total runtime: {round((time.time() - start)/60, 2)} mins")