"""
Train flood risk prediction model
Predicts P(flood) for each road segment
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_auc_score, mean_squared_error, r2_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import config


class FloodRiskModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        Path(config.MODEL_DIR).mkdir(parents=True, exist_ok=True)
    
    def load_training_data(self):
        """Load processed training data"""
        print("📂 Loading training data...")
        data_path = f"{config.PROCESSED_DATA_DIR}/training_data.csv"
        df = pd.read_csv(data_path)
        print(f"✅ Loaded {len(df)} samples")
        return df
    
    def prepare_features(self, df):
        """Select and prepare features for training"""
        print("🔧 Preparing features...")
        
        # Define feature columns (exclude identifiers and targets)
        self.feature_columns = [
            'rainfall_mm', 'rainfall_mm_avg', 'elevation_m', 'slope',
            'lc_flood_susceptibility', 'historic_flood_count', 
            'historic_flood_nearby', 'road_type_risk', 'drainage_capacity',
            'low_elevation_risk', 'high_slope_safety', 'heavy_rainfall',
            'water_proximity_risk', 'rainfall_x_susceptibility',
            'elevation_x_rainfall', 'historic_flood_norm', 'length_km'
        ]
        
        # Handle missing values
        X = df[self.feature_columns].fillna(0)
        
        # Target: flood probability (continuous 0-1)
        y = df['flood_probability'].fillna(0.5)
        
        print(f"✅ Features prepared: {X.shape[1]} features, {len(X)} samples")
        return X, y
    
    def train(self, X, y):
        """Train the flood risk prediction model"""
        print("\n🎯 Training model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.TEST_SIZE, 
            random_state=config.RANDOM_STATE, stratify=(y > 0.5).astype(int)
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Gradient Boosting Regressor (outputs probability)
        print("Training Gradient Boosting model...")
        self.model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            subsample=0.8,
            random_state=config.RANDOM_STATE,
            verbose=1
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        y_pred = np.clip(y_pred, 0, 1)  # Ensure valid probabilities
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"\n✅ Training complete!")
        print(f"📊 RMSE: {rmse:.4f}")
        print(f"📊 R² Score: {r2:.4f}")
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train, 
            cv=5, scoring='neg_mean_squared_error'
        )
        print(f"📊 Cross-val RMSE: {np.sqrt(-cv_scores.mean()):.4f} (+/- {np.sqrt(cv_scores.std()):.4f})")
        
        # Feature importance
        self.plot_feature_importance(X.columns)
        
        return X_test, y_test, y_pred
    
    def plot_feature_importance(self, feature_names):
        """Plot top feature importances"""
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False).head(15)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance_df, x='importance', y='feature')
        plt.title('Top 15 Feature Importances')
        plt.tight_layout()
        plt.savefig(f"{config.MODEL_DIR}/feature_importance.png", dpi=150)
        print(f"💾 Feature importance plot saved")
        plt.close()
    
    def evaluate(self, X_test, y_test, y_pred):
        """Detailed model evaluation"""
        print("\n📊 Model Evaluation:")
        
        # Binary classification metrics (threshold at 0.5)
        y_test_binary = (y_test > 0.5).astype(int)
        y_pred_binary = (y_pred > 0.5).astype(int)
        
        print("\nClassification Report (threshold=0.5):")
        print(classification_report(y_test_binary, y_pred_binary, 
                                   target_names=['Safe', 'Flood Risk']))
        
        # ROC-AUC
        auc = roc_auc_score(y_test_binary, y_pred)
        print(f"\nROC-AUC Score: {auc:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test_binary, y_pred_binary)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Safe', 'Flood Risk'],
                   yticklabels=['Safe', 'Flood Risk'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f"{config.MODEL_DIR}/confusion_matrix.png", dpi=150)
        print(f"💾 Confusion matrix saved")
        plt.close()
        
        # Prediction distribution
        plt.figure(figsize=(10, 5))
        plt.hist(y_test, bins=30, alpha=0.5, label='True', color='blue')
        plt.hist(y_pred, bins=30, alpha=0.5, label='Predicted', color='red')
        plt.xlabel('Flood Probability')
        plt.ylabel('Frequency')
        plt.title('Distribution of Flood Probabilities')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{config.MODEL_DIR}/prediction_distribution.png", dpi=150)
        print(f"💾 Prediction distribution saved")
        plt.close()
    
    def save_model(self):
        """Save trained model and scaler"""
        model_path = f"{config.MODEL_DIR}/flood_risk_model.pkl"
        scaler_path = f"{config.MODEL_DIR}/scaler.pkl"
        features_path = f"{config.MODEL_DIR}/feature_columns.pkl"
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.feature_columns, features_path)
        
        print(f"\n💾 Model saved to {model_path}")
        print(f"💾 Scaler saved to {scaler_path}")
        print(f"💾 Features saved to {features_path}")
    
    def load_model(self):
        """Load trained model"""
        model_path = f"{config.MODEL_DIR}/flood_risk_model.pkl"
        scaler_path = f"{config.MODEL_DIR}/scaler.pkl"
        features_path = f"{config.MODEL_DIR}/feature_columns.pkl"
        
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.feature_columns = joblib.load(features_path)
        
        print("✅ Model loaded successfully")
    
    def predict(self, X):
        """
        Predict flood probability for new segments
        X: DataFrame with same features as training data
        Returns: array of probabilities [0, 1]
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Ensure features are in correct order
        X_features = X[self.feature_columns].fillna(0)
        
        # Scale and predict
        X_scaled = self.scaler.transform(X_features)
        predictions = self.model.predict(X_scaled)
        
        # Clip to valid probability range
        predictions = np.clip(predictions, 0, 1)
        
        return predictions
    
    def run_training_pipeline(self):
        """Execute complete training pipeline"""
        print("\n🚀 Starting ML Training Pipeline\n")
        
        # Load data
        df = self.load_training_data()
        
        # Prepare features
        X, y = self.prepare_features(df)
        
        # Train model
        X_test, y_test, y_pred = self.train(X, y)
        
        # Evaluate
        self.evaluate(X_test, y_test, y_pred)
        
        # Save model
        self.save_model()
        
        print("\n✅ Training pipeline complete!")
        
        return self


if __name__ == "__main__":
    # Train the model
    model = FloodRiskModel()
    model.run_training_pipeline()
    
    # Test prediction on sample
    print("\n🧪 Testing prediction...")
    df = model.load_training_data()
    sample = df.head(5)
    predictions = model.predict(sample)
    
    print("\nSample Predictions:")
    for i, pred in enumerate(predictions):
        risk_level = "High" if pred > 0.6 else "Moderate" if pred > 0.3 else "Low"
        print(f"Segment {i+1}: P(flood) = {pred:.3f} ({risk_level} risk)")