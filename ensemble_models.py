"""
Ensemble Model - Loads and combines predictions from all trained models
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
import os
import warnings
warnings.filterwarnings('ignore')

def calculate_rmsle(y_true, y_pred):
    """Calculate RMSLE metric"""
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

def train_or_load_models(X_train, X_val, X_test, y_train):
    """Train models or load if already trained"""
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBRegressor
    
    predictions_val = {}
    predictions_test = {}
    
    print("Preparing models...")
    
    # 1. XGBoost model
    print("  Training XGBoost...")
    xgb_model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=8,
        subsample=0.991,
        colsample_bytree=0.738,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42
    )
    xgb_model.fit(X_train, y_train)
    predictions_val['xgboost'] = xgb_model.predict(X_val)
    predictions_test['xgboost'] = xgb_model.predict(X_test)
    print("    ✓ XGBoost ready")
    
    # 2. Try to load MLP PyTorch model if it exists
    if os.path.exists('best_resmlp.pt'):
        try:
            print("  Loading MLP PyTorch model...")
            # Define ResidualBlock and ResMLP architecture matching MLP_pytorch.ipynb
            class ResidualBlock(nn.Module):
                def __init__(self, dim, dropout=0.2):
                    super().__init__()
                    self.fc1 = nn.Linear(dim, dim)
                    self.bn1 = nn.BatchNorm1d(dim)
                    self.fc2 = nn.Linear(dim, dim)
                    self.bn2 = nn.BatchNorm1d(dim)
                    self.act = nn.ReLU()
                    self.dropout = nn.Dropout(dropout)

                def forward(self, x):
                    residual = x
                    out = self.fc1(x)
                    out = self.bn1(out)
                    out = self.act(out)
                    out = self.dropout(out)
                    out = self.fc2(out)
                    out = self.bn2(out)
                    out = out + residual
                    out = self.act(out)
                    return out

            class ResMLP(nn.Module):
                def __init__(self, input_dim, hidden_dim=256, num_blocks=3, dropout=0.2):
                    super().__init__()
                    self.input_fc = nn.Linear(input_dim, hidden_dim)
                    self.input_bn = nn.BatchNorm1d(hidden_dim)
                    self.act = nn.ReLU()
                    self.dropout = nn.Dropout(dropout)
                    self.blocks = nn.ModuleList(
                        [ResidualBlock(hidden_dim, dropout=dropout) for _ in range(num_blocks)]
                    )
                    self.out = nn.Linear(hidden_dim, 1)

                def forward(self, x):
                    x = self.input_fc(x)
                    x = self.input_bn(x)
                    x = self.act(x)
                    x = self.dropout(x)
                    for block in self.blocks:
                        x = block(x)
                    x = self.out(x).squeeze(-1)
                    return x
            
            # Need to scale data for MLP (same as in training)
            mlp_scaler = StandardScaler()
            X_train_mlp_scaled = mlp_scaler.fit_transform(X_train)
            X_val_mlp_scaled = mlp_scaler.transform(X_val)
            X_test_mlp_scaled = mlp_scaler.transform(X_test)
            
            mlp_model = ResMLP(X_val.shape[1], hidden_dim=256, num_blocks=3, dropout=0.2)
            mlp_model.load_state_dict(torch.load('best_resmlp.pt', map_location='cpu'))
            mlp_model.eval()
            
            with torch.no_grad():
                X_val_tensor = torch.FloatTensor(X_val_mlp_scaled)
                X_test_tensor = torch.FloatTensor(X_test_mlp_scaled)
                predictions_val['mlp'] = mlp_model(X_val_tensor).numpy().flatten()
                predictions_test['mlp'] = mlp_model(X_test_tensor).numpy().flatten()
            print("    ✓ MLP loaded")
        except Exception as e:
            print(f"    ⚠ Could not load MLP model: {e}")
    
    return predictions_val, predictions_test

def main():
    print("="*60)
    print("ENSEMBLE MODEL - Using All Trained Models")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    train_data = pd.read_csv('tmdb-box-office-data/train_processed.csv')
    test_data = pd.read_csv('tmdb-box-office-data/test_processed.csv')
    
    # Prepare features
    feature_cols = [col for col in train_data.select_dtypes(include=[np.number]).columns 
                   if col not in ['revenue_log', 'revenue', 'id']]
    
    X = train_data[feature_cols].fillna(0)
    y = train_data['revenue_log']
    X_test = test_data[feature_cols].fillna(0)
    
    # Split for validation (same random state as original models)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Features: {len(feature_cols)}, Validation samples: {len(X_val)}")
    
    # Train or load models
    predictions_val, predictions_test = train_or_load_models(X_train, X_val, X_test, y_train)
    
    # Evaluate individual models
    print(f"\n{'='*60}")
    print("INDIVIDUAL MODEL PERFORMANCE")
    print(f"{'='*60}")
    
    model_scores = {}
    for model_name, preds in predictions_val.items():
        rmsle = calculate_rmsle(y_val, preds)
        model_scores[model_name] = rmsle
        print(f"  {model_name:20s} RMSLE: {rmsle:.6f}")
    
    # Create ensemble predictions
    print(f"\n{'='*60}")
    print("ENSEMBLE STRATEGIES")
    print(f"{'='*60}")
    
    # Strategy 1: Simple average
    ensemble_val_avg = np.mean(list(predictions_val.values()), axis=0)
    ensemble_test_avg = np.mean(list(predictions_test.values()), axis=0)
    avg_rmsle = calculate_rmsle(y_val, ensemble_val_avg)
    print(f"\n1. Simple Average")
    print(f"   Validation RMSLE: {avg_rmsle:.6f}")
    
    # Strategy 2: Weighted by inverse RMSLE (better models get more weight)
    weights = np.array([1.0 / score for score in model_scores.values()])
    weights = weights / weights.sum()
    
    ensemble_val_weighted = np.average(list(predictions_val.values()), axis=0, weights=weights)
    ensemble_test_weighted = np.average(list(predictions_test.values()), axis=0, weights=weights)
    weighted_rmsle = calculate_rmsle(y_val, ensemble_val_weighted)
    
    print(f"\n2. Weighted Average (by performance)")
    print(f"   Weights:")
    for model_name, weight in zip(predictions_val.keys(), weights):
        print(f"     {model_name:20s} {weight:.4f}")
    print(f"   Validation RMSLE: {weighted_rmsle:.6f}")
    
    # Debug: Check prediction ranges before ensemble
    print(f"\n{'='*60}")
    print("PREDICTION RANGES (log scale)")
    print(f"{'='*60}")
    for model_name, preds in predictions_test.items():
        print(f"  {model_name:20s} Min: {preds.min():.2f}, Max: {preds.max():.2f}, Mean: {preds.mean():.2f}")
    
    # Choose best ensemble
    best_ensemble = 'weighted' if weighted_rmsle < avg_rmsle else 'average'
    best_rmsle = min(weighted_rmsle, avg_rmsle)
    final_predictions = ensemble_test_weighted if best_ensemble == 'weighted' else ensemble_test_avg
    
    print(f"\n{'='*60}")
    print(f"BEST ENSEMBLE: {best_ensemble.upper()}")
    print(f"Validation RMSLE: {best_rmsle:.6f}")
    print(f"{'='*60}")
    print(f"Ensemble predictions (log) - Min: {final_predictions.min():.2f}, Max: {final_predictions.max():.2f}")
    
    # Clip predictions to reasonable range (prevent overflow)
    # Max log revenue in training data is 21.14, so clip at 22 (exp(22) ≈ $3.6B)
    final_predictions = np.clip(final_predictions, 0, 22)
    
    # Convert from log to actual revenue
    final_revenue = np.expm1(final_predictions)
    
    # Create submission
    print("\nCreating submission file...")
    submission = pd.read_csv('tmdb-box-office-data/sample_submission.csv')
    submission['revenue'] = final_revenue
    submission.to_csv('submission_ensemble.csv', index=False)
    
    print("\n✓ Created submission_ensemble.csv")
    print(f"\nPrediction Statistics:")
    print(f"  Models combined: {len(predictions_val)}")
    print(f"  Min revenue:  ${final_revenue.min():,.0f}")
    print(f"  Max revenue:  ${final_revenue.max():,.0f}")
    print(f"  Mean revenue: ${final_revenue.mean():,.0f}")
    print(f"  Median revenue: ${np.median(final_revenue):,.0f}")

if __name__ == "__main__":
    main()