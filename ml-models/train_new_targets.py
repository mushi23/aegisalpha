#!/usr/bin/env python3
"""
Train models on the new 5-bar forward return targets and classification targets.
This script will train both regression and classification models.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, confusion_matrix
from xgboost import XGBRegressor, XGBClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

def load_and_prepare_data():
    """Load the updated dataset and prepare features and targets."""
    print("Loading updated dataset...")
    df = pd.read_csv("all_currencies_with_indicators_updated.csv")
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Feature columns (technical indicators)
    feature_cols = [
        'sma_20', 'ema_20', 'rsi', 'macd', 'macd_signal',
        'bb_upper', 'bb_lower', 'bb_mid', 'support', 'resistance'
    ]
    
    # Target columns
    regression_target = 'future_return_5'
    classification_target = 'target_class'
    binary_target = 'target_binary'
    multi_target = 'target_multi'
    
    # Remove rows with NaN values
    df_clean = df[feature_cols + [regression_target, classification_target, binary_target, multi_target]].dropna()
    
    print(f"Clean dataset shape: {df_clean.shape}")
    print(f"Feature columns: {feature_cols}")
    
    return df_clean, feature_cols, regression_target, classification_target, binary_target, multi_target

def train_regression_models(X_train, X_test, y_train, y_test, target_name):
    """Train regression models to predict future returns."""
    print(f"\n🔄 Training regression models for {target_name}...")
    
    models = {
        'XGBoost': XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42),
        'RandomForest': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_test - y_pred))
        
        # Direction accuracy (if prediction and actual have same sign)
        direction_accuracy = np.mean((y_pred > 0) == (y_test > 0))
        
        results[name] = {
            'model': model,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'direction_accuracy': direction_accuracy,
            'predictions': y_pred
        }
        
        print(f"  {name} - MSE: {mse:.6f}, RMSE: {rmse:.6f}, MAE: {mae:.6f}")
        print(f"  Direction Accuracy: {direction_accuracy:.3f}")
    
    return results

def train_classification_models(X_train, X_test, y_train, y_test, target_name):
    """Train classification models to predict trading signals."""
    print(f"\n🔄 Training classification models for {target_name}...")
    
    models = {
        'XGBoost': XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        print(f"  {name} - Accuracy: {accuracy:.3f}")
        
        # Classification report
        print(f"  Classification Report:")
        print(classification_report(y_test, y_pred))
    
    return results

def plot_results(regression_results, classification_results, y_test_reg, y_test_clf, target_name):
    """Plot model results."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Regression results
    if regression_results:
        ax1, ax2 = axes[0, 0], axes[0, 1]
        
        # Plot actual vs predicted
        ax1.scatter(y_test_reg, regression_results['XGBoost']['predictions'], alpha=0.5, label='XGBoost')
        ax1.scatter(y_test_reg, regression_results['RandomForest']['predictions'], alpha=0.5, label='RandomForest')
        ax1.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'r--', lw=2)
        ax1.set_xlabel('Actual Returns')
        ax1.set_ylabel('Predicted Returns')
        ax1.set_title(f'Actual vs Predicted Returns ({target_name})')
        ax1.legend()
        ax1.grid(True)
        
        # Plot prediction errors
        xgb_errors = y_test_reg - regression_results['XGBoost']['predictions']
        rf_errors = y_test_reg - regression_results['RandomForest']['predictions']
        ax2.hist(xgb_errors, alpha=0.7, label='XGBoost', bins=50)
        ax2.hist(rf_errors, alpha=0.7, label='RandomForest', bins=50)
        ax2.set_xlabel('Prediction Error')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'Prediction Error Distribution ({target_name})')
        ax2.legend()
        ax2.grid(True)
    
    # Classification results
    if classification_results:
        ax3, ax4 = axes[1, 0], axes[1, 1]
        
        # Confusion matrix for XGBoost
        cm_xgb = confusion_matrix(y_test_clf, classification_results['XGBoost']['predictions'])
        sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Blues', ax=ax3)
        ax3.set_title(f'XGBoost Confusion Matrix ({target_name})')
        ax3.set_xlabel('Predicted')
        ax3.set_ylabel('Actual')
        
        # Confusion matrix for RandomForest
        cm_rf = confusion_matrix(y_test_clf, classification_results['RandomForest']['predictions'])
        sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=ax4)
        ax4.set_title(f'RandomForest Confusion Matrix ({target_name})')
        ax4.set_xlabel('Predicted')
        ax4.set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig(f'model_results_{target_name}.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_models(results, target_name, model_type):
    """Save trained models."""
    models_dir = f"models_{target_name}_{model_type}"
    os.makedirs(models_dir, exist_ok=True)
    
    for name, result in results.items():
        model_path = os.path.join(models_dir, f"{name.lower()}_{target_name}_{model_type}.pkl")
        joblib.dump(result['model'], model_path)
        print(f"✅ Saved {name} model to {model_path}")

def main():
    """Main training function."""
    print("🚀 Starting model training with new targets...")
    
    # Load and prepare data
    df_clean, feature_cols, regression_target, classification_target, binary_target, multi_target = load_and_prepare_data()
    
    # Prepare features
    X = df_clean[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save scaler
    joblib.dump(scaler, 'feature_scaler.pkl')
    print("✅ Saved feature scaler")
    
    # Train/test split
    X_train, X_test, y_train_reg, y_test_reg, y_train_clf, y_test_clf, y_train_bin, y_test_bin, y_train_multi, y_test_multi = train_test_split(
        X_scaled, 
        df_clean[regression_target], 
        df_clean[classification_target], 
        df_clean[binary_target], 
        df_clean[multi_target],
        test_size=0.2, 
        random_state=42
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Train regression models
    regression_results = train_regression_models(X_train, X_test, y_train_reg, y_test_reg, regression_target)
    
    # Train classification models for binary target
    binary_results = train_classification_models(X_train, X_test, y_train_bin, y_test_bin, binary_target)
    
    # Train classification models for multi-class target (need to remap labels to 0-4)
    y_train_multi_remapped = y_train_multi + 2  # Convert [-2, -1, 0, 1, 2] to [0, 1, 2, 3, 4]
    y_test_multi_remapped = y_test_multi + 2
    multi_results = train_classification_models(X_train, X_test, y_train_multi_remapped, y_test_multi_remapped, multi_target)
    
    # Plot results
    plot_results(regression_results, binary_results, y_test_reg, y_test_bin, 'binary')
    plot_results(regression_results, multi_results, y_test_reg, y_test_multi, 'multi')
    
    # Save models
    save_models(regression_results, regression_target, 'regression')
    save_models(binary_results, binary_target, 'classification')
    save_models(multi_results, multi_target, 'classification')
    
    # Save results summary
    results_summary = {
        'regression': {name: {'mse': res['mse'], 'rmse': res['rmse'], 'direction_accuracy': res['direction_accuracy']} 
                      for name, res in regression_results.items()},
        'binary_classification': {name: {'accuracy': res['accuracy']} 
                                for name, res in binary_results.items()},
        'multi_classification': {name: {'accuracy': res['accuracy']} 
                               for name, res in multi_results.items()}
    }
    
    # Print final summary
    print("\n" + "="*50)
    print("📊 FINAL RESULTS SUMMARY")
    print("="*50)
    
    print("\n🔢 Regression Results (5-bar forward return):")
    for name, metrics in results_summary['regression'].items():
        print(f"  {name}: MSE={metrics['mse']:.6f}, RMSE={metrics['rmse']:.6f}, Direction Acc={metrics['direction_accuracy']:.3f}")
    
    print("\n🎯 Binary Classification Results (long vs not long):")
    for name, metrics in results_summary['binary_classification'].items():
        print(f"  {name}: Accuracy={metrics['accuracy']:.3f}")
    
    print("\n🎯 Multi-class Classification Results (5 classes):")
    for name, metrics in results_summary['multi_classification'].items():
        print(f"  {name}: Accuracy={metrics['accuracy']:.3f}")
    
    print("\n✅ Training completed successfully!")

if __name__ == "__main__":
    main() 