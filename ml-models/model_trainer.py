import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import os
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.create_directory_structure()
        
    def create_directory_structure(self):
        """Create necessary directories for storing models and results"""
        directories = [
            f"{self.models_dir}",
            f"{self.models_dir}/lstm",
            f"{self.models_dir}/xgboost", 
            f"{self.models_dir}/scalers",
            f"{self.models_dir}/predictions",
            f"{self.models_dir}/plots"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"✅ Created directory: {directory}")
    
    def prepare_data_with_indicators(self):
        """Load and prepare data with technical indicators"""
        print("🔄 Loading enhanced regime features data...")
        
        # Load the enhanced features data
        df = pd.read_csv('enhanced_regime_features.csv')
        
        # Convert datetime
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values(['pair', 'datetime']).reset_index(drop=True)
        
        print(f"✅ Loaded {len(df)} rows with {len(df.columns)} columns")
        print(f"📊 Currency pairs: {df['pair'].unique()}")
        
        return df
    
    def train_lstm_models(self, df):
        """Train LSTM models for each currency pair"""
        print("\n🔄 Training LSTM models...")
        
        # Define features (excluding datetime, OHLCV, pair, and return)
        feature_cols = [col for col in df.columns if col not in ['datetime', 'open', 'high', 'low', 'close', 'volume', 'pair', 'return']]
        target = 'return'
        
        print(f"📈 Using {len(feature_cols)} features for LSTM training")
        
        all_results = []
        
        for pair in df['pair'].unique():
            print(f"\nTraining LSTM for {pair}...")
            
            # Get data for this pair only
            pair_df = df[df['pair'] == pair].copy()
            pair_df = pair_df[feature_cols + [target]].dropna().reset_index(drop=True)
            
            if len(pair_df) < 100:
                print(f"Skipping {pair} - insufficient data")
                continue
            
            # Create sequences for LSTM
            sequence_length = 20
            X_sequences = []
            y_targets = []
            
            for i in range(sequence_length, len(pair_df)):
                feature_seq = pair_df.iloc[i-sequence_length:i][feature_cols].values
                X_sequences.append(feature_seq)
                y_targets.append(pair_df.iloc[i][target])
            
            X_sequences = np.array(X_sequences)
            y_targets = np.array(y_targets)
            
            # Train/test split (time series split)
            split_idx = int(len(X_sequences) * 0.8)
            X_train = X_sequences[:split_idx]
            X_test = X_sequences[split_idx:]
            y_train = y_targets[:split_idx]
            y_test = y_targets[split_idx:]
            
            # Scale features
            scaler = MinMaxScaler()
            X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
            X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
            
            X_train_scaled = scaler.fit_transform(X_train_reshaped)
            X_test_scaled = scaler.transform(X_test_reshaped)
            
            X_train_scaled = X_train_scaled.reshape(X_train.shape)
            X_test_scaled = X_test_scaled.reshape(X_test.shape)
            
            # Build LSTM model
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(sequence_length, len(feature_cols))),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])
            
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            
            # Train model
            history = model.fit(
                X_train_scaled, y_train,
                epochs=50,
                batch_size=32,
                validation_data=(X_test_scaled, y_test),
                verbose=0
            )
            
            # Predict and evaluate
            y_pred = model.predict(X_test_scaled).flatten()
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            print(f"{pair} LSTM - MSE: {mse:.6f}, MAE: {mae:.6f}, RMSE: {rmse:.6f}")
            
            # Store results
            all_results.append({
                'pair': pair,
                'mse': mse,
                'mae': mae,
                'rmse': rmse
            })
            
            # Save model and scaler
            model_path = f"{self.models_dir}/lstm/lstm_model_{pair}.keras"
            scaler_path = f"{self.models_dir}/scalers/lstm_scaler_{pair}.pkl"
            
            model.save(model_path)
            joblib.dump(scaler, scaler_path)
            
            # Save predictions
            np.save(f"{self.models_dir}/predictions/lstm_pred_{pair}.npy", y_pred)
            np.save(f"{self.models_dir}/predictions/lstm_true_{pair}.npy", y_test)
            
            print(f"✅ LSTM model saved: {model_path}")
        
        # Save results summary
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(f"{self.models_dir}/model_performance_summary.csv", index=False)
        print(f"\n✅ Model performance summary saved")
        
        return results_df
    
    def create_binary_labels(self, df):
        """Create binary labels from continuous returns"""
        df = df.copy()
        # Create binary labels: 1 for positive returns, 0 for negative/zero returns
        df['binary_return'] = (df['return'] > 0).astype(int)
        return df

    def create_cost_aware_labels(self, df, transaction_cost=0.0, slippage=0.002):
        """Create cost-aware binary labels: 1 if net return > 0 after costs, else 0"""
        df = df.copy()
        df['net_return'] = df['return'] - (transaction_cost + slippage)
        df['label'] = (df['net_return'] > 0).astype(int)
        return df
    
    def train_xgboost_models(self, df):
        """Train XGBoost models for each currency pair using cost-aware labels"""
        print("\n🔄 Training XGBoost models...")
        
        # Create cost-aware labels
        df = self.create_cost_aware_labels(df, transaction_cost=0.0, slippage=0.002)
        
        # Define features (excluding datetime, OHLCV, pair, and return)
        feature_cols = [col for col in df.columns if col not in ['datetime', 'open', 'high', 'low', 'close', 'volume', 'pair', 'return', 'net_return', 'label', 'binary_return']]
        target = 'label'
        
        print(f"📈 Using {len(feature_cols)} features for XGBoost training (cost-aware labels)")
        
        # Save feature list for extraction script
        with open('feature_list_enhanced.txt', 'w') as f:
            for feature in feature_cols:
                f.write(f"{feature}\n")
        print(f"✅ Saved feature list to feature_list_enhanced.txt")
        
        all_results = []
        
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        
        for pair in df['pair'].unique():
            print(f"\nTraining XGBoost for {pair}...")
            
            # Get data for this pair only
            pair_df = df[df['pair'] == pair].copy()
            pair_df = pair_df[feature_cols + [target]].dropna().reset_index(drop=True)
            
            if len(pair_df) < 100:
                print(f"Skipping {pair} - insufficient data")
                continue
            
            # Create lagged features
            sequence_length = 10
            X_sequences = []
            y_targets = []
            
            for i in range(sequence_length, len(pair_df)):
                feature_seq = pair_df.iloc[i-sequence_length:i][feature_cols].values
                X_sequences.append(feature_seq.flatten())
                y_targets.append(pair_df.iloc[i][target])
            
            X_sequences = np.array(X_sequences)
            y_targets = np.array(y_targets)
            
            # Train/test split (time series split)
            split_idx = int(len(X_sequences) * 0.8)
            X_train = X_sequences[:split_idx]
            X_test = X_sequences[split_idx:]
            y_train = y_targets[:split_idx]
            y_test = y_targets[split_idx:]
            
            # Scale features
            scaler = MinMaxScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Calculate scale_pos_weight for class imbalance
            n_zeros = (y_train == 0).sum()
            n_ones = (y_train == 1).sum()
            scale_pos_weight = n_zeros / n_ones if n_ones > 0 else 1
            print(f"Class balance: 0s={n_zeros}, 1s={n_ones}, scale_pos_weight={scale_pos_weight:.2f}")
            
            # Train XGBoost classifier with class weighting
            model = XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=4,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss',
                early_stopping_rounds=10,
                scale_pos_weight=scale_pos_weight
            )
            
            model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_test_scaled, y_test)],
                verbose=0
            )
            
            # Predict and evaluate
            y_pred = model.predict(X_test_scaled)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            print(f"Predicted label counts: {dict(zip(*np.unique(y_pred, return_counts=True)))}")
            print(f"{pair} XGBoost - Accuracy: {acc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
            
            # Store results
            all_results.append({
                'pair': pair,
                'accuracy': acc,
                'f1': f1,
                'precision': precision,
                'recall': recall
            })
            
            # Save model and scaler
            model_path = f"{self.models_dir}/xgboost/xgb_model_{pair}.pkl"
            scaler_path = f"{self.models_dir}/scalers/xgb_scaler_{pair}.pkl"
            
            joblib.dump(model, model_path)
            joblib.dump(scaler, scaler_path)
            
            # Save predictions
            np.save(f"{self.models_dir}/predictions/xgb_pred_{pair}.npy", y_pred)
            np.save(f"{self.models_dir}/predictions/xgb_true_{pair}.npy", y_test)
            
            print(f"✅ XGBoost model saved: {model_path}")
        
        # Save results summary
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(f"{self.models_dir}/model_performance_summary.csv", index=False)
        print(f"\n✅ Model performance summary saved")
        
        return results_df
    
    def train_xgboost_regressors(self, df):
        """Train XGBoost regressors for each currency pair to predict expected return"""
        print("\n🔄 Training XGBoost regressors (expected return)...")
        
        # Define regression target
        df = df.copy()
        df['target'] = df['return']
        
        # Define features (excluding datetime, OHLCV, pair, and return)
        feature_cols = [col for col in df.columns if col not in ['datetime', 'open', 'high', 'low', 'close', 'volume', 'pair', 'return', 'net_return', 'label', 'binary_return', 'target']]
        target = 'target'
        
        print(f"📈 Using {len(feature_cols)} features for XGBoost regression")
        
        # Save feature list for extraction script
        with open('feature_list_enhanced.txt', 'w') as f:
            for feature in feature_cols:
                f.write(f"{feature}\n")
        print(f"✅ Saved feature list to feature_list_enhanced.txt")
        
        all_results = []
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        
        for pair in df['pair'].unique():
            print(f"\nTraining XGBoost regressor for {pair}...")
            
            # Get data for this pair only
            pair_df = df[df['pair'] == pair].copy()
            pair_df = pair_df[feature_cols + [target]].dropna().reset_index(drop=True)
            
            if len(pair_df) < 100:
                print(f"Skipping {pair} - insufficient data")
                continue
            
            # Create lagged features
            sequence_length = 10
            X_sequences = []
            y_targets = []
            
            for i in range(sequence_length, len(pair_df)):
                feature_seq = pair_df.iloc[i-sequence_length:i][feature_cols].values
                X_sequences.append(feature_seq.flatten())
                y_targets.append(pair_df.iloc[i][target])
            
            X_sequences = np.array(X_sequences)
            y_targets = np.array(y_targets)
            
            # Train/test split (time series split)
            split_idx = int(len(X_sequences) * 0.8)
            X_train = X_sequences[:split_idx]
            X_test = X_sequences[split_idx:]
            y_train = y_targets[:split_idx]
            y_test = y_targets[split_idx:]
            
            # Scale features
            scaler = MinMaxScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train XGBoost regressor
            from xgboost import XGBRegressor
            model = XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
            model.fit(X_train_scaled, y_train, eval_set=[(X_test_scaled, y_test)], verbose=0)
            
            # Predict and evaluate
            y_pred = model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            print(f"{pair} XGBoost Regressor - MSE: {mse:.6f}, MAE: {mae:.6f}")
            
            # Store results
            all_results.append({
                'pair': pair,
                'mse': mse,
                'mae': mae
            })
            
            # Save model and scaler
            model_path = f"{self.models_dir}/xgboost/xgb_regressor_{pair}.pkl"
            scaler_path = f"{self.models_dir}/scalers/xgb_regressor_scaler_{pair}.pkl"
            joblib.dump(model, model_path)
            joblib.dump(scaler, scaler_path)
            
            # Save predictions
            np.save(f"{self.models_dir}/predictions/xgb_reg_pred_{pair}.npy", y_pred)
            np.save(f"{self.models_dir}/predictions/xgb_reg_true_{pair}.npy", y_test)
            print(f"✅ XGBoost regressor saved: {model_path}")
        
        # Save results summary
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(f"{self.models_dir}/xgb_regressor_performance_summary.csv", index=False)
        print(f"\n✅ XGBoost regressor performance summary saved")
        return results_df
    
    def create_model_comparison_plots(self):
        """Create comparison plots for all models (robust to regression/classification)"""
        print("\n🔄 Creating model comparison plots...")
        
        try:
            results_df = pd.read_csv(f"{self.models_dir}/model_performance_summary.csv")
            
            # Plot regression metrics if present (LSTM)
            regression_metrics = ['mse', 'mae', 'rmse']
            regression_metrics_present = [m for m in regression_metrics if m in results_df.columns]
            if regression_metrics_present:
                plt.figure(figsize=(5 * len(regression_metrics_present), 5))
                for i, metric in enumerate(regression_metrics_present, 1):
                    plt.subplot(1, len(regression_metrics_present), i)
                    plt.bar(results_df['pair'], results_df[metric])
                    plt.title(f'LSTM {metric.upper()}')
                    plt.ylabel(metric.upper())
            plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(f"{self.models_dir}/plots/lstm_performance_comparison.png", dpi=300, bbox_inches='tight')
                plt.show()
                print(f"✅ LSTM performance comparison plot saved")

            # Plot classification metrics if present (XGBoost)
            classification_metrics = ['accuracy', 'f1', 'precision', 'recall']
            classification_metrics_present = [m for m in classification_metrics if m in results_df.columns]
            if classification_metrics_present:
                plt.figure(figsize=(5 * len(classification_metrics_present), 5))
                for i, metric in enumerate(classification_metrics_present, 1):
                    plt.subplot(1, len(classification_metrics_present), i)
                    plt.bar(results_df['pair'], results_df[metric])
                    plt.title(f'XGBoost {metric.capitalize()}')
                    plt.ylabel(metric.capitalize())
            plt.xticks(rotation=45)
            plt.tight_layout()
                plt.savefig(f"{self.models_dir}/plots/xgboost_performance_comparison.png", dpi=300, bbox_inches='tight')
            plt.show()
                print(f"✅ XGBoost performance comparison plot saved")

            # Print best model for each type
            if 'mse' in results_df.columns:
                best_lstm = results_df.loc[results_df['mse'].idxmin()]
                print(f"\n🏆 Best LSTM Model: {best_lstm['pair']}")
                print(f"   MSE: {best_lstm['mse']:.6f}")
                if 'mae' in best_lstm: print(f"   MAE: {best_lstm['mae']:.6f}")
                if 'rmse' in best_lstm: print(f"   RMSE: {best_lstm['rmse']:.6f}")
            if 'accuracy' in results_df.columns:
                best_xgb = results_df.loc[results_df['accuracy'].idxmax()]
                print(f"\n🏆 Best XGBoost Model: {best_xgb['pair']}")
                print(f"   Accuracy: {best_xgb['accuracy']:.4f}")
                if 'f1' in best_xgb: print(f"   F1: {best_xgb['f1']:.4f}")
                if 'precision' in best_xgb: print(f"   Precision: {best_xgb['precision']:.4f}")
                if 'recall' in best_xgb: print(f"   Recall: {best_xgb['recall']:.4f}")
            
        except FileNotFoundError:
            print("⚠️ Performance summary not found. Run training first.")
    
    def run_complete_training(self):
        """Run the complete training pipeline"""
        print("🚀 Starting complete model training pipeline...")
        
        # Step 1: Prepare data with indicators
        df = self.prepare_data_with_indicators()
        
        # Step 2: Train LSTM models
        self.train_lstm_models(df)
        
        # Step 3: Train XGBoost models
        results_df = self.train_xgboost_models(df)
        
        # Step 4: Train XGBoost regressors for expected return
        self.train_xgboost_regressors(df)
        
        # Step 5: Create comparison plots
        self.create_model_comparison_plots()
        
        print("\n🎉 Complete training pipeline finished!")
        print(f"📁 All models saved in: {self.models_dir}/")
        
        return results_df

if __name__ == "__main__":
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Run complete training
    results = trainer.run_complete_training()
    
    # Print best model
    if not results.empty:
        if 'mse' in results.columns:
        best_model = results.loc[results['mse'].idxmin()]
            print(f"\n🏆 Best LSTM Model: {best_model['pair']}")
        print(f"   MSE: {best_model['mse']:.6f}")
            if 'mae' in best_model: print(f"   MAE: {best_model['mae']:.6f}")
            if 'rmse' in best_model: print(f"   RMSE: {best_model['rmse']:.6f}")
        if 'accuracy' in results.columns:
            best_model = results.loc[results['accuracy'].idxmax()]
            print(f"\n🏆 Best XGBoost Model: {best_model['pair']}")
            print(f"   Accuracy: {best_model['accuracy']:.4f}")
            if 'f1' in best_model: print(f"   F1: {best_model['f1']:.4f}")
            if 'precision' in best_model: print(f"   Precision: {best_model['precision']:.4f}")
            if 'recall' in best_model: print(f"   Recall: {best_model['recall']:.4f}") 