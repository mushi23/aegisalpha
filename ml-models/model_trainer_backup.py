#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor, XGBClassifier
import joblib
import ta
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras import Input
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    def __init__(self, models_dir="models"):
        """Initialize the model trainer with organized directory structure"""
        self.models_dir = models_dir
        self.create_directory_structure()
        
    def create_directory_structure(self):
        """Create organized directory structure for models"""
        directories = [
            f"{self.models_dir}/lstm",
            f"{self.models_dir}/xgboost", 
            f"{self.models_dir}/scalers",
            f"{self.models_dir}/data",
            f"{self.models_dir}/predictions",
            f"{self.models_dir}/plots"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"✅ Created directory: {directory}")
    
    def prepare_data_with_indicators(self):
        """Prepare dataset with technical indicators (from notebook cell 5)"""
        print("🔄 Preparing data with technical indicators...")
        
        # Load combined dataset
        df = pd.read_csv("all_currencies_combined.csv")
        df.columns = df.columns.str.strip().str.lower()
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Container for processed DataFrames
        enriched_dfs = []
        
        # Ensure 'return' is calculated for each group before indicators
        for pair in df.groupby("pair"):
            pair_name, group = pair
            group = group.sort_values("datetime").copy()
            group['return'] = group['close'].pct_change()
            
            print(f"Processing indicators for {pair_name}...")
            
            # Technical indicators
            group['sma_20'] = ta.trend.sma_indicator(group['close'], window=20)
            group['ema_20'] = ta.trend.ema_indicator(group['close'], window=20)
            group['rsi'] = ta.momentum.rsi(group['close'], window=14)
            group['macd'] = ta.trend.macd(group['close'])
            group['macd_signal'] = ta.trend.macd_signal(group['close'])
            
            bb = ta.volatility.BollingerBands(close=group['close'], window=20)
            group['bb_upper'] = bb.bollinger_hband()
            group['bb_lower'] = bb.bollinger_lband()
            group['bb_mid'] = bb.bollinger_mavg()
            
            # Support & resistance
            group['support'] = group['low'].rolling(window=20).min()
            group['resistance'] = group['high'].rolling(window=20).max()
            
            # Drop rows with NaNs from indicators
            group.dropna(inplace=True)
            enriched_dfs.append(group)
        
        # Combine all processed groups
        df_enriched = pd.concat(enriched_dfs, ignore_index=True)
        
        # Fill NaN in 'return' with 0
        if 'return' in df_enriched.columns:
            df_enriched['return'] = df_enriched['return'].fillna(0)
        
        print("Columns in df_enriched:", df_enriched.columns.tolist())
        print("Number of rows with NaN in 'return':", df_enriched['return'].isna().sum())
        
        # Add cost-aware label after all indicators are added
        cost_per_trade = 0.002 + 0.005
        if 'return' in df_enriched.columns:
            df_enriched['net_return'] = df_enriched['return'] - cost_per_trade
            df_enriched['label'] = (df_enriched['net_return'] > 0).astype(int)
        
        print("Number of rows with NaN in 'label':", df_enriched['label'].isna().sum())
        print("Label value counts:", df_enriched['label'].value_counts(dropna=False))
        
        # Save enriched dataset
        output_path = f"{self.models_dir}/data/all_currencies_with_indicators.csv"
        df_enriched.to_csv(output_path, index=False)
        print(f"✅ Enriched dataset saved to: {output_path}")
        
        return df_enriched
    
    def train_lstm_models(self, df):
        """Train LSTM models for each currency pair (from notebook cell 6 & 7)"""
        print("\n🔄 Training LSTM models...")
        
        # Features to use
        features = ['close', 'sma_20', 'ema_20', 'rsi', 'macd', 'macd_signal']
        sequence_length = 60
        
        # Group by currency pair
        for pair in df['pair'].unique():
            print(f"\nTraining LSTM for {pair}...")
            
            # Get data for this pair
            pair_df = df[df['pair'] == pair].copy()
            pair_df = pair_df.sort_values('datetime')
            pair_df = pair_df[features].dropna()
            
            if len(pair_df) < sequence_length + 100:  # Need sufficient data
                print(f"Skipping {pair} - insufficient data")
                continue
            
            # Scale features
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(pair_df)
            
            # Sequence generation
            X_lstm, y_lstm = [], []
            for i in range(sequence_length, len(scaled)):
                X_lstm.append(scaled[i - sequence_length:i])
                y_lstm.append(scaled[i, 0])  # predict 'close'
            
            X_lstm = np.array(X_lstm)
            y_lstm = np.array(y_lstm)
            
            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X_lstm, y_lstm, test_size=0.2, shuffle=False
            )
            
            # Model architecture
            model = Sequential([
                Input(shape=(X_lstm.shape[1], X_lstm.shape[2])),
                LSTM(64, return_sequences=True),
                Dropout(0.2),
                LSTM(64),
                Dropout(0.2),
                Dense(1)
            ])
            
            model.compile(optimizer='adam', loss='mse')
            
            # Train
            history = model.fit(
                X_train, y_train,
                epochs=20,
                batch_size=32,
                validation_data=(X_test, y_test),
                verbose=0
            )
            
            # Evaluate
            y_pred = model.predict(X_test, verbose=0)
            mse = mean_squared_error(y_test, y_pred)
            
            print(f"{pair} LSTM - MSE: {mse:.6f}")
            
            # Save model and scaler
            model_path = f"{self.models_dir}/lstm/lstm_model_{pair}.keras"
            scaler_path = f"{self.models_dir}/scalers/lstm_scaler_{pair}.pkl"
            
            model.save(model_path)
            joblib.dump(scaler, scaler_path)
            
            # Save predictions for comparison
            np.save(f"{self.models_dir}/predictions/lstm_pred_{pair}.npy", y_pred)
            np.save(f"{self.models_dir}/predictions/lstm_true_{pair}.npy", y_test)
            
            print(f"✅ LSTM model saved: {model_path}")
    
    def train_xgboost_models(self, df):
        """Train XGBoost models for each currency pair (improved version)"""
        print("\n🔄 Training XGBoost models...")
        
        # Use the exact features present in the current data
        features = [
            'sma_20', 'ema_20', 'rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower', 'bb_mid', 'support', 'resistance',
            'regime_hmm', 'bull_prob_hmm', 'regime_gmm', 'bull_prob_gmm', 'return', 'regime_hmm_vol', 'regime_gmm_vol',
            'regime_agreement', 'hmm_regime_duration', 'gmm_regime_duration', 'hmm_regime_change', 'gmm_regime_change',
            'regime_change_agreement', 'bull_prob_diff', 'bull_prob_ratio', 'bull_prob_avg', 'bull_prob_hmm_capped',
            'bull_prob_hmm_log', 'bull_prob_hmm_sqrt', 'bull_prob_hmm_zscore', 'vol_regime_hmm', 'vol_regime_gmm',
            'vol_bull_prob_hmm', 'vol_bull_prob_gmm', 'return_regime_hmm', 'return_regime_gmm', 'return_bull_prob_hmm',
            'return_bull_prob_gmm', 'regime_vol_interaction', 'regime_vol_diff', 'bull_prob_hmm_ma5', 'bull_prob_gmm_ma5',
            'bull_prob_hmm_std5', 'bull_prob_gmm_std5', 'price_range', 'volume_price_ratio', 'hour', 'day_of_week',
            'is_weekend', 'regime_strength_hmm', 'regime_strength_gmm', 'regime_strength_avg'
        ]
        # Save feature list for extraction
        with open('feature_list_enhanced.txt', 'w') as f:
            for feat in features:
                f.write(f"{feat}\n")

        target = 'label'  # Use cost-aware label for classification
        
        all_results = []
        
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        
        for pair in df['pair'].unique():
            print(f"\nTraining XGBoost for {pair}...")
            
            # Get data for this pair only
            pair_df = df[df['pair'] == pair].copy()
            pair_df = pair_df[features + [target]].dropna().reset_index(drop=True)
            
            if len(pair_df) < 100:
                print(f"Skipping {pair} - insufficient data")
                continue
            
            # Create lagged features
            sequence_length = 10
            X_sequences = []
            y_targets = []
            
            for i in range(sequence_length, len(pair_df)):
                feature_seq = pair_df.iloc[i-sequence_length:i][features].values
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
        
        # Step 4: Create comparison plots
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