import os
import logging
import warnings
import urllib.request
import zipfile
from datetime import timedelta
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Modeling
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, classification_report
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import calibration_and_holdout_data, summary_data_from_transaction_data

# Data Contracts
from pydantic import BaseModel, Field, field_validator

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
LOG_DIR = os.path.join(BASE_DIR, 'logs')
ASSET_DIR = os.path.join(BASE_DIR, 'assets')
RAW_DATA_PATH = os.path.join(DATA_DIR, "online_retail_II.xlsx")

for d in [DATA_DIR, LOG_DIR, ASSET_DIR]:
    if not os.path.exists(d): os.makedirs(d)

logger = logging.getLogger("RetentionEngine")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(os.path.join(LOG_DIR, "app.log"))
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(message)s'))
if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

warnings.filterwarnings("ignore")

class TransactionSchema(BaseModel):
    Invoice: str
    StockCode: str
    Quantity: int
    InvoiceDate: Any 
    Price: float
    CustomerID: str

    @field_validator('Price')
    def check_price(cls, v):
        if v < 0: raise ValueError("Price cannot be negative")
        return v

class AssetGenerator:
    
    @staticmethod
    def plot_model_comparison(results: pd.DataFrame, save_path: str):
        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")
        
        for name, data in results.iterrows():
            plt.plot(
                data['fpr'], data['tpr'], 
                label=f"{name} (AUC={data['auc']:.3f})",
                linewidth=2.5
            )
            
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.title('Model Comparison: ROC-AUC', fontsize=14)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

    @staticmethod
    def plot_feature_importance(model, feature_names: List[str], save_path: str):
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            return

        df = pd.DataFrame({'feature': feature_names, 'importance': importances})
        df = df.sort_values('importance', ascending=False).head(10)
        
        plt.figure(figsize=(10, 5))
        sns.barplot(x='importance', y='feature', data=df, palette="viridis")
        plt.title('Top 10 Drivers of Customer Churn', fontsize=14)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

    @staticmethod
    def plot_intervention_impact(plan: pd.DataFrame, save_path: str):
        if plan.empty: return
        
        plt.figure(figsize=(10, 6))
        sns.histplot(plan['value_at_risk'], kde=True, color='#2ecc71', label='Value at Risk')
        plt.axvline(plan['cost'].mean(), color='red', linestyle='--', label='Intervention Cost')
        plt.title('Why We Intervene: Value at Risk Distribution', fontsize=14)
        plt.xlabel('Potential Revenue Loss ($)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

class RetentionEngine:
    def __init__(self, obs_window=90):
        self.obs_window = obs_window
        self.scaler = RobustScaler()
        self.is_fitted = False
        self.best_model_name = None
        self.risk_model = None
        
        # Probabilistic Value Models
        self.bg_nbd = BetaGeoFitter(penalizer_coef=0.01)
        self.gamma_gamma = GammaGammaFitter(penalizer_coef=0.01)
        
        self.feature_cols = [
            'frequency_cal', 'recency_cal', 'T_cal', 'monetary_value_cal',
            'velocity', 'consistency', 'avg_interpurchase_days'
        ]

    def _ensure_data(self):
        if not os.path.exists(RAW_DATA_PATH):
            logger.info("Downloading UCI Online Retail II dataset...")
            url = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
            zip_path = os.path.join(DATA_DIR, "temp.zip")
            urllib.request.urlretrieve(url, zip_path)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(DATA_DIR)
            os.remove(zip_path)
            logger.info("Download complete.")

    def load_data(self) -> pd.DataFrame:
        self._ensure_data()
        logger.info("Loading Excel data (Year 2010-2011)...")
        df = pd.read_excel(RAW_DATA_PATH, sheet_name="Year 2010-2011")
        
        # Cleaning
        df = df.dropna(subset=['Customer ID', 'InvoiceDate'])
        df['Customer ID'] = df['Customer ID'].astype(str).str.split('.').str[0]
        df['TotalValue'] = df['Quantity'] * df['Price']
        
        # Remove cancelled orders (C prefix) for modeling stability
        df = df[~df['Invoice'].astype(str).str.startswith('C')]
        df = df[df['TotalValue'] > 0]
        
        logger.info(f"Loaded {len(df):,} transactions for {df['Customer ID'].nunique():,} customers.")
        return df

    def _enrich_features(self, summary_df: pd.DataFrame) -> pd.DataFrame:
        summary_df['velocity'] = summary_df['frequency_cal'] / (summary_df['T_cal'] + 1e-5)
        
        summary_df['avg_interpurchase_days'] = summary_df['recency_cal'] / (summary_df['frequency_cal'] + 1e-5)
        
        summary_df['consistency'] = summary_df['frequency_cal'] * summary_df['recency_cal']
        
        return summary_df

    def run_model_comparison(self, X, y) -> pd.DataFrame:
        logger.info("Running model comparison...")
        
        candidates = {
            "LogisticRegression": {
                "model": LogisticRegression(class_weight='balanced', solver='liblinear'),
                "params": {
                    "C": [0.01, 0.1, 1, 10], 
                    "penalty": ["l1", "l2"]
                }
            },
            "RandomForest": {
                "model": RandomForestClassifier(class_weight='balanced', n_jobs=-1),
                "params": {
                    "n_estimators": [100, 200],
                    "max_depth": [5, 10, 15],
                    "min_samples_split": [5, 10]
                }
            },
            "XGBoost": {
                "model": xgb.XGBClassifier(eval_metric='logloss', scale_pos_weight=5, n_jobs=-1),
                "params": {
                    "n_estimators": [100, 200, 300],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "max_depth": [3, 4, 5],
                    "colsample_bytree": [0.7, 0.8, 1.0]
                }
            }
        }

        results = []
        best_auc = 0
        X_scaled = self.scaler.fit_transform(X) # Scale once
        
        for name, config in candidates.items():
            logger.info(f"Tuning {name} (5-Fold CV)...")
            
            search = RandomizedSearchCV(
                config["model"],
                config["params"],
                n_iter=10,
                scoring="roc_auc",
                cv=StratifiedKFold(n_splits=5),
                n_jobs=-1,
                random_state=42,
                verbose=0
            )
            search.fit(X_scaled, y)
            
            # Metrics
            best_est = search.best_estimator_
            probs = best_est.predict_proba(X_scaled)[:, 1]
            auc = roc_auc_score(y, probs)
            fpr, tpr, _ = roc_curve(y, probs)
            
            logger.info(f"   -> Best AUC: {auc:.4f} | Params: {search.best_params_}")
            
            results.append({
                "name": name,
                "auc": auc,
                "fpr": fpr,
                "tpr": tpr,
                "model": best_est
            })
            
            if auc > best_auc:
                best_auc = auc
                self.risk_model = best_est
                self.best_model_name = name

        logger.info(f"Best model: {self.best_model_name} (AUC={best_auc:.4f})")
        return pd.DataFrame(results).set_index("name")

    def train(self, df: pd.DataFrame, tune_hyperparams=True):
        logger.info("Generating calibration/holdout features...")
        max_date = df['InvoiceDate'].max()
        split_date = max_date - timedelta(days=self.obs_window)
        
        summary = calibration_and_holdout_data(
            df, 'Customer ID', 'InvoiceDate',
            calibration_period_end=split_date,
            observation_period_end=max_date,
            monetary_value_col='TotalValue'
        )
        
        summary = self._enrich_features(summary)
        summary['churned'] = (summary['frequency_holdout'] == 0).astype(int)
        
        summary = summary[summary['frequency_cal'] > 0]
        X = summary[self.feature_cols]
        y = summary['churned']
        
        if tune_hyperparams:
            results_df = self.run_model_comparison(X, y)
            # Generate Assets
            AssetGenerator.plot_model_comparison(
                results_df, os.path.join(ASSET_DIR, "model_comparison.png")
            )
            AssetGenerator.plot_feature_importance(
                self.risk_model, self.feature_cols, os.path.join(ASSET_DIR, "feature_importance.png")
            )
        else:
            self.risk_model = xgb.XGBClassifier()
            self.risk_model.fit(self.scaler.fit_transform(X), y)
            
        logger.info("Fitting value models (BG/NBD + Gamma-Gamma)...")
        self.bg_nbd.fit(summary['frequency_cal'], summary['recency_cal'], summary['T_cal'])
        self.gamma_gamma.fit(summary['frequency_cal'], summary['monetary_value_cal'])
        
        self.is_fitted = True
        return summary

    def optimize_budget(self, df: pd.DataFrame, budget: float, cost: float) -> pd.DataFrame:
        if not self.is_fitted: raise RuntimeError("Train first!")
        
        logger.info("Running inference on current customer base...")
        
        current_rfm = summary_data_from_transaction_data(
            df, 'Customer ID', 'InvoiceDate', monetary_value_col='TotalValue'
        )
        current_rfm = self._enrich_features(current_rfm.rename(columns={
            'frequency': 'frequency_cal', 'recency': 'recency_cal', 
            'T': 'T_cal', 'monetary_value': 'monetary_value_cal'
        }))
        current_rfm = current_rfm[current_rfm['frequency_cal'] > 0]
        
        X_inf = current_rfm[self.feature_cols]
        
        X_scaled = self.scaler.transform(X_inf)
        churn_prob = self.risk_model.predict_proba(X_scaled)[:, 1]
        
        pred_ltv = self.gamma_gamma.customer_lifetime_value(
            self.bg_nbd, 
            current_rfm['frequency_cal'], 
            current_rfm['recency_cal'], 
            current_rfm['T_cal'], 
            current_rfm['monetary_value_cal'], 
            time=3,
            discount_rate=0.01
        )
        
        results = pd.DataFrame({
            'customer_id': current_rfm.index,
            'churn_prob': churn_prob,
            'predicted_ltv': pred_ltv
        })
        
        results['value_at_risk'] = results['churn_prob'] * results['predicted_ltv']
        
        candidates = results[results['value_at_risk'] > cost].copy()
        
        candidates = candidates.sort_values('value_at_risk', ascending=False)
        
        selected = []
        spent = 0
        for _, row in candidates.iterrows():
            if spent + cost <= budget:
                selected.append(row)
                spent += cost
            else:
                break
        
        plan = pd.DataFrame(selected)
        if not plan.empty:
            plan['action'] = 'Intervene'
            plan['cost'] = cost
            plan['roi'] = plan['value_at_risk'] / cost
            
            # Generate Report Asset
            AssetGenerator.plot_intervention_impact(
                plan, os.path.join(ASSET_DIR, "intervention_distribution.png")
            )
        
        return plan