"""
ML Trainer/Inferencer for market context produced by tendance_globale.py

Features source: SQLite DB table 'market_history' (written by tendance_globale.py)
Target: next-hour class of final_score_1h (Bullish / Neutral / Bearish)

Usage (train):
  python ml_model.py --train --db multi_indicator_macro_history.db --out models/market_classifier.pkl

Usage (predict one-off from current metrics dict):
  from ml_model import load_model_or_none, features_from_metrics
  bundle = load_model_or_none('models/market_classifier.pkl')
  if bundle:
      model = bundle['model']
      feats = features_from_metrics(metrics)
      y_prob = model.predict_proba(feats)[0]
      y_pred = model.classes_[y_prob.argmax()]
"""

from __future__ import annotations

import os
import sqlite3
import argparse
from typing import Dict, Optional

import pandas as pd

def _safe_import_sklearn():
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import classification_report
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        import joblib
        return {
            'RandomForestClassifier': RandomForestClassifier,
            'TimeSeriesSplit': TimeSeriesSplit,
            'classification_report': classification_report,
            'Pipeline': Pipeline,
            'StandardScaler': StandardScaler,
            'joblib': joblib,
        }
    except Exception:
        return None


FEATURES = [
    # 1h features (techniques)
    'rsi_1h_simple',
    'macd_histogram_1h_simple',
    'obv_momentum_1h_simple',
    'composite_score_1h',
    'final_score_1h',
    # Cadres plus longs (résumés)
    'final_score_1d', 'final_score_1w', 'final_score_1M',
    # Macro
    'macro_score', 'fear_greed', 'vix', 'dxy_change', 'sp500_change',
]


def load_data_from_db(db_path: str) -> pd.DataFrame:
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"DB introuvable: {db_path}")
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query("SELECT * FROM market_history ORDER BY timestamp", conn)
    # Parse timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp']).sort_values('timestamp')
    return df


def build_dataset(df: pd.DataFrame) -> pd.DataFrame:
    # Keep only necessary columns
    cols_needed = ['timestamp'] + FEATURES
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans DB: {missing}")

    data = df[cols_needed].copy()
    # Target: next-hour class of final_score_1h (shift -1)
    data['target_score_next'] = df['final_score_1h'].shift(-1)

    def score_to_class(x: float) -> str:
        try:
            if x > 20:
                return 'Bullish'
            if x < -20:
                return 'Bearish'
            return 'Neutral'
        except Exception:
            return None

    data['target_class'] = data['target_score_next'].apply(score_to_class)
    data = data.dropna(subset=['target_class'])
    return data


def train_model(db_path: str, out_path: str) -> Dict:
    sk = _safe_import_sklearn()
    if sk is None:
        raise RuntimeError("scikit-learn/joblib non installés. Installez-les avec: pip install scikit-learn joblib")

    df = load_data_from_db(db_path)
    data = build_dataset(df)

    X = data[FEATURES].fillna(0.0)
    y = data['target_class']

    # Time-based split: last 20% as test
    n = len(data)
    if n < 50:
        raise RuntimeError("Pas assez de données pour entraîner un modèle (>= 50 lignes recommandé).")
    split_idx = int(n * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = sk['RandomForestClassifier'](n_estimators=300, random_state=42, class_weight='balanced_subsample')
    pipe = sk['Pipeline']([
        ('scaler', sk['StandardScaler']()),
        ('clf', model),
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    report = sk['classification_report'](y_test, y_pred, digits=3)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sk['joblib'].dump({'model': pipe, 'features': FEATURES}, out_path)

    return {'report': report, 'model_path': out_path, 'train_size': len(X_train), 'test_size': len(X_test)}


def load_model_or_none(path: str) -> Optional[Dict]:
    sk = _safe_import_sklearn()
    if sk is None:
        return None
    if not os.path.exists(path):
        return None
    try:
        bundle = sk['joblib'].load(path)
        # sanity
        if not isinstance(bundle, dict) or 'model' not in bundle or 'features' not in bundle:
            return None
        return bundle
    except Exception:
        return None


def features_from_metrics(metrics: Dict) -> pd.DataFrame:
    # metrics is the dict computed by calculate_global_market_metrics
    # Map to the FEATURES order
    feat_vals = {
        'rsi_1h_simple': metrics['rsi_simple'].get('1h', 0.0),
        'macd_histogram_1h_simple': metrics['macd_histogram_simple'].get('1h', 0.0),
        'obv_momentum_1h_simple': metrics['obv_momentum_simple'].get('1h', 0.0),
        'composite_score_1h': metrics['composite_score_simple'].get('1h', 0.0),
        'final_score_1h': metrics['final_score_simple'].get('1h', 0.0),
        'final_score_1d': metrics['final_score_simple'].get('1d', 0.0),
        'final_score_1w': metrics['final_score_simple'].get('1w', 0.0),
        'final_score_1M': metrics['final_score_simple'].get('1M', 0.0),
        'macro_score': metrics['macro_analysis'].get('score', 0.0),
        'fear_greed': (metrics['macro_data'].get('fear_greed') or {}).get('value', 0.0) if metrics.get('macro_data') else 0.0,
        'vix': (metrics['macro_data'].get('vix') or {}).get('value', 0.0) if metrics.get('macro_data') else 0.0,
        'dxy_change': (metrics['macro_data'].get('dxy') or {}).get('change_pct', 0.0) if metrics.get('macro_data') else 0.0,
        'sp500_change': (metrics['macro_data'].get('sp500') or {}).get('change_pct', 0.0) if metrics.get('macro_data') else 0.0,
    }
    row = [[feat_vals.get(k, 0.0) for k in FEATURES]]
    return pd.DataFrame(row, columns=FEATURES)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Entraîner le modèle')
    parser.add_argument('--db', default='multi_indicator_macro_history.db', help='Chemin vers la base SQLite')
    parser.add_argument('--out', default=os.path.join('models', 'market_classifier.pkl'), help='Chemin de sortie du modèle')
    args = parser.parse_args()

    if args.train:
        res = train_model(args.db, args.out)
        print("=== Rapport de performance (test) ===")
        print(res['report'])
        print(f"Modèle enregistré → {res['model_path']}")
    else:
        print("Rien à faire. Utilisez --train pour entraîner un modèle.")


if __name__ == '__main__':
    main()
