#!/usr/bin/env python3
"""
Auto-trainer ML pour tendance_globale.py
Réentraîne automatiquement le modèle ML quand assez de nouvelles données sont disponibles.
"""

import os
import sqlite3
import json
import logging
from datetime import datetime, timedelta
from typing import Optional

def should_retrain_model(db_path: str = "multi_indicator_macro_history.db", 
                        model_path: str = "models/market_classifier.pkl",
                        min_new_rows: int = 4,   # 4h de nouvelles données
                        force_days: int = 3) -> tuple[bool, str]:
    """
    Détermine si le modèle doit être réentraîné.
    
    Args:
        db_path: Chemin vers la base SQLite
        model_path: Chemin vers le modèle ML
        min_new_rows: Nombre minimum de nouvelles lignes pour déclencher un réentraînement
        force_days: Force le réentraînement après X jours même sans nouvelles données
    
    Returns:
        (should_retrain: bool, reason: str)
    """
    
    # 1. Vérifier si le modèle existe
    if not os.path.exists(model_path):
        return True, f"Modèle inexistant: {model_path}"
    
    # 2. Vérifier si la DB existe
    if not os.path.exists(db_path):
        return False, f"Base de données inexistante: {db_path}"
    
    try:
        # 3. Obtenir l'âge du modèle
        model_mtime = datetime.fromtimestamp(os.path.getmtime(model_path))
        model_age_days = (datetime.now() - model_mtime).days
        
        # 4. Compter les nouvelles données depuis la dernière modification du modèle
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Total de lignes
            cursor.execute("SELECT COUNT(*) FROM market_history")
            total_rows = cursor.fetchone()[0]
            
            if total_rows == 0:
                return False, "Aucune donnée en base"
            
            # Nouvelles lignes depuis la modification du modèle
            model_timestamp = model_mtime.strftime('%Y-%m-%d %H:%M:%S')
            cursor.execute(
                "SELECT COUNT(*) FROM market_history WHERE timestamp > ?",
                (model_timestamp,)
            )
            new_rows = cursor.fetchone()[0]
            
            # 5. Décisions de réentraînement
            if new_rows >= min_new_rows:
                return True, f"{new_rows} nouvelles lignes (>= {min_new_rows} requis)"
            
            if model_age_days >= force_days:
                return True, f"Modèle ancien de {model_age_days} jours (>= {force_days} jours max)"
            
            return False, f"Pas besoin: {new_rows} nouvelles lignes, âge {model_age_days}j"
            
    except Exception as e:
        return False, f"Erreur vérification: {e}"


def auto_retrain_if_needed(force: bool = False, 
                          db_path: str = "multi_indicator_macro_history.db",
                          model_path: str = "models/market_classifier.pkl") -> bool:
    """
    Réentraîne automatiquement le modèle si nécessaire.
    
    Args:
        force: Force le réentraînement même si pas nécessaire
        db_path: Chemin vers la base
        model_path: Chemin vers le modèle
    
    Returns:
        True si réentraînement effectué, False sinon
    """
    
    print("🤖 AUTO-TRAINER ML")
    print("=" * 30)
    
    # Vérifier si réentraînement nécessaire
    if not force:
        should_retrain, reason = should_retrain_model(db_path, model_path)
        print(f"Évaluation: {reason}")
        
        if not should_retrain:
            print("✅ Pas de réentraînement nécessaire")
            return False
    else:
        print("🚀 Réentraînement forcé")
    
    print("🔄 Lancement du réentraînement...")
    
    try:
        # Import du module ML
        from ml_model import train_model
        
        # Sauvegarder l'ancien modèle
        if os.path.exists(model_path):
            backup_path = model_path.replace('.pkl', f'_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl')
            os.rename(model_path, backup_path)
            print(f"📦 Ancien modèle sauvé: {backup_path}")
        
        # Réentraîner
        result = train_model(db_path, model_path)
        
        print("✅ RÉENTRAÎNEMENT RÉUSSI !")
        print(f"📊 Données train: {result['train_size']}")
        print(f"📊 Données test: {result['test_size']}")
        print(f"💾 Modèle sauvé: {result['model_path']}")
        print("\n📈 Performance:")
        print(result['report'])
        
        # Log dans un fichier
        log_retrain_event(result, reason if not force else "Forcé")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur réentraînement: {e}")
        
        # Restaurer l'ancien modèle si erreur
        if 'backup_path' in locals() and os.path.exists(backup_path):
            os.rename(backup_path, model_path)
            print(f"🔄 Ancien modèle restauré: {model_path}")
        
        return False


def log_retrain_event(result: dict, reason: str):
    """Log les événements de réentraînement"""
    log_file = "ml_retrain_history.json"
    
    event = {
        "timestamp": datetime.now().isoformat(),
        "reason": reason,
        "train_size": result.get('train_size', 0),
        "test_size": result.get('test_size', 0),
        "model_path": result.get('model_path', ''),
        "success": True
    }
    
    # Charger l'historique existant
    history = []
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
        except Exception:
            pass
    
    # Ajouter le nouvel événement
    history.append(event)
    
    # Garder seulement les 50 derniers événements
    history = history[-50:]
    
    # Sauvegarder
    try:
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"⚠️ Impossible de logger l'événement: {e}")


def get_retrain_status() -> dict:
    """Obtient le statut du système de réentraînement"""
    db_path = "multi_indicator_macro_history.db"
    model_path = "models/market_classifier.pkl"
    
    status = {
        "model_exists": os.path.exists(model_path),
        "db_exists": os.path.exists(db_path),
        "should_retrain": False,
        "reason": "",
        "last_retrain": None,
        "total_retrains": 0
    }
    
    if status["model_exists"]:
        status["model_age_hours"] = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(model_path))).total_seconds() / 3600
    
    if status["db_exists"]:
        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM market_history")
                status["total_data_rows"] = cursor.fetchone()[0]
        except Exception:
            status["total_data_rows"] = 0
    
    # Vérifier si réentraînement nécessaire
    status["should_retrain"], status["reason"] = should_retrain_model(db_path, model_path)
    
    # Historique des réentraînements
    log_file = "ml_retrain_history.json"
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
            status["total_retrains"] = len(history)
            if history:
                status["last_retrain"] = history[-1]["timestamp"]
        except Exception:
            pass
    
    return status


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Auto-trainer ML pour tendance_globale.py")
    parser.add_argument("--force", action="store_true", help="Force le réentraînement")
    parser.add_argument("--status", action="store_true", help="Affiche le statut")
    parser.add_argument("--check", action="store_true", help="Vérifie seulement si réentraînement nécessaire")
    
    args = parser.parse_args()
    
    if args.status:
        status = get_retrain_status()
        print("📊 STATUT AUTO-TRAINER:")
        print("-" * 25)
        print(f"Modèle existe: {'✅' if status['model_exists'] else '❌'}")
        print(f"Base données: {'✅' if status['db_exists'] else '❌'}")
        if status['model_exists']:
            print(f"Âge modèle: {status['model_age_hours']:.1f}h")
        if status['db_exists']:
            print(f"Lignes données: {status['total_data_rows']}")
        print(f"Réentraînements: {status['total_retrains']}")
        if status['last_retrain']:
            print(f"Dernier: {status['last_retrain']}")
        print(f"Besoin réentraînement: {'✅' if status['should_retrain'] else '❌'}")
        print(f"Raison: {status['reason']}")
        
    elif args.check:
        should, reason = should_retrain_model()
        print(f"Réentraînement nécessaire: {'✅' if should else '❌'}")
        print(f"Raison: {reason}")
    else:
        auto_retrain_if_needed(force=args.force)