# 🚀 Trading Bot - Déploiement Railway

## Description
Bot d'analyse crypto 24/7 avec Machine Learning intégré.

## Fonctionnalités
- ✅ Analyse 497 paires Binance Futures
- 🤖 ML predictions avec auto-entraînement
- 📊 Indicateurs techniques multiples
- 🎯 Alertes RSI intelligentes
- 💾 Sauvegarde automatique SQLite

## Déploiement Railway
1. Fork ce repo
2. Connecter à Railway
3. Déployer automatiquement
4. Variables d'env configurées dans `.env`

## Variables d'environnement
- `SOUND_ALERTS=false` - Désactive les sons (serveur)
- `ML_AUTO_RETRAIN=true` - Auto-entraînement ML
- `LOG_LEVEL=INFO` - Niveau de logs

## Surveillance
Logs disponibles dans l'interface Railway.

## Stack
- Python 3.11
- Docker
- scikit-learn
- ccxt (Binance)
- pandas-ta