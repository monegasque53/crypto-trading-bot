#!/usr/bin/env python3
"""
Version serveur de tendance_globale.py
Optimisée pour déploiement cloud sans interface audio
"""

import os
import sys

# Désactiver les alertes sonores sur serveur
os.environ['SOUND_ALERTS'] = 'false'

# Import du script principal
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import tendance_globale
    import logging
    from datetime import datetime
    
    def setup_server_logging():
        """Configuration des logs pour serveur"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('logs/server.log', encoding='utf-8')
            ]
        )
    
    def main():
        """Point d'entrée serveur"""
        setup_server_logging()
        
        print("🚀 DÉMARRAGE BOT TRADING SERVEUR")
        print(f"📅 Heure: {datetime.now()}")
        print("🔇 Mode serveur: alertes sonores désactivées")
        
        try:
            # Lancement direct du script principal
            print("▶️ Lancement du monitoring continu...")
            tendance_globale.main()
            
        except KeyboardInterrupt:
            print("\n🛑 Arrêt demandé par l'utilisateur")
        except Exception as e:
            print(f"❌ Erreur critique: {e}")
            logging.error(f"Erreur critique: {e}", exc_info=True)
            sys.exit(1)
    
    if __name__ == "__main__":
        main()
        
except ImportError as e:
    print(f"❌ Erreur d'import: {e}")
    print("📥 Assurez-vous que tendance_globale.py est présent")
    sys.exit(1)
