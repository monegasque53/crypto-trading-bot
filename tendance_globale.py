#!/usr/bin/env python3
"""
Moniteur Multi-Indicateurs Binance Futures - Version Complète avec Données Macroéconomiques
- Combine RSI, MACD et OBV pour une analyse technique complète
- Intègre données macroéconomiques gratuites (FRED, Yahoo Finance, Alpha Vantage)
- Inclus versions Pondérées (Volume) et Simples (Arithmétriques)
- Unités de temps 1H, 1D, 1W et 1M
- Analyse fondamentale + technique pour signaux avancés
- Design amélioré avec la bibliothèque rich
- Persistance de l'historique via une base de données SQLite
- Alerte sonore lors des changements de tendance
"""

import requests
import argparse
import logging
from logging.handlers import RotatingFileHandler
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import statistics
# import schedule  # non utilisé
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
from collections import deque
import json
import os
from enum import Enum
import yfinance as yf
import sqlite3
from playsound import playsound
from requests import HTTPError

# Fallback possible via ccxt (import paresseux dans la méthode)

# Imports pour le design du terminal
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.text import Text
# from rich.align import Align  # non utilisé
# from rich.layout import Layout  # non utilisé

# Import ML optionnel (ne casse pas si scikit-learn non installé ou modèle absent)
try:
    from ml_model import load_model_or_none, features_from_metrics
except Exception:
    load_model_or_none = None
    features_from_metrics = None

class MarketSentiment(Enum):
    EXTREME_BEARISH = "extreme_bearish"
    BEARISH = "bearish"
    WEAK_BEARISH = "weak_bearish"
    NEUTRAL = "neutral"
    WEAK_BULLISH = "weak_bullish"
    BULLISH = "bullish"
    EXTREME_BULLISH = "extreme_bullish"

class TradingSignal(Enum):
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    WEAK_BUY = "weak_buy"
    HOLD = "hold"
    WEAK_SELL = "weak_sell"
    SELL = "sell"
    STRONG_SELL = "strong_sell"

class MacroIndicators:
    """Classe pour gérer les données macroéconomiques gratuites."""
    
    def __init__(self):
        self.console = Console()
        self.cache_duration = 3600  # Cache de 1 heure
        self.cache = {}
        # Cache persistant sur disque pour résilience réseau
        self.macro_cache_file = os.environ.get('MACRO_CACHE_FILE', 'macro_cache.json')
        try:
            self.macro_cache_ttl = int(os.environ.get('MACRO_CACHE_TTL_SECONDS', '86400'))  # 24h
        except Exception:
            self.macro_cache_ttl = 86400

    # ------------------ Helpers Résilience ------------------
    def _get_json_with_retries(self, url: str, timeout: int = 10, retries: int = 2) -> Optional[Dict]:
        """GET JSON avec quelques retries simples."""
        for i in range(max(1, retries)):
            try:
                resp = requests.get(url, timeout=timeout)
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                if i == retries - 1:
                    self.console.log(
                        f"[yellow]⚠️ Échec GET {url}: {e}[/yellow]"
                    )
                else:
                    time.sleep(1 + i)
        return None

    def _yf_history_try(self, tickers: List[str], periods: List[str]) -> Tuple[pd.DataFrame, Optional[str]]:
        """Essaye plusieurs tickers/périodes pour retourner un historique non vide."""
        for tk in tickers:
            try:
                t = yf.Ticker(tk)
                for per in periods:
                    hist = t.history(period=per)
                    if hist is not None and not hist.empty:
                        return hist, tk
            except Exception:
                continue
        return pd.DataFrame(), None
    
    def _load_macro_cache(self) -> Dict:
        try:
            if not os.path.exists(self.macro_cache_file):
                return {}
            with open(self.macro_cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            ts = data.get('timestamp')
            payload = data.get('macro_data') or {}
            if not ts or not payload:
                return {}
            ts_str = str(ts)
            if ts_str.endswith('Z'):
                ts_str = ts_str.replace('Z', '+00:00')
            try:
                cache_dt = datetime.fromisoformat(ts_str)
            except Exception:
                return {}
            age_seconds = max(0, int((datetime.utcnow() - cache_dt.replace(tzinfo=None)).total_seconds()))
            if age_seconds <= self.macro_cache_ttl:
                return payload
            return {}
        except Exception:
            return {}

    def _write_macro_cache(self, macro_data: Dict) -> None:
        try:
            payload = {
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'macro_data': macro_data
            }
            with open(self.macro_cache_file, 'w', encoding='utf-8') as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            self.console.log("[green]💾 Cache macro écrit → macro_cache.json[/green]")
        except Exception as e:
            self.console.log(f"[yellow]⚠️ Impossible d'écrire le cache macro: {e}[/yellow]")
        
    def get_fear_greed_index(self) -> Optional[Dict]:
        """Récupère l'indice Fear & Greed de CNN (API gratuite)."""
        try:
            url = "https://api.alternative.me/fng/"
            data = self._get_json_with_retries(url, timeout=10, retries=2)
            
            if data and 'data' in data and len(data['data']) > 0:
                latest = data['data'][0]
                return {
                    'value': int(latest['value']),
                    'classification': latest['value_classification'],
                    'timestamp': latest['timestamp']
                }
        except Exception as e:
            self.console.log(f"[yellow]⚠️ Erreur Fear & Greed Index: {e}[/yellow]")
        return None
    
    def get_vix_data(self) -> Optional[Dict]:
        """Récupère les données VIX via Yahoo Finance, avec fallback ETF SPY volatilité approximée."""
        try:
            hist, used = self._yf_history_try([
                "^VIX",  # principal
            ], ["5d", "1mo"])
            # Fallback approximatif via ETF volatilité (si aucune donnée VIX)
            if hist.empty:
                hist, used = self._yf_history_try(["VIXY", "VXX"], ["5d", "1mo"])
            if not hist.empty:
                current_vix = float(hist['Close'].iloc[-1])
                prev_vix = float(hist['Close'].iloc[-2]) if len(hist) > 1 else current_vix
                change = current_vix - prev_vix
                
                if current_vix < 20:
                    classification = "Faible Volatilité"
                    sentiment = "Bullish"
                elif current_vix < 30:
                    classification = "Volatilité Modérée"
                    sentiment = "Neutre"
                elif current_vix < 40:
                    classification = "Haute Volatilité"
                    sentiment = "Bearish"
                else:
                    classification = "Volatilité Extrême"
                    sentiment = "Très Bearish"
                
                return {
                    'value': round(current_vix, 2),
                    'change': round(change, 2),
                    'classification': classification,
                    'sentiment': sentiment
                }
        except Exception as e:
            self.console.log(f"[yellow]⚠️ Erreur VIX: {e}[/yellow]")
        return None
    
    def get_dxy_data(self) -> Optional[Dict]:
        """Récupère l'indice Dollar (DXY) via Yahoo Finance avec fallback DX=F (futures)."""
        try:
            hist, used = self._yf_history_try([
                "DX-Y.NYB",  # DXY index
                "DX=F",      # Dollar Index futures
            ], ["5d", "1mo"])
            if not hist.empty:
                current_dxy = float(hist['Close'].iloc[-1])
                prev_dxy = float(hist['Close'].iloc[-2]) if len(hist) > 1 else current_dxy
                change = current_dxy - prev_dxy
                change_pct = (change / prev_dxy) * 100 if prev_dxy != 0 else 0
                
                if change_pct > 0.5:
                    crypto_impact = "Négatif Fort"
                elif change_pct > 0.1:
                    crypto_impact = "Négatif Faible"
                elif change_pct < -0.5:
                    crypto_impact = "Positif Fort"
                elif change_pct < -0.1:
                    crypto_impact = "Positif Faible"
                else:
                    crypto_impact = "Neutre"
                
                return {
                    'value': round(current_dxy, 2),
                    'change': round(change, 3),
                    'change_pct': round(change_pct, 2),
                    'crypto_impact': crypto_impact
                }
        except Exception as e:
            self.console.log(f"[yellow]⚠️ Erreur DXY: {e}[/yellow]")
        return None
    
    def get_gold_data(self) -> Optional[Dict]:
        """Récupère les données de l'or via Yahoo Finance avec fallback ETF GLD."""
        try:
            hist, used = self._yf_history_try(["GC=F", "GLD"], ["5d", "1mo"])
            if not hist.empty:
                current_gold = float(hist['Close'].iloc[-1])
                prev_gold = float(hist['Close'].iloc[-2]) if len(hist) > 1 else current_gold
                change_pct = ((current_gold - prev_gold) / prev_gold) * 100 if prev_gold != 0 else 0
                
                if change_pct > 1:
                    risk_sentiment = "Risk-Off Fort"
                elif change_pct > 0.2:
                    risk_sentiment = "Risk-Off Modéré"
                elif change_pct < -1:
                    risk_sentiment = "Risk-On Fort"
                elif change_pct < -0.2:
                    risk_sentiment = "Risk-On Modéré"
                else:
                    risk_sentiment = "Neutre"
                
                return {
                    'value': round(current_gold, 2),
                    'change_pct': round(change_pct, 2),
                    'risk_sentiment': risk_sentiment
                }
        except Exception as e:
            self.console.log(f"[yellow]⚠️ Erreur Gold: {e}[/yellow]")
        return None
    
    def get_us_10y_yield(self) -> Optional[Dict]:
        """Récupère le rendement des obligations US 10 ans; fallback via ETF IEF (approx)."""
        try:
            hist, used = self._yf_history_try(["^TNX"], ["5d", "1mo"])
            if hist.empty:
                hist, used = self._yf_history_try(["IEF"], ["5d", "1mo"])  # approx via prix IEF
            if not hist.empty:
                current_yield = float(hist['Close'].iloc[-1])
                prev_yield = float(hist['Close'].iloc[-2]) if len(hist) > 1 else current_yield
                change = current_yield - prev_yield
                
                if change > 0.1:
                    impact = "Baissier Actifs Risqués"
                elif change > 0.05:
                    impact = "Légèrement Baissier"
                elif change < -0.1:
                    impact = "Haussier Actifs Risqués"
                elif change < -0.05:
                    impact = "Légèrement Haussier"
                else:
                    impact = "Neutre"
                
                return {
                    'value': round(current_yield, 3),
                    'change': round(change, 3),
                    'impact': impact
                }
        except Exception as e:
            self.console.log(f"[yellow]⚠️ Erreur US 10Y: {e}[/yellow]")
        return None
    
    def get_btc_dominance(self) -> Optional[Dict]:
        """Récupère la dominance Bitcoin via CoinGecko API gratuite."""
        try:
            url = "https://api.coingecko.com/api/v3/global"
            data = self._get_json_with_retries(url, timeout=10, retries=2)
            
            if 'data' in data and 'market_cap_percentage' in data['data']:
                btc_dominance = data['data']['market_cap_percentage'].get('btc', 0)
                
                if btc_dominance > 50:
                    trend = "Bitcoin Season"
                    altcoin_sentiment = "Bearish"
                elif btc_dominance > 45:
                    trend = "Bitcoin Dominant"
                    altcoin_sentiment = "Faible"
                elif btc_dominance < 40:
                    trend = "Alt Season"
                    altcoin_sentiment = "Bullish"
                else:
                    trend = "Transition"
                    altcoin_sentiment = "Neutre"
                
                return {
                    'value': round(btc_dominance, 2),
                    'trend': trend,
                    'altcoin_sentiment': altcoin_sentiment
                }
        except Exception as e:
            self.console.log(f"[yellow]⚠️ Erreur BTC Dominance: {e}[/yellow]")
        return None
    
    def get_sp500_data(self) -> Optional[Dict]:
        """Récupère les données S&P 500 pour la corrélation crypto; fallback ETF SPY."""
        try:
            hist, used = self._yf_history_try(["^GSPC", "SPY"], ["5d", "1mo"])
            if not hist.empty:
                current_sp500 = float(hist['Close'].iloc[-1])
                prev_sp500 = float(hist['Close'].iloc[-2]) if len(hist) > 1 else current_sp500
                change_pct = ((current_sp500 - prev_sp500) / prev_sp500) * 100 if prev_sp500 != 0 else 0
                
                if change_pct > 1:
                    crypto_correlation = "Haussier Fort"
                elif change_pct > 0.3:
                    crypto_correlation = "Haussier Modéré"
                elif change_pct < -1:
                    crypto_correlation = "Baissier Fort"
                elif change_pct < -0.3:
                    crypto_correlation = "Baissier Modéré"
                else:
                    crypto_correlation = "Neutre"
                
                return {
                    'value': round(current_sp500, 2),
                    'change_pct': round(change_pct, 2),
                    'crypto_correlation': crypto_correlation
                }
        except Exception as e:
            self.console.log(f"[yellow]⚠️ Erreur S&P500: {e}[/yellow]")
        return None
    
    def get_all_macro_data(self) -> Dict:
        """Récupère toutes les données macroéconomiques en parallèle."""
        macro_data = {}
        
        with ThreadPoolExecutor(max_workers=7) as executor:
            futures = {
                'fear_greed': executor.submit(self.get_fear_greed_index),
                'vix': executor.submit(self.get_vix_data),
                'dxy': executor.submit(self.get_dxy_data),
                'gold': executor.submit(self.get_gold_data),
                'us_10y': executor.submit(self.get_us_10y_yield),
                'btc_dominance': executor.submit(self.get_btc_dominance),
                'sp500': executor.submit(self.get_sp500_data)
            }
            
            for key, future in futures.items():
                try:
                    result = future.result(timeout=15)
                    macro_data[key] = result
                except Exception as e:
                    self.console.log(f"[yellow]⚠️ Erreur pendant la récupération de {key}: {e}[/yellow]")
                    macro_data[key] = None
        # Fallback: compléter les clés manquantes depuis le cache disque récent
        try:
            missing = [k for k, v in macro_data.items() if v is None]
            if missing:
                cached = self._load_macro_cache()
                if cached:
                    for k in missing:
                        if cached.get(k) is not None:
                            macro_data[k] = cached.get(k)
                    if any(macro_data[k] is not None for k in missing):
                        self.console.log("[dim]ℹ️ Macro: valeurs manquantes complétées depuis le cache local.[/dim]")
        except Exception:
            pass
        # Écrire le cache si au moins un indicateur est présent
        try:
            if any(v is not None for v in macro_data.values()):
                self._write_macro_cache(macro_data)
        except Exception:
            pass
        return macro_data
    
    def calculate_macro_score(self, macro_data: Dict) -> Dict:
        """Calcule un score macroéconomique composite pour les cryptos."""
        score = 0
        components = []
        
        if macro_data.get('fear_greed'):
            fg_value = macro_data['fear_greed']['value']
            if fg_value <= 25: fg_score = -30
            elif fg_value <= 45: fg_score = -15
            elif fg_value <= 55: fg_score = 0
            elif fg_value <= 75: fg_score = 15
            else: fg_score = 30
            score += fg_score
            components.append(f"F&G: {fg_score}")
        
        if macro_data.get('vix'):
            vix_value = macro_data['vix']['value']
            if vix_value > 30: vix_score = -20
            elif vix_value > 20: vix_score = -10
            elif vix_value < 15: vix_score = 20
            else: vix_score = 0
            score += vix_score
            components.append(f"VIX: {vix_score}")
        
        if macro_data.get('dxy'):
            dxy_change = macro_data['dxy']['change_pct']
            if dxy_change > 0.5: dxy_score = -20
            elif dxy_change > 0.1: dxy_score = -10
            elif dxy_change < -0.5: dxy_score = 20
            elif dxy_change < -0.1: dxy_score = 10
            else: dxy_score = 0
            score += dxy_score
            components.append(f"DXY: {dxy_score}")
        
        if macro_data.get('sp500'):
            sp500_change = macro_data['sp500']['change_pct']
            if sp500_change > 1: sp500_score = 20
            elif sp500_change > 0.3: sp500_score = 10
            elif sp500_change < -1: sp500_score = -20
            elif sp500_change < -0.3: sp500_score = -10
            else: sp500_score = 0
            score += sp500_score
            components.append(f"S&P500: {sp500_score}")
        
        if macro_data.get('us_10y'):
            yield_change = macro_data['us_10y']['change']
            if yield_change > 0.1: yield_score = -10
            elif yield_change < -0.1: yield_score = 10
            else: yield_score = 0
            score += yield_score
            components.append(f"10Y: {yield_score}")
        
        if score > 50: macro_sentiment = "Très Bullish"
        elif score > 20: macro_sentiment = "Bullish"
        elif score > 5: macro_sentiment = "Légèrement Bullish"
        elif score < -50: macro_sentiment = "Très Bearish"
        elif score < -20: macro_sentiment = "Bearish"
        elif score < -5: macro_sentiment = "Légèrement Bearish"
        else: macro_sentiment = "Neutre"
        
        return {'score': score, 'sentiment': macro_sentiment, 'components': components}

class BinanceFuturesMultiIndicatorMonitor:
    """
    Classe principale pour monitorer RSI, MACD et OBV simultanément
    avec intégration des données macroéconomiques et persistance SQLite.
    """
    def __init__(self, max_workers: int = 20, webhook_url: Optional[str] = None, db_file: str = "market_history.db"):
        self.base_url = "https://fapi.binance.com"
        self.max_workers = max_workers
        self.webhook_url = webhook_url
        self.console = Console()
        self.macro_indicators = MacroIndicators()
        self.db_file = db_file
        # Backoff exponentiel configurable pour atténuer les bans / rate limits
        try:
            self.backoff_base_seconds = int(os.environ.get('BINANCE_BACKOFF_BASE_SECONDS', '120'))
        except Exception:
            self.backoff_base_seconds = 120
        try:
            self.backoff_max_seconds = int(os.environ.get('BINANCE_BACKOFF_MAX_SECONDS', '3600'))
        except Exception:
            self.backoff_max_seconds = 3600
        self.backoff_seconds = self.backoff_base_seconds
        # Statut de la dernière récupération de symboles (ok | rate_limited | error)
        self.last_symbol_fetch_status: str = 'ok'
        # Cache local des symboles (permet d'opérer sous ban prolongé)
        self.symbols_cache_file = os.environ.get('BINANCE_SYMBOLS_CACHE_FILE', 'binance_symbols_cache.json')
        try:
            self.symbols_cache_ttl = int(os.environ.get('BINANCE_SYMBOLS_CACHE_TTL_SECONDS', '21600'))  # 6 heures
        except Exception:
            self.symbols_cache_ttl = 21600
        # Fichier seed de secours (liste brute de symboles ligne par ligne)
        self.symbols_seed_file = os.environ.get('BINANCE_SYMBOLS_SEED_FILE', 'binance_symbols_seed.txt')
        # Mettre à jour le seed automatiquement à chaque succès (1/0)
        self.update_seed_on_success = os.environ.get('BINANCE_UPDATE_SEED_ON_SUCCESS', '1') not in ('0', 'false', 'False')
        # Limite d'analyse lorsque l'on utilise un fallback (cache/seed/rate-limited)
        try:
            self.fallback_symbols_limit = int(os.environ.get('BINANCE_SYMBOLS_FALLBACK_LIMIT', '50'))
        except Exception:
            self.fallback_symbols_limit = 50
        
        self.rsi_period = 14
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        self.obv_sma_period = 20
        
        self.last_signal = None
        self.signal_history = deque(maxlen=10)
        self.last_trend_state: Optional[str] = None

        self._init_db()
        # Fichier de cache partagé pour les autres scripts
        self.market_context_cache_file = "market_context_cache.json"

    def check_and_auto_retrain(self):
        """Vérifie et lance un auto-entraînement ML si nécessaire (non bloquant)."""
        try:
            # Import optionnel de l'auto-trainer
            from ml_auto_trainer import should_retrain_model, auto_retrain_if_needed
            
            # Vérification rapide (non bloquante)
            should_retrain, reason = should_retrain_model()
            
            if should_retrain:
                self.console.log(f"[yellow]🤖 Auto-entraînement ML déclenché: {reason}[/yellow]")
                success = auto_retrain_if_needed()
                if success:
                    self.console.log("[green]✅ Modèle ML mis à jour avec succès ![/green]")
                else:
                    self.console.log("[red]❌ Échec auto-entraînement ML[/red]")
            # Pas de log si pas nécessaire (pour éviter le spam)
                    
        except Exception as e:
            # Silent fail - l'auto-entraînement est optionnel
            try:
                self.console.log(f"[dim]Auto-trainer ML indisponible: {e}[/dim]")
            except Exception:
                pass

    def write_market_context_cache(self, metrics: Dict, analysis: Dict, ml_info: Optional[Dict] = None) -> None:
        """Écrit un cache JSON résumant le contexte de marché pour réutilisation.
        Structure minimaliste pour être consommée par d'autres scripts.
        """
        try:
            payload = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "signal": analysis.get("signal").value if analysis.get("signal") else None,
                "sentiment": analysis.get("sentiment").value if analysis.get("sentiment") else None,
                "avg_final_score": analysis.get("avg_final_score"),
                "technical_score": analysis.get("technical_score"),
                "macro_score": analysis.get("macro_score"),
                "bullish_pct": analysis.get("bullish_pct"),
                "bearish_pct": analysis.get("bearish_pct"),
                # Quelques infos utiles pour debug/affinage
                "total_pairs": metrics.get("total_pairs"),
            }
            if ml_info:
                try:
                    payload["ml_prediction"] = ml_info.get("prediction")
                    payload["ml_probs"] = ml_info.get("probs")
                except Exception:
                    pass
            with open(self.market_context_cache_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            self.console.log("[green]💾 Contexte marché écrit dans market_context_cache.json[/green]")
        except Exception as e:
            self.console.log(f"[yellow]⚠️ Impossible d'écrire le cache contexte: {e}[/yellow]")
            try:
                logging.getLogger("tendance_globale").warning("Echec ecriture cache contexte: %s", e)
            except Exception:
                pass

    def _init_db(self):
        """Initialise la base de données SQLite et crée la table si elle n'existe pas."""
        try:
            with sqlite3.connect(self.db_file) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS market_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    rsi_1h_simple REAL, rsi_1d_simple REAL, rsi_1w_simple REAL, rsi_1M_simple REAL,
                    rsi_1h_weighted REAL, rsi_1d_weighted REAL, rsi_1w_weighted REAL, rsi_1M_weighted REAL,
                    macd_histogram_1h_simple REAL, macd_histogram_1d_simple REAL, macd_histogram_1w_simple REAL, macd_histogram_1M_simple REAL,
                    macd_histogram_1h_weighted REAL, macd_histogram_1d_weighted REAL, macd_histogram_1w_weighted REAL, macd_histogram_1M_weighted REAL,
                    obv_momentum_1h_simple REAL, obv_momentum_1d_simple REAL, obv_momentum_1w_simple REAL, obv_momentum_1M_simple REAL,
                    obv_momentum_1h_weighted REAL, obv_momentum_1d_weighted REAL, obv_momentum_1w_weighted REAL, obv_momentum_1M_weighted REAL,
                    composite_score_1h REAL, composite_score_1d REAL, composite_score_1w REAL, composite_score_1M REAL,
                    final_score_1h REAL, final_score_1d REAL, final_score_1w REAL, final_score_1M REAL,
                    macro_score REAL, fear_greed REAL, vix REAL, dxy_change REAL, sp500_change REAL
                )
                """)
            self.console.log("[bold green]✅ Base de données SQLite initialisée avec succès.[/bold green]")
        except Exception as e:
            self.console.log(f"[bold red]❌ Erreur lors de l'initialisation de la base de données: {e}[/bold red]")

    def save_metrics_to_db(self, metrics: Dict):
        """Sauvegarde les nouvelles métriques dans la base de données SQLite."""
        try:
            with sqlite3.connect(self.db_file) as conn:
                cursor = conn.cursor()
                
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                # Accès robustes avec défauts pour éviter les erreurs si certaines clés sont absentes
                rs = metrics.get('rsi_simple') or {}
                rsw = metrics.get('rsi_weighted') or {}
                macs = metrics.get('macd_histogram_simple') or {}
                macw = metrics.get('macd_histogram_weighted') or {}
                obvs = metrics.get('obv_momentum_simple') or {}
                obvw = metrics.get('obv_momentum_weighted') or {}
                comps = metrics.get('composite_score_simple') or {}
                finals = metrics.get('final_score_simple') or {}
                manalysis = metrics.get('macro_analysis') or {}
                mdata = metrics.get('macro_data') or {}
                
                data_tuple = (
                    timestamp,
                    rs.get('1h'), rs.get('1d'), rs.get('1w'), rs.get('1M'),
                    rsw.get('1h'), rsw.get('1d'), rsw.get('1w'), rsw.get('1M'),
                    macs.get('1h'), macs.get('1d'), macs.get('1w'), macs.get('1M'),
                    macw.get('1h'), macw.get('1d'), macw.get('1w'), macw.get('1M'),
                    obvs.get('1h'), obvs.get('1d'), obvs.get('1w'), obvs.get('1M'),
                    obvw.get('1h'), obvw.get('1d'), obvw.get('1w'), obvw.get('1M'),
                    comps.get('1h'), comps.get('1d'), comps.get('1w'), comps.get('1M'),
                    finals.get('1h'), finals.get('1d'), finals.get('1w'), finals.get('1M'),
                    manalysis.get('score'),
                    (mdata.get('fear_greed') or {}).get('value'),
                    (mdata.get('vix') or {}).get('value'),
                    (mdata.get('dxy') or {}).get('change_pct'),
                    (mdata.get('sp500') or {}).get('change_pct')
                )

                sql = f"""
                INSERT INTO market_history (
                    timestamp,
                    rsi_1h_simple, rsi_1d_simple, rsi_1w_simple, rsi_1M_simple,
                    rsi_1h_weighted, rsi_1d_weighted, rsi_1w_weighted, rsi_1M_weighted,
                    macd_histogram_1h_simple, macd_histogram_1d_simple, macd_histogram_1w_simple, macd_histogram_1M_simple,
                    macd_histogram_1h_weighted, macd_histogram_1d_weighted, macd_histogram_1w_weighted, macd_histogram_1M_weighted,
                    obv_momentum_1h_simple, obv_momentum_1d_simple, obv_momentum_1w_simple, obv_momentum_1M_simple,
                    obv_momentum_1h_weighted, obv_momentum_1d_weighted, obv_momentum_1w_weighted, obv_momentum_1M_weighted,
                    composite_score_1h, composite_score_1d, composite_score_1w, composite_score_1M,
                    final_score_1h, final_score_1d, final_score_1w, final_score_1M,
                    macro_score, fear_greed, vix, dxy_change, sp500_change
                ) VALUES ({', '.join(['?'] * 38)})
                """
                
                cursor.execute(sql, data_tuple)
            
            self.console.log("[bold green]💾 Données sauvegardées dans la base de données SQLite.[/bold green]")
        except Exception as e:
            self.console.log(f"[bold red]❌ Erreur lors de la sauvegarde dans la base de données: {e}[/bold red]")

    def get_all_futures_symbols(self) -> List[str]:
        """Récupère toutes les paires Binance Futures actives (contrats perpétuels USDT)."""
        url = f"{self.base_url}/fapi/v1/exchangeInfo"
        try:
            response = requests.get(url, timeout=10)
            # Gestion explicite des codes de rate-limit/bans
            if response.status_code in (418, 429):
                # Si possible, ajuster le backoff selon le timestamp 'until <epoch_ms>' renvoyé par Binance
                try:
                    txt = response.text or ""
                    import re, math, time as _t
                    m = re.search(r"until\s+(\d{13})", txt)
                    if m:
                        ban_until_ms = int(m.group(1))
                        wait_secs = max(0, math.ceil(ban_until_ms / 1000 - _t.time()))
                        if wait_secs > 0:
                            self.backoff_seconds = min(max(wait_secs, self.backoff_base_seconds), self.backoff_max_seconds)
                            mins = self.backoff_seconds // 60
                            self.console.log(
                                f"[bold yellow]⚠️ API bannie jusqu'à +{wait_secs}s. Backoff aligné à {self.backoff_seconds}s (~{mins} min).[/bold yellow]"
                            )
                except Exception:
                    pass
                self.console.log(
                    f"[bold yellow]⚠️ Rate limit/Ban API ({response.status_code}) sur exchangeInfo. Tentative de fallback via ccxt...[/bold yellow]"
                )
                syms = self._fallback_symbols_via_ccxt()
                if syms:
                    self.console.log(f"[bold green]✅ Fallback ccxt réussi: {len(syms)} paires récupérées[/bold green]")
                    # écrire le cache local
                    self._write_symbols_cache(syms)
                    self.last_symbol_fetch_status = 'ok'
                    return sorted(syms)
                else:
                    self.console.log("[bold red]❌ Fallback ccxt échoué. Pas de symboles disponibles.[/bold red]")
                    self.last_symbol_fetch_status = 'rate_limited'
                    # Dernier recours: cache local si frais
                    cached = self._load_symbols_cache()
                    if cached:
                        self.console.log(
                            f"[bold yellow]ℹ️ Utilisation du cache symboles: {len(cached)} paires[/bold yellow]"
                        )
                        self.last_symbol_fetch_status = 'cached'
                        return sorted(cached)
                    # Dernier secours: seed local
                    seeded = self._load_symbols_seed()
                    if seeded:
                        self.console.log(
                            f"[bold yellow]ℹ️ Utilisation du seed symboles: {len(seeded)} paires[/bold yellow]"
                        )
                        self.last_symbol_fetch_status = 'seeded'
                        return sorted(seeded)
                    return []

            response.raise_for_status()
            data = response.json()
            symbols = [
                s['symbol'] for s in data['symbols']
                if s.get('status') == 'TRADING' and
                s.get('contractType') == 'PERPETUAL' and
                s.get('quoteAsset') == 'USDT'
            ]
            self.console.log(f"[bold green]✅ {len(symbols)} paires Binance Futures USDT trouvées[/bold green]")
            # écrire le cache local
            self._write_symbols_cache(symbols)
            self.last_symbol_fetch_status = 'ok'
            return sorted(symbols)
        except HTTPError as e:
            code = getattr(getattr(e, 'response', None), 'status_code', None)
            if code in (418, 429):
                # Ajuster le backoff si possible avec le timestamp
                try:
                    txt = getattr(getattr(e, 'response', None), 'text', '') or ''
                    import re, math, time as _t
                    m = re.search(r"until\s+(\d{13})", txt)
                    if m:
                        ban_until_ms = int(m.group(1))
                        wait_secs = max(0, math.ceil(ban_until_ms / 1000 - _t.time()))
                        if wait_secs > 0:
                            self.backoff_seconds = min(max(wait_secs, self.backoff_base_seconds), self.backoff_max_seconds)
                            mins = self.backoff_seconds // 60
                            self.console.log(
                                f"[bold yellow]⚠️ API bannie jusqu'à +{wait_secs}s. Backoff aligné à {self.backoff_seconds}s (~{mins} min).[/bold yellow]"
                            )
                except Exception:
                    pass
                self.console.log(
                    f"[bold yellow]⚠️ Rate limit/Ban API (HTTP {code}) capturé. Tentative de fallback via ccxt...[/bold yellow]"
                )
                syms = self._fallback_symbols_via_ccxt()
                if syms:
                    self.console.log(f"[bold green]✅ Fallback ccxt réussi: {len(syms)} paires récupérées[/bold green]")
                    self._write_symbols_cache(syms)
                    self.last_symbol_fetch_status = 'ok'
                    return sorted(syms)
                self.last_symbol_fetch_status = 'rate_limited'
                # Dernier recours: cache local si frais
                cached = self._load_symbols_cache()
                if cached:
                    self.console.log(
                        f"[bold yellow]ℹ️ Utilisation du cache symboles: {len(cached)} paires[/bold yellow]"
                    )
                    self.last_symbol_fetch_status = 'cached'
                    return sorted(cached)
            else:
                self.console.log(f"[bold red]❌ HTTPError exchangeInfo: {e}[/bold red]")
                self.last_symbol_fetch_status = 'error'
            # Dernier recours: cache local si frais
            cached = self._load_symbols_cache()
            if cached:
                self.console.log(
                    f"[bold yellow]ℹ️ Utilisation du cache symboles: {len(cached)} paires[/bold yellow]"
                )
                self.last_symbol_fetch_status = 'cached'
                return sorted(cached)
            # Dernier recours: cache local si frais
            cached = self._load_symbols_cache()
            if cached:
                self.console.log(
                    f"[bold yellow]ℹ️ Utilisation du cache symboles: {len(cached)} paires[/bold yellow]"
                )
                self.last_symbol_fetch_status = 'cached'
                return sorted(cached)
            # Dernier secours: seed local
            seeded = self._load_symbols_seed()
            if seeded:
                self.console.log(
                    f"[bold yellow]ℹ️ Utilisation du seed symboles: {len(seeded)} paires[/bold yellow]"
                )
                self.last_symbol_fetch_status = 'seeded'
                return sorted(seeded)
            return []
        except Exception as e:
            self.console.log(f"[bold red]❌ Erreur lors de la récupération des symboles: {e}[/bold red]")
            self.last_symbol_fetch_status = 'error'
            # Dernier recours: cache local si frais
            cached = self._load_symbols_cache()
            if cached:
                self.console.log(
                    f"[bold yellow]ℹ️ Utilisation du cache symboles: {len(cached)} paires[/bold yellow]"
                )
                self.last_symbol_fetch_status = 'cached'
                return sorted(cached)
            # Dernier secours: seed local
            seeded = self._load_symbols_seed()
            if seeded:
                self.console.log(
                    f"[bold yellow]ℹ️ Utilisation du seed symboles: {len(seeded)} paires[/bold yellow]"
                )
                self.last_symbol_fetch_status = 'seeded'
                return sorted(seeded)
            return []

    def _fallback_symbols_via_ccxt(self) -> List[str]:
        """Fallback pour récupérer les symboles via ccxt (Binance USDM) avec throttle intégré.
        Retourne une liste de symboles au format Binance API (ex: BTCUSDT).
        """
        try:
            import ccxt  # import paresseux
            exchange = ccxt.binanceusdm({
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future',
                }
            })
            markets = exchange.load_markets()
            symbols: List[str] = []
            for m in markets.values():
                if not m.get('swap', False):
                    continue  # on garde uniquement les perpétuels
                if m.get('quote') != 'USDT':
                    continue
                info = m.get('info') or {}
                sym = info.get('symbol') or m.get('id') or m.get('symbol')
                if sym:
                    symbols.append(sym)
            return sorted(set(symbols))
        except Exception as e:
            self.console.log(f"[yellow]⚠️ Fallback ccxt a échoué: {e}[/yellow]")
            # Essayer de lire le timestamp de ban éventuel dans le message ccxt
            try:
                txt = str(e)
                import re, math, time as _t
                m = re.search(r"until\s+(\d{13})", txt)
                if m:
                    ban_until_ms = int(m.group(1))
                    wait_secs = max(0, math.ceil(ban_until_ms / 1000 - _t.time()))
                    if wait_secs > 0:
                        self.backoff_seconds = min(max(wait_secs, self.backoff_base_seconds), self.backoff_max_seconds)
                        mins = self.backoff_seconds // 60
                        self.console.log(
                            f"[bold yellow]ℹ️ Indice de ban dans ccxt. Backoff aligné à {self.backoff_seconds}s (~{mins} min).[/bold yellow]"
                        )
            except Exception:
                pass
            return []

    def _write_symbols_cache(self, symbols: List[str]) -> None:
        """Écrit le cache local des symboles avec un timestamp UTC ISO."""
        try:
            payload = {
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'symbols': sorted(list(set(symbols)))
            }
            with open(self.symbols_cache_file, 'w', encoding='utf-8') as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            self.console.log(
                f"[green]💾 Cache symboles écrit ({len(payload['symbols'])} paires) → {self.symbols_cache_file}[/green]"
            )
            # Option: mettre à jour aussi le seed avec la liste complète
            if self.update_seed_on_success:
                try:
                    with open(self.symbols_seed_file, 'w', encoding='utf-8') as sf:
                        sf.write("# Seed généré automatiquement à partir de la dernière récupération réussie\n")
                        for s in payload['symbols']:
                            sf.write(f"{s}\n")
                    self.console.log(
                        f"[green]📝 Seed symboles mis à jour ({len(payload['symbols'])} paires) → {self.symbols_seed_file}[/green]"
                    )
                except Exception as se:
                    self.console.log(
                        f"[yellow]⚠️ Impossible de mettre à jour le seed symboles: {se}[/yellow]"
                    )
        except Exception as e:
            self.console.log(
                f"[yellow]⚠️ Impossible d'écrire le cache symboles: {e}[/yellow]"
            )

    def _load_symbols_cache(self) -> List[str]:
        """Charge les symboles depuis le cache si encore frais selon TTL."""
        try:
            if not os.path.exists(self.symbols_cache_file):
                return []
            with open(self.symbols_cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            ts = data.get('timestamp')
            symbols = data.get('symbols') or []
            if not ts or not symbols:
                return []
            # Parser l'ISO en UTC (support du suffixe Z)
            ts_str = str(ts)
            if ts_str.endswith('Z'):
                ts_str = ts_str.replace('Z', '+00:00')
            try:
                cache_dt = datetime.fromisoformat(ts_str)
            except Exception:
                # fallback: ignorer si parse KO
                return []
            age_seconds = max(0, int((datetime.utcnow() - cache_dt.replace(tzinfo=None)).total_seconds()))
            if age_seconds <= self.symbols_cache_ttl:
                return symbols
            return []
        except Exception:
            return []

    def _load_symbols_seed(self) -> List[str]:
        """Charge un fichier seed (une paire par ligne)."""
        try:
            if not os.path.exists(self.symbols_seed_file):
                return []
            with open(self.symbols_seed_file, 'r', encoding='utf-8') as f:
                lines = [ln.strip() for ln in f.readlines()]
            symbols = [ln for ln in lines if ln and not ln.startswith('#')]
            return sorted(list(set(symbols)))
        except Exception:
            return []

    def calculate_rsi(self, prices: List[float], period: int = 14) -> Optional[float]:
        if len(prices) < period + 1: return None
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = pd.Series(gains).ewm(span=period, adjust=False).mean().iloc[-1]
        avg_loss = pd.Series(losses).ewm(span=period, adjust=False).mean().iloc[-1]
        if avg_loss == 0: return 100.0
        rs = avg_gain / avg_loss
        return round(100 - (100 / (1 + rs)), 2)

    def calculate_macd(self, prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Optional[Tuple[float, float, float]]:
        if len(prices) < slow + signal: return None
        prices_series = pd.Series(prices)
        ema_fast = prices_series.ewm(span=fast).mean()
        ema_slow = prices_series.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return (round(macd_line.iloc[-1], 6), round(signal_line.iloc[-1], 6), round(histogram.iloc[-1], 6))

    def calculate_obv_and_sma(self, closes: List[float], volumes: List[float], sma_period: int = 20) -> Optional[Tuple[float, float, float]]:
        if len(closes) < sma_period: return None
        df = pd.DataFrame({'close': closes, 'volume': volumes})
        price_change = df['close'].diff()
        direction = np.where(price_change > 0, 1, np.where(price_change < 0, -1, 0))
        volume_change = df['volume'] * direction
        df['obv'] = volume_change.cumsum()
        df['obv_sma'] = df['obv'].rolling(window=sma_period).mean()
        df['obv_momentum'] = df['obv'] - df['obv_sma']
        if pd.isna(df['obv_momentum'].iloc[-1]): return None
        return (round(df['obv'].iloc[-1], 2), round(df['obv_sma'].iloc[-1], 2), round(df['obv_momentum'].iloc[-1], 2))

    def get_klines_data(self, symbol: str, interval: str, limit: int = 100) -> Optional[List]:
        try:
            url = f"{self.base_url}/fapi/v1/klines"
            params = {'symbol': symbol, 'interval': interval, 'limit': limit}
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.RequestException: return None

    def get_24h_ticker(self, symbol: str) -> Optional[Dict]:
        try:
            url = f"{self.base_url}/fapi/v1/ticker/24hr"
            params = {'symbol': symbol}
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.RequestException: return None

    def calculate_composite_score(self, rsi: Optional[float], macd_histogram: Optional[float], obv_momentum: Optional[float]) -> float:
        score, indicator_count = 0.0, 0
        if rsi is not None:
            if rsi <= 20: rsi_score = -40
            elif rsi <= 30: rsi_score = -20
            elif rsi <= 45: rsi_score = -10
            elif rsi <= 55: rsi_score = 0
            elif rsi <= 70: rsi_score = 10
            elif rsi <= 80: rsi_score = 20
            else: rsi_score = 40
            score += rsi_score
            indicator_count += 1
        if macd_histogram is not None:
            if macd_histogram > 0.001: macd_score = 30
            elif macd_histogram > 0.0001: macd_score = 15
            elif macd_histogram > -0.0001: macd_score = 0
            elif macd_histogram > -0.001: macd_score = -15
            else: macd_score = -30
            score += macd_score
            indicator_count += 1
        if obv_momentum is not None:
            if obv_momentum > 1000: obv_score = 30
            elif obv_momentum > 100: obv_score = 15
            elif obv_momentum > -100: obv_score = 0
            elif obv_momentum > -1000: obv_score = -15
            else: obv_score = -30
            score += obv_score
            indicator_count += 1
        return round(score * (3 / indicator_count), 2) if indicator_count > 0 else 0.0

    def calculate_final_score(self, technical_score: float, macro_score: float, macro_weight: float = 0.3) -> float:
        return round((technical_score * (1 - macro_weight)) + (macro_score * macro_weight), 2)

    def analyze_single_pair(self, symbol: str) -> Optional[Dict]:
        ticker_24h = self.get_24h_ticker(symbol)
        if not ticker_24h: return None
        result = {
            'symbol': symbol,
            'price': float(ticker_24h['lastPrice']),
            'volume_24h': float(ticker_24h['quoteVolume']),
            'price_change_24h': float(ticker_24h['priceChangePercent']),
        }
        has_data = False
        for tf in ['1h', '1d', '1w', '1M']:
            klines = self.get_klines_data(symbol, tf, 100)
            if klines:
                closes = [float(k[4]) for k in klines]
                volumes = [float(k[5]) for k in klines]
                rsi = self.calculate_rsi(closes, self.rsi_period)
                result[f'rsi_{tf}'] = rsi
                macd_res = self.calculate_macd(closes, self.macd_fast, self.macd_slow, self.macd_signal)
                result[f'macd_histogram_{tf}'] = macd_res[2] if macd_res else None
                obv_res = self.calculate_obv_and_sma(closes, volumes, self.obv_sma_period)
                result[f'obv_momentum_{tf}'] = obv_res[2] if obv_res else None
                result[f'composite_score_{tf}'] = self.calculate_composite_score(rsi, result[f'macd_histogram_{tf}'], result[f'obv_momentum_{tf}'])
                if rsi is not None or macd_res is not None or obv_res is not None: has_data = True
        return result if has_data else None

    def analyze_all_pairs(self) -> List[Dict]:
        symbols = self.get_all_futures_symbols()
        # Réduire la charge quand on est en fallback (cache/seed/rate-limited)
        if (
            self.last_symbol_fetch_status in ('cached', 'seeded', 'rate_limited')
            and self.fallback_symbols_limit > 0
            and len(symbols) > self.fallback_symbols_limit
        ):
            original_count = len(symbols)
            symbols = symbols[: self.fallback_symbols_limit]
            self.console.log(
                f"[bold yellow]⚠️ Mode fallback ({self.last_symbol_fetch_status}). Limitation des paires: {original_count} → {len(symbols)}[/bold yellow]"
            )
        if not symbols: return []
        results = []
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(bar_width=None), TextColumn("[progress.percentage]{task.percentage:>3.0f}%"), transient=True) as progress:
            task = progress.add_task(f"[cyan]🔄 Analyse de {len(symbols)} paires...", total=len(symbols))
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_symbol = {executor.submit(self.analyze_single_pair, s): s for s in symbols}
                for future in as_completed(future_to_symbol):
                    res = future.result()
                    if res: results.append(res)
                    progress.update(task, advance=1)
        self.console.log(f"[bold green]✅ Analyse technique terminée: {len(results)}/{len(symbols)} paires réussies[/bold green]")
        return results

    def calculate_global_market_metrics(self, market_data: List[Dict], macro_data: Dict) -> Dict:
        if not market_data:
            return {}

        metrics = {
            'total_pairs': len(market_data),

            # RSI: on conserve simple (moyenne), on ajoute médiane et breadth,
            # on garde weighted pour compat mais on ne l'affiche plus
            'rsi_simple': {},
            'rsi_median': {},
            # {'1h': {'oversold_pct': x, 'overbought_pct': y}, ...}
            'rsi_breadth': {},
            'rsi_weighted': {},

            # MACD / OBV / Composite
            'macd_histogram_simple': {},
            'macd_histogram_weighted': {},
            'obv_momentum_simple': {},
            'obv_momentum_weighted': {},
            'composite_score_simple': {},
            'composite_score_weighted': {},
            'final_score_simple': {},
            'final_score_weighted': {},

            'top_gainers': sorted(
                market_data,
                key=lambda x: x['price_change_24h'],
                reverse=True
            )[:5],
            'top_losers': sorted(
                market_data,
                key=lambda x: x['price_change_24h']
            )[:5],
            'macro_data': macro_data,
            'macro_analysis': self.macro_indicators.calculate_macro_score(
                macro_data
            ),
        }

        macro_score = metrics['macro_analysis']['score']

        for tf in ['1h', '1d', '1w', '1M']:
            # 1) RSI: moyenne simple (pour compat), médiane, breadth
            rsi_key = f'rsi_{tf}'
            rsi_vals = [
                d[rsi_key]
                for d in market_data
                if d.get(rsi_key) is not None
            ]

            if rsi_vals:
                # Moyenne simple (compat)
                metrics['rsi_simple'][tf] = round(statistics.mean(rsi_vals), 6)
                # Médiane
                metrics['rsi_median'][tf] = round(
                    statistics.median(rsi_vals), 6
                )
                # Breadth
                total = len(rsi_vals)
                oversold = sum(1 for v in rsi_vals if v < 30)
                overbought = sum(1 for v in rsi_vals if v > 70)
                metrics['rsi_breadth'][tf] = {
                    'oversold_pct': round(oversold / total * 100, 1),
                    'overbought_pct': round(overbought / total * 100, 1),
                }
            else:
                metrics['rsi_simple'][tf] = 0.0
                metrics['rsi_median'][tf] = 0.0
                metrics['rsi_breadth'][tf] = {
                    'oversold_pct': 0.0,
                    'overbought_pct': 0.0,
                }

            # 2) Les autres indicateurs (on garde le fonctionnement existant)
            for ind in ['macd_histogram', 'obv_momentum', 'composite_score']:
                key = f'{ind}_{tf}'
                valid_data = [d for d in market_data if d.get(key) is not None]
                if valid_data:
                    values = [d[key] for d in valid_data]
                    metrics[f'{ind}_simple'][tf] = round(
                        statistics.mean(values), 6
                    )
                    total_volume = sum(d['volume_24h'] for d in valid_data)
                    if total_volume > 0:
                        weighted_sum = sum(
                            d[key] * d['volume_24h'] for d in valid_data
                        )
                        metrics[f'{ind}_weighted'][tf] = round(
                            weighted_sum / total_volume, 6
                        )
                    else:
                        metrics[f'{ind}_weighted'][tf] = 0.0
                else:
                    metrics[f'{ind}_simple'][tf] = 0.0
                    metrics[f'{ind}_weighted'][tf] = 0.0

            # 3) Scores finaux (pondération macro)
            metrics['final_score_simple'][tf] = self.calculate_final_score(
                metrics['composite_score_simple'][tf], macro_score
            )
            metrics['final_score_weighted'][tf] = self.calculate_final_score(
                metrics['composite_score_weighted'][tf], macro_score
            )

        # Distributions
        comp_1h = [
            d['composite_score_1h']
            for d in market_data
            if d.get('composite_score_1h') is not None
        ]
        metrics['distribution_1h'] = {
            'Très Bearish (< -50)': len([s for s in comp_1h if s < -50]),
            'Bearish (-50 à -20)': len(
                [s for s in comp_1h if -50 <= s < -20]
            ),
            'Faiblement Bearish (-20 à -5)': len(
                [s for s in comp_1h if -20 <= s < -5]
            ),
            'Neutre (-5 à 5)': len([s for s in comp_1h if -5 <= s <= 5]),
            'Faiblement Bullish (5 à 20)': len(
                [s for s in comp_1h if 5 < s <= 20]
            ),
            'Bullish (20 à 50)': len(
                [s for s in comp_1h if 20 < s <= 50]
            ),
            'Très Bullish (> 50)': len([s for s in comp_1h if s > 50]),
        }

        final_1h = [
            self.calculate_final_score(d['composite_score_1h'], macro_score)
            for d in market_data
            if d.get('composite_score_1h') is not None
        ]
        metrics['distribution_final'] = {
            'Très Bearish (< -50)': len([s for s in final_1h if s < -50]),
            'Bearish (-50 à -20)': len(
                [s for s in final_1h if -50 <= s < -20]
            ),
            'Faiblement Bearish (-20 à -5)': len(
                [s for s in final_1h if -20 <= s < -5]
            ),
            'Neutre (-5 à 5)': len([s for s in final_1h if -5 <= s <= 5]),
            'Faiblement Bullish (5 à 20)': len(
                [s for s in final_1h if 5 < s <= 20]
            ),
            'Bullish (20 à 50)': len(
                [s for s in final_1h if 20 < s <= 50]
            ),
            'Très Bullish (> 50)': len([s for s in final_1h if s > 50]),
        }

        return metrics

    def analyze_market_sentiment(self, metrics: Dict) -> Dict:
        final_1h = metrics['final_score_simple']['1h']
        final_1d = metrics['final_score_simple']['1d']
        avg_final = (final_1h + final_1d) / 2
        if avg_final < -50: sentiment = MarketSentiment.EXTREME_BEARISH
        elif avg_final < -20: sentiment = MarketSentiment.BEARISH
        elif avg_final < -5: sentiment = MarketSentiment.WEAK_BEARISH
        elif avg_final < 5: sentiment = MarketSentiment.NEUTRAL
        elif avg_final < 20: sentiment = MarketSentiment.WEAK_BULLISH
        elif avg_final < 50: sentiment = MarketSentiment.BULLISH
        else: sentiment = MarketSentiment.EXTREME_BULLISH
        dist = metrics['distribution_final']
        total = metrics['total_pairs'] if metrics['total_pairs'] > 0 else 1
        bearish_pct = (dist.get('Très Bearish (< -50)', 0) + dist.get('Bearish (-50 à -20)', 0)) / total * 100
        bullish_pct = (dist.get('Bullish (20 à 50)', 0) + dist.get('Très Bullish (> 50)', 0)) / total * 100
        ib = metrics['final_score_weighted']['1h'] - metrics['final_score_simple']['1h']
        macro_impact = metrics['macro_analysis']['score']
        signal = self.generate_trading_signal(sentiment, bearish_pct, bullish_pct, ib, macro_impact)
        return {'sentiment': sentiment, 'signal': signal, 'avg_final_score': round(avg_final, 2), 'technical_score': metrics['composite_score_simple']['1h'], 'macro_score': macro_impact, 'bearish_pct': round(bearish_pct, 1), 'bullish_pct': round(bullish_pct, 1), 'institutional_bias': round(ib, 2), 'strength': self.calculate_signal_strength(sentiment, bearish_pct, bullish_pct, macro_impact), 'recommendations': self.generate_recommendations(signal, sentiment, metrics['macro_data'])}

    def generate_trading_signal(self, sentiment: MarketSentiment, bearish_pct: float, bullish_pct: float, institutional_bias: float, macro_score: float) -> TradingSignal:
        if sentiment == MarketSentiment.EXTREME_BULLISH and bullish_pct > 60 and macro_score > 0: return TradingSignal.STRONG_BUY
        elif sentiment == MarketSentiment.BULLISH and (institutional_bias > 5 or macro_score > 20): return TradingSignal.BUY
        elif sentiment in [MarketSentiment.BULLISH, MarketSentiment.WEAK_BULLISH] and macro_score > -20: return TradingSignal.WEAK_BUY
        elif sentiment == MarketSentiment.EXTREME_BEARISH and bearish_pct > 60 and macro_score < 0: return TradingSignal.STRONG_SELL
        elif sentiment == MarketSentiment.BEARISH and (institutional_bias < -5 or macro_score < -20): return TradingSignal.SELL
        elif sentiment in [MarketSentiment.BEARISH, MarketSentiment.WEAK_BEARISH] and macro_score < 20: return TradingSignal.WEAK_SELL
        return TradingSignal.HOLD

    def calculate_signal_strength(self, sentiment: MarketSentiment, bearish_pct: float, bullish_pct: float, macro_score: float) -> str:
        base_strength = 0
        if sentiment in [MarketSentiment.EXTREME_BEARISH, MarketSentiment.EXTREME_BULLISH]: base_strength = 3
        elif sentiment in [MarketSentiment.BEARISH, MarketSentiment.BULLISH]: base_strength = 2
        else: base_strength = 1
        if bearish_pct > 70 or bullish_pct > 70: base_strength += 1
        if abs(macro_score) > 50: base_strength += 1
        elif abs(macro_score) > 30: base_strength += 0.5
        if base_strength >= 4: return "TRÈS FORTE"
        elif base_strength >= 3: return "FORTE"
        elif base_strength >= 2: return "MODÉRÉE"
        else: return "FAIBLE"

    def generate_recommendations(self, signal: TradingSignal, sentiment: MarketSentiment, macro_data: Dict) -> List[str]:
        recs = []
        if signal in [TradingSignal.STRONG_BUY, TradingSignal.BUY]: recs.extend(["🎯 LONGS prioritaires: Score final > +20", "📊 CONFLUENCE: RSI survente + MACD haussier + OBV croissant", "💰 SIZING: Élevé si Fear&Greed > 50 et VIX < 25"])
        elif signal in [TradingSignal.STRONG_SELL, TradingSignal.SELL]: recs.extend(["🎯 SHORTS prioritaires: Score final < -20", "📊 CONFLUENCE: RSI surachat + MACD baissier + OBV décroissant", "💰 SIZING: Modéré si VIX > 30"])
        else: recs.extend(["⏳ MARCHÉ MIXTE: Attendre confluence", "👀 SURVEILLANCE: Changements macro (Fed, DXY, VIX)"])
        if macro_data.get('fear_greed'):
            if macro_data['fear_greed']['value'] <= 25: recs.append("😨 FEAR EXTREME: Opportunités d'accumulation")
            elif macro_data['fear_greed']['value'] >= 75: recs.append("🤑 GREED EXTREME: Prudence, prises de profit")
        if macro_data.get('vix') and macro_data['vix']['value'] > 30: recs.append("⚡ VIX ÉLEVÉ: Volatilité, stops serrés")
        if macro_data.get('dxy'):
            if macro_data['dxy']['change_pct'] > 0.5: recs.append("💵 DOLLAR FORT: Pression baissière sur crypto")
            elif macro_data['dxy']['change_pct'] < -0.5: recs.append("💵 DOLLAR FAIBLE: Support haussier pour crypto")
        if macro_data.get('btc_dominance'):
            if macro_data['btc_dominance']['value'] > 50: recs.append("₿ BTC DOMINANT: Focus sur BTC, alts risqués")
            elif macro_data['btc_dominance']['value'] < 40: recs.append("🏃‍♂️ ALT SEASON: Opportunités sur altcoins")
        return recs

    def display_macro_panel(self, macro_data: Dict, macro_analysis: Dict):
        macro_table = Table(title="🌍 ENVIRONNEMENT MACROÉCONOMIQUE", show_header=True, header_style="bold cyan")
        macro_table.add_column("Indicateur", style="dim", width=20)
        macro_table.add_column("Valeur", justify="right", width=15)
        macro_table.add_column("Impact Crypto", justify="center", width=20)
        macro_table.add_column("Tendance", justify="center", width=15)
        if macro_data.get('fear_greed'):
            fg = macro_data['fear_greed']; color = "red" if fg['value']<25 else "yellow" if fg['value']<50 else "green"
            macro_table.add_row("Fear & Greed", f"[{color}]{fg['value']}[/]", f"[{color}]{fg['classification']}[/]", "📊")
        if macro_data.get('vix'):
            vix=macro_data['vix']; color="green" if vix['value']<20 else "yellow" if vix['value']<30 else "red"; arrow="📈" if vix['change']>0 else "📉"
            macro_table.add_row("VIX (Volatilité)", f"[{color}]{vix['value']}[/]", f"[{color}]{vix['sentiment']}[/]", f"{arrow} {vix['change']:+.2f}")
        if macro_data.get('dxy'):
            dxy=macro_data['dxy']; color="red" if dxy['change_pct']>0.1 else "green" if dxy['change_pct']<-0.1 else "white"; arrow="📈" if dxy['change']>0 else "📉"
            macro_table.add_row("DXY (Dollar)", f"[{color}]{dxy['value']}[/]", f"[{color}]{dxy['crypto_impact']}[/]", f"{arrow} {dxy['change_pct']:+.2f}%")
        if macro_data.get('sp500'):
            sp=macro_data['sp500']; color="green" if sp['change_pct']>0.3 else "red" if sp['change_pct']<-0.3 else "white"; arrow="📈" if sp['change_pct']>0 else "📉"
            macro_table.add_row("S&P 500", f"[{color}]{sp['value']:,.0f}[/]", f"[{color}]{sp['crypto_correlation']}[/]", f"{arrow} {sp['change_pct']:+.2f}%")
        score_color = "green" if macro_analysis['score'] > 20 else "red" if macro_analysis['score'] < -20 else "yellow"
        score_panel = Panel(f"[bold {score_color}]Score Macro: {macro_analysis['score']:+.0f}/100\nSentiment: {macro_analysis['sentiment']}[/]\n[dim]Composants: {' | '.join(macro_analysis['components'])}[/dim]", title="🎯 ANALYSE MACRO COMPOSITE", border_style=score_color)
        self.console.print(Panel(macro_table, title="[bold]Données Macroéconomiques[/bold]", border_style="cyan"))
        self.console.print(score_panel)

    def display_rich_report(self, metrics: Dict):
        if not metrics:
            return

        self.console.rule(f"[bold yellow] RAPPORT DU {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [/bold yellow]")

        # Panel macro
        self.display_macro_panel(metrics['macro_data'], metrics['macro_analysis'])

        # Signaux
        market_analysis = self.analyze_market_sentiment(metrics)
        self.display_trading_signals(market_analysis)

        # Tableau RSI: Moyenne, Médiane, Breadth (Survente/Surachat)
        from rich.table import Table
        rsi_table = Table(title="📊 RSI Global (Médiane + Breadth)", header_style="bold cyan")
        rsi_table.add_column("Période", style="dim")
        rsi_table.add_column("Moyenne", justify="right")
        rsi_table.add_column("Médiane", justify="right")
        rsi_table.add_column("Survente % (<30)", justify="right")
        rsi_table.add_column("Surachat % (>70)", justify="right")

        for p in ['1h', '1d', '1w', '1M']:
            mean_v = metrics['rsi_simple'][p]
            median_v = metrics['rsi_median'][p]
            breadth = metrics['rsi_breadth'][p]
            oversold = breadth['oversold_pct']
            overbought = breadth['overbought_pct']

            # Coloration basée sur la médiane (plus robuste)
            color = "green" if median_v < 30 else "red" if median_v > 70 else "white"
            rsi_table.add_row(
                p.upper(),
                f"[{color}]{mean_v:.2f}[/]",
                f"[{color}]{median_v:.2f}[/]",
                f"[cyan]{oversold:.1f}%[/]",
                f"[magenta]{overbought:.1f}%[/]",
            )

        # Tableau Scores (inchangé)
        score_table = Table(title=f"🎯 SCORES GLOBAUX ({metrics['total_pairs']} paires)", header_style="bold yellow")
        score_table.add_column("Période", style="dim")
        score_table.add_column("Score Tech", justify="right")
        score_table.add_column("Score Final", justify="right")
        score_table.add_column("Sentiment", justify="right")
        for p in ['1h', '1d', '1w', '1M']:
            ts = metrics['composite_score_simple'][p]
            fs = metrics['final_score_simple'][p]
            color = "green" if fs > 20 else "red" if fs < -20 else "yellow"
            sent = "🟢 BULLISH" if fs > 20 else "🔴 BEARISH" if fs < -20 else "🟡 NEUTRE"
            score_table.add_row(p.upper(), f"{ts:+.1f}", f"[{color}]{fs:+.1f}[/]", f"[{color}]{sent}[/]")

        # Distribution Final (inchangé)
        dist_final_table = Table(
            title="🌍 Distribution Score Final 1H",
            header_style="bold green"
        )
        dist_final_table.add_column("Zone", style="green", width=56, no_wrap=False, overflow="fold")
        dist_final_table.add_column("Paires", justify="right", width=12)
        dist_final_table.add_column("%", justify="right", width=12)

        colors = {
            "Très Bearish": "bright_red", "Bearish": "red", "Faiblement Bearish": "yellow",
            "Neutre": "white",
            "Faiblement Bullish": "yellow", "Bullish": "green", "Très Bullish": "bright_green"
        }
        total_paires = sum(metrics['distribution_final'].values()) or 1

        total_bullish_paires = 0
        total_bearish_paires = 0
        for zone, count in metrics['distribution_final'].items():
            key = zone.split(" (")[0]
            dist_final_table.add_row(
                f"[{colors.get(key, 'white')}]" + zone + "[/]",
                str(count),
                f"{(count / total_paires) * 100:.1f}%",
            )
            if "Bearish" in key:
                total_bearish_paires += count
            elif "Bullish" in key:
                total_bullish_paires += count

        total_bearish_pct = (total_bearish_paires / total_paires) * 100
        total_bullish_pct = (total_bullish_paires / total_paires) * 100
        
        # Stocker les données de distribution dans metrics pour l'alerte RSI
        metrics['distribution_final_1h'] = {
            'total_bullish_pct': total_bullish_pct,
            'total_bearish_pct': total_bearish_pct,
            'total_bullish_paires': total_bullish_paires,
            'total_bearish_paires': total_bearish_paires
        }
        
        dist_final_table.add_section()
        dist_final_table.add_row("[bold red]🔴 TOTAL BEARISH[/bold red]", f"[bold red]{total_bearish_paires}[/]", f"[bold red]{total_bearish_pct:.1f}%[/]")
        dist_final_table.add_row("[bold green]🟢 TOTAL BULLISH[/bold green]", f"[bold green]{total_bullish_paires}[/]", f"[bold green]{total_bullish_pct:.1f}%[/]")

        # Afficher le tableau des scores maintenant
        self.console.print(score_table)

        # En toute fin: afficher successivement RSI puis Distribution 1H
        self.console.rule("[bold cyan]Tableaux de fin[/bold cyan]")
        self.console.print(rsi_table)
        self.console.print(dist_final_table)
        self.console.rule()

    def display_trading_signals(self, analysis: Dict):
        """Affiche les signaux de trading avec style rich (technique + macro)."""
        signal = analysis['signal']
        colors = {TradingSignal.STRONG_BUY: "bright_green", TradingSignal.BUY: "green", TradingSignal.WEAK_BUY: "yellow", TradingSignal.HOLD: "white", TradingSignal.WEAK_SELL: "yellow", TradingSignal.SELL: "red", TradingSignal.STRONG_SELL: "bright_red"}
        icons = {TradingSignal.STRONG_BUY: "🚀🟢", TradingSignal.BUY: "📈🟢", TradingSignal.WEAK_BUY: "↗️🟡", TradingSignal.HOLD: "⏸️⚪", TradingSignal.WEAK_SELL: "↘️🟡", TradingSignal.SELL: "📉🔴", TradingSignal.STRONG_SELL: "💥🔴"}
        
        signal_name = signal.value.replace('_', ' ').upper()
        color = colors[signal]
        icon = icons[signal]
        
        is_new = self.is_significant_signal_change(signal)
        title = f"[bold blink]{icon} NOUVEAU SIGNAL {icon}[/]" if is_new else "[bold]🎯 ANALYSE TECHNIQUE + MACRO[/]"
        
        full_text = (
            f"Signal: [{color}]{signal_name}[/] | Force: [bold]{analysis['strength']}[/]\n"
            f"Sentiment: {analysis['sentiment'].value.replace('_', ' ').title()}\n"
            f"Score Final: {analysis['avg_final_score']:+.1f} (Tech: {analysis['technical_score']:+.1f}, Macro: {analysis['macro_score']:+.1f})\n"
            f"Marché: [{analysis['bullish_pct']:.1f}% Bullish vs {analysis['bearish_pct']:.1f}% Bearish]"
        )
        signal_content = Text(full_text, justify="center")
        
        self.console.print(Panel(signal_content, title=title, border_style=color if is_new else "blue", padding=(1, 2)))
        rec_text = "\n".join([f"• {rec}" for rec in analysis['recommendations']])
        self.console.print(Panel(rec_text, title="💡 STRATÉGIE RECOMMANDÉE", border_style="yellow"))
        
        self.last_signal = signal

    def is_significant_signal_change(self, current_signal: TradingSignal) -> bool:
        if self.last_signal is None: return True
        return current_signal != self.last_signal and current_signal != TradingSignal.HOLD
        
    def check_trend_and_play_sound(self, metrics: Dict):
        """Détermine la tendance de la distribution 1H et joue un son si elle change."""
        if 'distribution_final' not in metrics:
            return

        dist = metrics['distribution_final']
        
        # Calculer le poids de chaque camp
        bullish_count = dist.get('Très Bullish (> 50)', 0) + dist.get('Bullish (20 à 50)', 0)
        bearish_count = dist.get('Très Bearish (< -50)', 0) + dist.get('Bearish (-50 à -20)', 0)
        
        current_trend = "Neutral"
        if bullish_count > bearish_count:
            current_trend = "Bullish"
        elif bearish_count > bullish_count:
            current_trend = "Bearish"

        # Vérifier s'il y a un changement de tendance DIRECT (Bullish <-> Bearish)
        is_direct_flip = (self.last_trend_state == "Bullish" and current_trend == "Bearish") or \
                         (self.last_trend_state == "Bearish" and current_trend == "Bullish")

        if is_direct_flip:
            sound_file = 'trend_change_alert.mp3'
            
            if current_trend == "Bullish":
                self.console.log("[bold bright_green]🔔 INVERSION DE TENDANCE DÉTECTÉE : Passage en mode HAUSSIER ![/bold bright_green]")
            elif current_trend == "Bearish":
                self.console.log("[bold bright_red]🔔 INVERSION DE TENDANCE DÉTECTÉE : Passage en mode BAISSIER ![/bold bright_red]")
            
            if os.path.exists(sound_file):
                try:
                    playsound(sound_file, block=False)
                except Exception as e:
                    self.console.log(f"[yellow]⚠️ Impossible de jouer le son '{sound_file}': {e}[/yellow]")
            else:
                self.console.log(f"[yellow]⚠️ Fichier son '{sound_file}' non trouvé.[/yellow]")

        # Mettre à jour l'état pour la prochaine vérification
        self.last_trend_state = current_trend

    def check_rsi_alignment_signal(self, metrics: dict):
        """
        Détecte si les RSI horaire et journalier sont alignés avec la distribution
        des paires pour signaler une convergence technique forte.
        
        Conditions:
        - RSI 1H > 50 ET RSI 1D > 50 ET Distribution Bullish > 65%
        - RSI 1H < 50 ET RSI 1D < 50 ET Distribution Bearish > 65%
        """
        try:
            rsi_1h = metrics['rsi_median']['1h']
            rsi_1d = metrics['rsi_median']['1d']
            
            # Récupérer la distribution des paires (% bullish vs bearish)
            distribution = metrics.get('distribution_final_1h', {})
            total_bullish_pct = distribution.get('total_bullish_pct', 0)
            total_bearish_pct = distribution.get('total_bearish_pct', 0)
            
            alignment_detected = False
            alignment_type = ""
            sound_file = 'gentle_chime.wav'
            
            # Cas 1: Signal haussier fort
            # RSI 1H > 50 ET RSI 1D > 50 ET Distribution Bullish > 65%
            if (rsi_1h > 50 and rsi_1d > 50 and total_bullish_pct > 65):
                alignment_detected = True
                alignment_type = "SIGNAL HAUSSIER FORT"
                msg = ("[bold bright_green]� SIGNAL HAUSSIER CONFIRMÉ ! "
                       f"RSI 1H: {rsi_1h:.1f} | RSI 1D: {rsi_1d:.1f} | "
                       f"Paires Bullish: {total_bullish_pct:.1f}%[/bold bright_green]")
                
            # Cas 2: Signal baissier fort  
            # RSI 1H < 50 ET RSI 1D < 50 ET Distribution Bearish > 65%
            elif (rsi_1h < 50 and rsi_1d < 50 and total_bearish_pct > 65):
                alignment_detected = True
                alignment_type = "SIGNAL BAISSIER FORT"
                msg = ("[bold bright_red]� SIGNAL BAISSIER CONFIRMÉ ! "
                       f"RSI 1H: {rsi_1h:.1f} | RSI 1D: {rsi_1d:.1f} | "
                       f"Paires Bearish: {total_bearish_pct:.1f}%[/bold bright_red]")
            
            # Si signal détecté, afficher et jouer le son
            if alignment_detected:
                self.console.log(msg)
                self.console.log(
                    f"[dim]📊 Confluence technique: RSI alignés + "
                    f"Distribution dominante ({alignment_type})[/dim]"
                )
                
                if os.path.exists(sound_file):
                    try:
                        playsound(sound_file, block=False)
                        self.console.log(f"[dim]🔊 Alerte sonore: {alignment_type}[/dim]")
                    except Exception as e:
                        self.console.log(f"[yellow]⚠️ Erreur son: {e}[/yellow]")
                else:
                    self.console.log(f"[yellow]⚠️ Son '{sound_file}' introuvable[/yellow]")
                    
        except Exception as e:
            self.console.log(f"[red]❌ Erreur détection RSI+Distribution: {e}[/red]")

    def run_analysis(self):
        """Lance l'analyse complète du marché."""
        self.console.log("[cyan]🌍 Récupération des données macroéconomiques...[/cyan]")
        macro_data = self.macro_indicators.get_all_macro_data()
        self.console.log("[cyan]📊 Analyse technique des paires crypto...[/cyan]")
        market_data = self.analyze_all_pairs()
        if not market_data:
            # Backoff intelligent si rate-limit / erreur
            if self.last_symbol_fetch_status in ('rate_limited', 'error'):
                wait = self.backoff_seconds
                mins = wait // 60
                self.console.log(
                    f"[bold yellow]⏳ Rate limit/erreur API détectée. Backoff {wait}s (≈{mins} min), nouvel essai...[/bold yellow]"
                )
                try:
                    time.sleep(wait)
                except Exception:
                    pass
                # Retry une fois
                market_data = self.analyze_all_pairs()
                if not market_data:
                    # Échec: on augmente le backoff et on abandonne ce cycle
                    self.backoff_seconds = min(self.backoff_seconds * 2, self.backoff_max_seconds)
                    next_mins = self.backoff_seconds // 60
                    self.console.log(
                        f"[bold red]❌ Toujours aucune donnée après retry. Prochaine tentative dans {next_mins} min.[/bold red]"
                    )
                    return
                else:
                    # Succès: reset du backoff
                    self.backoff_seconds = self.backoff_base_seconds
            else:
                self.console.log("[bold red]❌ Aucune donnée de marché récupérée. Prochaine tentative dans une heure.[/bold red]")
                return
        global_metrics = self.calculate_global_market_metrics(market_data, macro_data)
        
        self.check_trend_and_play_sound(global_metrics)

        # Calculer le résumé de contexte (signal + sentiment + stats) et l'écrire dans un cache partagé
        market_analysis = self.analyze_market_sentiment(global_metrics)
        # Inférence ML optionnelle (si modèle disponible et dépendances installées)
        ml_info: Optional[Dict] = None
        try:
            if load_model_or_none and features_from_metrics:
                model_path = os.environ.get('MARKET_MODEL_PATH') or os.path.join('models', 'market_classifier.pkl')
                # Fallback: fichier à la racine si le dossier models n'existe pas
                if not os.path.exists(model_path) and os.path.exists('market_classifier.pkl'):
                    model_path = 'market_classifier.pkl'
                bundle = load_model_or_none(model_path)
                if bundle:
                    model = bundle.get('model')
                    feats = features_from_metrics(global_metrics)
                    try:
                        import numpy as _np  # pour argmax robuste
                    except Exception:
                        _np = None
                    y_prob = model.predict_proba(feats)[0]
                    classes = list(getattr(model, 'classes_', []))
                    if classes:
                        max_idx = int(y_prob.argmax()) if hasattr(y_prob, 'argmax') else int(max(range(len(y_prob)), key=lambda i: y_prob[i]))
                        pred = classes[max_idx]
                        ml_info = {
                            'prediction': str(pred),
                            'probs': {str(c): float(y_prob[i]) for i, c in enumerate(classes)}
                        }
                        
                        # Format joli des probabilités
                        prob_display = " | ".join([
                            f"{label}: {prob*100:.1f}%"
                            for label, prob in ml_info['probs'].items()
                        ])
                        
                        # Emoji basé sur la prédiction
                        pred_emoji = {
                            'Bearish': '🔴',
                            'Neutral': '🟡',
                            'Bullish': '🟢'
                        }.get(str(pred), '🤖')
                        
                        try:
                            msg = (f"[magenta]{pred_emoji} ML Prédiction (1h): "
                                   f"[bold]{pred}[/bold] | {prob_display}[/magenta]")
                            self.console.log(msg)
                        except Exception:
                            pass
                else:
                    try:
                        self.console.log(f"[dim]ML: modèle introuvable ({model_path}). Pas d'inférence.[/dim]")
                    except Exception:
                        pass
            else:
                try:
                    self.console.log("[dim]ML: dépendances indisponibles (scikit-learn/joblib). Pas d'inférence.[/dim]")
                except Exception:
                    pass
        except Exception as e:
            try:
                self.console.log(f"[yellow]⚠️ ML indisponible: {e}[/yellow]")
            except Exception:
                pass

        self.write_market_context_cache(global_metrics, market_analysis, ml_info)

        self.save_metrics_to_db(global_metrics)
        
        # Auto-entraînement ML optionnel (périodique)
        self.check_and_auto_retrain()
        
        self.display_rich_report(global_metrics)
        
        # Vérifier l'alignement des RSI horaire/journalier (après calcul des distributions)
        self.check_rsi_alignment_signal(global_metrics)

    def start_monitoring(self, interval_seconds: int | None = None, once: bool = False):
        """Démarre le monitoring en continu."""
        self.console.print(
            "[bold green]🚀 DÉMARRAGE DU MONITORING BINANCE FUTURES (TECHNIQUE + MACRO)[/bold green]"
        )
        try:
            logging.getLogger("tendance_globale").info("Demarrage monitoring Binance Futures")
        except Exception:
            pass
        # Premier run immediat
        self.run_analysis()
        if once:
            try:
                self.console.print("[dim]Exécution unique (--once): fin du programme.[/dim]")
            except Exception:
                pass
            return

        # Offset persistant (top de l'heure + offset en secondes) si aucune période fixe n'est donnée
        try:
            scan_offset = int(
                os.environ.get('SCAN_OFFSET_SECONDS_TENDANCE', '0')
            )
        except Exception:
            scan_offset = 0
        if scan_offset < 0:
            scan_offset = 0
        if scan_offset >= 3600:
            scan_offset = scan_offset % 3600

        while True:
            if interval_seconds and interval_seconds > 0:
                sleep_seconds = int(interval_seconds)
                target_str = f"~{sleep_seconds}s (période fixe)"
            else:
                now = datetime.now()
                next_hour = (now + timedelta(hours=1)).replace(
                    minute=0, second=0, microsecond=0
                )
                target_time = next_hour + timedelta(seconds=scan_offset)
                sleep_seconds = max(1, int((target_time - now).total_seconds()))
                target_str = target_time.strftime('%H:%M:%S')
            try:
                self.console.print(
                    f"[dim]⏰ Prochain run dans ~{sleep_seconds}s (heure cible {target_str}).[/dim]"
                )
                # Double log simple pour consoles minimales
                print(f"Prochain run dans ~{sleep_seconds}s (cible {target_str})")
            except Exception:
                pass
            time.sleep(sleep_seconds)
            self.run_analysis()

def main():
    # Arguments d'exécution: --once (une seule analyse) et --interval-seconds (période fixe)
    parser = argparse.ArgumentParser(description="Monitoring Binance Futures (technique + macro) avec ML optionnelle")
    parser.add_argument("--once", action="store_true", help="Exécute une seule analyse et quitte")
    parser.add_argument("--interval-seconds", type=int, default=None, help="Exécute en boucle avec une période fixe (en secondes)")
    args = parser.parse_args()

    try:
        # Initialiser un logger fichier léger (rotation) pour diagnostic
        try:
            logs_dir = os.path.join(os.path.dirname(__file__), 'logs')
            os.makedirs(logs_dir, exist_ok=True)
            logger = logging.getLogger("tendance_globale")
            if not logger.handlers:
                logger.setLevel(logging.INFO)
                handler = RotatingFileHandler(os.path.join(logs_dir, 'tendance_globale.log'), maxBytes=5*1024*1024, backupCount=3, encoding='utf-8')
                formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
                handler.setFormatter(formatter)
                logger.addHandler(handler)
        except Exception:
            pass
        # Décalage de démarrage pour échelonner les appels API.
        # Par défaut 180s; surch. via STARTUP_DELAY_SECONDS_TENDANCE
        # ou STARTUP_DELAY_SECONDS (global).
        try:
            _delay_env = (
                os.environ.get('STARTUP_DELAY_SECONDS_TENDANCE')
                or os.environ.get('STARTUP_DELAY_SECONDS', '0')
            )
            STARTUP_DELAY_SECONDS = int(_delay_env)
        except Exception:
            STARTUP_DELAY_SECONDS = 0
        if STARTUP_DELAY_SECONDS > 0:
            try:
                print(
                    "⏳ Démarrage différé de "
                    f"{STARTUP_DELAY_SECONDS} secondes (anti-ban API)."
                )
                logging.getLogger("tendance_globale").info("Startup delay %ss", STARTUP_DELAY_SECONDS)
            except Exception:
                pass
            time.sleep(STARTUP_DELAY_SECONDS)

        monitor = BinanceFuturesMultiIndicatorMonitor(
            max_workers=25,
            webhook_url=None,
            db_file="multi_indicator_macro_history.db"
        )
        monitor.start_monitoring(interval_seconds=args.interval_seconds, once=args.once)
    except KeyboardInterrupt:
        Console().print("\n[bold yellow]👋 Monitoring arrêté par l'utilisateur.[/bold yellow]")
    except Exception as e:
        Console().print(f"\n[bold red]❌ Une erreur critique est survenue: {e}[/bold red]")
        try:
            logging.getLogger("tendance_globale").exception("Erreur critique: %s", e)
        except Exception:
            pass

if __name__ == "__main__":
    main()
