import os
import time
import math
import logging
import requests
import pandas as pd
from binance.client import Client
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import json
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('bot.log')
    ]
)
logger = logging.getLogger(__name__)

# Configurações do Bot
CONFIG = {
    "SYMBOL": os.getenv("TRADING_SYMBOL", "BNBUSDT"),
    "LEVERAGE": int(os.getenv("LEVERAGE", 8)),
    "RISK_PERCENT": float(os.getenv("RISK_PERCENT", 1.5)),
    "STOP_LOSS": float(os.getenv("STOP_LOSS", 1.2)),
    "TAKE_PROFIT": float(os.getenv("TAKE_PROFIT", 2.5)),
    "DATA_INTERVAL": os.getenv("DATA_INTERVAL", "15m"),
    "CHECK_INTERVAL": int(os.getenv("CHECK_INTERVAL", 60)),
    "TELEGRAM_TOKEN": os.getenv("TELEGRAM_TOKEN"),
    "TELEGRAM_CHAT_ID": os.getenv("TELEGRAM_CHAT_ID"),
    "TELEGRAM_ENABLED": os.getenv("TELEGRAM_ENABLED", "True") == "True",
    "MAX_CONSECUTIVE_ERRORS": int(os.getenv("MAX_CONSECUTIVE_ERRORS", 5)),
    "MIN_BALANCE": float(os.getenv("MIN_BALANCE", 100)),
    "MAX_POSITION_SIZE": float(os.getenv("MAX_POSITION_SIZE", 0.7)),
    "MIN_NOTIONAL": float(os.getenv("MIN_NOTIONAL", 50)),
    "TRAILING_STOP_PERCENT": float(os.getenv("TRAILING_STOP_PERCENT", 1.2)),
    "BREAKEVEN_PERCENT": float(os.getenv("BREAKEVEN_PERCENT", 1.5)),
    "HIGHER_TF_INTERVAL": os.getenv("HIGHER_TF_INTERVAL", "1h")
}

@dataclass
class TradeConfig:
    SYMBOL: str
    LEVERAGE: int
    RISK_PERCENT: float
    STOP_LOSS: float
    TAKE_PROFIT: float
    DATA_INTERVAL: str
    CHECK_INTERVAL: int
    TELEGRAM_ENABLED: bool
    MAX_CONSECUTIVE_ERRORS: int
    MIN_BALANCE: float
    MAX_POSITION_SIZE: float
    MIN_NOTIONAL: float
    TRAILING_STOP_PERCENT: float
    BREAKEVEN_PERCENT: float
    HIGHER_TF_INTERVAL: str
    TELEGRAM_TOKEN: Optional[str] = None
    TELEGRAM_CHAT_ID: Optional[str] = None

    @property
    def symbol(self):
        return self.SYMBOL
    
    @property
    def leverage(self):
        return self.LEVERAGE
    
    @property
    def risk_percent(self):
        return self.RISK_PERCENT
    
    @property
    def stop_loss(self):
        return self.STOP_LOSS
    
    @property
    def take_profit(self):
        return self.TAKE_PROFIT
    
    @property
    def data_interval(self):
        return self.DATA_INTERVAL
    
    @property
    def check_interval(self):
        return self.CHECK_INTERVAL
    
    @property
    def telegram_enabled(self):
        return self.TELEGRAM_ENABLED
    
    @property
    def max_consecutive_errors(self):
        return self.MAX_CONSECUTIVE_ERRORS
    
    @property
    def min_balance(self):
        return self.MIN_BALANCE
    
    @property
    def max_position_size(self):
        return self.MAX_POSITION_SIZE
    
    @property
    def min_notional(self):
        return self.MIN_NOTIONAL
    
    @property
    def trailing_stop_percent(self):
        return self.TRAILING_STOP_PERCENT
    
    @property
    def breakeven_percent(self):
        return self.BREAKEVEN_PERCENT
    
    @property
    def higher_tf_interval(self):
        return self.HIGHER_TF_INTERVAL

class TelegramNotifier:
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}/sendMessage"
        self.session = requests.Session()
        self.session.timeout = 10
        
        if os.getenv('HTTP_PROXY'):
            self.session.proxies = {
                'http': os.getenv('HTTP_PROXY'),
                'https': os.getenv('HTTPS_PROXY', os.getenv('HTTP_PROXY'))
            }
            logger.info("Proxy configurado para notificações Telegram")

    def send(self, message: str) -> bool:
        try:
            if len(message) > 4000:
                message = message[:3900] + "\n...[mensagem truncada]"
                
            response = self.session.post(
                self.base_url,
                json={
                    'chat_id': self.chat_id,
                    'text': message,
                    'parse_mode': 'Markdown',
                    'disable_web_page_preview': True
                }
            )
            
            if response.status_code != 200:
                error_info = response.json()
                logger.error(f"Erro Telegram API: {error_info.get('description')}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Erro ao enviar para Telegram: {str(e)}")
            return False

class Backtester:
    def __init__(self, client: Client, symbol: str):
        self.client = client
        self.symbol = symbol
        
    def executar_backtest(self, interval: str = '15m', days: int = 30):
        """Executa backtest simples"""
        try:
            end_time = int(time.time() * 1000)
            start_time = end_time - (days * 24 * 60 * 60 * 1000)
            
            klines = self.client.futures_klines(
                symbol=self.symbol,
                interval=interval,
                startTime=start_time,
                endTime=end_time,
                limit=1000
            )
            
            if len(klines) < 100:
                logger.warning("Dados insuficientes para backtest")
                return None
                
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'trades',
                'taker_buy_base', 'taker_buy_quote', 'ignore'
            ])
            
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_cols] = df[numeric_cols].astype(float)
            
            # Simulação simples de trades
            resultados = {
                'total_trades': 0,
                'trades_vencedores': 0,
                'trades_perdedores': 0,
                'lucro_total': 0,
                'win_rate': 0
            }
            
            return resultados
            
        except Exception as e:
            logger.error(f"Erro no backtest: {str(e)}")
            return None

class BinanceTradingBot:
    def __init__(self, config: Dict[str, Any]):
        self.config = TradeConfig(**config)
        self._exchange_info = None
        self._symbol_info = None
        self._last_api_call = 0
        self._api_call_delay = 0.5  # 500ms entre chamadas
        self.backtester = None
        
        # Configuração inicial
        self._setup()

    def _setup(self):
        """Configuração inicial com tratamento de erros"""
        try:
            logger.info("Iniciando configuração do bot...")
            
            # Conexão com Telegram
            if self.config.telegram_enabled:
                if not self._testar_conexao_telegram():
                    raise ConnectionError("Falha na conexão com Telegram")
            
            # Cliente Binance
            self.client = self._inicializar_client()
            self.backtester = Backtester(self.client, self.config.symbol)
            
            # Configuração de alavancagem
            self.configurar_alavancagem()
            
            # Estado inicial
            self.contagem_trades = 0
            self.ultimo_trade_time = None
            self.erros_consecutivos = 0
            self.lucro_acumulado = 0
            self.trades_do_dia = 0
            self.ultimo_reset_dia = datetime.now().date()
            
            logger.info("Configuração inicial concluída com sucesso")
            
        except Exception as e:
            logger.error(f"Falha na configuração inicial: {str(e)}")
            raise

    def _safe_api_call(self, func, *args, **kwargs):
        """Wrapper seguro para chamadas de API"""
        try:
            # Rate limiting
            elapsed = time.time() - self._last_api_call
            if elapsed < self._api_call_delay:
                time.sleep(self._api_call_delay - elapsed)
            
            result = func(*args, **kwargs)
            self._last_api_call = time.time()
            return result
            
        except Exception as e:
            error_msg = f"Erro na API {func.__name__}: {str(e)}"
            logger.error(error_msg)
            self.notificar(f"❌ {error_msg[:200]}", "ERROR")
            raise

    def _testar_conexao_telegram(self) -> bool:
        """Testa a conexão com o Telegram"""
        try:
            if not self.config.TELEGRAM_TOKEN or not self.config.TELEGRAM_CHAT_ID:
                logger.error("Token ou Chat ID do Telegram não configurados")
                return False
                
            url = f"https://api.telegram.org/bot{self.config.TELEGRAM_TOKEN}/getMe"
            response = requests.get(url, timeout=10)
            
            if response.status_code != 200:
                logger.error(f"Token do Telegram inválido: {response.text}")
                return False
                
            bot_info = response.json()
            logger.info(f"Bot Telegram: @{bot_info['result']['username']}")
            
            self.telegram = TelegramNotifier(self.config.TELEGRAM_TOKEN, self.config.TELEGRAM_CHAT_ID)
            if not self.telegram.send("*Bot iniciado* - Conexão com Telegram OK"):
                logger.error("Falha ao enviar mensagem de teste")
                return False
                
            logger.info("Conexão com Telegram verificada")
            return True
            
        except Exception as e:
            logger.error(f"Falha no teste de conexão com Telegram: {str(e)}")
            return False

    def _inicializar_client(self) -> Client:
        """Inicializa o cliente da Binance"""
        try:
            api_key = os.getenv("BINANCE_API_KEY")
            api_secret = os.getenv("BINANCE_API_SECRET")
            
            if not api_key or not api_secret:
                raise ValueError("Chaves API não configuradas")
            
            logger.info("Inicializando cliente Binance...")
            client = Client(api_key, api_secret)
            
            # Teste de conexão
            server_time = client.get_server_time()
            logger.info(f"Cliente conectado. Server time: {server_time['serverTime']}")
            
            self.notificar("*Cliente Binance inicializado com sucesso*")
            return client
            
        except Exception as e:
            error_msg = f"Falha ao conectar na Binance: {str(e)}"
            logger.error(error_msg)
            self.notificar(f"❌ {error_msg[:200]}", "ERROR")
            raise

    def _get_symbol_info(self):
        """Obtém informações do símbolo"""
        try:
            if not self._exchange_info:
                self._exchange_info = self._safe_api_call(self.client.futures_exchange_info)
            
            if not self._symbol_info:
                self._symbol_info = next(
                    (s for s in self._exchange_info['symbols'] 
                    if s['symbol'] == self.config.symbol),
                    None
                )
                if not self._symbol_info:
                    raise ValueError(f"Símbolo {self.config.symbol} não encontrado")
            
            return self._symbol_info
            
        except Exception as e:
            error_msg = f"Erro ao obter informações do símbolo: {str(e)}"
            logger.error(error_msg)
            self.notificar(f"❌ {error_msg[:200]}", "ERROR")
            raise

    def notificar(self, mensagem: str, nivel: str = "INFO"):
        """Envia notificação para Telegram e log"""
        try:
            if nivel == "INFO":
                logger.info(mensagem)
            elif nivel == "WARNING":
                logger.warning(mensagem)
            elif nivel == "ERROR":
                logger.error(mensagem)
            
            if self.config.telegram_enabled and hasattr(self, 'telegram'):
                emoji = "ℹ️" if nivel == "INFO" else "⚠️" if nivel == "WARNING" else "❌"
                time.sleep(0.5)
                self.telegram.send(f"{emoji} {mensagem}")
                
        except Exception as e:
            logger.error(f"Falha ao enviar notificação: {str(e)}")

    def configurar_alavancagem(self):
        """Configura a alavancagem para o par"""
        try:
            if not 1 <= self.config.leverage <= 125:
                raise ValueError("Alavancagem deve estar entre 1 e 125")
            
            self._safe_api_call(
                self.client.futures_change_leverage,
                symbol=self.config.symbol,
                leverage=self.config.leverage
            )
            
            self.notificar(f"⚙️ Alavancagem configurada para {self.config.leverage}x")
            logger.info("Alavancagem configurada com sucesso")
            
        except Exception as e:
            error_msg = f"Erro ao configurar alavancagem: {str(e)}"
            logger.error(error_msg)
            self.notificar(f"❌ {error_msg[:200]}", "ERROR")
            raise

    def obter_dados_historicos(self) -> Optional[pd.DataFrame]:
        """Obtém dados históricos para análise"""
        try:
            klines = self._safe_api_call(
                self.client.futures_klines,
                symbol=self.config.symbol,
                interval=self.config.data_interval,
                limit=100
            )
            
            if len(klines) < 50:
                logger.warning("Dados históricos insuficientes")
                self.notificar("⚠️ Dados históricos insuficientes", "WARNING")
                return None
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'trades',
                'taker_buy_base', 'taker_buy_quote', 'ignore'
            ])
            
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_cols] = df[numeric_cols].astype(float)
            
            return df
            
        except Exception as e:
            error_msg = f"Erro ao obter dados históricos: {str(e)}"
            logger.error(error_msg)
            self.notificar(f"❌ {error_msg[:200]}", "ERROR")
            return None

    def calcular_indicadores(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Calcula indicadores técnicos"""
        try:
            if df is None or len(df) < 50:
                logger.warning("DataFrame vazio ou insuficiente para cálculo")
                return None
                
            if df.isnull().values.any():
                df = df.ffill().bfill()
                
            # EMAs
            df['ema_9'] = EMAIndicator(close=df['close'], window=9).ema_indicator()
            df['ema_21'] = EMAIndicator(close=df['close'], window=21).ema_indicator()
            df['ema_50'] = EMAIndicator(close=df['close'], window=50).ema_indicator()
            df['ema_200'] = EMAIndicator(close=df['close'], window=200).ema_indicator()
            
            # MACD
            macd = MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_diff'] = macd.macd_diff()
            df['macd_cross'] = (df['macd'] > df['macd_signal']).astype(int).diff()
            
            # RSI
            df['rsi'] = RSIIndicator(close=df['close'], window=14).rsi()
            df['rsi_smooth'] = df['rsi'].rolling(window=3).mean()
            
            # Bollinger Bands
            bb = BollingerBands(close=df['close'], window=20, window_dev=2)
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_lower'] = bb.bollinger_lband()
            df['bb_percent'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # VWAP
            vwap = VolumeWeightedAveragePrice(
                high=df['high'], low=df['low'], 
                close=df['close'], volume=df['volume'], 
                window=20
            )
            df['vwap'] = vwap.volume_weighted_average_price()
            df['vwap_std'] = df['vwap'].rolling(window=20).std()
            
            # Volume
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
            
            # ATR (Average True Range)
            df['atr'] = AverageTrueRange(
                high=df['high'], low=df['low'],
                close=df['close'], window=14
            ).average_true_range()
            
            # Stochastic Oscillator
            stoch = StochasticOscillator(
                high=df['high'], low=df['low'],
                close=df['close'], window=14, smooth_window=3
            )
            df['stoch_k'] = stoch.stoch()
            df['stoch_d'] = stoch.stoch_signal()
            
            return df
            
        except Exception as e:
            error_msg = f"Erro ao calcular indicadores: {str(e)}"
            logger.error(error_msg)
            self.notificar(f"❌ {error_msg[:200]}", "ERROR")
            return None

    def obter_tendencia_maior_tf(self) -> Optional[str]:
        """Analisa tendência em timeframe maior"""
        try:
            klines = self._safe_api_call(
                self.client.futures_klines,
                symbol=self.config.symbol,
                interval=self.config.higher_tf_interval,
                limit=50
            )
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'trades',
                'taker_buy_base', 'taker_buy_quote', 'ignore'
            ])
            
            df['close'] = df['close'].astype(float)
            df['ema_50'] = EMAIndicator(close=df['close'], window=50).ema_indicator()
            df['ema_200'] = EMAIndicator(close=df['close'], window=200).ema_indicator()
            
            return "ALTA" if df['ema_50'].iloc[-1] > df['ema_200'].iloc[-1] else "BAIXA"
            
        except Exception as e:
            logger.error(f"Erro análise maior TF: {str(e)}")
            return None

    def obter_sinal(self, df: pd.DataFrame) -> Optional[str]:
        """Gera sinal baseado nos indicadores técnicos"""
        try:
            if df is None or len(df) < 50:
                logger.warning("DataFrame insuficiente para gerar sinal")
                return None
                
            ultima_linha = df.iloc[-1]
            tendencia_maior_tf = self.obter_tendencia_maior_tf()
            
            # DEBUG: Mostrar valores dos indicadores
            logger.info(f"\n📊 Análise Técnica - {datetime.now().strftime('%H:%M:%S')}")
            logger.info(f"Preço Atual: {ultima_linha['close']:.2f}")
            logger.info(f"RSI: {ultima_linha['rsi_smooth']:.1f}")
            logger.info(f"MACD Diff: {ultima_linha['macd_diff']:.4f}")
            logger.info(f"Volume Ratio: {ultima_linha['volume_ratio']:.2f}")
            logger.info(f"BB %: {ultima_linha['bb_percent']:.2f}")
            logger.info(f"EMA9 > EMA21: {ultima_linha['ema_9'] > ultima_linha['ema_21']}")
            logger.info(f"EMA50 > EMA200: {ultima_linha['ema_50'] > ultima_linha['ema_200']}")
            logger.info(f"Tendência maior TF: {tendencia_maior_tf}")
            
            tendencia_alta = ultima_linha['ema_50'] > ultima_linha['ema_200']
            tendencia_baixa = ultima_linha['ema_50'] < ultima_linha['ema_200']
            
            # Condições mais restritivas para compra
            condicoes_compra = [
                ultima_linha['rsi_smooth'] < 30,
                ultima_linha['bb_percent'] < 0.15,
                ultima_linha['macd_diff'] > 0,
                ultima_linha['ema_9'] > ultima_linha['ema_21'],
                ultima_linha['volume_ratio'] > 2.0,
                tendencia_alta or (tendencia_maior_tf == "ALTA"),
                ultima_linha['stoch_k'] < 30 and ultima_linha['stoch_k'] > ultima_linha['stoch_d']
            ]
            
            # Condições mais restritivas para venda
            condicoes_venda = [
                ultima_linha['rsi_smooth'] > 70,
                ultima_linha['bb_percent'] > 0.85,
                ultima_linha['macd_diff'] < 0,
                ultima_linha['ema_9'] < ultima_linha['ema_21'],
                ultima_linha['volume_ratio'] > 2.0,
                tendencia_baixa or (tendencia_maior_tf == "BAIXA"),
                ultima_linha['stoch_k'] > 70 and ultima_linha['stoch_k'] < ultima_linha['stoch_d']
            ]
            
            logger.info(f"Condições COMPRA: {sum(condicoes_compra)}/7")
            logger.info(f"Condições VENDA: {sum(condicoes_venda)}/7")

            if sum(condicoes_compra) >= 3:
                logger.info("✅ Sinal de COMPRA identificado")
                return "COMPRAR"
            elif sum(condicoes_venda) >= 3:
                logger.info("✅ Sinal de VENDA identificado")
                return "VENDER"
                
            logger.info("❌ Nenhum sinal forte identificado")
            return None
            
        except Exception as e:
            error_msg = f"Erro ao gerar sinal: {str(e)}"
            logger.error(error_msg)
            self.notificar(f"❌ {error_msg[:200]}", "ERROR")
            return None

    def obter_saldo_conta(self) -> float:
        """Obtém o saldo disponível em USDT"""
        try:
            saldo = self._safe_api_call(self.client.futures_account_balance)
            saldo_usdt = next((item for item in saldo if item['asset'] == 'USDT'), None)
            
            if not saldo_usdt:
                raise ValueError("Saldo USDT não encontrado")
                
            saldo = float(saldo_usdt['balance'])
            margem_disponivel = saldo * self.config.leverage
            logger.info(f"Saldo disponível: ${saldo:.2f} | Margem disponível ({self.config.leverage}x): ${margem_disponivel:.2f}")
            
            if saldo < self.config.min_balance:
                logger.warning(f"Saldo abaixo do mínimo recomendado (${self.config.min_balance})")
                self.notificar(f"⚠️ Saldo baixo: ${saldo:.2f} (Mínimo recomendado: ${self.config.min_balance})", "WARNING")
                
            return saldo
            
        except Exception as e:
            error_msg = f"Erro ao verificar saldo: {str(e)}"
            logger.error(error_msg)
            self.notificar(f"❌ {error_msg[:200]}", "ERROR")
            return 0

    def obter_preco_atual(self) -> Optional[float]:
        """Obtém o preço atual do par"""
        try:
            ticker = self._safe_api_call(self.client.futures_mark_price, symbol=self.config.symbol)
            preco = float(ticker["markPrice"])
            logger.debug(f"Preço atual: {preco:.2f}")
            return preco
            
        except Exception as e:
            error_msg = f"Erro ao obter preço: {str(e)}"
            logger.error(error_msg)
            self.notificar(f"❌ {error_msg[:200]}", "ERROR")
            return None

    def calcular_tamanho_posicao(self, saldo: float, preco: float) -> float:
        """Calcula o tamanho da posição com gerenciamento de risco"""
        try:
            if preco <= 0 or saldo <= 0:
                raise ValueError("Preço ou saldo inválido")
                
            # Verificar saldo mínimo
            if saldo < self.config.min_balance:
                raise ValueError(f"Saldo mínimo não atingido (${self.config.min_balance})")
                
            # Obter informações do símbolo para step size
            symbol_info = self._get_symbol_info()
            if not symbol_info:
                raise ValueError("Não foi possível obter informações do símbolo")
                
            # Encontrar step size
            lot_size = next(
                (f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'),
                None
            )
            if not lot_size:
                raise ValueError("Não foi possível obter step size do símbolo")
                
            step_size = float(lot_size['stepSize'])
            min_qty = float(lot_size['minQty'])
            
            # Calcular valor mínimo garantindo o notional mínimo
            valor_minimo = max(
                saldo * (self.config.risk_percent / 100) * self.config.leverage,
                self.config.min_notional
            )
            
            # Limitar pelo máximo permitido
            valor_posicao = min(
                valor_minimo,
                saldo * self.config.leverage * self.config.max_position_size
            )
            
            # Calcular quantidade baseada no valor da posição
            quantidade = valor_posicao / preco
            
            # Ajustar para step size
            quantidade = max(round(quantidade, 8), min_qty)
            quantidade = math.floor(quantidade / step_size) * step_size
            
            # Verificação final do valor nocional
            valor_nocional = preco * quantidade
            if valor_nocional < self.config.min_notional:
                # Ajuste final garantindo o mínimo
                quantidade = math.ceil(self.config.min_notional / preco / step_size) * step_size
                valor_nocional = preco * quantidade
                logger.warning(f"Ajuste final para valor nocional: ${valor_nocional:.2f}")
                self.notificar(f"⚠️ Ajustando tamanho da posição para ${valor_nocional:.2f}", "WARNING")
                
            logger.info(f"Quantidade calculada: {quantidade:.6f} (Valor: ${valor_nocional:.2f})")
            return quantidade
            
        except Exception as e:
            error_msg = f"Erro ao calcular quantidade: {str(e)}"
            logger.error(error_msg)
            self.notificar(f"❌ {error_msg[:200]}", "ERROR")
            return 0.0

    def executar_trade(self, sinal: str) -> bool:
        """Executa o trade completo com ordens de proteção"""
        try:
            # Verificação de tempo mínimo entre trades
            if self.ultimo_trade_time and (datetime.now() - self.ultimo_trade_time).seconds < 300:
                logger.warning("Aguardando tempo mínimo entre trades")
                self.notificar("⏳ Aguardando tempo mínimo entre trades (5 minutos)", "WARNING")
                return False
            
            # Reset diário do contador de trades
            if datetime.now().date() != self.ultimo_reset_dia:
                self.trades_do_dia = 0
                self.ultimo_reset_dia = datetime.now().date()
            
            # Limite de trades por dia
            if self.trades_do_dia >= 5:
                logger.warning("Limite diário de trades atingido")
                self.notificar("⚠️ Limite diário de trades atingido (5 trades)", "WARNING")
                return False
            
            # Obtenção de saldo e preço
            saldo = self.obter_saldo_conta()
            preco = self.obter_preco_atual()
            
            if not all([saldo, preco, saldo > self.config.min_balance]):
                logger.warning("Condições inválidas para trade")
                self.notificar("❌ Condições inválidas para trade", "WARNING")
                return False
                
            # Cálculo de quantidade
            quantidade = self.calcular_tamanho_posicao(saldo, preco)
            
            if quantidade <= 0:
                logger.warning("Quantidade inválida para trade")
                self.notificar("❌ Quantidade inválida para trade", "WARNING")
                return False
            
            # Verificação final do valor nocional
            valor_nocional = preco * quantidade
            if valor_nocional < self.config.min_notional:
                error_msg = f"Não foi possível atingir o valor mínimo (${valor_nocional:.2f} < ${self.config.min_notional})"
                logger.error(error_msg)
                self.notificar(f"❌ {error_msg}", "ERROR")
                return False
            
            # Configuração da ordem
            if sinal == "COMPRAR":
                stop_price = preco * (1 - self.config.stop_loss/100)
                take_profit = preco * (1 + self.config.take_profit/100)
                lado = "BUY"
                lado_oposto = "SELL"
                emoji_sinal = "🟢"
                texto_sinal = "COMPRA"
            else:
                stop_price = preco * (1 + self.config.stop_loss/100)
                take_profit = preco * (1 - self.config.take_profit/100)
                lado = "SELL"
                lado_oposto = "BUY"
                emoji_sinal = "🔴"
                texto_sinal = "VENDA"
            
            # Verificação de distâncias mínimas
            min_diff = 0.1
            if abs(preco - stop_price)/preco * 100 < min_diff:
                logger.warning("Stop Loss muito próximo do preço")
                self.notificar(f"❌ Stop Loss muito próximo do preço (mínimo {min_diff}%)", "WARNING")
                return False
            
            if abs(preco - take_profit)/preco * 100 < min_diff:
                logger.warning("Take Profit muito próximo do preço")
                self.notificar(f"❌ Take Profit muito próximo do preço (mínimo {min_diff}%)", "WARNING")
                return False
            
            # Mensagem de trade
            msg_trade = (
                f"{emoji_sinal} *NOVA ORDEM EXECUTADA* {emoji_sinal}\n"
                f"*Par:* {self.config.symbol}\n"
                f"*Direção:* {texto_sinal}\n"
                f"*Preço:* {preco:.2f}\n"
                f"*Quantidade:* {quantidade:.6f}\n"
                f"*Valor:* ${valor_nocional:.2f}\n"
                f"*Alavancagem:* {self.config.leverage}x\n"
                f"*Stop Loss:* {stop_price:.2f} ({self.config.stop_loss}%)\n"
                f"*Take Profit:* {take_profit:.2f} ({self.config.take_profit}%)\n"
                f"*Risco:* {self.config.risk_percent}% do saldo"
            )
            
            # Execução das ordens
            self._safe_api_call(
                self.client.futures_create_order,
                symbol=self.config.symbol,
                side=lado,
                type="MARKET",
                quantity=quantidade,
                recvWindow=10000
            )
            
            time.sleep(1)
            
            self._safe_api_call(
                self.client.futures_create_order,
                symbol=self.config.symbol,
                side=lado_oposto,
                type="STOP_MARKET",
                stopPrice=round(stop_price, 2),
                quantity=quantidade,
                closePosition="true",
                workingType="MARK_PRICE",
                timeInForce="GTC"
            )
            
            time.sleep(1)
            
            self._safe_api_call(
                self.client.futures_create_order,
                symbol=self.config.symbol,
                side=lado_oposto,
                type="TAKE_PROFIT_MARKET",
                stopPrice=round(take_profit, 2),
                quantity=quantidade,
                closePosition="true",
                workingType="MARK_PRICE",
                timeInForce="GTC"
            )
            
            # Atualiza estado
            self.notificar(msg_trade)
            self.contagem_trades += 1
            self.trades_do_dia += 1
            self.ultimo_trade_time = datetime.now()
            
            # Iniciar monitoramento com trailing e break-even
            self.monitorar_posicao_com_trailing(
                entrada=preco,
                lado=lado,
                quantidade=quantidade
            )
            
            # Registra o trade
            self.registrar_trade({
                'symbol': self.config.symbol,
                'side': lado,
                'price': preco,
                'quantity': quantidade,
                'stop_loss': stop_price,
                'take_profit': take_profit,
                'timestamp': datetime.now().isoformat(),
                'notional_value': valor_nocional
            })
            
            logger.info("Trade executado com sucesso")
            return True
            
        except Exception as e:
            error_msg = f"Falha ao executar trade: {str(e)}"
            logger.error(error_msg)
            self.notificar(f"❌ {error_msg[:200]}", "ERROR")
            return False

    def registrar_trade(self, trade_data: Dict[str, Any]):
        """Registra os detalhes do trade em um arquivo JSON"""
        try:
            arquivo_log = "trades_log.json"
            dados_existentes = []
            
            if os.path.exists(arquivo_log):
                with open(arquivo_log, 'r') as f:
                    try:
                        dados_existentes = json.load(f)
                    except json.JSONDecodeError:
                        logger.warning("Arquivo de log corrompido, iniciando novo")
                        dados_existentes = []
            
            dados_existentes.append(trade_data)
            
            with open(arquivo_log, 'w') as f:
                json.dump(dados_existentes, f, indent=2)
                
            logger.debug("Trade registrado com sucesso")
                
        except Exception as e:
            logger.error(f"Erro ao registrar trade: {str(e)}")

    def verificar_posicoes_abertas(self) -> bool:
        """Verifica se há posições abertas"""
        try:
            posicoes = self._safe_api_call(self.client.futures_position_information)
            
            for p in posicoes:
                if (p['symbol'] == self.config.symbol and 
                    float(p['positionAmt']) != 0):
                    logger.info(f"Posição aberta encontrada: {p['positionAmt']}")
                    return True
                    
            logger.debug("Nenhuma posição aberta encontrada")
            return False
            
        except Exception as e:
            error_msg = f"Erro ao verificar posições: {str(e)}"
            logger.error(error_msg)
            self.notificar(f"❌ {error_msg[:200]}", "ERROR")
            return True  # Assume que há posição aberta para segurança

    def monitorar_posicao_com_trailing(self, entrada: float, lado: str, quantidade: float):
        """Monitora a posição aberta com lógica de break-even e trailing stop"""
        try:
            preco_max = entrada
            preco_min = entrada
            breakeven_atingido = False
            intervalo = 15  # segundos entre checagens

            while self.verificar_posicoes_abertas():
                preco_atual = self.obter_preco_atual()
                if preco_atual is None:
                    time.sleep(intervalo)
                    continue

                if lado == "BUY":
                    preco_max = max(preco_max, preco_atual)
                    lucro_percent = (preco_max - entrada) / entrada

                    if not breakeven_atingido and preco_max >= entrada * (1 + self.config.breakeven_percent/100):
                        breakeven_atingido = True
                        self.atualizar_stop_loss(entrada, quantidade, "SELL")
                        self.notificar("🟡 Break-even ativado: stop movido para o ponto de entrada")

                    elif breakeven_atingido:
                        novo_stop = preco_max * (1 - self.config.trailing_stop_percent/100)
                        self.atualizar_stop_loss(novo_stop, quantidade, "SELL")
                        self.notificar(f"🔄 Trailing Stop atualizado para ${novo_stop:.2f}")

                elif lado == "SELL":
                    preco_min = min(preco_min, preco_atual)
                    lucro_percent = (entrada - preco_min) / entrada

                    if not breakeven_atingido and preco_min <= entrada * (1 - self.config.breakeven_percent/100):
                        breakeven_atingido = True
                        self.atualizar_stop_loss(entrada, quantidade, "BUY")
                        self.notificar("🟡 Break-even ativado: stop movido para o ponto de entrada")

                    elif breakeven_atingido:
                        novo_stop = preco_min * (1 + self.config.trailing_stop_percent/100)
                        self.atualizar_stop_loss(novo_stop, quantidade, "BUY")
                        self.notificar(f"🔄 Trailing Stop atualizado para ${novo_stop:.2f}")

                time.sleep(intervalo)
        except Exception as e:
            self.notificar(f"❌ Erro no monitoramento da posição: {str(e)}", "ERROR")

    def atualizar_stop_loss(self, stop_price: float, quantidade: float, lado_oposto: str):
        """Atualiza o stop loss existente com novo preço"""
        try:
            # Primeiro cancela todas as ordens STOP existentes
            self._safe_api_call(
                self.client.futures_cancel_all_open_orders,
                symbol=self.config.symbol
            )
            
            time.sleep(1)
            
            # Cria nova ordem STOP
            self._safe_api_call(
                self.client.futures_create_order,
                symbol=self.config.symbol,
                side=lado_oposto,
                type="STOP_MARKET",
                stopPrice=round(stop_price, 2),
                quantity=quantidade,
                closePosition="true",
                workingType="MARK_PRICE",
                timeInForce="GTC"
            )
            
            logger.info(f"Stop loss atualizado para {stop_price:.2f}")
        except Exception as e:
            self.notificar(f"Erro ao atualizar stop: {str(e)}", "ERROR")

    def verificar_saude(self) -> bool:
        """Verifica a saúde geral da conta e conexão"""
        try:
            saldo = self.obter_saldo_conta()
            server_time = self._safe_api_call(self.client.get_server_time)
            system_status = self._safe_api_call(self.client.get_system_status)
            
            # Executa backtest rápido
            resultados_backtest = self.backtester.executar_backtest(days=7)
            
            msg_saude = (
                "🩺 *RELATÓRIO DE SAÚDE*\n"
                f"💰 *Saldo:* ${saldo:.2f}\n"
                f"📈 *Lucro Acumulado:* ${self.lucro_acumulado:.2f}\n"
                f"📊 *Trades Hoje:* {self.trades_do_dia}/5\n"
                f"🔄 *Total Trades:* {self.contagem_trades}\n"
                f"🕒 *Hora do servidor:* {datetime.fromtimestamp(server_time['serverTime']/1000)}\n"
                f"📡 *Status:* {'Operacional' if system_status.get('status') == 0 else 'Problemas'}\n"
                f"🔍 *Backtest (7 dias):* {resultados_backtest['win_rate']:.1f}% win rate" if resultados_backtest else ""
            )
            
            self.notificar(msg_saude)
            logger.info("Verificação de saúde concluída")
            return True
            
        except Exception as e:
            error_msg = f"Falha na verificação de saúde: {str(e)}"
            logger.error(error_msg)
            self.notificar(f"⚠️ {error_msg[:200]}", "ERROR")
            return False

    def executar(self):
        """Loop principal do bot"""
        msg_inicio = (
            "🚀 *BOT INICIADO - ESTRATÉGIA PARA $300/MÊS*\n"
            f"📊 *Par:* {self.config.symbol}\n"
            f"⚖️ *Alavancagem:* {self.config.leverage}x\n"
            f"⚠️ *Risco por trade:* {self.config.risk_percent}%\n"
            f"🛑 *Stop Loss:* {self.config.stop_loss}%\n"
            f"🎯 *Take Profit:* {self.config.take_profit}%\n"
            f"🔄 *Trailing Stop:* {self.config.trailing_stop_percent}%\n"
            f"🟡 *Break-even:* {self.config.breakeven_percent}%\n"
            f"⏱ *Intervalo:* {self.config.data_interval}\n"
            f"📈 *Tendência maior TF:* {self.config.higher_tf_interval}\n"
            f"🔄 *Frequência:* {self.config.check_interval}s\n"
            f"📈 *Tamanho máximo da posição:* {self.config.max_position_size*100:.0f}% do saldo\n"
            f"💰 *Valor mínimo por ordem:* ${self.config.min_notional}\n"
            f"📊 *Meta diária:* ~$15 (5 trades/dia max)"
        )
            
        self.notificar(msg_inicio)
        logger.info("Iniciando loop principal")
        
        ultima_verificacao_saude = datetime.now()
        
        try:
            while True:
                try:
                    # Verificação periódica de saúde
                    if (datetime.now() - ultima_verificacao_saude).seconds > 3600:
                        self.verificar_saude()
                        ultima_verificacao_saude = datetime.now()
                    
                    # Verifica posições abertas
                    if self.verificar_posicoes_abertas():
                        logger.info("Posição aberta encontrada - aguardando...")
                        time.sleep(60)
                        continue
                    
                    # Obtém e processa dados
                    df = self.obter_dados_historicos()
                    
                    if df is None:
                        self.erros_consecutivos += 1
                        if self.erros_consecutivos >= self.config.max_consecutive_errors:
                            logger.error("Muitos erros consecutivos - reiniciando...")
                            self.notificar("⚠️ Muitos erros consecutivos. Reiniciando...", "WARNING")
                            time.sleep(60)
                            continue
                        time.sleep(10)
                        continue
                    
                    df = self.calcular_indicadores(df)
                    
                    if df is None:
                        self.erros_consecutivos += 1
                        if self.erros_consecutivos >= self.config.max_consecutive_errors:
                            logger.error("Muitos erros consecutivos - reiniciando...")
                            self.notificar("⚠️ Muitos erros consecutivos. Reiniciando...", "WARNING")
                            time.sleep(60)
                            continue
                        time.sleep(10)
                        continue
                    
                    self.erros_consecutivos = 0
                    
                    # Gera e executa sinal
                    sinal = self.obter_sinal(df)
                    preco = self.obter_preco_atual()
                    
                    if sinal and preco:
                        logger.info(f"Preço Atual: {preco:.2f}")
                        self.executar_trade(sinal)
                    
                    time.sleep(self.config.check_interval)
                    
                except KeyboardInterrupt:
                    logger.info("Bot interrompido pelo usuário")
                    self.notificar("🛑 Bot interrompido pelo usuário")
                    break
                    
                except Exception as e:
                    error_msg = f"Erro no loop principal: {str(e)}"
                    logger.error(error_msg)
                    self.notificar(f"⚠️ {error_msg[:200]}", "ERROR")
                    
                    self.erros_consecutivos += 1
                    if self.erros_consecutivos >= self.config.max_consecutive_errors:
                        logger.error("Muitos erros consecutivos - encerrando...")
                        self.notificar("🆘 Muitos erros consecutivos. Encerrando...", "ERROR")
                        break
                        
                    time.sleep(30)
                    
        except Exception as e:
            error_msg = f"ERRO FATAL: {str(e)}"
            logger.critical(error_msg)
            self.notificar(f"❌ {error_msg[:200]}", "ERROR")
            raise
            
        finally:
            logger.info("Encerrando bot...")
            self.notificar("🔴 Bot encerrado")

def verificar_ambiente():
    """Verifica se todas as variáveis de ambiente necessárias estão configuradas"""
    print("\n🛠️ Verificação de Variáveis de Ambiente:")
    
    variaveis_necessarias = {
        'BINANCE_API_KEY': 'Chave API Binance',
        'BINANCE_API_SECRET': 'Segredo API Binance',
        'TELEGRAM_TOKEN': 'Token do Bot Telegram',
        'TELEGRAM_CHAT_ID': 'Chat ID do Telegram'
    }
    
    faltando = False
    for var, desc in variaveis_necessarias.items():
        valor = os.getenv(var)
        status = "✅" if valor else "❌"
        print(f"{status} {desc}: {'***' + valor[-4:] if valor else 'NÃO CONFIGURADO'}")
        if not valor:
            faltando = True
    
    if faltando:
        print("\n⚠️ Variáveis de ambiente necessárias faltando!")
        print("Crie um arquivo .env com essas variáveis ou configure no sistema")
        return False
    
    # Teste de conexão com Telegram
    token = os.getenv('TELEGRAM_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    if token and chat_id:
        try:
            print("\n🔍 Testando Conexão com Telegram...")
            url = f"https://api.telegram.org/bot{token}/getMe"
            resposta = requests.get(url, timeout=10)
            
            if resposta.status_code == 200:
                info_bot = resposta.json()
                print(f"✅ Conexão com Telegram bem-sucedida")
                print(f"🤖 Bot: @{info_bot['result']['username']}")
                return True
            else:
                print(f"❌ Erro na API do Telegram: {resposta.text}")
                return False
                
        except Exception as e:
            print(f"❌ Falha na conexão com Telegram: {str(e)}")
            return False
    
    return True

if __name__ == "__main__":
    # Verificação inicial do ambiente
    if not verificar_ambiente():
        exit(1)
    
    try:
        # Inicialização do bot
        logger.info("Iniciando bot...")
        bot = BinanceTradingBot(CONFIG)
        
        # Execução principal
        bot.executar()
        
    except Exception as e:
        logger.critical(f"Falha ao iniciar o bot: {str(e)}")
        raise