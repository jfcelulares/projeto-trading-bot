import streamlit as st
import pandas as pd
import json
import os
import time
from datetime import datetime
from bot_auto_final1 import BinanceTradingBot, CONFIG

# Inicia o bot (sem executar loop principal)
bot = BinanceTradingBot(CONFIG)

# Streamlit config
st.set_page_config(page_title="Painel do Bot de Trading", layout="wide")
st.title("📈 Painel em Tempo Real - Bot de Trading")

# Sessão de tempo real
col1, col2 = st.columns(2)

with col1:
    st.subheader("⚙️ Estratégia")
    for k, v in {
        "Par": CONFIG["SYMBOL"],
        "Alavancagem": f'{CONFIG["LEVERAGE"]}x',
        "Risco": f'{CONFIG["RISK_PERCENT"]}%',
        "SL": f'{CONFIG["STOP_LOSS"]}%',
        "TP": f'{CONFIG["TAKE_PROFIT"]}%',
        "Break-even": f'{CONFIG["BREAKEVEN_PERCENT"]}%',
        "Trailing": f'{CONFIG["TRAILING_STOP_PERCENT"]}%',
    }.items():
        st.text(f"{k}: {v}")

with col2:
    st.subheader("📊 Dados em Tempo Real")
    preco = bot.obter_preco_atual()
    saldo = bot.obter_saldo_conta()
    margem = saldo * CONFIG["LEVERAGE"]

    st.metric("Preço Atual", f"${preco:.2f}" if preco else "Erro")
    st.metric("Saldo", f"${saldo:.2f}")
    st.metric("Margem Disponível", f"${margem:.2f}")

    df = bot.obter_dados_historicos()
    if df is not None:
        df = bot.calcular_indicadores(df)
        sinal = bot.obter_sinal(df)
        status_sinal = "🟢 COMPRA" if sinal == "COMPRAR" else "🔴 VENDA" if sinal == "VENDER" else "⚪ NEUTRO"
        st.subheader("🧠 Sinal Atual")
        st.markdown(f"### {status_sinal}")
    else:
        st.warning("Sem dados suficientes para análise técnica.")

st.divider()

# Mostrar histórico de trades
st.subheader("📋 Histórico de Trades")
if os.path.exists("trades_log.json"):
    with open("trades_log.json", "r") as f:
        try:
            dados = json.load(f)
            df_trades = pd.DataFrame(dados)
            df_trades['timestamp'] = pd.to_datetime(df_trades['timestamp'])
            df_trades = df_trades.sort_values("timestamp", ascending=False)
            st.dataframe(df_trades[['timestamp', 'symbol', 'side', 'price', 'quantity', 'notional_value']],
                         use_container_width=True)
        except json.JSONDecodeError:
            st.warning("Erro ao carregar trades.")
else:
    st.info("Nenhum trade registrado ainda.")

st.caption(f"📅 Última atualização: {datetime.now().strftime('%H:%M:%S')} — recarregue a página para atualizar.")
