import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import get_kline_data, get_all_usdt_pairs, get_suggested_coins, is_favorite, add_to_favorites, remove_from_favorites, initialize_favorites

def show_correlation_page():
    st.header("Korrelationsanalyse")
    initialize_favorites()
    suggested_coins = get_suggested_coins()

    col_coin, col_button = st.columns([3, 1.5])
    coin_to_correlate = col_coin.text_input("Basiskoin (z.B. BTCUSDT)", value=st.session_state.get("coin_corr", suggested_coins[0] if suggested_coins else ""), key="coin_corr")
    
    # Buttons für vorgeschlagene Coins
    cols_suggest = col_coin.columns(min(5, len(suggested_coins)))
    for i, coin in enumerate(suggested_coins):
        if cols_suggest[i % len(cols_suggest)].button(coin, key=f"suggest_corr_{coin}"):
            st.session_state["coin_corr"] = coin
            st.rerun()

    interval = st.sidebar.selectbox("Intervall", ['1m', '5m', '15m', '30m', '1h', '4h', '1d'], index=4)
    num_coins = st.sidebar.slider("Anzahl der Coins", 5, 100, 20, 5)

    if col_button.button("Analyse starten", use_container_width=True):
        if coin_to_correlate not in get_all_usdt_pairs():
            st.error(f"{coin_to_correlate} ist kein gültiges USDT-Paar")
        else:
            try:
                analyze_correlations(coin_to_correlate, interval, num_coins)
            except Exception as e:
                st.error(f"Fehler: {e}")

def analyze_correlations(base_coin, interval, num_coins):
    base_df = get_kline_data(base_coin, interval=interval)
    if base_df.empty:
        st.warning("Keine Daten für Basis-Coin gefunden")
        return

    correlation_data = {}
    all_pairs = [p for p in get_all_usdt_pairs() if p != base_coin][:num_coins]
    
    progress_bar = st.progress(0)
    for i, coin in enumerate(all_pairs):
        try:
            df = get_kline_data(coin, interval=interval)
            if not df.empty:
                merged = pd.merge(base_df, df, left_index=True, right_index=True, suffixes=(f'_{base_coin}', f'_{coin}'))
                corr = merged[f'close_{base_coin}'].corr(merged[f'close_{coin}'])
                correlation_data[coin] = corr
        except:
            continue
        progress_bar.progress((i + 1) / len(all_pairs))

    if correlation_data:
        corr_series = pd.Series(correlation_data).sort_values(ascending=False)
        st.dataframe(corr_series)

        # Heatmap für Top 10
        top_coins = corr_series.head(10).index.tolist()
        heatmap_data = pd.DataFrame()
        for coin in [base_coin] + top_coins:
            df = get_kline_data(coin, interval=interval)
            if not df.empty:
                heatmap_data[coin] = df['close']

        if len(heatmap_data.columns) > 1:
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(heatmap_data.corr(), annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)