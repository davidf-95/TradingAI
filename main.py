# main.py
import streamlit as st
#from NewsSources import NewsAggregator


# Display in Streamlit
#st.dataframe(news_df)


# --- Seitenkonfiguration (MUSS das erste Streamlit-Kommando sein) ---
st.set_page_config(
    layout="wide",
    page_title="Trading Analysen",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

from FavoritenListe import show_favorites_page
from SpreadAnalyse import show_spread_page
from KorrelationsAnalyse import show_correlation_page
from MachineLearning import show_ai_prognosen_page

# --- Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Gehe zu", [
    "Spread Analyse", 
    "Korrelationsanalyse", 
    "Favoriten",
    "AI Prognosen"
])

# --- Hauptteil ---
if page == "Favoriten":
    show_favorites_page()
elif page == "Spread Analyse":
    show_spread_page()
elif page == "Korrelationsanalyse":
    show_correlation_page()
elif page == "AI Prognosen":
    show_ai_prognosen_page()