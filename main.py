import streamlit as st
from FavoritenListe import show_favorites_page
from SpreadAnalyse import show_spread_page
from KorrelationsAnalyse import show_correlation_page

# --- Seitenkonfiguration ---
st.set_page_config(layout="wide")  # Dies muss der erste Streamlit-Befehl sein!

# --- Sidebar für Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Gehe zu", ["Spread Analyse", "Korrelationsanalyse", "Favoriten", "Machine Learning"])

# --- Hauptteil der Anwendung ---
if page == "Favoriten":
    show_favorites_page()
elif page == "Spread Analyse":
    show_spread_page()
elif page == "Korrelationsanalyse":
    show_correlation_page()
elif page == "Machine Learning":
    st.header("Machine Learning (Coming Soon)")
    st.write("This section will be implemented later.")