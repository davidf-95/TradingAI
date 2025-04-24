import streamlit as st
from utils import get_all_usdt_pairs, add_to_favorites, remove_from_favorites, is_favorite, initialize_favorites

def show_favorites_page():
    st.header("Favoriten Liste")
    initialize_favorites()

    new_favorite = st.text_input("Neuen Favoriten hinzufügen:")
    if st.button("➕ Hinzufügen"):
        new_favorite = new_favorite.upper()
        if not new_favorite.endswith("USDT"):
            new_favorite += "USDT"
        
        if new_favorite in get_all_usdt_pairs():
            add_to_favorites(new_favorite)
            st.rerun()
        else:
            st.warning(f"{new_favorite} ist kein gültiges USDT-Paar")

    favorites = st.session_state.get("favorites", [])
    if favorites:
        for coin in sorted(favorites):
            col1, col2 = st.columns([3, 1])
            col1.write(coin)
            if col2.button("➖ Entfernen", key=f"remove_{coin}"):
                remove_from_favorites(coin)
                st.rerun()
    else:
        st.info("Keine Favoriten gespeichert")