# TradingAI
MaschineLearning, Monitoring, Spread Chart, Korrelation zur Kursvorhersage

pip install streamlit pandas numpy matplotlib seaborn requests streamlit-js-eval plotly ta-lib

py -m pip install .\trading_app\ta_lib-0.6.3-cp313-cp313-win_amd64.whl

streamlit run .\trading_app\main.py






Zukunftliche Wichtige Erweiterungen:

    Datenquellen Integration:

        Echte Kursdaten von Binance API

        Funding Rates für Futures

        Simuliertes Sentiment (kann später durch echte API ersetzt werden)

    Feature Engineering:

        Technische Indikatoren (RSI, MACD, Bollinger Bands)

        Volumenanalyse

        Zeitbasierte Features (Stunde, Wochentag)

    Machine Learning Modell:

        Random Forest Regressor für erste Implementierung

        Modell-Evaluation mit MAE und R²

        Feature Importance Analyse

    Visualisierung:

        Interaktive Plotly Charts

        Klare Darstellung von historischen Daten vs. Prognose

        Wichtigste Einflussfaktoren als Tabelle

    Benutzerinterface:

        Einfache Coin-Auswahl

        Fortschrittsanzeige während der Berechnung

        Erklärungen für Benutzer

Nächste Entwicklungsschritte:

    Echte Sentiment Integration:

        Alternative APIs: LunarCrush, Santiment, etc.

        Social Media Daten (Twitter, Reddit)

    Modellverbesserungen:

        LSTM/GRU Netzwerke für Zeitreihen

        Hyperparameter-Tuning

        Ensemble-Methoden

    Erweiterte Features:

        On-Chain Daten

        Liquiditätsanalysen

        News-Analyse

    Backtesting:

        Historische Performance-Tests

        Risikoanalyse