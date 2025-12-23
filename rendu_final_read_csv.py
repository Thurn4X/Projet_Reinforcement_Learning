import pandas as pd
import sys
import os

def evaluer_csv(file_path):
    if not os.path.exists(file_path):
        print(f"❌ Erreur : Le fichier {file_path} n'existe pas.")
        return

    # 1. Lecture du CSV
    df = pd.read_csv(file_path)
    
    # --- AJUSTEMENT DES NOMS DE COLONNES ---
    # On utilise les noms que tu as fournis dans ton exemple
    col_portefeuille = 'Valeur_Portefeuille'
    col_prix = 'Prix'
    
    # 2. Calcul des performances du Bot
    initial_val = df[col_portefeuille].iloc[0]
    final_val = df[col_portefeuille].iloc[-1]
    bot_return_pct = ((final_val - initial_val) / initial_val) * 100
    
    # 3. Calcul de la performance du Marché (Buy & Hold)
    initial_price = df[col_prix].iloc[0]
    final_price = df[col_prix].iloc[-1]
    market_return_pct = ((final_price - initial_price) / initial_price) * 100
    
    # 4. Calcul de l'Alpha (Surperformance)
    alpha = bot_return_pct - market_return_pct
    
    # 5. Calcul du Max Drawdown (La pire chute de ton capital)
    # Formule : (Valeur actuelle / Maximum historique) - 1
    rolling_max = df[col_portefeuille].cummax()
    drawdown = (df[col_portefeuille] / rolling_max) - 1
    max_drawdown = drawdown.min() * 100

    # --- AFFICHAGE DU RAPPORT ---
    print("\n" + "="*45)
    print(f"📊 RAPPORT DE PERFORMANCE : {os.path.basename(file_path)}")
    print("="*45)
    print(f"💰 Valeur Initiale   : {initial_val:,.2f} $")
    print(f"💰 Valeur Finale     : {final_val:,.2f} $")
    print("-" * 45)
    print(f"🤖 Performance BOT   : {bot_return_pct:.2f} %")
    print(f"🌍 Marché (B&H)      : {market_return_pct:.2f} %")
    print(f"⭐ ALPHA NET         : {alpha:.2f} % " + ("(SUCCÈS ✅)" if alpha > 0 else "(SOUS-PERF ❌)"))
    print("-" * 45)
    print(f"📉 Max Drawdown      : {max_drawdown:.2f} % (Risque de perte max)")
    print(f"🔢 Nombre de bougies : {len(df)} (Format 4H)")
    print("="*45 + "\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluer_performance.py RAPPORT_TRADES_votre_fichier.csv")
    else:
        evaluer_csv(sys.argv[1])