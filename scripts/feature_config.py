"""
Definição centralizada das colunas de features usadas pelo modelo.
Importado por api/main.py, scripts/train_model.py e scripts/build_features.py.
"""

FEATURE_COLS = [
    # Form geral
    "home_form_pts", "away_form_pts",
    "home_form_gf",  "away_form_gf",
    "home_form_ga",  "away_form_ga",
    # Form por mando
    "home_home_pts", "away_away_pts",
    "home_home_gf",  "away_away_gf",
    "home_home_ga",  "away_away_ga",
    # H2H
    "h2h_home_wins", "h2h_draws", "h2h_away_wins",
    "h2h_home_gf_avg", "h2h_away_gf_avg",
    # Tabela
    "home_table_pos", "away_table_pos",
    "home_table_pts", "away_table_pts",
    "pos_diff", "pts_diff",
    # xG médio por time (FBref/API-Football — preenchido com 0 se não disponível)
    "home_avg_xg", "away_avg_xg",
    "home_avg_xga", "away_avg_xga",
    "home_xg_net", "away_xg_net",
    "home_avg_poss",
    # Valor e público (Base dos Dados)
    "squad_value_ratio",
    "home_attendance_pct",
    # Contexto
    "matchday",
]

# Features específicas do mandante (perspectiva casa)
HOME_SPLIT_COLS = [
    "home_form_pts", "home_form_gf", "home_form_ga",
    "home_home_pts", "home_home_gf", "home_home_ga",
    "h2h_home_wins", "h2h_draws", "h2h_home_gf_avg",
    "home_table_pos", "home_table_pts",
    "pos_diff", "pts_diff",
    "home_avg_xg", "home_avg_xga", "home_xg_net",
    "home_avg_poss",
    "squad_value_ratio",
    "home_attendance_pct",
    "matchday",
]

# Features específicas do visitante (perspectiva fora)
AWAY_SPLIT_COLS = [
    "away_form_pts", "away_form_gf", "away_form_ga",
    "away_away_pts", "away_away_gf", "away_away_ga",
    "h2h_away_wins", "h2h_draws", "h2h_away_gf_avg",
    "away_table_pos", "away_table_pts",
    "pos_diff", "pts_diff",
    "away_avg_xg", "away_avg_xga",
    "squad_value_ratio",
    "home_attendance_pct",
    "matchday",
]

# Ligas com temporada cross-year (temporada começa no ano anterior ao término)
CROSS_YEAR_LEAGUES = {"PL", "PD", "SA", "FL1", "BL1", "CL", "DED", "PPL", "ELC"}
