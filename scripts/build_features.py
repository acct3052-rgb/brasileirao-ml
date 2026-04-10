"""
Engenharia de features para o modelo de predição do Brasileirão.

Para cada partida, calcula:
- Form recente dos times (últimos 5 jogos, geral e por mando)
- Histórico H2H (últimos 5 confrontos diretos)
- Posição e pontuação na tabela na época do jogo

Uso:
    python scripts/build_features.py --season 2024
    python scripts/build_features.py --all   (recalcula tudo)
"""

import os
import argparse
import logging
from datetime import datetime, timezone

import pandas as pd
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def get_supabase() -> Client:
    return create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])


# ── Carrega dados do Supabase ──────────────────────────────────────────────────

def load_xg_profiles(sb: Client) -> pd.DataFrame:
    """Carrega perfis xG por time/temporada (do FBref via SoccerData)."""
    resp = sb.table("team_xg_profiles").select("*").execute()
    if not resp.data:
        return pd.DataFrame()
    return pd.DataFrame(resp.data)


def load_team_season_stats(sb: Client) -> pd.DataFrame:
    """Carrega stats de público e valor de elenco (da Base dos Dados)."""
    resp = sb.table("team_season_stats").select("*").execute()
    if not resp.data:
        return pd.DataFrame()
    return pd.DataFrame(resp.data)


def load_historical_matches(sb: Client, seasons: list[int]) -> pd.DataFrame:
    """Carrega partidas históricas da Base dos Dados (2006+)."""
    resp = (
        sb.table("matches_historical")
        .select("*")
        .in_("season", seasons)
        .execute()
    )
    if not resp.data:
        return pd.DataFrame()
    df = pd.DataFrame(resp.data)
    df["match_date"] = pd.to_datetime(df["match_date"], utc=True)
    # Normaliza para o mesmo formato da tabela matches
    df["home_team_id"] = None
    df["away_team_id"] = None
    return df


def load_matches(sb: Client, seasons: list[int]) -> pd.DataFrame:
    all_data = []
    page_size = 1000
    offset = 0
    while True:
        resp = (
            sb.table("matches")
            .select("*")
            .in_("season", seasons)
            .eq("status", "FINISHED")
            .order("match_date")
            .range(offset, offset + page_size - 1)
            .execute()
        )
        if not resp.data:
            break
        all_data.extend(resp.data)
        if len(resp.data) < page_size:
            break
        offset += page_size
    log.info(f"  {len(all_data)} partidas (football-data) carregadas")
    df = pd.DataFrame(all_data)
    df["match_date"] = pd.to_datetime(df["match_date"], utc=True)
    return df


def load_scheduled(sb: Client, season: int) -> pd.DataFrame:
    """Carrega jogos ainda não disputados (para predição futura)."""
    resp = (
        sb.table("matches")
        .select("*")
        .eq("season", season)
        .in_("status", ["SCHEDULED", "TIMED"])
        .order("match_date")
        .execute()
    )
    df = pd.DataFrame(resp.data)
    if not df.empty:
        df["match_date"] = pd.to_datetime(df["match_date"], utc=True)
    return df


# ── Funções auxiliares de form ─────────────────────────────────────────────────

def get_team_form(df: pd.DataFrame, team_id: int, before_date: pd.Timestamp,
                  n: int = 5, home_only: bool = False, away_only: bool = False) -> dict:
    """
    Calcula form de um time nos últimos N jogos antes de uma data.
    Retorna: pontos médios, gols marcados médios, gols sofridos médios.
    """
    mask = (
        ((df["home_team_id"] == team_id) | (df["away_team_id"] == team_id))
        & (df["match_date"] < before_date)
    )

    if home_only:
        mask &= df["home_team_id"] == team_id
    elif away_only:
        mask &= df["away_team_id"] == team_id

    games = df[mask].sort_values("match_date", ascending=False).head(n)

    if games.empty:
        return {"pts": 0.0, "gf": 0.0, "ga": 0.0, "n": 0}

    pts_list, gf_list, ga_list = [], [], []

    for _, g in games.iterrows():
        if g["home_team_id"] == team_id:
            gf = g["home_goals"] or 0
            ga = g["away_goals"] or 0
        else:
            gf = g["away_goals"] or 0
            ga = g["home_goals"] or 0

        result = g["result"]
        if (g["home_team_id"] == team_id and result == "H") or \
           (g["away_team_id"] == team_id and result == "A"):
            pts = 3
        elif result == "D":
            pts = 1
        else:
            pts = 0

        pts_list.append(pts)
        gf_list.append(gf)
        ga_list.append(ga)

    return {
        "pts": sum(pts_list) / len(pts_list),
        "gf":  sum(gf_list) / len(gf_list),
        "ga":  sum(ga_list) / len(ga_list),
        "n":   len(games),
    }


def get_h2h(df: pd.DataFrame, home_id: int, away_id: int,
            before_date: pd.Timestamp, n: int = 5) -> dict:
    """Histórico head-to-head entre dois times."""
    mask = (
        (
            ((df["home_team_id"] == home_id) & (df["away_team_id"] == away_id)) |
            ((df["home_team_id"] == away_id) & (df["away_team_id"] == home_id))
        )
        & (df["match_date"] < before_date)
    )

    games = df[mask].sort_values("match_date", ascending=False).head(n)

    if games.empty:
        return {"home_wins": 0, "draws": 0, "away_wins": 0, "home_gf": 0.0, "away_gf": 0.0}

    home_wins, draws, away_wins = 0, 0, 0
    home_gf_list, away_gf_list = [], []

    for _, g in games.iterrows():
        # Normaliza: home_id sempre como "home" para H2H
        if g["home_team_id"] == home_id:
            hg = g["home_goals"] or 0
            ag = g["away_goals"] or 0
            r = g["result"]
        else:
            hg = g["away_goals"] or 0
            ag = g["home_goals"] or 0
            r = "A" if g["result"] == "H" else ("H" if g["result"] == "A" else "D")

        home_gf_list.append(hg)
        away_gf_list.append(ag)

        if r == "H":
            home_wins += 1
        elif r == "D":
            draws += 1
        else:
            away_wins += 1

    return {
        "home_wins": home_wins,
        "draws":     draws,
        "away_wins": away_wins,
        "home_gf":   sum(home_gf_list) / len(home_gf_list),
        "away_gf":   sum(away_gf_list) / len(away_gf_list),
    }


def get_table_position(df: pd.DataFrame, team_id: int,
                       before_date: pd.Timestamp, season: int) -> dict:
    """Calcula posição e pontuação na tabela até a data do jogo."""
    games = df[
        ((df["home_team_id"] == team_id) | (df["away_team_id"] == team_id))
        & (df["match_date"] < before_date)
        & (df["season"] == season)
    ]

    pts = 0
    for _, g in games.iterrows():
        is_home = g["home_team_id"] == team_id
        result = g["result"]
        if (is_home and result == "H") or (not is_home and result == "A"):
            pts += 3
        elif result == "D":
            pts += 1

    # Para posição, precisamos calcular para todos os times e ranquear
    # Simplificação: retorna pontos; posição relativa calculada depois
    return {"pts": pts}


def get_xg_profile(df_xg: pd.DataFrame, team_name: str, season: int) -> dict:
    """
    Retorna perfil xG médio do time na temporada.
    Se não houver dados, retorna dict com Nones.
    """
    if df_xg.empty or team_name is None:
        return {"avg_xg_home": None, "avg_xg_away": None,
                "avg_xga_home": None, "avg_xga_away": None, "avg_poss_home": None}

    row = df_xg[(df_xg["team_name"] == team_name) & (df_xg["season"] == season)]
    if row.empty:
        # Tenta temporada anterior como fallback
        row = df_xg[(df_xg["team_name"] == team_name) & (df_xg["season"] == season - 1)]

    if row.empty:
        return {"avg_xg_home": None, "avg_xg_away": None,
                "avg_xga_home": None, "avg_xga_away": None, "avg_poss_home": None}

    r = row.iloc[0]
    return {
        "avg_xg_home":   r.get("avg_xg_home"),
        "avg_xg_away":   r.get("avg_xg_away"),
        "avg_xga_home":  r.get("avg_xga_home"),
        "avg_xga_away":  r.get("avg_xga_away"),
        "avg_poss_home": r.get("avg_poss_home"),
    }


def get_squad_value_ratio(df_stats: pd.DataFrame, home_name: str,
                          away_name: str, season: int) -> float | None:
    """
    Retorna razão de valor de elenco: home_value / away_value.
    Proxy de qualidade relativa dos times.
    """
    if df_stats.empty or not home_name or not away_name:
        return None

    home_row = df_stats[(df_stats["team_name"] == home_name) & (df_stats["season"] == season)]
    away_row = df_stats[(df_stats["team_name"] == away_name) & (df_stats["season"] == season)]

    if home_row.empty or away_row.empty:
        return None

    hv = home_row.iloc[0].get("avg_home_squad_value")
    av = away_row.iloc[0].get("avg_away_squad_value")

    if hv and av and av > 0:
        return float(hv) / float(av)
    return None


def get_attendance_profile(df_stats: pd.DataFrame, team_name: str, season: int) -> float | None:
    """Retorna % média de ocupação do estádio do mandante."""
    if df_stats.empty or not team_name:
        return None

    row = df_stats[(df_stats["team_name"] == team_name) & (df_stats["season"] == season)]
    if row.empty:
        return None
    return row.iloc[0].get("avg_attendance_pct")


def compute_standings(df: pd.DataFrame, season: int, before_date: pd.Timestamp) -> dict:
    """
    Retorna dict {team_id: posição} para todos os times até a data.
    """
    season_games = df[(df["season"] == season) & (df["match_date"] < before_date)]
    teams = set(season_games["home_team_id"].tolist() + season_games["away_team_id"].tolist())

    pts_map = {}
    for team_id in teams:
        pts_map[team_id] = get_table_position(df, team_id, before_date, season)["pts"]

    # Ordena por pontos (desc) e atribui posição
    sorted_teams = sorted(pts_map.items(), key=lambda x: x[1], reverse=True)
    positions = {team_id: pos + 1 for pos, (team_id, _) in enumerate(sorted_teams)}
    return positions, pts_map


# ── Pipeline de features ───────────────────────────────────────────────────────

def build_features_for_matches(
    df_all: pd.DataFrame,
    matches: pd.DataFrame,
    df_xg: pd.DataFrame = None,
    df_stats: pd.DataFrame = None,
    team_name_map: dict = None,
) -> list[dict]:
    """
    Calcula features para uma lista de partidas.
    df_all:       todos os dados históricos (para form/h2h)
    matches:      partidas para calcular
    df_xg:        perfis xG por time/temporada (opcional, FBref)
    df_stats:     stats de público/valor por time/temporada (opcional, BDD)
    team_name_map: mapeamento de team_id → nome (para lookup nos perfis)
    """
    if df_xg is None:
        df_xg = pd.DataFrame()
    if df_stats is None:
        df_stats = pd.DataFrame()
    if team_name_map is None:
        team_name_map = {}

    rows = []

    for _, match in matches.iterrows():
        match_date  = match["match_date"]
        home_id     = match["home_team_id"]
        away_id     = match["away_team_id"]
        season      = match["season"]

        # Resolve nome dos times para lookup nos perfis
        home_name = team_name_map.get(home_id)
        away_name = team_name_map.get(away_id)

        # Form geral
        hf  = get_team_form(df_all, home_id, match_date, n=5)
        af  = get_team_form(df_all, away_id, match_date, n=5)

        # Form por mando
        hfh = get_team_form(df_all, home_id, match_date, n=5, home_only=True)
        afa = get_team_form(df_all, away_id, match_date, n=5, away_only=True)

        # H2H
        h2h = get_h2h(df_all, home_id, away_id, match_date, n=5)

        # Tabela
        positions, pts_map = compute_standings(df_all, season, match_date)
        home_pos       = positions.get(home_id, 20)
        away_pos       = positions.get(away_id, 20)
        home_pts_table = pts_map.get(home_id, 0)
        away_pts_table = pts_map.get(away_id, 0)

        # ── Novas features ────────────────────────────────────────────────────

        # xG profiles (FBref)
        home_xg = get_xg_profile(df_xg, home_name, season)
        away_xg = get_xg_profile(df_xg, away_name, season)

        # Razão de valor de elenco
        squad_value_ratio = get_squad_value_ratio(df_stats, home_name, away_name, season)

        # % ocupação do estádio (pressão da torcida)
        home_attendance_pct = get_attendance_profile(df_stats, home_name, season)

        # xG differential histórico do time (mandante gera mais xG que concede?)
        home_xg_net = None
        if home_xg["avg_xg_home"] and home_xg["avg_xga_home"]:
            home_xg_net = home_xg["avg_xg_home"] - home_xg["avg_xga_home"]

        away_xg_net = None
        if away_xg["avg_xg_away"] and away_xg["avg_xga_away"]:
            away_xg_net = away_xg["avg_xg_away"] - away_xg["avg_xga_away"]

        rows.append({
            "match_id":           match["id"],
            # Form geral
            "home_form_pts":      hf["pts"],
            "away_form_pts":      af["pts"],
            "home_form_gf":       hf["gf"],
            "away_form_gf":       af["gf"],
            "home_form_ga":       hf["ga"],
            "away_form_ga":       af["ga"],
            # Form por mando
            "home_home_pts":      hfh["pts"],
            "away_away_pts":      afa["pts"],
            "home_home_gf":       hfh["gf"],
            "away_away_gf":       afa["gf"],
            "home_home_ga":       hfh["ga"],
            "away_away_ga":       afa["ga"],
            # H2H
            "h2h_home_wins":      h2h["home_wins"],
            "h2h_draws":          h2h["draws"],
            "h2h_away_wins":      h2h["away_wins"],
            "h2h_home_gf_avg":    h2h["home_gf"],
            "h2h_away_gf_avg":    h2h["away_gf"],
            # Tabela
            "home_table_pos":     home_pos,
            "away_table_pos":     away_pos,
            "home_table_pts":     home_pts_table,
            "away_table_pts":     away_pts_table,
            "pos_diff":           away_pos - home_pos,
            "pts_diff":           home_pts_table - away_pts_table,
            # ── Novas features ────────────────────────────────────────
            # xG médio por mando (FBref — None se não disponível)
            "home_avg_xg":        home_xg.get("avg_xg_home"),
            "away_avg_xg":        away_xg.get("avg_xg_away"),
            "home_avg_xga":       home_xg.get("avg_xga_home"),
            "away_avg_xga":       away_xg.get("avg_xga_away"),
            "home_xg_net":        home_xg_net,
            "away_xg_net":        away_xg_net,
            # Posse média
            "home_avg_poss":      home_xg.get("avg_poss_home"),
            # Valor de elenco (Base dos Dados)
            "squad_value_ratio":  squad_value_ratio,   # > 1 = mandante mais valioso
            "home_attendance_pct": home_attendance_pct,
            # Contexto
            "matchday":           match.get("matchday"),
            "season":             season,
        })

    log.info(f"  {len(rows)} features calculadas")
    return rows


def save_features(sb: Client, rows: list[dict]) -> None:
    batch_size = 50
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        sb.table("match_features").upsert(batch, on_conflict="match_id").execute()
    log.info(f"  {len(rows)} features salvas no Supabase")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calcula features para o modelo ML")
    parser.add_argument("--season", type=int, action="append", dest="seasons")
    parser.add_argument("--all", action="store_true", help="Recalcula todas as temporadas")
    parser.add_argument("--upcoming", action="store_true", help="Calcula features para jogos futuros")
    args = parser.parse_args()

    sb = get_supabase()

    if args.all:
        seasons = [2020, 2021, 2022, 2023, 2024]
    else:
        seasons = args.seasons or [datetime.now().year]

    log.info(f"Calculando features para temporadas: {seasons}")

    # Carrega dados históricos (football-data.org)
    all_seasons = list(range(min(seasons) - 2, max(seasons) + 1))
    df_all = load_matches(sb, all_seasons)
    log.info(f"  {len(df_all)} partidas (football-data) carregadas")

    # Carrega dados enriquecidos (Base dos Dados + FBref)
    df_xg    = load_xg_profiles(sb)
    df_stats = load_team_season_stats(sb)
    log.info(f"  {len(df_xg)} perfis xG | {len(df_stats)} stats de público/valor")

    # Mapa team_id → nome (para lookup nos perfis por nome)
    resp_teams = sb.table("teams").select("id, name").execute()
    team_name_map = {r["id"]: r["name"] for r in resp_teams.data} if resp_teams.data else {}

    if args.upcoming:
        current_year = datetime.now().year
        df_scheduled = load_scheduled(sb, current_year)
        if not df_scheduled.empty:
            log.info(f"  {len(df_scheduled)} jogos futuros para calcular features")
            rows = build_features_for_matches(
                df_all, df_scheduled, df_xg, df_stats, team_name_map
            )
            save_features(sb, rows)
    else:
        df_target = df_all[df_all["season"].isin(seasons)]
        rows = build_features_for_matches(
            df_all, df_target, df_xg, df_stats, team_name_map
        )
        save_features(sb, rows)

    log.info("Features concluídas")
