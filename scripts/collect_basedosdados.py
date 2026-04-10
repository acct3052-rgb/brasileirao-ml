"""
Coleta de dados históricos do Brasileirão via Base dos Dados (BigQuery).
Fonte: basedosdados.org — tabela `basedosdados.mundo_transfermarkt_competicoes.brasileirao_serie_a`

Cobre: partidas desde 2006 com gols, público, árbitro, estádio e mais.
Dados estruturados e limpos, muito mais histórico que a football-data.org.

Pré-requisitos:
    1. Conta Google Cloud (gratuita)
    2. Projeto GCP com BigQuery API habilitada
    3. Arquivo de credenciais JSON (service account) OU Application Default Credentials
    4. pip install google-cloud-bigquery db-dtypes

Variáveis de ambiente:
    GOOGLE_APPLICATION_CREDENTIALS  — caminho para o JSON da service account
    GCP_PROJECT_ID                  — ID do projeto GCP (para billing das queries)
    SUPABASE_URL / SUPABASE_KEY

Uso:
    python scripts/collect_basedosdados.py --seasons 2006 2024
    python scripts/collect_basedosdados.py --seasons 2018 2024  (seletivo)
"""

import os
import logging
import argparse
from datetime import datetime, timezone

import pandas as pd
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── Queries BigQuery ───────────────────────────────────────────────────────────

QUERY_MATCHES = """
SELECT
    CAST(ano_campeonato AS INT64)               AS season,
    CAST(rodada AS INT64)                       AS matchday,
    data                                        AS match_date,
    hora                                        AS match_time,
    time_mandante                               AS home_team_name,
    time_visitante                              AS away_team_name,
    CAST(gols_mandante AS INT64)                AS home_goals,
    CAST(gols_visitante AS INT64)               AS away_goals,
    tecnico_mandante                            AS home_coach,
    tecnico_visitante                           AS away_coach,
    estadio                                     AS stadium,
    arbitro                                     AS referee,
    CAST(publico AS INT64)                      AS attendance,
    CAST(publico_max AS INT64)                  AS capacity,
    SAFE_DIVIDE(CAST(publico AS FLOAT64),
                CAST(publico_max AS FLOAT64))   AS attendance_pct,
    valor_equipe_titular_mandante               AS home_squad_value,
    valor_equipe_titular_visitante              AS away_squad_value
FROM `basedosdados.mundo_transfermarkt_competicoes.brasileirao_serie_a`
WHERE ano_campeonato BETWEEN {year_start} AND {year_end}
  AND gols_mandante IS NOT NULL
  AND gols_visitante IS NOT NULL
ORDER BY ano_campeonato, rodada, data
"""

QUERY_TEAM_STATS = """
SELECT
    ano_campeonato                              AS season,
    time_mandante                               AS team_name,
    COUNT(*)                                    AS games_home,
    SUM(gols_mandante)                          AS goals_scored_home,
    SUM(gols_visitante)                         AS goals_conceded_home,
    AVG(CAST(publico AS FLOAT64))               AS avg_attendance_home,
    AVG(SAFE_DIVIDE(CAST(publico AS FLOAT64),
                    CAST(publico_max AS FLOAT64))) AS avg_occupancy_home
FROM `basedosdados.mundo_transfermarkt_competicoes.brasileirao_serie_a`
WHERE gols_mandante IS NOT NULL
GROUP BY ano_campeonato, time_mandante

UNION ALL

SELECT
    ano_campeonato,
    time_visitante,
    COUNT(*),
    SUM(gols_visitante),
    SUM(gols_mandante),
    NULL,
    NULL
FROM `basedosdados.mundo_transfermarkt_competicoes.brasileirao_serie_a`
WHERE gols_visitante IS NOT NULL
GROUP BY ano_campeonato, time_visitante
"""


# ── BigQuery client ────────────────────────────────────────────────────────────

def get_bq_client():
    try:
        from google.cloud import bigquery
        project_id = os.environ.get("GCP_PROJECT_ID")
        if not project_id:
            raise ValueError("GCP_PROJECT_ID não definido no .env")
        return bigquery.Client(project=project_id)
    except ImportError:
        raise ImportError(
            "google-cloud-bigquery não instalado.\n"
            "Execute: pip install google-cloud-bigquery db-dtypes"
        )


def run_query(client, query: str) -> pd.DataFrame:
    log.info("Executando query BigQuery...")
    job = client.query(query)
    df = job.to_dataframe()
    log.info(f"  {len(df)} linhas retornadas")
    return df


# ── Normalização de nomes de times ────────────────────────────────────────────

# Mapeamento para padronizar nomes entre Base dos Dados e football-data.org
TEAM_NAME_MAP = {
    "Atletico-MG":          "Atlético Mineiro",
    "Atletico-GO":          "Atlético Goianiense",
    "Atletico-PR":          "Athletico Paranaense",
    "Athletico-PR":         "Athletico Paranaense",
    "Sport":                "Sport Recife",
    "América-MG":           "América Mineiro",
    "America-MG":           "América Mineiro",
    "Vasco":                "Vasco da Gama",
    "Bragantino":           "Red Bull Bragantino",
    "Red Bull Bragantino":  "Red Bull Bragantino",
    "RB Bragantino":        "Red Bull Bragantino",
    "Chapecoense-SC":       "Chapecoense",
}

def normalize_team_name(name: str) -> str:
    return TEAM_NAME_MAP.get(name, name)


# ── Processamento ──────────────────────────────────────────────────────────────

def parse_result(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "H"
    if home_goals == away_goals:
        return "D"
    return "A"


def process_matches(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["home_team_name"] = df["home_team_name"].apply(normalize_team_name)
    df["away_team_name"] = df["away_team_name"].apply(normalize_team_name)

    df["result"] = df.apply(
        lambda r: parse_result(r["home_goals"], r["away_goals"]), axis=1
    )

    # Cria ID único baseado em temporada + rodada + times (sem depender de ID externo)
    df["match_id_bdd"] = (
        df["season"].astype(str) + "_"
        + df["matchday"].astype(str).str.zfill(2) + "_"
        + df["home_team_name"].str.replace(" ", "").str.lower() + "_"
        + df["away_team_name"].str.replace(" ", "").str.lower()
    )

    # Converte data
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df["match_date"] = df["match_date"].dt.tz_localize("America/Sao_Paulo", ambiguous="NaT")
    df["match_date"] = df["match_date"].dt.tz_convert("UTC")

    return df


# ── Persistência no Supabase ───────────────────────────────────────────────────

def upsert_historical_matches(sb: Client, df: pd.DataFrame) -> int:
    rows = []
    for _, r in df.iterrows():
        match_date = r["match_date"]
        if pd.isna(match_date):
            continue

        rows.append({
            "match_id_bdd":     r["match_id_bdd"],
            "season":           int(r["season"]),
            "matchday":         int(r["matchday"]) if pd.notna(r["matchday"]) else None,
            "match_date":       match_date.isoformat(),
            "status":           "FINISHED",
            "home_team_name":   r["home_team_name"],
            "away_team_name":   r["away_team_name"],
            "home_goals":       int(r["home_goals"]),
            "away_goals":       int(r["away_goals"]),
            "result":           r["result"],
            "stadium":          r.get("stadium"),
            "referee":          r.get("referee"),
            "attendance":       int(r["attendance"]) if pd.notna(r.get("attendance")) else None,
            "attendance_pct":   float(r["attendance_pct"]) if pd.notna(r.get("attendance_pct")) else None,
            "home_squad_value": float(r["home_squad_value"]) if pd.notna(r.get("home_squad_value")) else None,
            "away_squad_value": float(r["away_squad_value"]) if pd.notna(r.get("away_squad_value")) else None,
            "source":           "basedosdados",
        })

    if not rows:
        log.warning("Nenhum registro válido para inserir")
        return 0

    batch_size = 100
    total = 0
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        sb.table("matches_historical").upsert(batch, on_conflict="match_id_bdd").execute()
        total += len(batch)
        if (i // batch_size + 1) % 5 == 0:
            log.info(f"  Progresso: {total}/{len(rows)} registros")

    return total


# ── Features extras da Base dos Dados ─────────────────────────────────────────

def build_attendance_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gera features de público e valor de elenco por time/temporada.
    Essas features capturam:
    - Pressão do torcedor (público alto = vantagem casa)
    - Qualidade do elenco (valor de mercado = proxy de qualidade)
    """
    features = []

    for (season, team), group in df.groupby(["season", "home_team_name"]):
        home_games = group
        away_games = df[df["away_team_name"] == team]

        features.append({
            "season":               season,
            "team_name":            team,
            "avg_attendance_home":  home_games["attendance"].mean(),
            "avg_attendance_pct":   home_games["attendance_pct"].mean(),
            "avg_home_squad_value": home_games["home_squad_value"].mean(),
            "avg_away_squad_value": away_games["away_squad_value"].mean(),
        })

    return pd.DataFrame(features)


def upsert_team_season_stats(sb: Client, df: pd.DataFrame) -> None:
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "season":               int(r["season"]),
            "team_name":            r["team_name"],
            "avg_attendance_home":  float(r["avg_attendance_home"]) if pd.notna(r.get("avg_attendance_home")) else None,
            "avg_attendance_pct":   float(r["avg_attendance_pct"]) if pd.notna(r.get("avg_attendance_pct")) else None,
            "avg_home_squad_value": float(r["avg_home_squad_value"]) if pd.notna(r.get("avg_home_squad_value")) else None,
            "avg_away_squad_value": float(r["avg_away_squad_value"]) if pd.notna(r.get("avg_away_squad_value")) else None,
        })

    batch_size = 100
    for i in range(0, len(rows), batch_size):
        sb.table("team_season_stats").upsert(
            rows[i : i + batch_size],
            on_conflict="season,team_name"
        ).execute()

    log.info(f"  {len(rows)} stats de times/temporada salvas")


# ── Pipeline principal ─────────────────────────────────────────────────────────

def run_basedosdados_collection(year_start: int, year_end: int) -> None:
    log.info(f"── Base dos Dados: {year_start}–{year_end} ─────────────────")

    bq = get_bq_client()
    sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

    # 1. Busca partidas
    query = QUERY_MATCHES.format(year_start=year_start, year_end=year_end)
    df_raw = run_query(bq, query)

    # 2. Processa
    df = process_matches(df_raw)
    log.info(f"  {len(df)} partidas processadas ({df['season'].min()}–{df['season'].max()})")

    # 3. Persiste partidas históricas
    total = upsert_historical_matches(sb, df)
    log.info(f"  {total} partidas salvas em matches_historical")

    # 4. Gera e persiste features de público/valor
    df_stats = build_attendance_features(df)
    upsert_team_season_stats(sb, df_stats)

    log.info("Base dos Dados: coleta concluída")
    log.info(f"  Distribuição por temporada:\n{df.groupby('season').size().to_string()}")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Coleta histórica via Base dos Dados (BigQuery)")
    parser.add_argument(
        "--seasons", nargs=2, type=int, metavar=("START", "END"),
        default=[2006, datetime.now().year],
        help="Intervalo de temporadas (ex: --seasons 2006 2024)"
    )
    args = parser.parse_args()
    run_basedosdados_collection(args.seasons[0], args.seasons[1])
