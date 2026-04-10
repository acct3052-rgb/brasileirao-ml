"""
Coleta de stats avançadas do Brasileirão via SoccerData (FBref).
Fonte: github.com/probberechts/soccerdata

Dados coletados:
- xG (expected goals) por partida — mandante e visitante
- xGA (expected goals against)
- Chutes a gol, chutes totais
- Posse de bola
- Passes completados / tentados
- Pressão alta (PPDA proxy)
- Distância média de chutes

FBref cobre o Brasileirão a partir de ~2017.
Taxa de scraping: conservadora (10s entre requests) para não ser bloqueado.

Pré-requisitos:
    pip install soccerdata

Variáveis de ambiente:
    SUPABASE_URL / SUPABASE_KEY

Uso:
    python scripts/collect_soccerdata.py --season 2024
    python scripts/collect_soccerdata.py --season 2024 --season 2023
    python scripts/collect_soccerdata.py --all  (2017 em diante)
"""

import os
import time
import logging
import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Diretório de cache do soccerdata (evita re-scraping)
CACHE_DIR = Path("data/soccerdata_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# FBref identifica o Brasileirão como "BRA-Serie A"
FBREF_LEAGUE = "BRA-Serie A"

# Temporadas disponíveis no FBref para o Brasileirão
FBREF_MIN_SEASON = 2017


# ── SoccerData client ──────────────────────────────────────────────────────────

def get_fbref(seasons: list[int]):
    try:
        import soccerdata as sd
        fbref = sd.FBref(
            leagues=FBREF_LEAGUE,
            seasons=seasons,
            data_dir=CACHE_DIR,
        )
        return fbref
    except ImportError:
        raise ImportError(
            "soccerdata não instalado.\n"
            "Execute: pip install soccerdata"
        )


# ── Coleta de dados ────────────────────────────────────────────────────────────

def fetch_match_stats(fbref, season: int) -> pd.DataFrame | None:
    """
    Busca stats de partida do FBref para uma temporada.
    Retorna DataFrame com xG, posse, chutes, etc.
    """
    log.info(f"  FBref: buscando stats de partidas {season}...")
    try:
        df = fbref.read_schedule(force_cache=False)
        if df is None or df.empty:
            log.warning(f"  Sem dados para {season}")
            return None

        # Filtra pela temporada
        df = df.reset_index()
        if "season" in df.columns:
            df = df[df["season"] == season]

        log.info(f"  {len(df)} partidas encontradas")
        return df
    except Exception as e:
        log.error(f"  Erro ao buscar schedule {season}: {e}")
        return None


def fetch_team_stats(fbref, stat_type: str) -> pd.DataFrame | None:
    """
    Busca stats agregadas por time.
    stat_type: 'shooting', 'passing', 'possession', 'defense', 'misc'
    """
    log.info(f"  FBref: buscando {stat_type}...")
    try:
        if stat_type == "shooting":
            return fbref.read_team_season_stats(stat_type="shooting")
        elif stat_type == "passing":
            return fbref.read_team_season_stats(stat_type="passing")
        elif stat_type == "possession":
            return fbref.read_team_season_stats(stat_type="possession")
        elif stat_type == "defense":
            return fbref.read_team_season_stats(stat_type="defense")
        else:
            return None
    except Exception as e:
        log.error(f"  Erro ao buscar {stat_type}: {e}")
        return None


def fetch_match_xg(fbref) -> pd.DataFrame | None:
    """
    Busca xG por partida — a feature mais valiosa do FBref.
    """
    log.info("  FBref: buscando xG por partida...")
    try:
        df = fbref.read_schedule()
        if df is None or df.empty:
            return None

        df = df.reset_index()

        # Colunas de xG variam por versão — busca as disponíveis
        xg_cols = [c for c in df.columns if "xg" in c.lower() or "xGA" in c]
        log.info(f"  Colunas xG encontradas: {xg_cols}")

        return df
    except Exception as e:
        log.error(f"  Erro ao buscar xG: {e}")
        return None


# ── Processamento ──────────────────────────────────────────────────────────────

def extract_match_advanced_stats(df: pd.DataFrame, season: int) -> list[dict]:
    """
    Extrai e normaliza stats avançadas por partida do DataFrame do FBref.
    """
    rows = []

    # Normaliza nomes das colunas (FBref usa MultiIndex às vezes)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(filter(None, c)).strip() for c in df.columns]

    df = df.copy()
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    for _, r in df.iterrows():
        try:
            # Identifica colunas de xG disponíveis
            home_xg = _safe_float(r, ["home_xg", "xg_home", "xg", "home_expected"])
            away_xg = _safe_float(r, ["away_xg", "xg_away", "xg_away_1", "away_expected"])

            home_team = _safe_str(r, ["home_team", "home", "squad_home"])
            away_team = _safe_str(r, ["away_team", "away", "squad_away"])

            if not home_team or not away_team:
                continue

            row = {
                "season":           season,
                "home_team_name":   home_team,
                "away_team_name":   away_team,
                "xg_home":          home_xg,
                "xg_away":          away_xg,
                "xg_diff":          (home_xg - away_xg) if home_xg and away_xg else None,
            }

            # Posse
            row["possession_home"] = _safe_float(r, ["home_possession", "poss_home", "possession"])
            row["possession_away"] = 100 - row["possession_home"] if row["possession_home"] else None

            # Chutes
            row["shots_home"]    = _safe_int(r, ["home_shots", "sh_home", "shots"])
            row["shots_away"]    = _safe_int(r, ["away_shots", "sh_away"])
            row["shots_ot_home"] = _safe_int(r, ["home_shots_on_target", "sot_home"])
            row["shots_ot_away"] = _safe_int(r, ["away_shots_on_target", "sot_away"])

            # Data
            match_date = _safe_str(r, ["date", "match_date", "datetime"])
            row["match_date"] = match_date

            # ID para upsert
            row["match_id_fbref"] = (
                f"{season}_"
                + str(r.get("matchday", r.get("round", ""))).zfill(2) + "_"
                + home_team.replace(" ", "").lower() + "_"
                + away_team.replace(" ", "").lower()
            )

            rows.append(row)

        except Exception as e:
            log.debug(f"  Linha ignorada: {e}")
            continue

    return rows


def _safe_float(row, keys: list) -> float | None:
    for k in keys:
        v = row.get(k)
        if v is not None and str(v) not in ("nan", "None", ""):
            try:
                return float(v)
            except (ValueError, TypeError):
                continue
    return None


def _safe_int(row, keys: list) -> int | None:
    v = _safe_float(row, keys)
    return int(v) if v is not None else None


def _safe_str(row, keys: list) -> str | None:
    for k in keys:
        v = row.get(k)
        if v and str(v) not in ("nan", "None", ""):
            return str(v).strip()
    return None


# ── Stats de xG por time/temporada (agregado) ─────────────────────────────────

def build_team_xg_profile(match_rows: list[dict]) -> list[dict]:
    """
    Calcula perfil de xG médio por time por temporada.
    Usado como feature: time que gera/concede muito xG consistentemente.
    """
    df = pd.DataFrame(match_rows)
    if df.empty:
        return []

    profiles = []

    all_teams = set(df["home_team_name"].tolist() + df["away_team_name"].tolist())

    for team in all_teams:
        home = df[df["home_team_name"] == team]
        away = df[df["away_team_name"] == team]

        season = df["season"].iloc[0] if not df.empty else None

        profiles.append({
            "season":           season,
            "team_name":        team,
            "avg_xg_home":      home["xg_home"].mean() if not home.empty else None,
            "avg_xg_away":      away["xg_away"].mean() if not away.empty else None,
            "avg_xga_home":     home["xg_away"].mean() if not home.empty else None,
            "avg_xga_away":     away["xg_home"].mean() if not away.empty else None,
            "avg_poss_home":    home["possession_home"].mean() if not home.empty else None,
            "avg_shots_home":   home["shots_home"].mean() if not home.empty else None,
            "avg_shots_away":   away["shots_away"].mean() if not away.empty else None,
        })

    return profiles


# ── Persistência ───────────────────────────────────────────────────────────────

def upsert_match_advanced(sb: Client, rows: list[dict]) -> int:
    if not rows:
        return 0

    batch_size = 100
    total = 0
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        sb.table("match_advanced_stats").upsert(
            batch, on_conflict="match_id_fbref"
        ).execute()
        total += len(batch)

    log.info(f"  {total} stats avançadas salvas")
    return total


def upsert_team_xg_profiles(sb: Client, profiles: list[dict]) -> None:
    if not profiles:
        return

    batch_size = 100
    for i in range(0, len(profiles), batch_size):
        sb.table("team_xg_profiles").upsert(
            profiles[i : i + batch_size],
            on_conflict="season,team_name"
        ).execute()

    log.info(f"  {len(profiles)} perfis xG de times salvos")


# ── Pipeline principal ─────────────────────────────────────────────────────────

def run_soccerdata_collection(seasons: list[int]) -> None:
    # Filtra temporadas disponíveis no FBref
    valid_seasons = [s for s in seasons if s >= FBREF_MIN_SEASON]
    if not valid_seasons:
        log.warning(f"FBref só tem dados do Brasileirão a partir de {FBREF_MIN_SEASON}")
        return

    log.info(f"── SoccerData/FBref: {valid_seasons} ──────────────────")
    sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

    for season in valid_seasons:
        log.info(f"\n── Temporada {season} ──")

        try:
            fbref = get_fbref([season])

            # xG por partida
            df_schedule = fetch_match_xg(fbref)
            if df_schedule is not None and not df_schedule.empty:
                match_rows = extract_match_advanced_stats(df_schedule, season)
                log.info(f"  {len(match_rows)} partidas com stats extraídas")

                upsert_match_advanced(sb, match_rows)

                # Perfil xG por time
                profiles = build_team_xg_profile(match_rows)
                upsert_team_xg_profiles(sb, profiles)
            else:
                log.warning(f"  Sem dados de xG para {season}")

            # Rate limiting respeitoso
            time.sleep(15)

        except Exception as e:
            log.error(f"  Erro na temporada {season}: {e}")
            time.sleep(30)
            continue

    log.info("\nSoccerData: coleta concluída")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Coleta stats avançadas via SoccerData/FBref")
    parser.add_argument("--season", type=int, action="append", dest="seasons")
    parser.add_argument(
        "--all", action="store_true",
        help=f"Coleta todas as temporadas disponíveis ({FBREF_MIN_SEASON}–{datetime.now().year})"
    )
    args = parser.parse_args()

    if args.all:
        seasons = list(range(FBREF_MIN_SEASON, datetime.now().year + 1))
    else:
        seasons = args.seasons or [datetime.now().year]

    run_soccerdata_collection(seasons)
