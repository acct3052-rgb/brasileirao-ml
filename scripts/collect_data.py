"""
Coleta de dados de futebol — football-data.org
Suporta múltiplas ligas via --league.

Uso:
    python scripts/collect_data.py --league BSA --season 2024
    python scripts/collect_data.py --league PL  --season 2024 --season 2023
    python scripts/collect_data.py --league BSA --incremental

Variáveis de ambiente necessárias:
    FOOTBALL_DATA_API_KEY  — chave gratuita em football-data.org
    SUPABASE_URL / SUPABASE_KEY
"""

import os
import time
import argparse
import logging
from datetime import datetime, timezone, timedelta

import requests
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Ligas suportadas ───────────────────────────────────────────────────────────

LEAGUES: dict[str, dict] = {
    "BSA": {"name": "Brasileirão Série A", "flag": "🇧🇷"},
    "BSB": {"name": "Brasileirão Série B", "flag": "🇧🇷"},
    "PL":  {"name": "Premier League",      "flag": "🏴󠁧󠁢󠁥󠁮󠁧󠁿"},
    "PD":  {"name": "La Liga",             "flag": "🇪🇸"},
    "SA":  {"name": "Serie A",             "flag": "🇮🇹"},
    "FL1": {"name": "Ligue 1",             "flag": "🇫🇷"},
    "BL1": {"name": "Bundesliga",          "flag": "🇩🇪"},
    "CL":  {"name": "Champions League",    "flag": "🏆"},
    "DED": {"name": "Eredivisie",          "flag": "🇳🇱"},
    "PPL": {"name": "Primeira Liga",       "flag": "🇵🇹"},
    "ELC": {"name": "Championship",        "flag": "🏴󠁧󠁢󠁥󠁮󠁧󠁿"},
}

FOOTBALL_DATA_BASE = "https://api.football-data.org/v4"
HEADERS            = {"X-Auth-Token": os.environ["FOOTBALL_DATA_API_KEY"]}
REQUEST_DELAY      = 7  # segundos entre chamadas (plano gratuito: 10 req/min)


# ── Cliente Supabase ───────────────────────────────────────────────────────────

def get_supabase() -> Client:
    return create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])


# ── Funções de coleta ──────────────────────────────────────────────────────────

def fetch_teams(league: str, season: int) -> list[dict]:
    url = f"{FOOTBALL_DATA_BASE}/competitions/{league}/teams?season={season}"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json().get("teams", [])


def fetch_matches(league: str, season: int) -> list[dict]:
    url = f"{FOOTBALL_DATA_BASE}/competitions/{league}/matches?season={season}"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json().get("matches", [])


def parse_result(home_goals: int | None, away_goals: int | None) -> str | None:
    if home_goals is None or away_goals is None:
        return None
    if home_goals > away_goals:
        return "H"
    if home_goals == away_goals:
        return "D"
    return "A"


# ── Funções de persistência ────────────────────────────────────────────────────

def upsert_teams(sb: Client, teams: list[dict]) -> None:
    rows = [
        {
            "id":         t["id"],
            "name":       t["name"],
            "short_name": t.get("shortName"),
            "tla":        t.get("tla"),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        for t in teams
    ]
    sb.table("teams").upsert(rows, on_conflict="id").execute()
    log.info(f"  {len(rows)} times inseridos/atualizados")


def upsert_matches(sb: Client, matches: list[dict], season: int, league: str) -> int:
    rows = []
    for m in matches:
        score  = m.get("score", {})
        full   = score.get("fullTime", {})
        home_g = full.get("home")
        away_g = full.get("away")
        result = parse_result(home_g, away_g) if m["status"] == "FINISHED" else None

        rows.append({
            "id":           m["id"],
            "season":       season,
            "matchday":     m.get("matchday"),
            "match_date":   m.get("utcDate"),
            "status":       m.get("status"),
            "home_team_id": m["homeTeam"]["id"],
            "away_team_id": m["awayTeam"]["id"],
            "home_goals":   home_g,
            "away_goals":   away_g,
            "result":       result,
            "league":       league,
        })

    if not rows:
        return 0

    batch_size = 50
    total = 0
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        sb.table("matches").upsert(batch, on_conflict="id").execute()
        total += len(batch)
        log.info(f"  Lote {i // batch_size + 1}: {len(batch)} partidas inseridas")

    return total


def ensure_teams_from_matches(sb: Client, matches: list[dict]) -> None:
    """Garante que todos os times das partidas existem na tabela teams."""
    inline_teams: dict[int, dict] = {}
    for m in matches:
        for side in ("homeTeam", "awayTeam"):
            t   = m.get(side, {})
            tid = t.get("id")
            if tid and tid not in inline_teams:
                inline_teams[tid] = {
                    "id":         tid,
                    "name":       t.get("name", f"Time {tid}"),
                    "short_name": t.get("shortName"),
                    "tla":        t.get("tla"),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
    if inline_teams:
        rows = list(inline_teams.values())
        for i in range(0, len(rows), 50):
            sb.table("teams").upsert(rows[i:i+50], on_conflict="id").execute()
        log.info(f"  {len(inline_teams)} times garantidos via partidas")


# ── Pipeline principal ─────────────────────────────────────────────────────────

def run_collection(league: str, seasons: list[int]) -> None:
    info = LEAGUES.get(league, {})
    log.info(f"Liga: {info.get('flag','')} {info.get('name', league)} ({league})")
    sb = get_supabase()

    for season in sorted(seasons):
        log.info(f"── Temporada {season} ──────────────────────")

        log.info("Buscando times...")
        try:
            teams = fetch_teams(league, season)
            upsert_teams(sb, teams)
        except Exception as e:
            log.warning(f"  Times de {season} indisponíveis ({e}), pulando")
        time.sleep(REQUEST_DELAY)

        log.info("Buscando partidas...")
        try:
            matches = fetch_matches(league, season)
            ensure_teams_from_matches(sb, matches)
            total = upsert_matches(sb, matches, season, league)
            log.info(f"  Total: {total} partidas processadas")
        except Exception as e:
            log.error(f"  Erro ao buscar partidas de {season}: {e}")
        time.sleep(REQUEST_DELAY)

    log.info("Coleta concluída")


def run_incremental(sb: Client, league: str) -> None:
    current_year = datetime.now().year
    log.info(f"Atualização incremental — {league} temporada {current_year}")

    matches = fetch_matches(league, current_year)

    now          = datetime.now(timezone.utc)
    window_start = now - timedelta(days=7)
    window_end   = now + timedelta(days=14)

    recent = []
    for m in matches:
        date_str = m.get("utcDate")
        if not date_str:
            continue
        try:
            match_date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            if window_start <= match_date <= window_end:
                recent.append(m)
        except ValueError:
            continue

    log.info(f"  {len(recent)} jogos na janela de atualização")
    ensure_teams_from_matches(sb, recent)
    upsert_matches(sb, recent, current_year, league)
    log.info("Atualização incremental concluída")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Coleta de dados — football-data.org")
    parser.add_argument(
        "--league", type=str, default="BSA",
        choices=list(LEAGUES.keys()),
        help="Código da liga (ex: BSA, PL, PD)"
    )
    parser.add_argument(
        "--season", type=int, action="append", dest="seasons",
        help="Temporada (ex: 2024). Pode repetir para múltiplas."
    )
    parser.add_argument(
        "--incremental", action="store_true",
        help="Atualiza apenas jogos recentes/próximos"
    )
    args = parser.parse_args()

    if args.incremental:
        run_incremental(get_supabase(), args.league)
    else:
        seasons = args.seasons or [datetime.now().year]
        run_collection(args.league, seasons)
