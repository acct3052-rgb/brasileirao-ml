"""
Coleta xG e estatísticas avançadas via API-Football (RapidAPI).
Plano gratuito: 100 requests/dia.

Estratégia:
  - Busca fixtures de uma liga/temporada (1 request)
  - Para cada fixture, busca statistics (1 request) → tem expected_goals
  - Salva em match_advanced_stats e team_xg_profiles
  - Controla progresso para retomar no dia seguinte

Uso:
    python scripts/collect_apifootball.py --league PL --season 2024
    python scripts/collect_apifootball.py --league PL --season 2024 --season 2023
    python scripts/collect_apifootball.py --league BSA --season 2025

Mapeamento de ligas (API-Football league IDs):
    BSA → 71  (Brasileirão Série A)
    PL  → 39  (Premier League)
    PD  → 140 (La Liga)
    SA  → 135 (Serie A)
    FL1 → 61  (Ligue 1)
    BL1 → 78  (Bundesliga)
    DED → 88  (Eredivisie)
    PPL → 94  (Primeira Liga)
    ELC → 40  (Championship)
"""

import os
import time
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path

import requests
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

API_KEY  = os.environ.get("APIFOOTBALL_KEY", "cc606a843ba9238c8c5e85c5ba0863d7")
API_BASE = "https://v3.football.api-sports.io"
HEADERS  = {"x-apisports-key": API_KEY}

# Arquivo de progresso — evita re-coletar fixtures já processados
PROGRESS_FILE = Path("data/apifootball_progress.json")
PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)

# Mapeamento liga → ID na API-Football
LEAGUE_IDS: dict[str, int] = {
    "BSA": 71,
    "PL":  39,
    "PD":  140,
    "SA":  135,
    "FL1": 61,
    "BL1": 78,
    "DED": 88,
    "PPL": 94,
    "ELC": 40,
    "CL":  2,
}


def get_supabase() -> Client:
    return create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])


def load_progress() -> tuple[set[int], list[dict]]:
    """Carrega IDs de fixtures já processados e dados coletados."""
    if PROGRESS_FILE.exists():
        data = json.loads(PROGRESS_FILE.read_text())
        return set(data.get("done", [])), data.get("rows", [])
    return set(), []


def save_progress(done: set[int], rows: list[dict]) -> None:
    PROGRESS_FILE.write_text(json.dumps({"done": list(done), "rows": rows}))


def fetch_fixtures(league_id: int, season: int) -> list[dict]:
    """Busca todos os fixtures de uma liga/temporada (1 request)."""
    resp = requests.get(
        f"{API_BASE}/fixtures",
        headers=HEADERS,
        params={"league": league_id, "season": season, "status": "FT"},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    remaining = data.get("response", [])
    log.info(f"  {len(remaining)} fixtures encontrados (league={league_id}, season={season})")
    return remaining


def fetch_statistics(fixture_id: int) -> list[dict] | None:
    """Busca estatísticas de um fixture (1 request). Retorna lista [home_stats, away_stats]."""
    resp = requests.get(
        f"{API_BASE}/fixtures/statistics",
        headers=HEADERS,
        params={"fixture": fixture_id},
        timeout=30,
    )
    if resp.status_code != 200:
        log.warning(f"  Fixture {fixture_id}: HTTP {resp.status_code}")
        return None
    data = resp.json()
    return data.get("response") or None


def check_remaining_requests() -> int:
    """Verifica quantas requests restam hoje."""
    resp = requests.get(f"{API_BASE}/status", headers=HEADERS, timeout=10)
    data = resp.json()
    response = data.get("response", {})
    if isinstance(response, list):
        response = response[0] if response else {}
    req = response.get("requests", {})
    current = req.get("current", 0)
    limit = req.get("limit_day", 100)
    remaining = limit - current
    log.info(f"  Requests: {current}/{limit} usados, {remaining} restantes")
    return remaining


def parse_stats(stats_list: list[dict], fixture: dict, season: int, league: str) -> dict | None:
    """Extrai xG e stats das duas equipes de um fixture."""
    if not stats_list or len(stats_list) < 2:
        return None

    def get_stat(team_stats: dict, stat_type: str):
        for s in team_stats.get("statistics", []):
            if s["type"] == stat_type:
                v = s["value"]
                if v is None or str(v) in ("", "None"):
                    return None
                try:
                    return float(str(v).replace("%", ""))
                except (ValueError, TypeError):
                    return None
        return None

    home = stats_list[0]
    away = stats_list[1]

    xg_home = get_stat(home, "expected_goals")
    xg_away = get_stat(away, "expected_goals")

    match_info = fixture.get("fixture", {})
    teams = fixture.get("teams", {})
    goals = fixture.get("goals", {})

    def to_int(v):
        return int(v) if v is not None else None

    return {
        "fixture_id":       match_info.get("id"),
        "league":           league,
        "season":           season,
        "match_date":       match_info.get("date"),
        "home_team_name":   teams.get("home", {}).get("name"),
        "away_team_name":   teams.get("away", {}).get("name"),
        "home_goals":       to_int(goals.get("home")),
        "away_goals":       to_int(goals.get("away")),
        # xG
        "xg_home":          xg_home,
        "xg_away":          xg_away,
        "xg_diff":          round(xg_home - xg_away, 3) if xg_home and xg_away else None,
        # Posse
        "possession_home":  get_stat(home, "Ball Possession"),
        "possession_away":  get_stat(away, "Ball Possession"),
        # Chutes (inteiros)
        "shots_home":       to_int(get_stat(home, "Total Shots")),
        "shots_away":       to_int(get_stat(away, "Total Shots")),
        "shots_ot_home":    to_int(get_stat(home, "Shots on Goal")),
        "shots_ot_away":    to_int(get_stat(away, "Shots on Goal")),
    }


def build_xg_profiles(rows: list[dict]) -> list[dict]:
    """Agrega xG médio por time/temporada/liga para o team_xg_profiles."""
    from collections import defaultdict
    import statistics

    # Agrupa por (league, season, team)
    home_data: dict = defaultdict(list)
    away_data: dict = defaultdict(list)

    for r in rows:
        league  = r["league"]
        season  = r["season"]
        ht      = r["home_team_name"]
        at      = r["away_team_name"]
        if ht:
            home_data[(league, season, ht)].append(r)
        if at:
            away_data[(league, season, at)].append(r)

    profiles = {}
    for (league, season, team), games in home_data.items():
        key = (league, season, team)
        xg_h  = [g["xg_home"] for g in games if g.get("xg_home") is not None]
        xga_h = [g["xg_away"] for g in games if g.get("xg_away") is not None]
        poss  = [g["possession_home"] for g in games if g.get("possession_home") is not None]
        shots = [g["shots_home"] for g in games if g.get("shots_home") is not None]
        profiles[key] = {
            "league":         league,
            "season":         season,
            "team_name":      team,
            "avg_xg_home":    round(statistics.mean(xg_h), 3)  if xg_h  else None,
            "avg_xga_home":   round(statistics.mean(xga_h), 3) if xga_h else None,
            "avg_poss_home":  round(statistics.mean(poss), 1)  if poss  else None,
            "avg_shots_home": round(statistics.mean(shots), 1) if shots else None,
        }

    for (league, season, team), games in away_data.items():
        key = (league, season, team)
        xg_a  = [g["xg_away"] for g in games if g.get("xg_away") is not None]
        xga_a = [g["xg_home"] for g in games if g.get("xg_home") is not None]
        shots = [g["shots_away"] for g in games if g.get("shots_away") is not None]
        if key not in profiles:
            profiles[key] = {"league": league, "season": season, "team_name": team}
        profiles[key]["avg_xg_away"]    = round(statistics.mean(xg_a), 3)  if xg_a  else None
        profiles[key]["avg_xga_away"]   = round(statistics.mean(xga_a), 3) if xga_a else None
        profiles[key]["avg_shots_away"] = round(statistics.mean(shots), 1) if shots else None

    return list(profiles.values())


def save_to_supabase(sb: Client, rows: list[dict], profiles: list[dict]) -> None:
    """Salva stats avançadas e perfis xG no Supabase."""
    if rows:
        batch_size = 50
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i+batch_size]
            # Usa fixture_id como chave de upsert
            sb.table("match_advanced_stats").upsert(
                [{**r, "match_id_fbref": f"apif_{r['fixture_id']}"} for r in batch],
                on_conflict="match_id_fbref"
            ).execute()
        log.info(f"  {len(rows)} stats de partidas salvas")

    if profiles:
        for i in range(0, len(profiles), 50):
            sb.table("team_xg_profiles").upsert(
                profiles[i:i+50],
                on_conflict="season,team_name"
            ).execute()
        log.info(f"  {len(profiles)} perfis xG salvos")


def run(league: str, seasons: list[int], max_requests: int = 90) -> None:
    league_id = LEAGUE_IDS.get(league)
    if not league_id:
        log.error(f"Liga {league} não suportada. Disponíveis: {list(LEAGUE_IDS.keys())}")
        return

    sb = get_supabase()
    done, all_rows = load_progress()
    log.info(f"Progresso: {len(done)} fixtures já processados, {len(all_rows)} rows em cache")

    remaining = check_remaining_requests()
    if remaining < 5:
        log.warning("Menos de 5 requests restantes hoje. Tente amanhã.")
        return

    # Limite seguro: deixa 5 de margem
    budget = min(max_requests, remaining - 5)
    used = 0

    for season in seasons:
        log.info(f"\n── Liga {league} (id={league_id}) Temporada {season} ──")

        # 1 request para listar fixtures
        if used >= budget:
            log.info("Budget de requests atingido. Continue amanhã.")
            break
        fixtures = fetch_fixtures(league_id, season)
        used += 1
        time.sleep(1)

        # Filtra fixtures ainda não processados
        pending = [f for f in fixtures if f["fixture"]["id"] not in done]
        log.info(f"  {len(pending)} fixtures pendentes de {len(fixtures)} total")

        for fixture in pending:
            if used >= budget:
                log.info(f"  Budget atingido ({used}/{budget}). Salvando progresso...")
                break

            fid = fixture["fixture"]["id"]
            stats = fetch_statistics(fid)
            used += 1
            time.sleep(2.5)  # respeita rate limit (max ~24 req/min no free)

            if stats:
                row = parse_stats(stats, fixture, season, league)
                if row:
                    all_rows.append(row)
                    done.add(fid)
                    if len(all_rows) % 20 == 0:
                        log.info(f"  {len(all_rows)} partidas processadas...")

            save_progress(done, all_rows)
            time.sleep(0.5)  # pausa extra após salvar progresso

    # Salva tudo no Supabase
    if all_rows:
        profiles = build_xg_profiles(all_rows)
        save_to_supabase(sb, all_rows, profiles)
        log.info(f"\nTotal: {len(all_rows)} partidas, {len(profiles)} perfis de times")
    else:
        log.info("Nenhum dado novo para salvar.")

    log.info(f"Requests usados nesta sessão: {used}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Coleta xG via API-Football")
    parser.add_argument("--league", type=str, default="PL", help="Código da liga (ex: PL, BSA)")
    parser.add_argument("--season", type=int, action="append", dest="seasons")
    parser.add_argument("--max-requests", type=int, default=90, help="Máximo de requests nesta sessão")
    args = parser.parse_args()

    seasons = args.seasons or [datetime.now().year]
    run(args.league, seasons, args.max_requests)
