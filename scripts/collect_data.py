"""
Coleta de dados do Brasileirão — football-data.org
Busca partidas das temporadas configuradas e insere no Supabase.

Uso:
    python scripts/collect_data.py --season 2024
    python scripts/collect_data.py --season 2024 --season 2023  (múltiplas)

Variáveis de ambiente necessárias (.env ou Railway):
    FOOTBALL_DATA_API_KEY  — chave gratuita em football-data.org
    SUPABASE_URL           — URL do projeto Supabase
    SUPABASE_KEY           — chave service_role do Supabase
"""

import os
import time
import argparse
import logging
from datetime import datetime, timezone

import requests
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Configurações ──────────────────────────────────────────────────────────────

FOOTBALL_DATA_BASE = "https://api.football-data.org/v4"
BRASILEIRAO_CODE   = "BSA"   # Série A — código na API
HEADERS            = {"X-Auth-Token": os.environ["FOOTBALL_DATA_API_KEY"]}

# Taxa de requisições do plano gratuito: 10 req/min
REQUEST_DELAY = 7  # segundos entre chamadas


# ── Cliente Supabase ───────────────────────────────────────────────────────────

def get_supabase() -> Client:
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_KEY"]
    return create_client(url, key)


# ── Funções de coleta ──────────────────────────────────────────────────────────

def fetch_teams(season: int) -> list[dict]:
    """Busca times da competição para a temporada."""
    url = f"{FOOTBALL_DATA_BASE}/competitions/{BRASILEIRAO_CODE}/teams?season={season}"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json().get("teams", [])


def fetch_matches(season: int) -> list[dict]:
    """Busca todas as partidas da temporada."""
    url = f"{FOOTBALL_DATA_BASE}/competitions/{BRASILEIRAO_CODE}/matches?season={season}"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json().get("matches", [])


def parse_result(home_goals: int | None, away_goals: int | None) -> str | None:
    """Converte placar em resultado: H / D / A."""
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


def upsert_matches(sb: Client, matches: list[dict], season: int) -> int:
    rows = []
    for m in matches:
        score   = m.get("score", {})
        full    = score.get("fullTime", {})
        home_g  = full.get("home")
        away_g  = full.get("away")
        result  = parse_result(home_g, away_g) if m["status"] == "FINISHED" else None

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
        })

    if not rows:
        return 0

    # Upsert em lotes de 50 (limite do Supabase free)
    batch_size = 50
    total = 0
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        sb.table("matches").upsert(batch, on_conflict="id").execute()
        total += len(batch)
        log.info(f"  Lote {i // batch_size + 1}: {len(batch)} partidas inseridas")

    return total


# ── Pipeline principal ─────────────────────────────────────────────────────────

def run_collection(seasons: list[int]) -> None:
    sb = get_supabase()
    log.info("Supabase conectado")

    for season in sorted(seasons):
        log.info(f"── Temporada {season} ──────────────────────")

        # Times (tenta sempre — plano gratuito pode bloquear temporadas antigas)
        log.info("Buscando times...")
        try:
            teams = fetch_teams(season)
            upsert_teams(sb, teams)
        except Exception as e:
            log.warning(f"  Times de {season} indisponíveis ({e}), pulando times")
        time.sleep(REQUEST_DELAY)

        # Partidas — extrai times inline das partidas se FK falhar
        log.info("Buscando partidas...")
        try:
            matches = fetch_matches(season)

            # Garante que todos os times das partidas existem na tabela
            inline_teams = {}
            for m in matches:
                for side in ("homeTeam", "awayTeam"):
                    t = m.get(side, {})
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

            total = upsert_matches(sb, matches, season)
            log.info(f"  Total: {total} partidas processadas")
        except Exception as e:
            log.error(f"  Erro ao buscar partidas de {season}: {e}")
        time.sleep(REQUEST_DELAY)

    log.info("Coleta concluída")


# ── Coleta incremental (próximos jogos) ────────────────────────────────────────

def run_incremental(sb: Client) -> None:
    """
    Atualiza apenas os jogos mais recentes (última rodada + próximos).
    Usado no cron diário — mais rápido que reprocessar tudo.
    """
    current_year = datetime.now().year
    log.info(f"Atualização incremental — temporada {current_year}")

    matches = fetch_matches(current_year)

    # Filtra jogos dos últimos 7 dias + próximos 14 dias
    from datetime import timedelta
    now = datetime.now(timezone.utc)
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
    upsert_matches(sb, recent, current_year)
    log.info("Atualização incremental concluída")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Coleta de dados do Brasileirão")
    parser.add_argument(
        "--season", type=int, action="append", dest="seasons",
        help="Temporada para coletar (ex: 2024). Pode repetir para múltiplas."
    )
    parser.add_argument(
        "--incremental", action="store_true",
        help="Modo incremental: atualiza apenas jogos recentes/próximos"
    )
    args = parser.parse_args()

    if args.incremental:
        run_incremental(get_supabase())
    else:
        seasons = args.seasons or [datetime.now().year]
        run_collection(seasons)
