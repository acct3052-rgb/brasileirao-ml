"""
Coleta lesões, suspensões e escalações via API-Football (RapidAPI).

Busca para cada jogo SCHEDULED/TIMED nas próximas 72h:
- Lesões/suspensões confirmadas (injuries endpoint)
- Escalações confirmadas ~1h antes (lineups endpoint)

Uso:
    python scripts/collect_injuries.py              # próximas 72h
    python scripts/collect_injuries.py --all        # todos os jogos futuros
    python scripts/collect_injuries.py --lineups    # só escalações (usar ~1h antes)

Configurar no .env:
    API_FOOTBALL_KEY=sua_chave_rapidapi
    (Obter em: https://rapidapi.com/api-sports/api/api-football)

Notas:
    - Plano Free: 100 req/dia. Plano Pro: 7500 req/dia.
    - Lineups só ficam disponíveis ~60 min antes do jogo.
    - Injuries disponíveis dias antes.
"""

import os
import sys
import argparse
import logging
from datetime import datetime, timezone, timedelta

import httpx
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

API_FOOTBALL_BASE = "https://api-football-v1.p.rapidapi.com/v3"
BRASILEIRAO_LEAGUE_ID = 71  # Série A
BRASILEIRAO_SEASON = 2026

# Mapeamento de nomes API-Football → nosso banco
# (Completar conforme necessário)
TEAM_ID_MAP = {
    # API-Football ID → nosso team_id no Supabase
    # Preenchido dinamicamente via consulta
}


def get_headers() -> dict:
    api_key = os.environ.get("API_FOOTBALL_KEY")
    if not api_key:
        log.error("API_FOOTBALL_KEY não configurada no .env")
        sys.exit(1)
    return {
        "X-RapidAPI-Key": api_key,
        "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com",
    }


def get_supabase():
    return create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])


def fetch_upcoming_fixtures(hours_ahead: int = 72) -> list:
    """Busca jogos futuros do Brasileirão na API-Football."""
    headers = get_headers()
    now = datetime.now(timezone.utc)
    date_from = now.strftime("%Y-%m-%d")
    date_to = (now + timedelta(hours=hours_ahead)).strftime("%Y-%m-%d")

    url = f"{API_FOOTBALL_BASE}/fixtures"
    params = {
        "league": BRASILEIRAO_LEAGUE_ID,
        "season": BRASILEIRAO_SEASON,
        "from": date_from,
        "to": date_to,
        "status": "NS",  # Not Started
    }
    resp = httpx.get(url, headers=headers, params=params, timeout=15)
    if resp.status_code != 200:
        log.error(f"API-Football fixtures erro: {resp.status_code}")
        return []
    data = resp.json()
    fixtures = data.get("response", [])
    log.info(f"  {len(fixtures)} jogos encontrados ({date_from} → {date_to})")
    return fixtures


def match_our_fixture(api_home: str, api_away: str, sb) -> int | None:
    """Encontra o match_id no nosso banco cruzando pelo nome dos times."""
    resp = (
        sb.table("upcoming_predictions")
        .select("match_id, home_team, away_team")
        .execute()
    )

    def norm(s: str) -> str:
        import unicodedata
        s = unicodedata.normalize("NFD", s.lower())
        s = "".join(c for c in s if unicodedata.category(c) != "Mn")
        return s.replace(" ", "")

    api_h = norm(api_home)
    api_a = norm(api_away)

    for row in resp.data:
        db_h = norm(row["home_team"])
        db_a = norm(row["away_team"])
        if (api_h in db_h or db_h in api_h) and (api_a in db_a or db_a in api_a):
            return row["match_id"]
    return None


def match_our_team(api_team_name: str, sb) -> int | None:
    """Encontra team_id no nosso banco."""
    import unicodedata

    def norm(s: str) -> str:
        s = unicodedata.normalize("NFD", s.lower())
        s = "".join(c for c in s if unicodedata.category(c) != "Mn")
        return s.replace(" ", "")

    resp = sb.table("teams").select("id, name").execute()
    api_n = norm(api_team_name)
    for t in resp.data:
        db_n = norm(t["name"])
        if api_n in db_n or db_n in api_n:
            return t["id"]
    return None


def collect_injuries(fixture_id: int, match_id: int, sb) -> int:
    """Coleta lesões/suspensões para um jogo específico."""
    headers = get_headers()
    url = f"{API_FOOTBALL_BASE}/injuries"
    params = {"fixture": fixture_id}

    resp = httpx.get(url, headers=headers, params=params, timeout=15)
    if resp.status_code != 200:
        log.warning(f"    Injuries API erro {resp.status_code} para fixture {fixture_id}")
        return 0

    data = resp.json()
    injuries = data.get("response", [])
    now_iso = datetime.now(timezone.utc).isoformat()
    count = 0

    for injury in injuries:
        player = injury.get("player", {})
        team = injury.get("team", {})
        team_id = match_our_team(team.get("name", ""), sb)

        row = {
            "match_id": match_id,
            "team_id": team_id,
            "player_name": player.get("name", ""),
            "player_position": player.get("type", ""),  # "Goalkeeper", "Defender", etc.
            "injury_type": player.get("reason", ""),    # "Injury", "Suspension"
            "is_starter": False,  # Atualizado quando lineup confirmar
            "collected_at": now_iso,
        }
        try:
            # Upsert by match_id + team_id + player_name
            existing = (
                sb.table("player_injuries")
                .select("id")
                .eq("match_id", match_id)
                .eq("player_name", row["player_name"])
                .maybe_single()
                .execute()
            )
            if existing.data:
                sb.table("player_injuries").update(row).eq("id", existing.data["id"]).execute()
            else:
                sb.table("player_injuries").insert(row).execute()
            count += 1
        except Exception as e:
            log.warning(f"    Erro ao salvar lesão {player.get('name')}: {e}")

    return count


def collect_lineup(fixture_id: int, match_id: int, sb) -> bool:
    """
    Coleta escalação confirmada (~1h antes do jogo).
    Calcula key_players_out cruzando com player_injuries.
    """
    headers = get_headers()
    url = f"{API_FOOTBALL_BASE}/fixtures/lineups"
    params = {"fixture": fixture_id}

    resp = httpx.get(url, headers=headers, params=params, timeout=15)
    if resp.status_code != 200:
        log.warning(f"    Lineups API erro {resp.status_code} para fixture {fixture_id}")
        return False

    data = resp.json()
    lineups = data.get("response", [])

    if not lineups:
        log.info(f"    Escalação ainda não disponível para fixture {fixture_id}")
        return False

    now_iso = datetime.now(timezone.utc).isoformat()

    for lineup in lineups:
        team_name = lineup.get("team", {}).get("name", "")
        team_id = match_our_team(team_name, sb)
        is_confirmed = len(lineup.get("startXI", [])) == 11

        # Busca lesões deste time neste jogo
        inj_resp = (
            sb.table("player_injuries")
            .select("player_name, player_position")
            .eq("match_id", match_id)
            .eq("team_id", team_id)
            .execute()
        ) if team_id else None

        injured_names = {r["player_name"] for r in (inj_resp.data if inj_resp else [])}

        # Marca titulares lesionados como is_starter=True
        starters = [p.get("player", {}).get("name", "") for p in lineup.get("startXI", [])]
        key_players_out = 0
        for name in injured_names:
            # Conta como key player out se está no banco de lesões mas não na escalação
            if name not in starters:
                key_players_out += 1
                # Atualiza flag is_starter=False (confirmado ausente)
                try:
                    sb.table("player_injuries").update({"is_starter": False}).eq(
                        "match_id", match_id
                    ).eq("player_name", name).execute()
                except Exception:
                    pass

        row = {
            "match_id": match_id,
            "team_id": team_id,
            "is_confirmed": is_confirmed,
            "lineup_json": lineup,
            "key_players_out": key_players_out,
            "collected_at": now_iso,
        }

        try:
            sb.table("match_lineups").upsert(row, on_conflict="match_id,team_id").execute()
            log.info(f"    Escalação salva: {team_name} | confirmada={is_confirmed} | ausências={key_players_out}")
        except Exception as e:
            log.warning(f"    Erro ao salvar lineup {team_name}: {e}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Coleta lesões e escalações via API-Football")
    parser.add_argument("--all", action="store_true", help="Todos os jogos futuros (padrão: 72h)")
    parser.add_argument("--lineups", action="store_true", help="Só escalações (para rodar ~1h antes)")
    parser.add_argument("--injuries-only", action="store_true", help="Só lesões/suspensões")
    parser.add_argument("--hours", type=int, default=72, help="Janela em horas (padrão: 72)")
    args = parser.parse_args()

    sb = get_supabase()
    hours = 24 * 30 if args.all else args.hours

    log.info(f"Buscando jogos nas próximas {hours}h...")
    fixtures = fetch_upcoming_fixtures(hours_ahead=hours)

    if not fixtures:
        log.info("Nenhum jogo encontrado.")
        return

    total_injuries = 0
    total_lineups = 0

    for fix in fixtures:
        fixture_id = fix["fixture"]["id"]
        home = fix["teams"]["home"]["name"]
        away = fix["teams"]["away"]["name"]
        kickoff = fix["fixture"]["date"]

        log.info(f"  {home} vs {away} ({kickoff[:10]})")

        # Encontra match_id no nosso banco
        match_id = match_our_fixture(home, away, sb)
        if not match_id:
            log.warning(f"    Jogo não encontrado no banco: {home} vs {away}")
            continue

        if not args.lineups:
            # Coleta lesões
            n = collect_injuries(fixture_id, match_id, sb)
            total_injuries += n
            if n > 0:
                log.info(f"    {n} lesões/suspensões registradas")
            else:
                log.info(f"    Nenhuma lesão registrada")

        if not args.injuries_only:
            # Coleta escalações (só disponível ~1h antes)
            ok = collect_lineup(fixture_id, match_id, sb)
            if ok:
                total_lineups += 1

    log.info(f"\nConcluído: {total_injuries} lesões registradas, {total_lineups} escalações coletadas")
    log.info("Para usar no modelo, execute: python api/main.py (reload automático)")


if __name__ == "__main__":
    main()
