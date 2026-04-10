"""
Cron worker para Railway — executa o pipeline pós-rodada chamando
os endpoints da própria API e o ETL diretamente (sem subprocess).

Configurar no Railway como Cron Job:
    Schedule: 0 23 * * 0   (domingo 23h UTC = domingo 20h BRT)
    Command:  python scripts/cron_worker.py
"""

import os
import logging
import httpx
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000")
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "")
FOOTBALL_DATA_API_KEY = os.environ.get("FOOTBALL_DATA_API_KEY", "")


def api_post(path: str) -> dict | None:
    try:
        headers = {"Authorization": f"Bearer {ADMIN_TOKEN}"}
        resp = httpx.post(f"{API_BASE}{path}", headers=headers, timeout=60)
        if resp.status_code == 200:
            return resp.json()
        log.warning(f"  {path} → {resp.status_code}: {resp.text[:200]}")
        return None
    except Exception as e:
        log.error(f"  Erro em {path}: {e}")
        return None


def collect_data():
    """ETL incremental via football-data.org."""
    if not FOOTBALL_DATA_API_KEY:
        log.warning("FOOTBALL_DATA_API_KEY não configurada — pulando ETL")
        return

    import requests
    from supabase import create_client

    sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
    season = datetime.now().year
    headers = {"X-Auth-Token": FOOTBALL_DATA_API_KEY}

    log.info(f"Coletando partidas da temporada {season}...")
    try:
        url = f"https://api.football-data.org/v4/competitions/BSA/matches?season={season}"
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        matches = resp.json().get("matches", [])
        log.info(f"  {len(matches)} partidas encontradas")

        updated = 0
        for m in matches:
            if m.get("status") != "FINISHED":
                continue
            score = m.get("score", {}).get("fullTime", {})
            home_goals = score.get("home")
            away_goals = score.get("away")
            if home_goals is None or away_goals is None:
                continue

            match_id_ext = str(m["id"])
            # Tenta atualizar pelo external_id
            existing = sb.table("matches").select("id").eq("external_id", match_id_ext).execute()
            if not existing.data:
                continue

            if home_goals > away_goals:
                result = "H"
            elif home_goals < away_goals:
                result = "A"
            else:
                result = "D"

            sb.table("matches").update({
                "home_goals": home_goals,
                "away_goals": away_goals,
                "result": result,
                "status": "finished",
            }).eq("external_id", match_id_ext).execute()
            updated += 1

        log.info(f"  {updated} partidas atualizadas no Supabase")
    except Exception as e:
        log.error(f"  Erro no ETL: {e}")


def main():
    start = datetime.now(timezone.utc)
    log.info(f"=== Cron worker iniciado: {start.strftime('%Y-%m-%d %H:%M UTC')} ===")

    # 1. ETL — coleta resultados reais
    log.info("1/4 — ETL: coletando resultados...")
    collect_data()

    # 2. Atualiza predições e apostas
    log.info("2/4 — Atualizando resultados nas predições...")
    result = api_post("/api/update-results")
    if result:
        log.info(f"  predições atualizadas: {result.get('updated', 0)}")
        log.info(f"  apostas resolvidas: {result.get('bets_updated', 0)}")

    # 3. Gera predições para próxima rodada
    log.info("3/4 — Gerando predições batch...")
    result = api_post("/api/predict/batch")
    if result:
        log.info(f"  predições geradas: {result.get('predicted', 0)}")

    # 4. Alertas Telegram (via endpoint)
    log.info("4/4 — Alertas de value bets...")
    result = api_post("/api/alerts/value-bets")
    if result:
        log.info(f"  alertas enviados: {result.get('sent', 0)}")
    else:
        log.info("  endpoint de alertas não disponível (opcional)")

    elapsed = (datetime.now(timezone.utc) - start).total_seconds()
    log.info(f"=== Cron worker concluído em {elapsed:.0f}s ===")


if __name__ == "__main__":
    main()
