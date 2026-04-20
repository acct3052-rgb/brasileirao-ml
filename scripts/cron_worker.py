"""
Cron worker para Railway — executa o pipeline pós-rodada automaticamente.

Roda para todas as ligas ativas, em sequência:
1. Sincroniza resultados reais (ETL + marca predições + resolve apostas)
2. Gera predições para próximos jogos (batch)

Configurar no Railway como Cron Job:
    Schedule: 0 6 * * *   (todo dia 06:00 UTC = 03:00 BRT)
    Command:  python scripts/cron_worker.py

Rodadas terminam em horários variados, então rodar 1x/dia de manhã
garante que qualquer jogo finalizado na noite anterior seja processado.
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

# Ligas que devem ser processadas automaticamente
ACTIVE_LEAGUES = ["BSA", "PL"]


def api_post(path: str, timeout: int = 120) -> dict | None:
    try:
        headers = {}
        if ADMIN_TOKEN:
            headers["Authorization"] = f"Bearer {ADMIN_TOKEN}"
        resp = httpx.post(f"{API_BASE}{path}", headers=headers, timeout=timeout)
        if resp.status_code == 200:
            return resp.json()
        log.warning(f"  {path} -> {resp.status_code}: {resp.text[:300]}")
        return None
    except Exception as e:
        log.error(f"  Erro em {path}: {e}")
        return None


def process_league(league: str):
    log.info(f"--- [{league}] Iniciando processamento ---")

    # 1. Sincroniza resultados reais + atualiza predições + resolve apostas
    log.info(f"  [{league}] Sincronizando resultados...")
    result = api_post(f"/api/sync-results?league={league}")
    if result:
        log.info(f"  [{league}] Jogos sincronizados: {result.get('matches_synced', 0)}")
        log.info(f"  [{league}] Predições atualizadas: {result.get('predictions_updated', 0)}")
        log.info(f"  [{league}] Apostas resolvidas: {result.get('bets_updated', 0)}")
        if result.get("etl_error"):
            log.warning(f"  [{league}] ETL warning: {result['etl_error']}")
    else:
        log.warning(f"  [{league}] sync-results falhou")

    # 2. Gera predições para próximos jogos
    log.info(f"  [{league}] Gerando predições batch...")
    result = api_post(f"/api/predict/batch?league={league}")
    if result:
        log.info(f"  [{league}] Predições geradas: {result.get('predicted', 0)}")
        log.info(f"  [{league}] Já existiam: {result.get('already_predicted', 0)}")
    else:
        log.warning(f"  [{league}] predict/batch falhou")

    log.info(f"--- [{league}] Concluído ---")


def main():
    start = datetime.now(timezone.utc)
    log.info(f"=== Cron worker iniciado: {start.strftime('%Y-%m-%d %H:%M UTC')} ===")
    log.info(f"    API: {API_BASE}")
    log.info(f"    Ligas: {', '.join(ACTIVE_LEAGUES)}")

    for league in ACTIVE_LEAGUES:
        try:
            process_league(league)
        except Exception as e:
            log.error(f"  [{league}] Erro fatal: {e}")

    elapsed = (datetime.now(timezone.utc) - start).total_seconds()
    log.info(f"=== Cron worker concluído em {elapsed:.0f}s ===")


if __name__ == "__main__":
    main()
