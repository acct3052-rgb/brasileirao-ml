"""
Pipeline automático pós-rodada do Brasileirão ML.

Orquestra as etapas em sequência:
1. Coleta dados (ETL incremental)
2. Atualiza resultados nas predições e apostas
3. Reconstrói features dos próximos jogos
4. Gera novas predições (batch)
5. Coleta lesões/escalações (se API_FOOTBALL_KEY configurada)
6. Envia alertas de value bets (se Telegram configurado)

Uso:
    python scripts/cron_pipeline.py                  # pipeline completo
    python scripts/cron_pipeline.py --skip-etl       # só predições + alertas
    python scripts/cron_pipeline.py --only-alerts    # só alertas
    python scripts/cron_pipeline.py --dry-run        # sem enviar Telegram

Configurar como tarefa agendada (Windows Task Scheduler):
    Trigger: Após cada rodada (ex: domingo 23:00)
    Ação: python C:\\caminho\\scripts\\cron_pipeline.py
"""

import os
import sys
import argparse
import logging
import subprocess
import httpx
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/cron_pipeline.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000")
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "")


def step(name: str):
    log.info(f"\n{'='*50}")
    log.info(f"ETAPA: {name}")
    log.info(f"{'='*50}")


def run(cmd: list[str], check: bool = True) -> bool:
    log.info(f"  Executando: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.stdout:
        for line in result.stdout.strip().split("\n"):
            log.info(f"    {line}")
    if result.stderr:
        for line in result.stderr.strip().split("\n"):
            if line.strip():
                log.info(f"    {line}")
    if check and result.returncode != 0:
        log.error(f"  Falhou com código {result.returncode}")
        return False
    return True


def api_post(path: str) -> dict | None:
    try:
        headers = {"Authorization": f"Bearer {ADMIN_TOKEN}"}
        resp = httpx.post(f"{API_BASE}{path}", headers=headers, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        log.warning(f"  API {path} retornou {resp.status_code}")
        return None
    except Exception as e:
        log.error(f"  Erro ao chamar {path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Pipeline automático do Brasileirão ML")
    parser.add_argument("--skip-etl", action="store_true", help="Pula coleta de dados")
    parser.add_argument("--only-alerts", action="store_true", help="Só alertas Telegram")
    parser.add_argument("--dry-run", action="store_true", help="Alertas sem enviar Telegram")
    parser.add_argument("--ev-min", type=float, default=0.03, help="EV mínimo para alertas")
    args = parser.parse_args()

    # Garante que o diretório de logs existe
    os.makedirs("logs", exist_ok=True)

    start = datetime.now(timezone.utc)
    log.info(f"Pipeline iniciado: {start.strftime('%Y-%m-%d %H:%M UTC')}")

    if args.only_alerts:
        step("Alertas de Value Bets")
        alert_args = ["python", "scripts/alert_value_bets.py", f"--ev-min={args.ev_min}"]
        if args.dry_run:
            alert_args.append("--dry-run")
        run(alert_args, check=False)
        log.info("Pipeline concluído (só alertas).")
        return

    # ── 1. ETL incremental ─────────────────────────────────────────────────────
    if not args.skip_etl:
        step("1/6 — ETL incremental (coleta de dados)")
        ok = run(["python", "scripts/collect_data.py", "--incremental"], check=False)
        if not ok:
            log.warning("  ETL falhou — continuando mesmo assim")

    # ── 2. Atualiza resultados ─────────────────────────────────────────────────
    step("2/6 — Atualiza resultados e apostas")
    result = api_post("/api/update-results")
    if result:
        log.info(f"  Predições atualizadas: {result.get('updated', 0)}")
        log.info(f"  Apostas resolvidas: {result.get('bets_updated', 0)}")
    else:
        log.warning("  update-results falhou — API offline?")

    # ── 3. Reconstrói features ─────────────────────────────────────────────────
    if not args.skip_etl:
        step("3/6 — Reconstrói features dos próximos jogos")
        run(["python", "scripts/build_features.py", "--upcoming"], check=False)

    # ── 4. Gera predições batch ────────────────────────────────────────────────
    step("4/6 — Gera predições (batch)")
    result = api_post("/api/predict/batch")
    if result:
        log.info(f"  Predições geradas: {result.get('predicted', 0)}")

    # ── 5. Lesões/escalações (se API_FOOTBALL_KEY configurada) ─────────────────
    step("5/6 — Coleta lesões e suspensões")
    if os.environ.get("API_FOOTBALL_KEY"):
        run(["python", "scripts/collect_injuries.py", "--hours=72"], check=False)
    else:
        log.info("  API_FOOTBALL_KEY não configurada — pulando coleta de lesões")

    # ── 6. Alertas Telegram ────────────────────────────────────────────────────
    step("6/6 — Alertas de Value Bets")
    has_telegram = os.environ.get("TELEGRAM_BOT_TOKEN") and os.environ.get("TELEGRAM_CHAT_ID")
    if has_telegram or args.dry_run:
        alert_args = ["python", "scripts/alert_value_bets.py", f"--ev-min={args.ev_min}"]
        if args.dry_run:
            alert_args.append("--dry-run")
        run(alert_args, check=False)
    else:
        log.info("  TELEGRAM_BOT_TOKEN/CHAT_ID não configurados — pulando alertas")

    elapsed = (datetime.now(timezone.utc) - start).total_seconds()
    log.info(f"\nPipeline concluído em {elapsed:.0f}s")


if __name__ == "__main__":
    main()
