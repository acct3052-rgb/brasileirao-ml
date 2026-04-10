"""
Alertas Telegram para Value Bets do Brasileirão.

Busca odds da The Odds API, cruza com predições do modelo e envia mensagem
no Telegram quando encontra value bets (EV > threshold).

Uso:
    python scripts/alert_value_bets.py
    python scripts/alert_value_bets.py --ev-min 0.05  (só EV > 5%)
    python scripts/alert_value_bets.py --dry-run       (exibe sem enviar)

Configurar no .env:
    TELEGRAM_BOT_TOKEN=seu_token
    TELEGRAM_CHAT_ID=seu_chat_id
"""

import os
import sys
import argparse
import logging
import httpx
from datetime import datetime, timezone
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TEAM_NAME_MAP = {
    "Remo":              "Clube do Remo",
    "Vasco da Gama":     "CR Vasco da Gama",
    "Vitoria":           "EC Vitória",
    "Sao Paulo":         "São Paulo FC",
    "Mirassol":          "Mirassol FC",
    "Bahia":             "EC Bahia",
    "Fluminense":        "Fluminense FC",
    "Flamengo":          "CR Flamengo",
    "Santos":            "Santos FC",
    "Atletico Mineiro":  "CA Mineiro",
    "Internacional":     "SC Internacional",
    "Gremio":            "Grêmio FBPA",
    "Grêmio":            "Grêmio FBPA",
    "Atletico Paranaense": "CA Paranaense",
    "Chapecoense":       "Chapecoense AF",
    "Botafogo":          "Botafogo FR",
    "Coritiba":          "Coritiba FBC",
    "Cruzeiro":          "Cruzeiro EC",
    "Bragantino-SP":     "RB Bragantino",
    "Corinthians":       "SC Corinthians Paulista",
    "Palmeiras":         "SE Palmeiras",
}

BOOKMAKER_PRIORITY = [
    "pinnacle", "betfair_ex_eu", "matchbook",
    "williamhill", "betsson", "marathonbet",
]

OUTCOME_LABELS = {"H": "Casa 🏠", "D": "Empate 🤝", "A": "Visitante ✈️"}


def normalize(name: str) -> str:
    return TEAM_NAME_MAP.get(name, name)


def best_bookmaker(bookmakers: list) -> dict | None:
    bk_map = {b["key"]: b for b in bookmakers}
    for key in BOOKMAKER_PRIORITY:
        if key in bk_map:
            return bk_map[key]
    return bookmakers[0] if bookmakers else None


def calc_ev(prob: float, odd: float) -> float:
    return prob * odd - 1


def fetch_odds() -> list:
    api_key = os.environ.get("ODDS_API_KEY")
    if not api_key:
        log.error("ODDS_API_KEY não configurada")
        return []

    url = "https://api.the-odds-api.com/v4/sports/soccer_brazil_campeonato/odds"
    params = {
        "apiKey": api_key,
        "regions": "eu",
        "markets": "h2h",
        "oddsFormat": "decimal",
    }
    resp = httpx.get(url, params=params, timeout=15)
    if resp.status_code != 200:
        log.error(f"Odds API erro: {resp.status_code}")
        return []
    return resp.json()


def fetch_predictions() -> dict:
    """Retorna mapa (home_team, away_team) → prediction."""
    sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
    resp = (
        sb.table("upcoming_predictions")
        .select("*")
        .execute()
    )
    preds = {}
    for p in resp.data:
        key = (p["home_team"].lower(), p["away_team"].lower())
        preds[key] = p
    return preds


def send_telegram(token: str, chat_id: str, message: str) -> bool:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "HTML"}
    try:
        resp = httpx.post(url, json=payload, timeout=10)
        return resp.status_code == 200
    except Exception as e:
        log.error(f"Erro ao enviar Telegram: {e}")
        return False


def find_value_bets(games: list, preds: dict, ev_min: float) -> list:
    value_bets = []

    for game in games:
        home_raw = game["home_team"]
        away_raw = game["away_team"]
        home_name = normalize(home_raw)
        away_name = normalize(away_raw)

        pred = preds.get((home_name.lower(), away_name.lower()))
        if not pred:
            continue

        # Melhor odd H2H
        bk = best_bookmaker([b for b in game.get("bookmakers", [])
                              if any(m["key"] == "h2h" for m in b["markets"])])
        if not bk:
            continue

        h2h_market = next((m for m in bk["markets"] if m["key"] == "h2h"), None)
        if not h2h_market:
            continue

        outcomes = {o["name"]: o["price"] for o in h2h_market["outcomes"]}
        odd_home = outcomes.get(home_raw) or outcomes.get(home_name)
        odd_away = outcomes.get(away_raw) or outcomes.get(away_name)
        odd_draw = next((v for k, v in outcomes.items()
                         if k not in (home_raw, away_raw, home_name, away_name)), None)

        odds_map = {"H": odd_home, "D": odd_draw, "A": odd_away}
        probs_map = {
            "H": pred["prob_home"],
            "D": pred["prob_draw"],
            "A": pred["prob_away"],
        }

        for outcome in ["H", "D", "A"]:
            odd = odds_map.get(outcome)
            prob = probs_map.get(outcome)
            if not odd or not prob:
                continue
            ev = calc_ev(prob, odd)
            if ev >= ev_min:
                value_bets.append({
                    "home_team": home_name,
                    "away_team": away_name,
                    "commence_time": game["commence_time"],
                    "outcome": outcome,
                    "prob": prob,
                    "odd": odd,
                    "ev": ev,
                    "fair_odd": round(1 / prob, 2),
                    "bookmaker": bk["key"],
                    "predicted_result": pred.get("predicted_result"),
                })

    # Ordena por EV decrescente
    return sorted(value_bets, key=lambda x: x["ev"], reverse=True)


def format_alert(value_bets: list, ev_min: float) -> str:
    now = datetime.now(timezone.utc).strftime("%d/%m/%Y %H:%M UTC")
    lines = [
        f"🏆 <b>Brasileirão ML — Value Bets</b>",
        f"📅 {now} · EV mínimo: {ev_min*100:.0f}%",
        f"✅ {len(value_bets)} value bet{'s' if len(value_bets) != 1 else ''} encontrada{'s' if len(value_bets) != 1 else ''}",
        "",
    ]
    for vb in value_bets:
        match_time = vb["commence_time"][:10]
        ev_str = f"+{vb['ev']*100:.1f}%"
        model_match = "⭐" if vb["outcome"] == vb["predicted_result"] else ""
        lines += [
            f"<b>{vb['home_team']} vs {vb['away_team']}</b> ({match_time})",
            f"  🎯 {OUTCOME_LABELS[vb['outcome']]} {model_match}",
            f"  Prob: {vb['prob']*100:.1f}% · Odd: {vb['odd']:.2f} (Justa: {vb['fair_odd']:.2f})",
            f"  EV: <b>{ev_str}</b> · Casa: {vb['bookmaker']}",
            "",
        ]
    return "\n".join(lines).strip()


def main():
    parser = argparse.ArgumentParser(description="Alertas Telegram de value bets")
    parser.add_argument("--ev-min", type=float, default=0.03,
                        help="EV mínimo para considerar value bet (padrão: 0.03 = 3%%)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Exibe a mensagem sem enviar para o Telegram")
    args = parser.parse_args()

    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    if not args.dry_run and (not bot_token or not chat_id):
        log.error("Configure TELEGRAM_BOT_TOKEN e TELEGRAM_CHAT_ID no .env ou use --dry-run")
        sys.exit(1)

    log.info("Buscando odds da The Odds API...")
    games = fetch_odds()
    log.info(f"  {len(games)} jogos encontrados")

    log.info("Buscando predições do banco...")
    preds = fetch_predictions()
    log.info(f"  {len(preds)} predições carregadas")

    log.info(f"Procurando value bets (EV >= {args.ev_min*100:.0f}%)...")
    value_bets = find_value_bets(games, preds, args.ev_min)
    log.info(f"  {len(value_bets)} value bets encontradas")

    if not value_bets:
        log.info("Nenhuma value bet encontrada. Nenhum alerta enviado.")
        return

    message = format_alert(value_bets, args.ev_min)
    import sys
    sep = "-" * 50
    out = f"\n{sep}\n{message}\n{sep}\n"
    sys.stdout.buffer.write(out.encode("utf-8", errors="replace"))
    sys.stdout.buffer.flush()

    if args.dry_run:
        log.info("Dry-run: mensagem não enviada.")
        return

    log.info("Enviando alerta Telegram...")
    ok = send_telegram(bot_token, chat_id, message)
    if ok:
        log.info("✅ Alerta enviado com sucesso!")
    else:
        log.error("❌ Falha ao enviar alerta.")
        sys.exit(1)


if __name__ == "__main__":
    main()
