"""
API FastAPI — Brasileirão ML
Serve predições, fixtures e métricas do modelo.

Endpoints:
    GET  /health                    → status da API
    GET  /api/fixtures              → próximos jogos com predições
    POST /api/predict               → prediz um jogo específico
    GET  /api/accuracy              → acurácia histórica do modelo
    POST /api/retrain               → retreina modelo (protegido) com status
    GET  /api/retrain/status        → status do retreinamento
    GET  /api/bets                  → lista apostas do usuário
    POST /api/bets                  → registra nova aposta
    DELETE /api/bets/{id}           → remove aposta
    GET  /api/bets/metrics          → métricas de bankroll
    POST /api/run-etl               → dispara coleta + features (protegido)
"""

import os
import pickle
import logging
import subprocess
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Literal

import httpx
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from scipy.stats import poisson
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODELS_DIR = "models"
FEATURE_COLS = [
    "home_form_pts", "away_form_pts",
    "home_form_gf",  "away_form_gf",
    "home_form_ga",  "away_form_ga",
    "home_home_pts", "away_away_pts",
    "home_home_gf",  "away_away_gf",
    "home_home_ga",  "away_away_ga",
    "h2h_home_wins", "h2h_draws", "h2h_away_wins",
    "h2h_home_gf_avg", "h2h_away_gf_avg",
    "home_table_pos", "away_table_pos",
    "home_table_pts", "away_table_pts",
    "pos_diff", "pts_diff",
    "home_avg_xg", "away_avg_xg",
    "home_avg_xga", "away_avg_xga",
    "home_xg_net", "away_xg_net",
    "home_avg_poss",
    "squad_value_ratio",
    "home_attendance_pct",
    "matchday",
]

HOME_SPLIT_COLS = [
    "home_form_pts", "home_form_gf", "home_form_ga",
    "home_home_pts", "home_home_gf", "home_home_ga",
    "h2h_home_wins", "h2h_draws", "h2h_home_gf_avg",
    "home_table_pos", "home_table_pts",
    "pos_diff", "pts_diff",
    "home_avg_xg", "home_avg_xga", "home_xg_net",
    "home_avg_poss",
    "squad_value_ratio",
    "home_attendance_pct",
    "matchday",
]

AWAY_SPLIT_COLS = [
    "away_form_pts", "away_form_gf", "away_form_ga",
    "away_away_pts", "away_away_gf", "away_away_ga",
    "h2h_away_wins", "h2h_draws", "h2h_away_gf_avg",
    "away_table_pos", "away_table_pts",
    "pos_diff", "pts_diff",
    "away_avg_xg", "away_avg_xga",
    "squad_value_ratio",
    "home_attendance_pct",
    "matchday",
]

# Pesos do blend: [resultado_geral, home_split, away_split]
_BLEND_WEIGHTS = [0.5, 0.25, 0.25]

# ── Ligas suportadas ──────────────────────────────────────────────────────────

LEAGUES_META: dict[str, dict] = {
    "BSA": {"name": "Brasileirão Série A", "flag": "🇧🇷", "active": True},
    "BSB": {"name": "Brasileirão Série B", "flag": "🇧🇷", "active": False},
    "PL":  {"name": "Premier League",      "flag": "🏴󠁧󠁢󠁥󠁮󠁧󠁿", "active": True},
    "PD":  {"name": "La Liga",             "flag": "🇪🇸", "active": False},
    "SA":  {"name": "Serie A",             "flag": "🇮🇹", "active": False},
    "FL1": {"name": "Ligue 1",             "flag": "🇫🇷", "active": False},
    "BL1": {"name": "Bundesliga",          "flag": "🇩🇪", "active": False},
    "CL":  {"name": "Champions League",    "flag": "🏆",  "active": False},
    "DED": {"name": "Eredivisie",          "flag": "🇳🇱", "active": False},
    "PPL": {"name": "Primeira Liga",       "flag": "🇵🇹", "active": False},
    "ELC": {"name": "Championship",        "flag": "🏴󠁧󠁢󠁥󠁮󠁧󠁿", "active": False},
}

# ── Estado global dos modelos (por liga) ──────────────────────────────────────

# models_by_league["BSA"] = {"result_model": ..., "label_encoder": ..., ...}
models_by_league: dict[str, dict] = {}

# ── Estado do retreinamento ───────────────────────────────────────────────────

_retrain_lock = threading.Lock()
_retrain_state: dict = {
    "status": "idle",   # idle | running | done | error
    "started_at": None,
    "finished_at": None,
    "error": None,
}


def load_models_for_league(league: str) -> dict:
    """Carrega os modelos de uma liga específica. Retorna dict com os modelos."""
    league_models: dict = {}
    model_names = ["result_model", "label_encoder", "home_goals_model", "away_goals_model",
                   "home_split_model", "away_split_model"]

    # Tenta pasta models/{league}/ primeiro
    league_dir = os.path.join(MODELS_DIR, league)
    # Fallback: raiz de models/ (legado BSA)
    fallback_dir = MODELS_DIR

    for name in model_names:
        path = os.path.join(league_dir, f"{name}_latest.pkl")
        if not os.path.exists(path):
            path = os.path.join(fallback_dir, f"{name}_latest.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                league_models[name] = pickle.load(f)
            log.info(f"[{league}] Modelo carregado: {name}")
        else:
            league_models[name] = None
            if name not in ("home_split_model", "away_split_model"):
                log.warning(f"[{league}] Modelo não encontrado: {path}")

    return league_models


def load_all_models():
    """Carrega modelos de todas as ligas ativas na inicialização."""
    for league, meta in LEAGUES_META.items():
        if meta["active"]:
            models_by_league[league] = load_models_for_league(league)
            log.info(f"Liga {league} ({meta['name']}) — modelos carregados")


def get_models(league: str = "BSA") -> dict:
    """Retorna os modelos da liga. Carrega sob demanda se não estiver em cache."""
    if league not in models_by_league:
        log.info(f"Carregando modelos para {league} sob demanda...")
        models_by_league[league] = load_models_for_league(league)
    return models_by_league[league]


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_all_models()
    yield


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Brasileirão ML API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer(auto_error=False)


def get_supabase() -> Client:
    return create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])


def verify_admin(credentials: HTTPAuthorizationCredentials = Depends(security)):
    admin_token = os.environ.get("ADMIN_TOKEN", "")
    if not credentials or credentials.credentials != admin_token:
        raise HTTPException(status_code=401, detail="Token inválido")
    return True


# ── Helpers de predição ────────────────────────────────────────────────────────

# Parâmetro rho da correção Dixon-Coles (estimado empiricamente para o Brasileirão).
# Valores típicos: -0.10 a -0.20. Corrige sub-contagem de 0-0 e super-contagem de 1-1.
_DIXON_COLES_RHO = -0.13


def _dc_tau(home_goals: int, away_goals: int, lh: float, la: float, rho: float) -> float:
    """
    Fator de correção Dixon-Coles para placares baixos.
    Só afeta os 4 placares: (0,0), (0,1), (1,0), (1,1).
    """
    if home_goals == 0 and away_goals == 0:
        return 1 - lh * la * rho
    elif home_goals == 0 and away_goals == 1:
        return 1 + lh * rho
    elif home_goals == 1 and away_goals == 0:
        return 1 + la * rho
    elif home_goals == 1 and away_goals == 1:
        return 1 - rho
    return 1.0


def _score_matrix_dc(lh: float, la: float, max_goals: int = 10) -> list[list[float]]:
    """
    Retorna matriz de probabilidade P(h, a) com correção Dixon-Coles.
    matrix[h][a] = probabilidade do placar h–a.
    """
    matrix = []
    for h in range(max_goals + 1):
        row = []
        for a in range(max_goals + 1):
            p = poisson.pmf(h, lh) * poisson.pmf(a, la)
            p *= _dc_tau(h, a, lh, la, _DIXON_COLES_RHO)
            row.append(float(p))
        matrix.append(row)
    return matrix


def over_n_prob(lambda_home: float, lambda_away: float, n: float) -> float:
    """P(total de gols > n) com correção Dixon-Coles para placares baixos."""
    matrix = _score_matrix_dc(lambda_home, lambda_away)
    prob_over = 0.0
    for h, row in enumerate(matrix):
        for a, p in enumerate(row):
            if h + a > n:
                prob_over += p
    return float(prob_over)


def btts_prob(lambda_home: float, lambda_away: float) -> float:
    """P(ambos marcam) com correção Dixon-Coles."""
    matrix = _score_matrix_dc(lambda_home, lambda_away)
    # P(h>=1 e a>=1) = 1 - P(h=0) - P(a=0) + P(h=0,a=0)
    prob_h0 = sum(matrix[0])           # home marca 0
    prob_a0 = sum(row[0] for row in matrix)  # away marca 0
    prob_00 = matrix[0][0]
    return float(1 - prob_h0 - prob_a0 + prob_00)


def over25_prob(lambda_home: float, lambda_away: float) -> float:
    return over_n_prob(lambda_home, lambda_away, 2.5)


def result_probs_dc(lambda_home: float, lambda_away: float) -> tuple[float, float, float]:
    """
    Probabilidades H/D/A calculadas via Dixon-Coles.
    Útil para validar / complementar o XGBoost.
    """
    matrix = _score_matrix_dc(lambda_home, lambda_away)
    p_home = p_draw = p_away = 0.0
    for h, row in enumerate(matrix):
        for a, p in enumerate(row):
            if h > a:
                p_home += p
            elif h == a:
                p_draw += p
            else:
                p_away += p
    total = p_home + p_draw + p_away
    if total > 0:
        p_home, p_draw, p_away = p_home / total, p_draw / total, p_away / total
    return p_home, p_draw, p_away


def predict_from_features(features: dict, league: str = "BSA") -> dict:
    m = get_models(league)
    if not m.get("result_model"):
        raise HTTPException(status_code=503, detail=f"Modelo não carregado para liga {league}")

    X_full = pd.DataFrame([features])[FEATURE_COLS].fillna(0)
    classes = m["label_encoder"].classes_  # ['A', 'D', 'H']

    # Resultado — blend entre modelo geral + splits home/away (se disponíveis)
    proba_general = m["result_model"].predict_proba(X_full)[0]

    home_split = m.get("home_split_model")
    away_split = m.get("away_split_model")

    if home_split is not None and away_split is not None:
        X_home = pd.DataFrame([features])[HOME_SPLIT_COLS].fillna(0)
        X_away = pd.DataFrame([features])[AWAY_SPLIT_COLS].fillna(0)
        proba_home_split = home_split.predict_proba(X_home)[0]
        proba_away_split = away_split.predict_proba(X_away)[0]
        w = _BLEND_WEIGHTS
        proba = (w[0] * proba_general + w[1] * proba_home_split + w[2] * proba_away_split)
    else:
        proba = proba_general

    # Gols esperados (necessário antes do blend DC)
    lambda_home = float(m["home_goals_model"].predict(X_full)[0])
    lambda_away = float(m["away_goals_model"].predict(X_full)[0])
    lambda_home = max(0.1, lambda_home)
    lambda_away = max(0.1, lambda_away)

    # Dixon-Coles: blend 70% XGBoost + 30% DC para suavizar placares baixos
    dc_h, dc_d, dc_a = result_probs_dc(lambda_home, lambda_away)
    # classes = ['A', 'D', 'H'] → índices: A=0, D=1, H=2
    xgb_h = float(proba[np.where(classes == 'H')[0][0]])
    xgb_d = float(proba[np.where(classes == 'D')[0][0]])
    xgb_a = float(proba[np.where(classes == 'A')[0][0]])

    DC_WEIGHT = 0.20  # 20% Dixon-Coles, 80% XGBoost blend
    final_h = (1 - DC_WEIGHT) * xgb_h + DC_WEIGHT * dc_h
    final_d = (1 - DC_WEIGHT) * xgb_d + DC_WEIGHT * dc_d
    final_a = (1 - DC_WEIGHT) * xgb_a + DC_WEIGHT * dc_a

    prob_map = {"H": final_h, "D": final_d, "A": final_a}
    proba_final = np.array([final_a, final_d, final_h])  # ['A','D','H']

    predicted_result = classes[np.argmax(proba_final)]
    confidence = float(np.max(proba_final))

    return {
        "prob_home":              prob_map.get("H", 0.0),
        "prob_draw":              prob_map.get("D", 0.0),
        "prob_away":              prob_map.get("A", 0.0),
        "predicted_result":       predicted_result,
        "confidence":             confidence,
        "expected_goals_home":    round(lambda_home, 2),
        "expected_goals_away":    round(lambda_away, 2),
        "expected_total_goals":   round(lambda_home + lambda_away, 2),
        "over_05_prob":           round(over_n_prob(lambda_home, lambda_away, 0.5), 3),
        "over_15_prob":           round(over_n_prob(lambda_home, lambda_away, 1.5), 3),
        "over_25_prob":           round(over25_prob(lambda_home, lambda_away), 3),
        "over_35_prob":           round(over_n_prob(lambda_home, lambda_away, 3.5), 3),
        "over_45_prob":           round(over_n_prob(lambda_home, lambda_away, 4.5), 3),
        "btts_prob":              round(btts_prob(lambda_home, lambda_away), 3),
    }


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    summary = {}
    for lg, m in models_by_league.items():
        loaded = [k for k, v in m.items() if v is not None]
        summary[lg] = loaded
    return {
        "status": "ok",
        "models_by_league": summary,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/leagues")
def get_leagues():
    """Lista todas as ligas com status de modelos disponíveis."""
    result = []
    for code, meta in LEAGUES_META.items():
        has_model = code in models_by_league and models_by_league[code].get("result_model") is not None
        result.append({
            "code": code,
            "name": meta["name"],
            "flag": meta["flag"],
            "active": meta["active"],
            "has_model": has_model,
        })
    return {"leagues": result}


@app.get("/api/fixtures")
def get_fixtures(limit: int = 50, league: str = "BSA", sb: Client = Depends(get_supabase)):
    """Próximos jogos com predições, filtrado por liga."""
    resp = (
        sb.table("upcoming_predictions")
        .select("*")
        .eq("league", league)
        .limit(limit)
        .execute()
    )
    return {"fixtures": resp.data, "count": len(resp.data)}


@app.get("/api/fixtures/current-round")
def get_current_round_fixtures(league: str = "BSA", sb: Client = Depends(get_supabase)):
    """
    Retorna todos os jogos da rodada atual (passados e futuros).
    Usado no formulário de registro de apostas.
    """
    season = datetime.now(timezone.utc).year

    # Descobre a rodada atual — menor matchday com jogos futuros ou o maior matchday já jogado
    upcoming = (
        sb.table("upcoming_predictions")
        .select("matchday")
        .eq("league", league)
        .order("match_date")
        .limit(1)
        .execute()
    )
    if upcoming.data:
        current_matchday = upcoming.data[0]["matchday"]
    else:
        last = (
            sb.table("matches")
            .select("matchday")
            .eq("season", season)
            .eq("league", league)
            .order("matchday", desc=True)
            .limit(1)
            .execute()
        )
        current_matchday = last.data[0]["matchday"] if last.data else 1

    # Busca predições da rodada atual (com ou sem resultado)
    match_ids_resp = (
        sb.table("matches")
        .select("id")
        .eq("season", season)
        .eq("league", league)
        .eq("matchday", current_matchday)
        .execute()
    )
    match_ids = [m["id"] for m in (match_ids_resp.data or [])]
    if not match_ids:
        return {"fixtures": [], "matchday": current_matchday}

    preds_resp = (
        sb.table("predictions")
        .select("match_id,prob_home,prob_draw,prob_away,predicted_result,confidence,expected_goals_home,expected_goals_away,over_15_prob,over_25_prob,matches!inner(match_date,matchday,season,home_team:home_team_id(name),away_team:away_team_id(name))")
        .in_("match_id", match_ids)
        .execute()
    )

    fixtures = []
    for r in (preds_resp.data or []):
        m = r.get("matches", {})
        fixtures.append({
            "match_id": r["match_id"],
            "match_date": m.get("match_date"),
            "matchday": m.get("matchday"),
            "season": m.get("season"),
            "home_team": (m.get("home_team") or {}).get("name", "?"),
            "away_team": (m.get("away_team") or {}).get("name", "?"),
            "prob_home": r.get("prob_home"),
            "prob_draw": r.get("prob_draw"),
            "prob_away": r.get("prob_away"),
            "predicted_result": r.get("predicted_result"),
            "confidence": r.get("confidence"),
            "expected_goals_home": r.get("expected_goals_home"),
            "expected_goals_away": r.get("expected_goals_away"),
            "over_15_prob": r.get("over_15_prob"),
            "over_25_prob": r.get("over_25_prob"),
        })
    fixtures.sort(key=lambda x: x["match_date"] or "")
    return {"fixtures": fixtures, "matchday": current_matchday, "count": len(fixtures)}


@app.get("/api/accuracy")
def get_accuracy(league: str = "BSA", sb: Client = Depends(get_supabase)):
    """Acurácia histórica do modelo por liga."""
    resp = sb.table("model_accuracy").select("*").eq("league", league).execute()
    return resp.data[0] if resp.data else {"total_predictions": 0, "accuracy_pct": None}


@app.get("/api/accuracy/by-round")
def get_accuracy_by_round(league: str = "BSA", season: int | None = None, sb: Client = Depends(get_supabase)):
    """Acurácia do modelo por rodada."""
    query = sb.table("round_accuracy").select("*").eq("league", league)
    if season:
        query = query.eq("season", season)
    resp = query.order("season", desc=True).order("matchday", desc=True).execute()
    return {"rounds": resp.data, "count": len(resp.data)}


@app.get("/api/goals-lines/{match_id}")
def get_goals_lines(match_id: int, sb: Client = Depends(get_supabase)):
    """
    Retorna todas as linhas de gols (Over 0.5 → 4.5) com probabilidade,
    odd justa e nível de destaque para um jogo específico.
    """
    resp = (
        sb.table("upcoming_predictions")
        .select("expected_goals_home,expected_goals_away,over_05_prob,over_15_prob,over_25_prob,over_35_prob,over_45_prob,btts_prob")
        .eq("match_id", match_id)
        .limit(1)
        .execute()
    )
    if not resp.data:
        raise HTTPException(404, "Predição não encontrada")

    r = resp.data[0]
    lh = r.get("expected_goals_home") or 0
    la = r.get("expected_goals_away") or 0

    # Se colunas ainda não existem na view, recalcula ao vivo
    def get_prob(key: str, n: float) -> float:
        v = r.get(key)
        if v is not None:
            return float(v)
        return round(over_n_prob(lh, la, n), 3)

    lines = []
    configs = [
        ("Over 0.5", get_prob("over_05_prob", 0.5), 0.95, 0.90),
        ("Over 1.5", get_prob("over_15_prob", 1.5), 0.80, 0.72),
        ("Over 2.5", get_prob("over_25_prob", 2.5), 0.60, 0.52),
        ("Over 3.5", get_prob("over_35_prob", 3.5), 0.40, 0.30),
        ("Over 4.5", get_prob("over_45_prob", 4.5), 0.25, 0.18),
        ("BTTS",     get_prob("btts_prob",    -1),   0.65, 0.55),
    ]
    for label, prob, hot_threshold, good_threshold in configs:
        if label == "BTTS" and r.get("btts_prob") is None:
            prob = round(btts_prob(lh, la), 3)
        fair_odd = round(1 / prob, 2) if prob > 0 else None
        highlight = (
            "hot"    if prob >= hot_threshold  else
            "good"   if prob >= good_threshold else
            "normal"
        )
        lines.append({
            "label":      label,
            "prob":       round(prob, 3),
            "prob_pct":   round(prob * 100),
            "fair_odd":   fair_odd,
            "_sort":      prob,
            "highlight":  highlight,
        })

    lines.sort(key=lambda x: x.pop("_sort"), reverse=True)

    return {
        "match_id":             match_id,
        "expected_goals_home":  lh,
        "expected_goals_away":  la,
        "lines":                lines,
    }


@app.get("/api/accuracy/by-round/{season}/{matchday}")
def get_round_detail(season: int, matchday: int, sb: Client = Depends(get_supabase)):
    """Detalhe jogo a jogo de uma rodada — acertos e erros."""
    # 1. Busca match_ids da rodada
    matches_resp = (
        sb.table("matches")
        .select("id, match_date, home_goals, away_goals, home_team:home_team_id(name), away_team:away_team_id(name)")
        .eq("season", season)
        .eq("matchday", matchday)
        .execute()
    )
    match_ids = [m["id"] for m in (matches_resp.data or [])]
    if not match_ids:
        return {"season": season, "matchday": matchday, "games": []}

    match_map = {m["id"]: m for m in matches_resp.data}

    # 2. Busca predições para esses match_ids
    preds_resp = (
        sb.table("predictions")
        .select("match_id, predicted_result, confidence, correct")
        .in_("match_id", match_ids)
        .not_.is_("correct", "null")
        .execute()
    )

    games = []
    for r in (preds_resp.data or []):
        m = match_map.get(r["match_id"], {})
        hg = m.get("home_goals")
        ag = m.get("away_goals")
        actual = None
        if hg is not None and ag is not None:
            actual = "H" if hg > ag else ("A" if ag > hg else "D")
        games.append({
            "home_team": (m.get("home_team") or {}).get("name", "?"),
            "away_team": (m.get("away_team") or {}).get("name", "?"),
            "match_date": m.get("match_date"),
            "predicted_result": r.get("predicted_result"),
            "actual_result": actual,
            "home_goals": hg,
            "away_goals": ag,
            "confidence": r.get("confidence"),
            "correct": r.get("correct"),
        })
    games.sort(key=lambda g: g["match_date"] or "")
    return {"season": season, "matchday": matchday, "games": games}


@app.get("/api/accuracy/calibration")
def get_calibration(sb: Client = Depends(get_supabase)):
    """
    Retorna a acurácia real por faixa de confiança do modelo.
    Usado para calibrar o nível de destaque nos cards.
    """
    resp = sb.table("confidence_calibration").select("*").order("confidence_bucket").execute()
    # Monta lookup: dado confidence 0.0-1.0 → acurácia real histórica
    calibration = []
    for r in resp.data:
        calibration.append({
            "confidence_min": round(float(r["confidence_bucket"]) / 100 - 0.05, 2),
            "confidence_max": round(float(r["confidence_bucket"]) / 100 + 0.05, 2),
            "confidence_bucket_pct": float(r["confidence_bucket"]),
            "total": r["total"],
            "correct": r["correct"],
            "actual_accuracy_pct": float(r["actual_accuracy_pct"]),
        })
    return {"calibration": calibration}


@app.get("/api/predictions/recent")
def get_recent_predictions(limit: int = 20, league: str = "BSA", sb: Client = Depends(get_supabase)):
    """Últimas predições com resultado real (para validação)."""
    resp = (
        sb.table("predictions")
        .select("*, matches!inner(match_date, home_team:home_team_id(name), away_team:away_team_id(name), home_goals, away_goals)")
        .eq("league", league)
        .not_.is_("actual_result", "null")
        .order("predicted_at", desc=True)
        .limit(limit)
        .execute()
    )
    return {"predictions": resp.data}


class PredictRequest(BaseModel):
    match_id: int | None = None
    features: dict | None = None


@app.post("/api/predict")
def predict_match(req: PredictRequest, sb: Client = Depends(get_supabase)):
    if req.match_id:
        resp = sb.table("match_features").select("*").eq("match_id", req.match_id).execute()
        if not resp.data:
            raise HTTPException(404, f"Features não encontradas para match_id={req.match_id}")
        features = resp.data[0]
    elif req.features:
        features = req.features
    else:
        raise HTTPException(400, "Forneça match_id ou features")

    result = predict_from_features(features)

    if req.match_id:
        sb.table("predictions").upsert(
            {**result, "match_id": req.match_id, "model_version": "1.0"},
            on_conflict="match_id"
        ).execute()

    return result


@app.post("/api/predict/batch")
def predict_batch(season: int | None = None, league: str = "BSA", sb: Client = Depends(get_supabase)):
    m = get_models(league)
    if not m.get("result_model"):
        raise HTTPException(503, f"Modelo não carregado para liga {league}")

    query = sb.table("match_features").select("*, matches!inner(status, season)")
    if season:
        query = query.eq("season", season)
    resp = query.execute()

    predicted_ids = {
        r["match_id"] for r in sb.table("predictions").select("match_id").execute().data
    }

    to_predict = [
        r for r in resp.data
        if r["match_id"] not in predicted_ids
        and (
            season is not None  # retroativo: ignora filtro de status
            or r.get("matches", {}).get("status") in ("SCHEDULED", "TIMED")
        )
    ]

    results = []
    for features in to_predict:
        try:
            pred = predict_from_features(features)
            sb.table("predictions").upsert(
                {**pred, "match_id": features["match_id"], "model_version": "1.0"},
                on_conflict="match_id"
            ).execute()
            results.append({"match_id": features["match_id"], "status": "ok"})
        except Exception as e:
            results.append({"match_id": features["match_id"], "status": f"erro: {e}"})

    return {"predicted": len(results), "details": results}


@app.post("/api/update-results")
def update_results(sb: Client = Depends(get_supabase)):
    """Atualiza actual_result nas predições e status nas apostas do usuário."""
    resp = (
        sb.table("predictions")
        .select("match_id, predicted_result, matches!inner(result, status)")
        .is_("actual_result", "null")
        .execute()
    )

    updated = 0
    finished_match_ids = []
    for row in resp.data:
        match_data = row.get("matches", {})
        if match_data.get("status") != "FINISHED":
            continue
        actual = match_data.get("result")
        if not actual:
            continue

        correct = actual == row["predicted_result"]
        sb.table("predictions").update(
            {"actual_result": actual, "correct": correct}
        ).eq("match_id", row["match_id"]).execute()
        finished_match_ids.append((row["match_id"], actual))
        updated += 1

    # Atualiza apostas pendentes cujos jogos terminaram
    bets_updated = 0
    for match_id, actual_result in finished_match_ids:
        bets_resp = (
            sb.table("user_bets")
            .select("id, bet_outcome")
            .eq("match_id", match_id)
            .eq("status", "pending")
            .execute()
        )
        for bet in bets_resp.data:
            new_status = "won" if bet["bet_outcome"] == actual_result else "lost"
            sb.table("user_bets").update({"status": new_status}).eq("id", bet["id"]).execute()
            bets_updated += 1

    return {"updated": updated, "bets_updated": bets_updated}


# ── Retreinamento com status ───────────────────────────────────────────────────

@app.post("/api/retrain")
def retrain(background_tasks: BackgroundTasks, league: str = "BSA", _=Depends(verify_admin)):
    """Retreina o modelo de uma liga: build_features + train_model. Retorna imediatamente."""
    with _retrain_lock:
        if _retrain_state["status"] == "running":
            raise HTTPException(409, "Treinamento já em andamento")
        _retrain_state.update({
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": None,
            "error": None,
        })

    def _run():
        try:
            log.info(f"Retreinamento [{league}]: build_features --all")
            subprocess.run(["python", "scripts/build_features.py", "--all", f"--league={league}"], check=True)
            log.info(f"Retreinamento [{league}]: train_model")
            subprocess.run(["python", "scripts/train_model.py", f"--league={league}"], check=True)
            models_by_league[league] = load_models_for_league(league)
            with _retrain_lock:
                _retrain_state.update({
                    "status": "done",
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                })
            log.info(f"Retreinamento [{league}] concluído")
        except Exception as e:
            with _retrain_lock:
                _retrain_state.update({"status": "error", "error": str(e)})
            log.error(f"Retreinamento falhou: {e}")

    background_tasks.add_task(_run)
    return {"status": "running"}


@app.get("/api/retrain/status")
def retrain_status(_=Depends(verify_admin)):
    """Retorna o estado atual do retreinamento."""
    with _retrain_lock:
        return dict(_retrain_state)


# ── Apostas (Bankroll Tracker) ────────────────────────────────────────────────

class BetCreate(BaseModel):
    match_id: int | None = None          # None quando é múltipla
    bet_outcome: str                      # H, D, A, over_15, over_25, btts, combo
    odd: float
    stake: float
    notes: str | None = None
    market: str = "result"               # result | over_15 | over_25 | btts | combo
    is_combo: bool = False               # múltipla
    combo_description: str | None = None # ex: "Flamengo + Over 1.5 + Santos"


@app.get("/api/bets")
def list_bets(sb: Client = Depends(get_supabase)):
    """Lista todas as apostas do usuário com dados do jogo."""
    resp = (
        sb.table("user_bets")
        .select("*, matches(match_date, status, result, home_team:home_team_id(name), away_team:away_team_id(name))")
        .order("created_at", desc=True)
        .execute()
    )
    return {"bets": resp.data}


@app.post("/api/bets")
def create_bet(req: BetCreate, sb: Client = Depends(get_supabase)):
    """Registra uma nova aposta, capturando snapshot das probabilidades do modelo."""
    model_prob = None
    model_pick = None

    if req.match_id and not req.is_combo:
        pred_resp = (
            sb.table("predictions")
            .select("prob_home, prob_draw, prob_away, predicted_result, over_15_prob, over_25_prob, btts_prob")
            .eq("match_id", req.match_id)
            .maybe_single()
            .execute()
        )
        pred = pred_resp.data or {}
        prob_map = {
            "H": pred.get("prob_home"),
            "D": pred.get("prob_draw"),
            "A": pred.get("prob_away"),
            "over_15": pred.get("over_15_prob"),
            "over_25": pred.get("over_25_prob"),
            "btts": pred.get("btts_prob"),
        }
        model_prob = prob_map.get(req.bet_outcome)
        model_pick = pred.get("predicted_result")

    data: dict = {
        "bet_outcome": req.bet_outcome,
        "odd": req.odd,
        "stake": req.stake,
        "notes": req.notes,
        "model_prob": model_prob,
        "model_pick": model_pick,
        "market": req.market,
        "is_combo": req.is_combo,
        "combo_description": req.combo_description,
    }
    if req.match_id is not None:
        data["match_id"] = req.match_id
    try:
        resp = sb.table("user_bets").insert(data).execute()
        return resp.data[0]
    except Exception as e:
        log.error(f"create_bet error: {e} | data: {data}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/bets/{bet_id}")
def delete_bet(bet_id: str, sb: Client = Depends(get_supabase)):
    """Remove uma aposta."""
    sb.table("user_bets").delete().eq("id", bet_id).execute()
    return {"deleted": bet_id}


@app.get("/api/bets/metrics")
def bets_metrics(sb: Client = Depends(get_supabase)):
    """Métricas de bankroll: ROI, acerto, lucro/prejuízo, acordo com modelo."""
    resp = sb.table("user_bets").select("stake, odd, status, bet_outcome, model_pick").execute()
    bets = resp.data
    total = len(bets)
    if total == 0:
        return {
            "total_bets": 0, "won": 0, "lost": 0, "pending": 0,
            "total_stake": 0, "profit_loss": 0, "roi_pct": None,
            "model_agreement_pct": None,
        }

    won = sum(1 for b in bets if b["status"] == "won")
    lost = sum(1 for b in bets if b["status"] == "lost")
    resolved = won + lost
    total_stake = sum(float(b["stake"]) for b in bets)
    total_pl = sum(
        float(b["stake"]) * float(b["odd"]) - float(b["stake"]) if b["status"] == "won"
        else -float(b["stake"]) if b["status"] == "lost"
        else 0.0
        for b in bets
    )
    stake_resolved = sum(float(b["stake"]) for b in bets if b["status"] in ("won", "lost"))
    roi = (total_pl / stake_resolved * 100) if stake_resolved > 0 else None
    model_agreement = (
        sum(1 for b in bets if b["bet_outcome"] == b["model_pick"]) / total
    )

    return {
        "total_bets": total,
        "won": won,
        "lost": lost,
        "pending": total - won - lost,
        "total_stake": round(total_stake, 2),
        "profit_loss": round(total_pl, 2),
        "roi_pct": round(roi, 2) if roi is not None else None,
        "model_agreement_pct": round(model_agreement * 100, 1),
    }


# ── Odds de mercado (The Odds API) ────────────────────────────────────────────

# Mapeamento de nomes: The Odds API → nomes no nosso banco
_TEAM_NAME_MAP = {
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

# Ordem de preferência das casas (menor margem primeiro)
_BOOKMAKER_PRIORITY = [
    "pinnacle", "betfair_ex_eu", "matchbook",
    "williamhill", "betsson", "marathonbet",
    "nordicbet", "unibet_nl", "unibet_fr", "unibet_se",
]


def _normalize(name: str) -> str:
    return _TEAM_NAME_MAP.get(name, name)


def _best_bookmaker(bookmakers: list) -> dict | None:
    """Retorna o bookmaker de maior prioridade disponível."""
    bk_map = {b["key"]: b for b in bookmakers}
    for key in _BOOKMAKER_PRIORITY:
        if key in bk_map:
            return bk_map[key]
    return bookmakers[0] if bookmakers else None


@app.get("/api/odds")
async def get_odds(sb: Client = Depends(get_supabase)):
    """
    Busca odds de mercado (h2h + totals) da The Odds API para o Brasileirão.
    Complementa com cálculos Poisson do modelo para over 1.5 e BTTS.
    """
    api_key = os.environ.get("ODDS_API_KEY")
    if not api_key:
        raise HTTPException(503, "ODDS_API_KEY não configurada")

    url = "https://api.the-odds-api.com/v4/sports/soccer_brazil_campeonato/odds"
    params = {
        "apiKey": api_key,
        "regions": "eu",
        "markets": "h2h,totals",
        "oddsFormat": "decimal",
    }

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(url, params=params)

    if resp.status_code != 200:
        raise HTTPException(502, f"The Odds API retornou {resp.status_code}")

    games = resp.json()

    # Busca predições com expected goals para os jogos disponíveis
    preds_resp = (
        sb.table("predictions")
        .select("match_id, expected_goals_home, expected_goals_away, matches!inner(home_team:home_team_id(name), away_team:away_team_id(name), status)")
        .in_("matches.status", ["TIMED", "SCHEDULED"])
        .execute()
    )
    # Indexa por (home_team, away_team)
    preds_map: dict[tuple, dict] = {}
    for p in preds_resp.data:
        m = p.get("matches", {})
        ht = m.get("home_team", {}).get("name", "")
        at = m.get("away_team", {}).get("name", "")
        if ht and at:
            preds_map[(ht.lower(), at.lower())] = p

    result = []
    for game in games:
        home_raw = game["home_team"]
        away_raw = game["away_team"]
        home_name = _normalize(home_raw)
        away_name = _normalize(away_raw)

        # ── Agrega odds de TODAS as casas disponíveis ──────────────────────────
        all_h2h_odds: list[dict] = []
        all_totals_odds: list[dict] = []

        for bk in game.get("bookmakers", []):
            for market in bk.get("markets", []):
                if market["key"] == "h2h":
                    outcomes = {o["name"]: o["price"] for o in market["outcomes"]}
                    o_home = outcomes.get(home_raw) or outcomes.get(home_name)
                    o_away = outcomes.get(away_raw) or outcomes.get(away_name)
                    o_draw = next((v for k, v in outcomes.items()
                                   if k not in (home_raw, away_raw, home_name, away_name)), None)
                    if o_home and o_away:
                        all_h2h_odds.append({
                            "bookmaker": bk["key"],
                            "odd_home": o_home,
                            "odd_draw": o_draw,
                            "odd_away": o_away,
                        })
                elif market["key"] == "totals":
                    over25 = under25 = None
                    for o in market["outcomes"]:
                        if o.get("point") == 2.5:
                            if o["name"] == "Over":
                                over25 = o["price"]
                            elif o["name"] == "Under":
                                under25 = o["price"]
                    if over25 or under25:
                        all_totals_odds.append({
                            "bookmaker": bk["key"],
                            "odd_over25": over25,
                            "odd_under25": under25,
                        })

        # ── Melhor odd disponível (máximo entre casas) ─────────────────────────
        best_odd_home  = max((b["odd_home"] for b in all_h2h_odds if b["odd_home"]), default=None)
        best_odd_draw  = max((b["odd_draw"] for b in all_h2h_odds if b["odd_draw"]), default=None)
        best_odd_away  = max((b["odd_away"] for b in all_h2h_odds if b["odd_away"]), default=None)
        best_over25    = max((b["odd_over25"] for b in all_totals_odds if b["odd_over25"]), default=None)
        best_under25   = max((b["odd_under25"] for b in all_totals_odds if b["odd_under25"]), default=None)

        # Casa com melhor odd para H2H (preferência: Pinnacle)
        bk_h2h = _best_bookmaker([b for b in game.get("bookmakers", [])
                                   if any(m["key"] == "h2h" for m in b["markets"])])
        bk_totals = _best_bookmaker([b for b in game.get("bookmakers", [])
                                     if any(m["key"] == "totals" for m in b["markets"])])

        # Casas que oferecem a melhor odd
        def _best_bk_for(field: str, items: list) -> str | None:
            best = max((b[field] for b in items if b.get(field)), default=None)
            if best is None:
                return None
            for prio in _BOOKMAKER_PRIORITY:
                if any(b["bookmaker"] == prio and b.get(field) == best for b in items):
                    return prio
            return next((b["bookmaker"] for b in items if b.get(field) == best), None)

        # Cálculos Poisson do modelo
        pred = preds_map.get((home_name.lower(), away_name.lower()))
        model_over15 = model_over25 = model_btts = None
        if pred and pred.get("expected_goals_home") and pred.get("expected_goals_away"):
            lh = float(pred["expected_goals_home"])
            la = float(pred["expected_goals_away"])
            model_over15 = round(over_n_prob(lh, la, 1.5), 4)
            model_over25 = round(over_n_prob(lh, la, 2.5), 4)
            model_btts   = round(btts_prob(lh, la), 4)

        result.append({
            "home_team":          home_name,
            "away_team":          away_name,
            "commence_time":      game["commence_time"],
            # H2H — melhor odd disponível entre todas as casas
            "h2h_bookmaker":      bk_h2h["key"] if bk_h2h else None,
            "odd_home":           best_odd_home,
            "odd_draw":           best_odd_draw,
            "odd_away":           best_odd_away,
            "best_home_bk":       _best_bk_for("odd_home", all_h2h_odds),
            "best_draw_bk":       _best_bk_for("odd_draw", all_h2h_odds),
            "best_away_bk":       _best_bk_for("odd_away", all_h2h_odds),
            # Totals de mercado (Over/Under 2.5)
            "totals_bookmaker":   bk_totals["key"] if bk_totals else None,
            "odd_over25_market":  best_over25,
            "odd_under25_market": best_under25,
            "best_over25_bk":     _best_bk_for("odd_over25", all_totals_odds),
            "best_under25_bk":    _best_bk_for("odd_under25", all_totals_odds),
            # Todas as casas (para comparação)
            "all_h2h_odds":       all_h2h_odds,
            "all_totals_odds":    all_totals_odds,
            # Modelo Poisson
            "model_over15":       model_over15,
            "model_over25":       model_over25,
            "model_btts":         model_btts,
            # Fair odds do modelo
            "fair_over15":        round(1 / model_over15, 2) if model_over15 else None,
            "fair_over25":        round(1 / model_over25, 2) if model_over25 else None,
            "fair_btts":          round(1 / model_btts,   2) if model_btts   else None,
            "fair_under25":       round(1 / (1 - model_over25), 2) if model_over25 else None,
            "fair_no_btts":       round(1 / (1 - model_btts),   2) if model_btts   else None,
        })

    # ── Salva snapshot no histórico (em background, ignora falhas) ────────────
    try:
        now_iso = datetime.now(timezone.utc).isoformat()
        snapshot_rows = []
        for item in result:
            # Busca match_id pelo nome dos times (join com upcoming_predictions)
            match_row = next(
                (p for p in preds_resp.data
                 if (p.get("matches", {}).get("home_team", {}).get("name", "") == item["home_team"] and
                     p.get("matches", {}).get("away_team", {}).get("name", "") == item["away_team"])),
                None
            )
            match_id = match_row["match_id"] if match_row else None

            # H2H rows por bookmaker
            for bk_entry in item.get("all_h2h_odds", []):
                for outcome, odd_val in [
                    ("H", bk_entry.get("odd_home")),
                    ("D", bk_entry.get("odd_draw")),
                    ("A", bk_entry.get("odd_away")),
                ]:
                    if odd_val:
                        snapshot_rows.append({
                            "match_id": match_id,
                            "bookmaker": bk_entry["bookmaker"],
                            "market": "h2h",
                            "outcome": outcome,
                            "odd": odd_val,
                            "captured_at": now_iso,
                        })

            # Totals rows por bookmaker
            for bk_entry in item.get("all_totals_odds", []):
                if bk_entry.get("odd_over25"):
                    snapshot_rows.append({
                        "match_id": match_id,
                        "bookmaker": bk_entry["bookmaker"],
                        "market": "totals",
                        "outcome": "over25",
                        "odd": bk_entry["odd_over25"],
                        "captured_at": now_iso,
                    })
                if bk_entry.get("odd_under25"):
                    snapshot_rows.append({
                        "match_id": match_id,
                        "bookmaker": bk_entry["bookmaker"],
                        "market": "totals",
                        "outcome": "under25",
                        "odd": bk_entry["odd_under25"],
                        "captured_at": now_iso,
                    })

        if snapshot_rows:
            sb.table("odds_history").insert(snapshot_rows).execute()
            log.info(f"  Snapshot de odds salvo: {len(snapshot_rows)} linhas")
    except Exception as e:
        log.warning(f"  Falha ao salvar snapshot de odds: {e}")

    return {"odds": result, "count": len(result)}


@app.get("/api/odds/history")
def get_odds_history(
    home_team: str | None = None,
    away_team: str | None = None,
    market: str = "h2h",
    outcome: str = "H",
    limit: int = 200,
    sb: Client = Depends(get_supabase),
):
    """Histórico de odds para um jogo/mercado específico."""
    query = (
        sb.table("odds_history")
        .select("*")
        .eq("market", market)
        .eq("outcome", outcome)
        .order("captured_at", desc=True)
        .limit(limit)
    )
    if home_team:
        query = query.eq("home_team", home_team)
    if away_team:
        query = query.eq("away_team", away_team)

    resp = query.execute()
    return {"history": resp.data, "count": len(resp.data)}


# ── Análise por time ──────────────────────────────────────────────────────────

@app.get("/api/teams")
def list_teams(sb: Client = Depends(get_supabase)):
    """Lista todos os times com dados da temporada atual."""
    resp = sb.table("teams").select("id, name").order("name").execute()
    return {"teams": resp.data}


@app.get("/api/teams/{team_name}/profile")
def team_profile(team_name: str, season: int = 2026, sb: Client = Depends(get_supabase)):
    """
    Perfil completo de um time: forma recente, gols esperados,
    rendimento casa/fora e próximos jogos com predição.
    """
    # Busca o time
    team_resp = sb.table("teams").select("id, name").ilike("name", f"%{team_name}%").limit(1).execute()
    if not team_resp.data:
        raise HTTPException(404, f"Time não encontrado: {team_name}")
    team = team_resp.data[0]
    team_id = team["id"]

    # Últimas 10 partidas disputadas (com resultado)
    home_resp = (
        sb.table("matches")
        .select("id, match_date, matchday, result, home_goals, away_goals, status, away_team:away_team_id(name)")
        .eq("home_team_id", team_id)
        .eq("season", season)
        .eq("status", "FINISHED")
        .order("match_date", desc=True)
        .limit(10)
        .execute()
    )
    away_resp = (
        sb.table("matches")
        .select("id, match_date, matchday, result, home_goals, away_goals, status, home_team:home_team_id(name)")
        .eq("away_team_id", team_id)
        .eq("season", season)
        .eq("status", "FINISHED")
        .order("match_date", desc=True)
        .limit(10)
        .execute()
    )

    home_matches = [
        {**m, "venue": "home", "opponent": m.pop("away_team", {}).get("name", ""),
         "team_goals": m["home_goals"], "opp_goals": m["away_goals"],
         "pts": 3 if m["result"] == "H" else 1 if m["result"] == "D" else 0}
        for m in home_resp.data
    ]
    away_matches = [
        {**m, "venue": "away", "opponent": m.pop("home_team", {}).get("name", ""),
         "team_goals": m["away_goals"], "opp_goals": m["home_goals"],
         "pts": 3 if m["result"] == "A" else 1 if m["result"] == "D" else 0}
        for m in away_resp.data
    ]

    all_matches = sorted(home_matches + away_matches, key=lambda x: x["match_date"], reverse=True)[:10]

    # Próximos jogos com predição
    next_home = (
        sb.table("upcoming_predictions")
        .select("*")
        .eq("home_team", team["name"])
        .limit(3)
        .execute()
    )
    next_away = (
        sb.table("upcoming_predictions")
        .select("*")
        .eq("away_team", team["name"])
        .limit(3)
        .execute()
    )
    upcoming = sorted(
        next_home.data + next_away.data,
        key=lambda x: x.get("match_date", "")
    )[:5]

    # Features mais recentes do time (para forma atual)
    feat_home = (
        sb.table("match_features")
        .select("home_form_pts,home_form_gf,home_form_ga,home_home_pts,home_home_gf,home_home_ga,home_table_pos,home_table_pts,home_avg_xg,home_avg_xga,matches!inner(match_date,status)")
        .eq("matches.status", "SCHEDULED")
        .execute()
    )
    # Pega features mais recentes para este time como mandante
    latest_feat = next(
        (f for f in sorted(feat_home.data, key=lambda x: x.get("matches", {}).get("match_date", ""), reverse=True)
         if True), None
    )

    # Resumo estatístico
    def stats(matches: list, venue: str | None = None):
        ms = [m for m in matches if venue is None or m["venue"] == venue]
        if not ms:
            return {"jogos": 0, "pts": 0, "gf": 0, "ga": 0, "wins": 0, "draws": 0, "losses": 0}
        pts_total = sum(m["pts"] for m in ms)
        gf = sum(m.get("team_goals", 0) or 0 for m in ms)
        ga = sum(m.get("opp_goals", 0) or 0 for m in ms)
        wins = sum(1 for m in ms if m["pts"] == 3)
        draws = sum(1 for m in ms if m["pts"] == 1)
        losses = sum(1 for m in ms if m["pts"] == 0)
        return {
            "jogos": len(ms), "pts": pts_total, "pts_pj": round(pts_total / len(ms), 2),
            "gf": gf, "ga": ga, "gf_pj": round(gf / len(ms), 2), "ga_pj": round(ga / len(ms), 2),
            "wins": wins, "draws": draws, "losses": losses,
        }

    return {
        "team": team,
        "season": season,
        "recent_matches": all_matches,
        "upcoming": upcoming,
        "stats_overall": stats(all_matches),
        "stats_home": stats(home_matches),
        "stats_away": stats(away_matches),
        "features": latest_feat,
    }


# ── Lesões, suspensões e escalações (#4 e #8) ────────────────────────────────

# Impacto estimado por posição no λ (gols esperados):
# Ausência de titular reduz capacidade ofensiva/defensiva.
_INJURY_IMPACT = {
    "Goalkeeper":    {"home_ga": +0.08, "away_ga": +0.08},   # goleiro ausente → mais gols sofridos
    "Defender":      {"home_ga": +0.06, "away_ga": +0.06},
    "Midfielder":    {"home_gf": -0.05, "home_ga": +0.03, "away_gf": -0.05, "away_ga": +0.03},
    "Attacker":      {"home_gf": -0.10, "away_gf": -0.10},
    "Forward":       {"home_gf": -0.10, "away_gf": -0.10},
}


def _apply_injury_adjustment(
    lambda_home: float,
    lambda_away: float,
    home_injuries: list[dict],
    away_injuries: list[dict],
) -> tuple[float, float]:
    """
    Ajusta λ_home e λ_away com base nas ausências confirmadas.
    Cada jogador ausente modifica os lambdas de acordo com a posição.
    """
    for inj in home_injuries:
        pos = inj.get("player_position", "")
        impact = _INJURY_IMPACT.get(pos, {})
        lambda_home = max(0.1, lambda_home + impact.get("home_gf", 0))
        lambda_away = max(0.1, lambda_away - impact.get("home_ga", 0) * -1)

    for inj in away_injuries:
        pos = inj.get("player_position", "")
        impact = _INJURY_IMPACT.get(pos, {})
        lambda_away = max(0.1, lambda_away + impact.get("away_gf", 0))
        lambda_home = max(0.1, lambda_home - impact.get("away_ga", 0) * -1)

    return lambda_home, lambda_away


@app.get("/api/injuries/{match_id}")
def get_injuries(match_id: int, sb: Client = Depends(get_supabase)):
    """Retorna lesões/suspensões registradas para um jogo."""
    resp = (
        sb.table("player_injuries")
        .select("*, teams!inner(name)")
        .eq("match_id", match_id)
        .order("player_position")
        .execute()
    )
    return {"injuries": resp.data, "count": len(resp.data)}


@app.get("/api/lineups/{match_id}")
def get_lineups(match_id: int, sb: Client = Depends(get_supabase)):
    """Retorna escalações confirmadas para um jogo."""
    resp = (
        sb.table("match_lineups")
        .select("*, teams!inner(name)")
        .eq("match_id", match_id)
        .execute()
    )
    return {"lineups": resp.data, "confirmed": any(r.get("is_confirmed") for r in resp.data)}


@app.post("/api/predict/with-lineup/{match_id}")
def predict_with_lineup(match_id: int, sb: Client = Depends(get_supabase)):
    """
    Recalcula probabilidades levando em conta lesões/escalações confirmadas.
    Retorna predição original + predição ajustada + delta.
    Ideal para rodar ~1h antes do jogo quando escalações estão confirmadas.
    """
    # Busca features do jogo
    feat_resp = (
        sb.table("match_features")
        .select("*")
        .eq("match_id", match_id)
        .maybe_single()
        .execute()
    )
    if not feat_resp.data:
        raise HTTPException(404, f"Features não encontradas para match_id={match_id}")

    features = feat_resp.data

    # Predição base (sem ajuste)
    base_pred = predict_from_features(features)
    lh_base = base_pred["expected_goals_home"]
    la_base = base_pred["expected_goals_away"]

    # Busca o time mandante e visitante
    match_resp = sb.table("matches").select("home_team_id, away_team_id").eq("id", match_id).execute()
    if not match_resp.data:
        raise HTTPException(404, f"Jogo {match_id} não encontrado")

    home_team_id = match_resp.data[0]["home_team_id"]
    away_team_id = match_resp.data[0]["away_team_id"]

    # Busca lesões por time
    inj_home = (
        sb.table("player_injuries")
        .select("player_name, player_position, injury_type")
        .eq("match_id", match_id)
        .eq("team_id", home_team_id)
        .execute()
    ).data or []

    inj_away = (
        sb.table("player_injuries")
        .select("player_name, player_position, injury_type")
        .eq("match_id", match_id)
        .eq("team_id", away_team_id)
        .execute()
    ).data or []

    # Busca info de escalação (key_players_out)
    lu_home_resp = (
        sb.table("match_lineups")
        .select("is_confirmed, key_players_out")
        .eq("match_id", match_id)
        .eq("team_id", home_team_id)
        .execute()
    )
    lu_away_resp = (
        sb.table("match_lineups")
        .select("is_confirmed, key_players_out")
        .eq("match_id", match_id)
        .eq("team_id", away_team_id)
        .execute()
    )
    lineup_home = lu_home_resp.data[0] if lu_home_resp.data else None
    lineup_away = lu_away_resp.data[0] if lu_away_resp.data else None

    # Aplica ajuste
    lh_adj, la_adj = _apply_injury_adjustment(lh_base, la_base, inj_home, inj_away)

    # Recalcula probabilidades com lambdas ajustados via Dixon-Coles
    dc_h, dc_d, dc_a = result_probs_dc(lh_adj, la_adj)

    # Blend: 70% predição base (XGBoost já calibrado) + 30% DC ajustado
    ADJ_WEIGHT = 0.30
    adj_h = (1 - ADJ_WEIGHT) * base_pred["prob_home"] + ADJ_WEIGHT * dc_h
    adj_d = (1 - ADJ_WEIGHT) * base_pred["prob_draw"] + ADJ_WEIGHT * dc_d
    adj_a = (1 - ADJ_WEIGHT) * base_pred["prob_away"] + ADJ_WEIGHT * dc_a

    # Normaliza
    total = adj_h + adj_d + adj_a
    adj_h, adj_d, adj_a = adj_h / total, adj_d / total, adj_a / total

    lineup_confirmed = (lineup_home and lineup_home.get("is_confirmed")) or \
                       (lineup_away and lineup_away.get("is_confirmed"))

    return {
        "match_id": match_id,
        "lineup_confirmed": lineup_confirmed,
        "injuries": {
            "home": inj_home,
            "away": inj_away,
            "home_key_out": lineup_home.get("key_players_out", 0) if lineup_home else len(inj_home),
            "away_key_out": lineup_away.get("key_players_out", 0) if lineup_away else len(inj_away),
        },
        "base_prediction": {
            "prob_home": base_pred["prob_home"],
            "prob_draw": base_pred["prob_draw"],
            "prob_away": base_pred["prob_away"],
            "expected_goals_home": lh_base,
            "expected_goals_away": la_base,
        },
        "adjusted_prediction": {
            "prob_home": round(adj_h, 4),
            "prob_draw": round(adj_d, 4),
            "prob_away": round(adj_a, 4),
            "expected_goals_home": round(lh_adj, 2),
            "expected_goals_away": round(la_adj, 2),
            "over_25_prob": round(over25_prob(lh_adj, la_adj), 3),
        },
        "delta": {
            "prob_home": round(adj_h - base_pred["prob_home"], 4),
            "prob_draw": round(adj_d - base_pred["prob_draw"], 4),
            "prob_away": round(adj_a - base_pred["prob_away"], 4),
        },
    }


@app.post("/api/collect-injuries")
def trigger_collect_injuries(
    background_tasks: BackgroundTasks,
    hours: int = 72,
    lineups_only: bool = False,
    _=Depends(verify_admin),
):
    """Dispara coleta de lesões/escalações via API-Football em background."""
    def _run():
        args = ["python", "scripts/collect_injuries.py", f"--hours={hours}"]
        if lineups_only:
            args.append("--lineups")
        subprocess.run(args, check=True)
        log.info("Coleta de lesões/escalações concluída")

    background_tasks.add_task(_run)
    return {"status": f"Coleta iniciada (janela: {hours}h, só_lineups={lineups_only})"}


# ── ETL ───────────────────────────────────────────────────────────────────────

@app.post("/api/run-etl")
def run_etl(background_tasks: BackgroundTasks, _=Depends(verify_admin)):
    """Dispara coleta de dados + recálculo de features em background."""
    def _run():
        subprocess.run(["python", "scripts/collect_data.py", "--incremental"], check=True)
        subprocess.run(["python", "scripts/build_features.py", "--upcoming"], check=True)
        log.info("ETL concluído")

    background_tasks.add_task(_run)
    return {"status": "ETL iniciado em background"}


@app.post("/api/sync-results")
def sync_results(sb: Client = Depends(get_supabase)):
    """
    Busca resultados reais na football-data.org e atualiza o Supabase,
    depois marca predições e apostas como corretas/erradas.
    Endpoint público — não precisa de token.
    """
    import requests as req

    football_key = os.environ.get("FOOTBALL_DATA_API_KEY", "")
    etl_updated = 0
    etl_error = None

    # ── 1. ETL: busca placares finalizados na football-data.org ────────────────
    if football_key:
        try:
            season = datetime.now(timezone.utc).year
            url = f"https://api.football-data.org/v4/competitions/BSA/matches?season={season}&status=FINISHED"
            resp = req.get(url, headers={"X-Auth-Token": football_key}, timeout=30)
            resp.raise_for_status()
            finished = resp.json().get("matches", [])
            log.info(f"sync-results: {len(finished)} jogos finalizados na API externa")

            for m in finished:
                ext_id = str(m["id"])
                score = m.get("score", {}).get("fullTime", {})
                hg = score.get("home")
                ag = score.get("away")
                if hg is None or ag is None:
                    continue

                result = "H" if hg > ag else ("A" if ag > hg else "D")

                # Verifica se já está atualizado
                existing = sb.table("matches").select("id, status, result").eq("id", int(ext_id)).execute()
                if not existing.data:
                    continue
                row = existing.data[0]
                if row.get("status") == "FINISHED" and row.get("result") == result:
                    continue  # já atualizado

                sb.table("matches").update({
                    "home_goals": hg,
                    "away_goals": ag,
                    "result": result,
                    "status": "FINISHED",
                }).eq("id", int(ext_id)).execute()
                etl_updated += 1

        except Exception as e:
            etl_error = str(e)
            log.warning(f"sync-results ETL erro: {e}")
    else:
        etl_error = "FOOTBALL_DATA_API_KEY não configurada"
        log.warning("sync-results: FOOTBALL_DATA_API_KEY não configurada")

    # ── 2. Atualiza predições ──────────────────────────────────────────────────
    resp = (
        sb.table("predictions")
        .select("match_id, predicted_result, matches!inner(result, status)")
        .is_("actual_result", "null")
        .execute()
    )

    predictions_updated = 0
    finished_match_ids = []
    for row in resp.data:
        match_data = row.get("matches", {})
        if match_data.get("status") != "FINISHED":
            continue
        actual = match_data.get("result")
        if not actual:
            continue
        correct = actual == row["predicted_result"]
        sb.table("predictions").update(
            {"actual_result": actual, "correct": correct}
        ).eq("match_id", row["match_id"]).execute()
        finished_match_ids.append((row["match_id"], actual))
        predictions_updated += 1

    # ── 3. Atualiza apostas ────────────────────────────────────────────────────
    bets_updated = 0
    for match_id, actual_result in finished_match_ids:
        bets_resp = (
            sb.table("user_bets")
            .select("id, bet_outcome")
            .eq("match_id", match_id)
            .eq("status", "pending")
            .execute()
        )
        for bet in bets_resp.data:
            new_status = "won" if bet["bet_outcome"] == actual_result else "lost"
            sb.table("user_bets").update({"status": new_status}).eq("id", bet["id"]).execute()
            bets_updated += 1

    return {
        "matches_synced": etl_updated,
        "predictions_updated": predictions_updated,
        "bets_updated": bets_updated,
        "etl_error": etl_error,
    }


# ── Chat com Claude ────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str

def _build_chat_context(message: str, sb: Client) -> str:
    """
    Detecta o tema da pergunta e busca só os dados relevantes.
    Mantém o contexto pequeno para economizar tokens.
    """
    msg = message.lower()
    parts: list[str] = []

    # Sempre inclui data atual
    hoje = datetime.now(timezone.utc).strftime("%d/%m/%Y")
    parts.append(f"Data atual: {hoje}")

    # Fixtures + odds de mercado + EV calculado
    if any(w in msg for w in ["hoje", "jogo", "aposta", "melhor", "ouro", "over", "xg", "gol", "confia", "rodada", "próxim", "elite", "ev", "valor"]):
        try:
            rows = (
                sb.table("upcoming_predictions")
                .select("match_id,match_date,matchday,home_team,away_team,prob_home,prob_draw,prob_away,predicted_result,confidence,expected_goals_home,expected_goals_away,over_15_prob,over_25_prob")
                .order("match_date")
                .limit(20)
                .execute()
            ).data or []

            # Busca odds de mercado da The Odds API
            odds_rows = []
            try:
                import requests as req
                odds_key = os.environ.get("ODDS_API_KEY", "")
                if odds_key:
                    r = req.get(
                        "https://api.the-odds-api.com/v4/sports/soccer_brazil_campeonato/odds",
                        params={"apiKey": odds_key, "regions": "eu", "markets": "h2h,totals", "oddsFormat": "decimal"},
                        timeout=10
                    )
                    if r.ok:
                        for game in r.json():
                            best_h = best_d = best_a = best_o25 = None
                            for bk in game.get("bookmakers", []):
                                for mkt in bk.get("markets", []):
                                    if mkt["key"] == "h2h":
                                        for o in mkt["outcomes"]:
                                            if o["name"] == game.get("home_team"): best_h = max(best_h or 0, o["price"])
                                            elif o["name"] == game.get("away_team"): best_a = max(best_a or 0, o["price"])
                                            else: best_d = max(best_d or 0, o["price"])
                                    elif mkt["key"] == "totals":
                                        for o in mkt["outcomes"]:
                                            if o["name"] == "Over": best_o25 = max(best_o25 or 0, o["price"])
                            odds_rows.append({
                                "home_team": game.get("home_team", ""),
                                "away_team": game.get("away_team", ""),
                                "odd_home": best_h, "odd_draw": best_d,
                                "odd_away": best_a, "odd_over25_market": best_o25,
                            })
            except Exception as e:
                log.warning(f"chat context odds fetch: {e}")

            # Monta lookup de odds por times normalizados
            def _norm(s: str) -> str:
                import unicodedata
                s = unicodedata.normalize('NFD', s.lower())
                s = ''.join(c for c in s if unicodedata.category(c) != 'Mn')
                return s.replace(' ', '')

            odds_map: dict = {}
            for o in odds_rows:
                key = (_norm(o.get("home_team", "")), _norm(o.get("away_team", "")))
                odds_map[key] = o

            if rows:
                lines = ["PRÓXIMOS JOGOS com odds e EV:"]
                for r in rows:
                    conf = round((r.get("confidence") or 0) * 100)
                    o15 = round((r.get("over_15_prob") or 0) * 100)
                    o25 = round((r.get("over_25_prob") or 0) * 100)
                    xgh = r.get("expected_goals_home") or 0
                    xga = r.get("expected_goals_away") or 0
                    tier = "Elite" if conf >= 70 else "Alta" if conf >= 60 else "Média" if conf >= 50 else "Baixa"
                    pred = r.get("predicted_result", "")

                    # EV do resultado previsto
                    ev_str = ""
                    key = (_norm(r.get("home_team", "")), _norm(r.get("away_team", "")))
                    mkt = odds_map.get(key)
                    if mkt:
                        prob = r.get({"H": "prob_home", "D": "prob_draw", "A": "prob_away"}.get(pred, "prob_home")) or 0
                        odd = mkt.get({"H": "odd_home", "D": "odd_draw", "A": "odd_away"}.get(pred, "odd_home"))
                        if odd and prob:
                            ev = round((prob * odd - 1) * 100, 1)
                            ev_str = f" | EV:{'+' if ev>0 else ''}{ev}%"
                        odd_o25 = mkt.get("odd_over25_market")
                        if odd_o25:
                            prob_o25 = r.get("over_25_prob") or 0
                            ev_o25 = round((prob_o25 * odd_o25 - 1) * 100, 1)
                            ev_str += f" | O2.5 odd:{odd_o25:.2f} EV:{'+' if ev_o25>0 else ''}{ev_o25}%"

                    lines.append(
                        f"Rd{r['matchday']} {r['match_date'][:10]} | {r['home_team']} vs {r['away_team']} | "
                        f"prev:{pred} conf:{conf}% ({tier}) | "
                        f"xG:{xgh:.1f}-{xga:.1f} | O1.5:{o15}% O2.5:{o25}%{ev_str}"
                    )
                parts.append("\n".join(lines))
        except Exception as e:
            log.warning(f"chat context fixtures: {e}")

    # Acurácia geral
    if any(w in msg for w in ["precisão", "acurácia", "acerto", "errou", "acertou", "histór", "resultado", "modelo"]):
        try:
            acc = (sb.table("predictions").select("correct").not_.is_("correct", "null").execute()).data or []
            if acc:
                total = len(acc)
                certos = sum(1 for r in acc if r["correct"])
                parts.append(f"ACURÁCIA GERAL: {certos}/{total} = {round(certos/total*100, 1)}% de acerto histórico")
        except Exception as e:
            log.warning(f"chat context accuracy: {e}")

    # Acurácia por rodada — detecta número específico ou carrega todas
    if any(w in msg for w in ["rodada", "round", "última", "recente", "acerto", "errou", "acertou"]):
        import re
        round_numbers = [int(n) for n in re.findall(r'\b(\d{1,2})\b', msg)]
        try:
            # Todas as rodadas com resultado
            all_rounds = (
                sb.table("round_accuracy")
                .select("season,matchday,total,correct,wrong,accuracy_pct,avg_confidence")
                .order("season", desc=True)
                .order("matchday", desc=True)
                .execute()
            ).data or []
            if all_rounds:
                lines = ["ACURÁCIA POR RODADA:"]
                for r in all_rounds:
                    acc = round((r.get("accuracy_pct") or 0))
                    conf = round((r.get("avg_confidence") or 0) * 100)
                    lines.append(f"  {r['season']} Rd{r['matchday']}: {r['correct']}/{r['total']} acertos = {acc}% (conf média {conf}%)")
                parts.append("\n".join(lines))

            # Se perguntou sobre rodada específica, busca jogo a jogo
            if round_numbers:
                season_now = datetime.now(timezone.utc).year
                for rd in round_numbers[:2]:  # máx 2 rodadas para não explodir contexto
                    games = (
                        sb.table("predictions")
                        .select("predicted_result,correct,confidence,matches!inner(home_team:home_team_id(name),away_team:away_team_id(name),home_goals,away_goals,matchday,season)")
                        .eq("matches.season", season_now)
                        .eq("matches.matchday", rd)
                        .not_.is_("correct", "null")
                        .execute()
                    ).data or []
                    if games:
                        lines = [f"DETALHES Rd{rd} ({season_now}):"]
                        for g in games:
                            m = g.get("matches", {})
                            ht = m.get("home_team", {}).get("name", "?")
                            at = m.get("away_team", {}).get("name", "?")
                            hg = m.get("home_goals")
                            ag = m.get("away_goals")
                            score = f"{hg}-{ag}" if hg is not None else "?"
                            pred = g.get("predicted_result", "?")
                            ok = "✓" if g.get("correct") else "✗"
                            conf = round((g.get("confidence") or 0) * 100)
                            lines.append(f"  {ok} {ht} vs {at} [{score}] prev:{pred} conf:{conf}%")
                        parts.append("\n".join(lines))
        except Exception as e:
            log.warning(f"chat context round_accuracy: {e}")

    # Calibração
    if any(w in msg for w in ["elite", "alta", "calibr", "confia", "tier", "faixa"]):
        parts.append(
            "CALIBRAÇÃO DO MODELO (dados reais 2023-2026):\n"
            "  Elite (≥70% conf): 88.3% acerto histórico — 60 jogos\n"
            "  Alta  (≥60% conf): 73.5% acerto histórico — 238 jogos\n"
            "  Média (≥50% conf): 52.9% acerto histórico — 548 jogos\n"
            "  Baixa (<50% conf): 47.8% acerto histórico — 638 jogos"
        )

    return "\n\n".join(parts)


SYSTEM_PROMPT = """Você é o assistente do Brasileirão ML, um sistema de predição de jogos do Campeonato Brasileiro.

Seu papel:
- Responder perguntas sobre predições, apostas, acurácia e análise dos jogos
- Ser direto e preciso — máximo 3-4 linhas por resposta
- Usar os dados fornecidos no contexto, nunca inventar números
- Destacar apostas de alto valor (Elite + EV positivo) quando relevante
- Falar em português brasileiro

Regras:
- Não recomendar apostas de forma irresponsável
- Se não tiver dados suficientes, dizer claramente
- Não repetir o contexto inteiro, só o que responde a pergunta"""

IS_PICKS_QUESTION = [
    "6 melhores", "melhores apostas", "apostas elite", "ev positivo",
    "monte", "cards", "picks", "selecione", "quais são as apostas",
    "melhores oportunidades", "apostas da rodada"
]

def _build_picks(sb: Client) -> list[dict]:
    """
    Monta as 6 melhores apostas da rodada atual, ordenadas por probabilidade decrescente.
    Ignora Over 0.5 (sempre alto). Inclui resultado H/D/A e Over 1.5/2.5/BTTS.
    """
    import unicodedata

    def _norm(s: str) -> str:
        s = unicodedata.normalize('NFD', s.lower())
        s = ''.join(c for c in s if unicodedata.category(c) != 'Mn')
        return s.replace(' ', '')

    # Busca todos os fixtures futuros
    all_fixtures = (
        sb.table("upcoming_predictions")
        .select("match_id,match_date,matchday,home_team,away_team,prob_home,prob_draw,prob_away,predicted_result,confidence,expected_goals_home,expected_goals_away,over_15_prob,over_25_prob,btts_prob")
        .order("match_date")
        .limit(50)
        .execute()
    ).data or []

    # Filtra só a rodada mais próxima (menor matchday disponível)
    if not all_fixtures:
        return []
    next_matchday = min(f.get("matchday") or 99 for f in all_fixtures)
    fixtures = [f for f in all_fixtures if f.get("matchday") == next_matchday]

    # Busca odds de mercado da The Odds API
    odds_map: dict = {}
    try:
        import requests as req
        odds_key = os.environ.get("ODDS_API_KEY", "")
        if odds_key:
            r = req.get(
                "https://api.the-odds-api.com/v4/sports/soccer_brazil_campeonato/odds",
                params={"apiKey": odds_key, "regions": "eu", "markets": "h2h,totals", "oddsFormat": "decimal"},
                timeout=10
            )
            if r.ok:
                for game in r.json():
                    best_h = best_d = best_a = best_o25 = None
                    for bk in game.get("bookmakers", []):
                        for mkt in bk.get("markets", []):
                            if mkt["key"] == "h2h":
                                for o in mkt["outcomes"]:
                                    if o["name"] == game.get("home_team"): best_h = max(best_h or 0, o["price"])
                                    elif o["name"] == game.get("away_team"): best_a = max(best_a or 0, o["price"])
                                    else: best_d = max(best_d or 0, o["price"])
                            elif mkt["key"] == "totals":
                                for o in mkt["outcomes"]:
                                    if o["name"] == "Over": best_o25 = max(best_o25 or 0, o["price"])
                    ht = game.get("home_team", "")
                    at = game.get("away_team", "")
                    odds_map[(_norm(ht), _norm(at))] = {
                        "odd_home": best_h, "odd_draw": best_d,
                        "odd_away": best_a, "odd_over25_market": best_o25,
                    }
    except Exception as e:
        log.warning(f"_build_picks odds fetch: {e}")

    candidates: list[dict] = []

    for f in fixtures:
        conf = f.get("confidence") or 0
        tier = "Elite" if conf >= 0.70 else "Alta" if conf >= 0.60 else "Média" if conf >= 0.50 else "Baixa"
        key = (_norm(f.get("home_team", "")), _norm(f.get("away_team", "")))
        mkt = odds_map.get(key)

        pred = f.get("predicted_result", "H")
        prob_map = {"H": f.get("prob_home") or 0, "D": f.get("prob_draw") or 0, "A": f.get("prob_away") or 0}
        odd_map_keys = {"H": "odd_home", "D": "odd_draw", "A": "odd_away"}

        # 1. Resultado previsto
        prob_result = prob_map[pred]
        odd_result = mkt.get(odd_map_keys[pred]) if mkt else None
        ev_result = round((prob_result * odd_result - 1) * 100, 1) if odd_result else None
        fair_result = round(1 / prob_result, 2) if prob_result > 0 else None

        candidates.append({
            "match": f"{f['home_team']} vs {f['away_team']}",
            "matchday": f.get("matchday"),
            "match_date": (f.get("match_date") or "")[:10],
            "market": f"Resultado: {'Casa' if pred=='H' else 'Empate' if pred=='D' else 'Visitante'}",
            "prob": round(prob_result * 100),
            "tier": tier,
            "fair_odd": fair_result,
            "market_odd": odd_result,
            "ev": ev_result,
            "score": conf * 0.7 + prob_result * 0.3 + (0.2 if ev_result and ev_result > 0 else 0),
        })

        # 2. Over 1.5 (sem odd de mercado — mostra odd justa)
        o15 = f.get("over_15_prob") or 0
        if o15 >= 0.65:
            fair_o15 = round(1 / o15, 2) if o15 > 0 else None
            candidates.append({
                "match": f"{f['home_team']} vs {f['away_team']}",
                "matchday": f.get("matchday"),
                "match_date": (f.get("match_date") or "")[:10],
                "market": "Over 1.5 gols",
                "prob": round(o15 * 100),
                "tier": tier,
                "fair_odd": fair_o15,
                "market_odd": None,  # sem odd de mercado disponível
                "ev": None,
                "score": o15 * 0.8 + conf * 0.2,
            })

        # 3. Over 2.5 com odd de mercado
        o25 = f.get("over_25_prob") or 0
        odd_o25 = mkt.get("odd_over25_market") if mkt else None
        if o25 >= 0.45 and odd_o25:
            ev_o25 = round((o25 * odd_o25 - 1) * 100, 1)
            fair_o25 = round(1 / o25, 2) if o25 > 0 else None
            candidates.append({
                "match": f"{f['home_team']} vs {f['away_team']}",
                "matchday": f.get("matchday"),
                "match_date": (f.get("match_date") or "")[:10],
                "market": "Over 2.5 gols",
                "prob": round(o25 * 100),
                "tier": tier,
                "fair_odd": fair_o25,
                "market_odd": odd_o25,
                "ev": ev_o25,
                "score": o25 * 0.5 + (0.3 if ev_o25 > 0 else 0) + conf * 0.2,
            })

        # 4. BTTS
        btts = f.get("btts_prob") or 0
        if btts >= 0.55:
            fair_btts = round(1 / btts, 2) if btts > 0 else None
            candidates.append({
                "match": f"{f['home_team']} vs {f['away_team']}",
                "matchday": f.get("matchday"),
                "match_date": (f.get("match_date") or "")[:10],
                "market": "Ambos marcam",
                "prob": round(btts * 100),
                "tier": tier,
                "fair_odd": fair_btts,
                "market_odd": None,
                "ev": None,
                "score": btts * 0.7 + conf * 0.3,
            })

    # Ordena por probabilidade decrescente (mais provável primeiro)
    candidates.sort(key=lambda x: x["prob"], reverse=True)
    top6 = candidates[:6]
    for c in top6:
        c.pop("score", None)
    return top6


@app.post("/api/chat")
async def chat(req: ChatRequest, sb: Client = Depends(get_supabase)):
    """Chat com Claude Haiku usando contexto dos dados do sistema."""
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(503, "ANTHROPIC_API_KEY não configurada")

    # Detecta se é pergunta de picks
    is_picks = any(w in req.message.lower() for w in IS_PICKS_QUESTION)
    picks: list[dict] = []

    if is_picks:
        picks = _build_picks(sb)
        picks_text = "TOP 6 APOSTAS SELECIONADAS:\n"
        for i, p in enumerate(picks, 1):
            ev_str = f" EV:{'+' if (p['ev'] or 0)>0 else ''}{p['ev']}%" if p['ev'] is not None else " (sem odd de mercado — use odd justa)"
            picks_text += (
                f"{i}. {p['match']} | Rd{p['matchday']} {p['match_date']}\n"
                f"   Mercado: {p['market']} | Prob: {p['prob']}% | Tier: {p['tier']}\n"
                f"   Odd justa: {p['fair_odd']} | Odd mercado: {p['market_odd'] or 'N/D'}{ev_str}\n"
            )
        context = _build_chat_context(req.message, sb) + "\n\n" + picks_text
    else:
        context = _build_chat_context(req.message, sb)

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=400,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Contexto do sistema:\n{context}\n\nPergunta: {req.message}"
            }
        ]
    )

    return {"reply": response.content[0].text, "picks": picks if is_picks else []}


# ── Entrypoint ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
