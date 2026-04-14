"""
Treinamento do modelo de predição do Brasileirão.

Modelos:
1. XGBoost Classifier (ensemble) → prediz resultado (H/D/A) com probabilidades
   - Treinado em perspectiva mista (todos os features)
   - home_split_model: features focadas no mandante
   - away_split_model: features focadas no visitante
   - Probabilidade final = blend ponderado dos 3 modelos
2. Poisson Regression → prediz gols esperados de cada time

Exporta todos para pickle (uso direto na API).
Salva métricas de avaliação no log.

Uso:
    python scripts/train_model.py
    python scripts/train_model.py --season-test 2024  (testa em uma temporada)
"""

import os
import pickle
import argparse
import logging
from datetime import datetime

import numpy as np
import pandas as pd
from numpy import exp
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.linear_model import PoissonRegressor
import xgboost as xgb
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

FEATURE_COLS = [
    # Form geral
    "home_form_pts", "away_form_pts",
    "home_form_gf",  "away_form_gf",
    "home_form_ga",  "away_form_ga",
    # Form por mando
    "home_home_pts", "away_away_pts",
    "home_home_gf",  "away_away_gf",
    "home_home_ga",  "away_away_ga",
    # H2H
    "h2h_home_wins", "h2h_draws", "h2h_away_wins",
    "h2h_home_gf_avg", "h2h_away_gf_avg",
    # Tabela
    "home_table_pos", "away_table_pos",
    "home_table_pts", "away_table_pts",
    "pos_diff", "pts_diff",
    # xG médio por time (FBref — preenchido com 0 se não disponível)
    "home_avg_xg", "away_avg_xg",
    "home_avg_xga", "away_avg_xga",
    "home_xg_net", "away_xg_net",
    "home_avg_poss",
    # Valor e público (Base dos Dados)
    "squad_value_ratio",
    "home_attendance_pct",
    # Contexto
    "matchday",
]

# Features específicas do mandante (perspectiva casa)
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

# Features específicas do visitante (perspectiva fora)
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


# ── Carrega dados ──────────────────────────────────────────────────────────────

def load_training_data(league: str = "BSA") -> pd.DataFrame:
    sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

    # Carrega features filtradas pela liga + join com resultado real
    all_data = []
    page_size = 1000
    offset = 0
    while True:
        resp = (
            sb.table("match_features")
            .select("*, matches!inner(result, home_goals, away_goals, season, match_date)")
            .eq("league", league)
            .range(offset, offset + page_size - 1)
            .execute()
        )
        if not resp.data:
            break
        all_data.extend(resp.data)
        if len(resp.data) < page_size:
            break
        offset += page_size

    df = pd.DataFrame(all_data)
    if df.empty:
        raise ValueError(f"Nenhuma feature encontrada para liga {league}. "
                         f"Execute build_features.py --league {league} primeiro.")

    # Flatten JSON aninhado — remove season duplicada antes do join
    matches_df = pd.json_normalize(df["matches"]).drop(columns=["season"], errors="ignore")
    df = df.drop(columns=["matches"]).join(matches_df)

    # Remove jogos sem resultado
    df = df[df["result"].notna()].copy()

    log.info(f"  {len(df)} partidas ({league}) com features e resultado carregadas")
    return df


# ── Treinamento do classificador de resultado ──────────────────────────────────

def compute_sample_weights(df: pd.DataFrame, decay_lambda: float = 0.003) -> np.ndarray:
    """
    Calcula pesos exponenciais por data do jogo.
    Jogos mais recentes recebem peso maior: w = exp(-lambda * dias_atras)
    Lambda padrão 0.003 → meia-vida ~230 dias (~uma temporada).
    """
    dates = pd.to_datetime(df["match_date"], utc=True)
    max_date = dates.max()
    days_ago = (max_date - dates).dt.days.values
    weights = exp(-decay_lambda * days_ago)
    # Normaliza para que a soma seja igual ao número de amostras
    weights = weights / weights.mean()
    log.info(f"  Peso temporal: lambda={decay_lambda} | meia-vida≈{int(0.693/decay_lambda)}d | "
             f"peso min={weights.min():.3f} max={weights.max():.3f}")
    return weights


def train_result_model(df: pd.DataFrame, test_season: int | None = None, decay_lambda: float = 0.003):
    """
    Treina XGBoost para prever H/D/A com peso temporal exponencial.
    Se test_season for fornecido, usa como hold-out para avaliação.
    """
    le = LabelEncoder()
    df = df.copy()
    df["target"] = le.fit_transform(df["result"])  # A=0, D=1, H=2

    X = df[FEATURE_COLS].fillna(0)
    y = df["target"]

    if test_season:
        train_mask = df["season"] != test_season
        test_mask  = df["season"] == test_season
        X_train, y_train = X[train_mask], y[train_mask]
        X_test,  y_test  = X[test_mask],  y[test_mask]
        weights_train = compute_sample_weights(df[train_mask], decay_lambda)
        log.info(f"  Treino: {len(X_train)} jogos | Teste (temporada {test_season}): {len(X_test)} jogos")
    else:
        X_train, y_train = X, y
        X_test,  y_test  = None, None
        weights_train = compute_sample_weights(df, decay_lambda)
        log.info(f"  Treino: {len(X_train)} jogos (sem hold-out)")

    # Modelo base
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )

    # Calibração com pesos temporais
    model = CalibratedClassifierCV(xgb_model, cv=3, method="isotonic")
    model.fit(X_train, y_train, sample_weight=weights_train)

    # Avaliação em hold-out
    if X_test is not None and len(X_test) > 0:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        acc = accuracy_score(y_test, y_pred)
        ll  = log_loss(y_test, y_proba)
        log.info(f"  Acurácia hold-out: {acc:.3f} ({acc*100:.1f}%)")
        log.info(f"  Log loss: {ll:.4f}")
        log.info("\n" + classification_report(y_test, y_pred, target_names=le.classes_))
    else:
        # Cross-validation se não tem hold-out
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy",
                                  fit_params={"sample_weight": weights_train})
        log.info(f"  Acurácia CV 5-fold: {scores.mean():.3f} ± {scores.std():.3f}")

    return model, le


# ── Treinamento dos modelos home/away split ────────────────────────────────────

def _build_xgb_calibrated(X_train, y_train, weights_train):
    """Treina um XGBoost calibrado com os dados e pesos fornecidos."""
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )
    model = CalibratedClassifierCV(xgb_model, cv=3, method="isotonic")
    model.fit(X_train, y_train, sample_weight=weights_train)
    return model


def train_split_models(df: pd.DataFrame, test_season: int | None = None, decay_lambda: float = 0.003):
    """
    Treina dois modelos XGBoost com perspectivas distintas:
    - home_split_model: features focadas no mandante
    - away_split_model: features focadas no visitante

    Retorna os dois modelos e o LabelEncoder (compartilhado).
    """
    le = LabelEncoder()
    df = df.copy()
    df["target"] = le.fit_transform(df["result"])

    if test_season:
        train_mask = df["season"] != test_season
        test_mask  = df["season"] == test_season
        weights_train = compute_sample_weights(df[train_mask], decay_lambda)

        X_home_train = df.loc[train_mask, HOME_SPLIT_COLS].fillna(0)
        X_away_train = df.loc[train_mask, AWAY_SPLIT_COLS].fillna(0)
        y_train = df.loc[train_mask, "target"]

        X_home_test = df.loc[test_mask, HOME_SPLIT_COLS].fillna(0)
        X_away_test = df.loc[test_mask, AWAY_SPLIT_COLS].fillna(0)
        y_test = df.loc[test_mask, "target"]

        log.info(f"  Split treino: {len(y_train)} | Teste ({test_season}): {len(y_test)}")
    else:
        weights_train = compute_sample_weights(df, decay_lambda)
        X_home_train = df[HOME_SPLIT_COLS].fillna(0)
        X_away_train = df[AWAY_SPLIT_COLS].fillna(0)
        y_train = df["target"]
        X_home_test = X_away_test = y_test = None
        log.info(f"  Split treino: {len(y_train)} jogos (sem hold-out)")

    log.info("  Treinando home_split_model...")
    home_model = _build_xgb_calibrated(X_home_train, y_train, weights_train)

    log.info("  Treinando away_split_model...")
    away_model = _build_xgb_calibrated(X_away_train, y_train, weights_train)

    # Avaliação: blend 50/50 home + away split
    if y_test is not None and len(y_test) > 0:
        proba_home = home_model.predict_proba(X_home_test)
        proba_away = away_model.predict_proba(X_away_test)
        proba_blend = (proba_home + proba_away) / 2
        y_pred_blend = np.argmax(proba_blend, axis=1)
        acc_blend = accuracy_score(y_test, y_pred_blend)
        ll_blend   = log_loss(y_test, proba_blend)
        log.info(f"  Blend (home+away) — Acurácia: {acc_blend:.3f} ({acc_blend*100:.1f}%) | Log loss: {ll_blend:.4f}")
        log.info("\n" + classification_report(y_test, y_pred_blend, target_names=le.classes_))

    return home_model, away_model, le


# ── Treinamento do modelo de gols (Poisson) ────────────────────────────────────

def train_goals_model(df: pd.DataFrame, decay_lambda: float = 0.003):
    """
    Treina dois modelos Poisson com peso temporal:
    - Um para prever gols do time mandante
    - Um para prever gols do time visitante
    """
    df = df.copy()
    df["home_goals"] = pd.to_numeric(df["home_goals"], errors="coerce").fillna(0)
    df["away_goals"] = pd.to_numeric(df["away_goals"], errors="coerce").fillna(0)

    X = df[FEATURE_COLS].fillna(0)
    weights = compute_sample_weights(df, decay_lambda)

    # Modelo para gols do mandante
    model_home_goals = PoissonRegressor(alpha=0.5, max_iter=500)
    model_home_goals.fit(X, df["home_goals"], sample_weight=weights)

    # Modelo para gols do visitante
    model_away_goals = PoissonRegressor(alpha=0.5, max_iter=500)
    model_away_goals.fit(X, df["away_goals"], sample_weight=weights)

    # Avalia com MAE simples
    home_pred = model_home_goals.predict(X)
    away_pred = model_away_goals.predict(X)
    home_mae = np.abs(home_pred - df["home_goals"]).mean()
    away_mae = np.abs(away_pred - df["away_goals"]).mean()
    log.info(f"  MAE gols mandante: {home_mae:.3f} | MAE gols visitante: {away_mae:.3f}")

    return model_home_goals, model_away_goals


# ── Probabilidade Over 2.5 via simulação de Poisson ───────────────────────────

def over25_prob_from_poisson(lambda_home: float, lambda_away: float) -> float:
    """
    Calcula P(total_gols > 2.5) usando distribuição de Poisson.
    Simula todas as combinações de placar de 0 a 10 gols por time.
    """
    from scipy.stats import poisson

    prob_over = 0.0
    for h in range(11):
        for a in range(11):
            if h + a > 2:
                prob_over += poisson.pmf(h, lambda_home) * poisson.pmf(a, lambda_away)
    return prob_over


# ── Salva e carrega modelos ────────────────────────────────────────────────────

def get_models_dir(league: str) -> str:
    """Retorna o diretório de modelos para a liga. Cria se não existir."""
    path = os.path.join(MODELS_DIR, league)
    os.makedirs(path, exist_ok=True)
    return path


def save_models(result_model, label_encoder, home_goals_model, away_goals_model,
                home_split_model=None, away_split_model=None, league: str = "BSA"):
    version = datetime.now().strftime("%Y%m%d_%H%M")
    models_dir = get_models_dir(league)

    objects = [
        ("result_model",     result_model),
        ("label_encoder",    label_encoder),
        ("home_goals_model", home_goals_model),
        ("away_goals_model", away_goals_model),
    ]
    if home_split_model is not None:
        objects.append(("home_split_model", home_split_model))
    if away_split_model is not None:
        objects.append(("away_split_model", away_split_model))

    for name, obj in objects:
        path_versioned = os.path.join(models_dir, f"{name}_{version}.pkl")
        path_latest    = os.path.join(models_dir, f"{name}_latest.pkl")

        with open(path_versioned, "wb") as f:
            pickle.dump(obj, f)
        with open(path_latest, "wb") as f:
            pickle.dump(obj, f)

        log.info(f"  Salvo: {path_latest}")

    log.info(f"  Liga: {league} | Versão: {version}")
    return version


def load_models(league: str = "BSA") -> tuple:
    """Carrega os modelos mais recentes da liga. Usado pela API."""
    models_dir = get_models_dir(league)
    models = {}
    for name in ["result_model", "label_encoder", "home_goals_model", "away_goals_model",
                 "home_split_model", "away_split_model"]:
        path = os.path.join(models_dir, f"{name}_latest.pkl")
        # fallback: modelos antigos na raiz (BSA legado)
        if not os.path.exists(path) and league == "BSA":
            path = os.path.join(MODELS_DIR, f"{name}_latest.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                models[name] = pickle.load(f)
        else:
            models[name] = None
    return (
        models["result_model"],
        models["label_encoder"],
        models["home_goals_model"],
        models["away_goals_model"],
        models["home_split_model"],
        models["away_split_model"],
    )


# ── Feature importance ─────────────────────────────────────────────────────────

def log_feature_importance(model, label_encoder):
    """Loga as features mais importantes do XGBoost."""
    try:
        base_model = model.calibrated_classifiers_[0].estimator
        importances = base_model.feature_importances_
        feat_imp = sorted(zip(FEATURE_COLS, importances), key=lambda x: x[1], reverse=True)
        log.info("\nTop 10 features mais importantes:")
        for feat, imp in feat_imp[:10]:
            bar = "█" * int(imp * 200)
            log.info(f"  {feat:<25} {imp:.4f} {bar}")
    except Exception:
        pass  # Calibrated model pode não expor diretamente


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treina os modelos ML por liga")
    parser.add_argument("--league", type=str, default="BSA",
                        help="Código da liga (ex: BSA, PL, PD)")
    parser.add_argument("--season-test", type=int, help="Temporada para usar como hold-out")
    parser.add_argument("--decay-lambda", type=float, default=0.003,
                        help="Lambda do decay exponencial (padrão: 0.003 ≈ meia-vida 230 dias)")
    args = parser.parse_args()

    log.info(f"── Liga: {args.league} ──────────────────────────────────")
    log.info("── Carregando dados ──────────────────────")
    df = load_training_data(league=args.league)

    log.info(f"── Treinando modelo de resultado (XGBoost + decay λ={args.decay_lambda}) ──")
    result_model, label_encoder = train_result_model(df, test_season=args.season_test,
                                                      decay_lambda=args.decay_lambda)
    log_feature_importance(result_model, label_encoder)

    log.info("── Treinando modelo de gols (Poisson + decay) ──")
    home_goals_model, away_goals_model = train_goals_model(df, decay_lambda=args.decay_lambda)

    log.info(f"── Treinando modelos split home/away (λ={args.decay_lambda}) ──")
    home_split_model, away_split_model, _ = train_split_models(
        df, test_season=args.season_test, decay_lambda=args.decay_lambda
    )

    log.info("── Salvando modelos ──────────────────────")
    version = save_models(
        result_model, label_encoder, home_goals_model, away_goals_model,
        home_split_model, away_split_model, league=args.league
    )

    log.info(f"\nTreinamento concluído. Liga: {args.league} | Versão: {version}")
    log.info("Execute 'python api/main.py' para servir as predições.")
