-- ============================================================
-- BRASILEIRAO ML — Schema Supabase
-- Cole esse SQL inteiro no SQL Editor do Supabase e execute
-- ============================================================

-- Tabela de times
CREATE TABLE IF NOT EXISTS teams (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    short_name  TEXT,
    tla         TEXT,
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);

-- Tabela de partidas históricas
CREATE TABLE IF NOT EXISTS matches (
    id              INTEGER PRIMARY KEY,
    season          INTEGER NOT NULL,
    matchday        INTEGER,
    match_date      TIMESTAMPTZ,
    status          TEXT,  -- 'FINISHED', 'SCHEDULED', 'IN_PLAY'
    home_team_id    INTEGER REFERENCES teams(id),
    away_team_id    INTEGER REFERENCES teams(id),
    home_goals      INTEGER,
    away_goals      INTEGER,
    result          TEXT,  -- 'H' (home), 'D' (draw), 'A' (away)
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_matches_season    ON matches(season);
CREATE INDEX IF NOT EXISTS idx_matches_date      ON matches(match_date);
CREATE INDEX IF NOT EXISTS idx_matches_home_team ON matches(home_team_id);
CREATE INDEX IF NOT EXISTS idx_matches_away_team ON matches(away_team_id);

-- Tabela de features calculadas por jogo (usada pelo modelo)
CREATE TABLE IF NOT EXISTS match_features (
    id                      SERIAL PRIMARY KEY,
    match_id                INTEGER REFERENCES matches(id) UNIQUE,
    -- Form recente (últimos 5 jogos)
    home_form_pts           FLOAT,   -- pontos nos últimos 5 jogos (casa+fora)
    away_form_pts           FLOAT,
    home_form_gf            FLOAT,   -- média de gols marcados últimos 5
    away_form_gf            FLOAT,
    home_form_ga            FLOAT,   -- média de gols sofridos últimos 5
    away_form_ga            FLOAT,
    -- Desempenho separado (mando de campo)
    home_home_pts           FLOAT,   -- pontos jogando em casa (últimos 5 em casa)
    away_away_pts           FLOAT,   -- pontos jogando fora (últimos 5 fora)
    home_home_gf            FLOAT,
    away_away_gf            FLOAT,
    home_home_ga            FLOAT,
    away_away_ga            FLOAT,
    -- Head-to-head (últimos 5 confrontos diretos)
    h2h_home_wins           INTEGER,
    h2h_draws               INTEGER,
    h2h_away_wins           INTEGER,
    h2h_home_gf_avg         FLOAT,
    h2h_away_gf_avg         FLOAT,
    -- Posição e pontuação na tabela
    home_table_pos          INTEGER,
    away_table_pos          INTEGER,
    home_table_pts          INTEGER,
    away_table_pts          INTEGER,
    pos_diff                INTEGER,  -- diferença de posição
    pts_diff                INTEGER,  -- diferença de pontos
    -- Rodada e temporada
    matchday                INTEGER,
    season                  INTEGER,
    created_at              TIMESTAMPTZ DEFAULT NOW()
);

-- Tabela de predições do modelo
CREATE TABLE IF NOT EXISTS predictions (
    id              SERIAL PRIMARY KEY,
    match_id        INTEGER REFERENCES matches(id) UNIQUE,
    -- Probabilidades resultado
    prob_home       FLOAT,  -- % vitória mandante
    prob_draw       FLOAT,  -- % empate
    prob_away       FLOAT,  -- % vitória visitante
    predicted_result TEXT,  -- 'H', 'D', 'A' (resultado mais provável)
    confidence      FLOAT,  -- max(prob_home, prob_draw, prob_away)
    -- Gols esperados
    expected_goals_home  FLOAT,
    expected_goals_away  FLOAT,
    expected_total_goals FLOAT,
    over_25_prob    FLOAT,  -- probabilidade de mais de 2.5 gols
    -- Resultado real (preenchido após o jogo)
    actual_result   TEXT,
    correct         BOOLEAN,
    -- Metadados
    model_version   TEXT,
    predicted_at    TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_predictions_match    ON predictions(match_id);
CREATE INDEX IF NOT EXISTS idx_predictions_correct  ON predictions(correct);

-- View de acurácia do modelo (útil para o dashboard)
CREATE OR REPLACE VIEW model_accuracy AS
SELECT
    COUNT(*)                                        AS total_predictions,
    COUNT(*) FILTER (WHERE correct = TRUE)          AS correct_predictions,
    ROUND(
        COUNT(*) FILTER (WHERE correct = TRUE)::NUMERIC
        / NULLIF(COUNT(*) FILTER (WHERE actual_result IS NOT NULL), 0) * 100
    , 1)                                            AS accuracy_pct,
    ROUND(AVG(confidence)::NUMERIC * 100, 1)        AS avg_confidence_pct,
    MAX(predicted_at)                               AS last_prediction
FROM predictions
WHERE actual_result IS NOT NULL;

-- View de próximos jogos com predição
CREATE OR REPLACE VIEW upcoming_predictions AS
SELECT
    m.id            AS match_id,
    m.match_date,
    m.matchday,
    m.season,
    ht.name         AS home_team,
    at.name         AS away_team,
    p.prob_home,
    p.prob_draw,
    p.prob_away,
    p.predicted_result,
    p.confidence,
    p.expected_total_goals,
    p.over_25_prob
FROM matches m
JOIN teams ht ON ht.id = m.home_team_id
JOIN teams at ON at.id = m.away_team_id
LEFT JOIN predictions p ON p.match_id = m.id
WHERE m.status = 'SCHEDULED'
ORDER BY m.match_date ASC;
