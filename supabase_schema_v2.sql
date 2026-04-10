-- ============================================================
-- ADIÇÃO AO SCHEMA — execute após o supabase_schema.sql base
-- Novas tabelas: Base dos Dados + SoccerData/FBref
-- ============================================================

-- Partidas históricas da Base dos Dados (2006+)
-- Separado da tabela matches para não conflitar com IDs do football-data.org
CREATE TABLE IF NOT EXISTS matches_historical (
    id                  SERIAL PRIMARY KEY,
    match_id_bdd        TEXT UNIQUE NOT NULL,   -- ID próprio: "2023_01_flamengo_palmeiras"
    season              INTEGER NOT NULL,
    matchday            INTEGER,
    match_date          TIMESTAMPTZ,
    status              TEXT DEFAULT 'FINISHED',
    home_team_name      TEXT NOT NULL,
    away_team_name      TEXT NOT NULL,
    home_goals          INTEGER,
    away_goals          INTEGER,
    result              TEXT,                   -- 'H', 'D', 'A'
    stadium             TEXT,
    referee             TEXT,
    attendance          INTEGER,
    attendance_pct      FLOAT,                  -- % de ocupação do estádio
    home_squad_value    FLOAT,                  -- valor de mercado do elenco titular (€M)
    away_squad_value    FLOAT,
    source              TEXT DEFAULT 'basedosdados',
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_hist_season    ON matches_historical(season);
CREATE INDEX IF NOT EXISTS idx_hist_date      ON matches_historical(match_date);
CREATE INDEX IF NOT EXISTS idx_hist_home_team ON matches_historical(home_team_name);
CREATE INDEX IF NOT EXISTS idx_hist_away_team ON matches_historical(away_team_name);

-- Stats avançadas por partida (FBref via SoccerData)
CREATE TABLE IF NOT EXISTS match_advanced_stats (
    id              SERIAL PRIMARY KEY,
    match_id_fbref  TEXT UNIQUE NOT NULL,       -- "2024_01_flamengo_palmeiras"
    season          INTEGER NOT NULL,
    match_date      TEXT,
    home_team_name  TEXT NOT NULL,
    away_team_name  TEXT NOT NULL,
    -- Expected Goals
    xg_home         FLOAT,
    xg_away         FLOAT,
    xg_diff         FLOAT,                      -- xg_home - xg_away
    -- Posse
    possession_home FLOAT,                      -- % posse mandante
    possession_away FLOAT,
    -- Chutes
    shots_home      INTEGER,
    shots_away      INTEGER,
    shots_ot_home   INTEGER,                    -- chutes a gol (on target)
    shots_ot_away   INTEGER,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_adv_season    ON match_advanced_stats(season);
CREATE INDEX IF NOT EXISTS idx_adv_home_team ON match_advanced_stats(home_team_name);

-- Perfil xG médio por time/temporada
CREATE TABLE IF NOT EXISTS team_xg_profiles (
    id              SERIAL PRIMARY KEY,
    season          INTEGER NOT NULL,
    team_name       TEXT NOT NULL,
    avg_xg_home     FLOAT,       -- xG médio gerado jogando em casa
    avg_xg_away     FLOAT,       -- xG médio gerado jogando fora
    avg_xga_home    FLOAT,       -- xG médio concedido em casa
    avg_xga_away    FLOAT,       -- xG médio concedido fora
    avg_poss_home   FLOAT,       -- posse média em casa
    avg_shots_home  FLOAT,
    avg_shots_away  FLOAT,
    UNIQUE(season, team_name)
);

-- Stats de público e valor de elenco por time/temporada (Base dos Dados)
CREATE TABLE IF NOT EXISTS team_season_stats (
    id                    SERIAL PRIMARY KEY,
    season                INTEGER NOT NULL,
    team_name             TEXT NOT NULL,
    avg_attendance_home   FLOAT,    -- público médio em casa
    avg_attendance_pct    FLOAT,    -- % ocupação média
    avg_home_squad_value  FLOAT,    -- valor médio do elenco titular em casa (€M)
    avg_away_squad_value  FLOAT,    -- valor médio do elenco titular adversário
    UNIQUE(season, team_name)
);

-- ── Views úteis ──────────────────────────────────────────────────────────────

-- View unificada de partidas (football-data + histórico)
-- Usa nomes de times como chave de junção
CREATE OR REPLACE VIEW all_matches AS
-- Partidas do football-data.org (com IDs de times)
SELECT
    m.id::TEXT          AS match_ref,
    m.season,
    m.matchday,
    m.match_date,
    m.status,
    ht.name             AS home_team_name,
    at.name             AS away_team_name,
    m.home_goals,
    m.away_goals,
    m.result,
    NULL::TEXT          AS stadium,
    NULL::INTEGER       AS attendance,
    NULL::FLOAT         AS home_squad_value,
    NULL::FLOAT         AS away_squad_value,
    'football-data'     AS source
FROM matches m
JOIN teams ht ON ht.id = m.home_team_id
JOIN teams at ON at.id = m.away_team_id
WHERE m.status = 'FINISHED'

UNION ALL

-- Partidas da Base dos Dados (histórico 2006+)
SELECT
    match_id_bdd        AS match_ref,
    season,
    matchday,
    match_date,
    status,
    home_team_name,
    away_team_name,
    home_goals,
    away_goals,
    result,
    stadium,
    attendance,
    home_squad_value,
    away_squad_value,
    source
FROM matches_historical
ORDER BY match_date;

-- View de features completas para o modelo
-- Junta match_features + xG + stats de público/valor
CREATE OR REPLACE VIEW full_match_features AS
SELECT
    mf.*,
    -- xG features
    adv.xg_home,
    adv.xg_away,
    adv.xg_diff,
    adv.possession_home,
    adv.shots_home,
    adv.shots_away,
    adv.shots_ot_home,
    adv.shots_ot_away,
    -- Perfil xG do time mandante
    hxg.avg_xg_home         AS home_avg_xg_home,
    hxg.avg_xga_home        AS home_avg_xga_home,
    hxg.avg_poss_home       AS home_avg_poss,
    -- Perfil xG do time visitante
    axg.avg_xg_away         AS away_avg_xg_away,
    axg.avg_xga_away        AS away_avg_xga_away,
    -- Stats de público e valor
    hss.avg_attendance_pct  AS home_avg_attendance_pct,
    hss.avg_home_squad_value AS home_squad_value_season,
    ass_.avg_away_squad_value AS away_squad_value_season
FROM match_features mf
-- xG da partida específica (quando disponível)
LEFT JOIN match_advanced_stats adv
    ON adv.match_id_fbref LIKE '%' ||
       LOWER(REPLACE((SELECT name FROM teams WHERE id = (
           SELECT home_team_id FROM matches WHERE id = mf.match_id
       )), ' ', '')) || '%'
-- Perfil xG do mandante
LEFT JOIN team_xg_profiles hxg
    ON hxg.season = mf.season
    AND hxg.team_name = (SELECT name FROM teams WHERE id = (
        SELECT home_team_id FROM matches WHERE id = mf.match_id
    ))
-- Perfil xG do visitante
LEFT JOIN team_xg_profiles axg
    ON axg.season = mf.season
    AND axg.team_name = (SELECT name FROM teams WHERE id = (
        SELECT away_team_id FROM matches WHERE id = mf.match_id
    ))
-- Stats de público/valor do mandante
LEFT JOIN team_season_stats hss
    ON hss.season = mf.season
    AND hss.team_name = (SELECT name FROM teams WHERE id = (
        SELECT home_team_id FROM matches WHERE id = mf.match_id
    ))
-- Stats de público/valor do visitante
LEFT JOIN team_season_stats ass_
    ON ass_.season = mf.season
    AND ass_.team_name = (SELECT name FROM teams WHERE id = (
        SELECT away_team_id FROM matches WHERE id = mf.match_id
    ));
