-- Adiciona colunas de xG e stats avançadas à tabela match_features
-- Execute no SQL Editor do Supabase

ALTER TABLE match_features
    ADD COLUMN IF NOT EXISTS home_avg_xg       FLOAT,
    ADD COLUMN IF NOT EXISTS away_avg_xg       FLOAT,
    ADD COLUMN IF NOT EXISTS home_avg_xga      FLOAT,
    ADD COLUMN IF NOT EXISTS away_avg_xga      FLOAT,
    ADD COLUMN IF NOT EXISTS home_xg_net       FLOAT,
    ADD COLUMN IF NOT EXISTS away_xg_net       FLOAT,
    ADD COLUMN IF NOT EXISTS home_avg_poss     FLOAT,
    ADD COLUMN IF NOT EXISTS squad_value_ratio FLOAT,
    ADD COLUMN IF NOT EXISTS home_attendance_pct FLOAT;
