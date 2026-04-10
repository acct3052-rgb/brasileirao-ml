# Brasileirão ML — Sistema de Predição de Partidas

API de machine learning para predição de resultados e gols do Brasileirão Série A.

## Stack

- **Railway** — hospedagem da API Python
- **Supabase** — banco de dados PostgreSQL
- **FastAPI** — framework da API
- **XGBoost** — modelo de predição de resultado (H/D/A)
- **Poisson Regression** — modelo de gols esperados / Over-Under
- **football-data.org** — fonte de dados gratuita

---

## Setup passo a passo

### 1. Supabase

1. Crie uma conta em [supabase.com](https://supabase.com)
2. Crie um novo projeto (anote a senha do banco)
3. Vá em **SQL Editor** e cole o conteúdo de `supabase_schema.sql`
4. Execute o SQL
5. Vá em **Settings → API** e copie:
   - `Project URL` → `SUPABASE_URL`
   - `service_role` key → `SUPABASE_KEY`

### 2. football-data.org

1. Cadastre-se em [football-data.org/client/register](https://www.football-data.org/client/register)
2. Confirme o email
3. Copie a API key do dashboard → `FOOTBALL_DATA_API_KEY`

### 3. Instalação local

```bash
git clone <seu-repositorio>
cd brasileirao-ml

# Crie ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Instala dependências
pip install -r requirements.txt

# Copia e preenche as variáveis de ambiente
cp .env.example .env
# Edite .env com suas chaves
```

### 4. Coleta de dados históricos

```bash
# Coleta dados de 2022, 2023 e 2024
python scripts/collect_data.py --season 2022 --season 2023 --season 2024

# Verifica no Supabase: tabelas 'teams' e 'matches' devem ter dados
```

### 5. Calcula features

```bash
# Calcula features para todas as temporadas coletadas
python scripts/build_features.py --all

# Verifica tabela 'match_features' no Supabase
```

### 6. Treina o modelo

```bash
# Treina usando 2022-2023 como treino e 2024 como hold-out
python scripts/train_model.py --season-test 2024

# Modelos salvos em models/
# Verifique a acurácia no log
```

### 7. Roda a API localmente

```bash
python api/main.py
# API disponível em http://localhost:8000
# Documentação: http://localhost:8000/docs
```

### 8. Testa a API

```bash
# Health check
curl http://localhost:8000/health

# Próximos jogos com predições
curl http://localhost:8000/api/fixtures

# Acurácia do modelo
curl http://localhost:8000/api/accuracy

# Gerar predições em lote para próximos jogos
curl -X POST http://localhost:8000/api/predict/batch
```

---

## Deploy no Railway

### Pré-requisitos
- Conta no [Railway](https://railway.app) (login com GitHub)
- Repositório no GitHub com o código

### Passos

1. No Railway: **New Project → Deploy from GitHub repo**
2. Selecione o repositório
3. Vá em **Variables** e adicione todas as variáveis do `.env.example`
4. Railway detecta o `Procfile` automaticamente e faz o deploy
5. Acesse a URL gerada (ex: `https://brasileirao-ml.up.railway.app`)

### Cron job no Railway (ETL diário)

1. No Railway: **New → Cron Job**
2. Configure:
   - Schedule: `0 3 * * *` (todo dia às 3h da manhã)
   - Command: `curl -X POST https://sua-api.up.railway.app/api/run-etl -H "Authorization: Bearer $ADMIN_TOKEN"`

---

## Endpoints da API

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| GET | `/health` | Status da API e modelos |
| GET | `/api/fixtures` | Próximos jogos com predições |
| GET | `/api/accuracy` | Acurácia histórica do modelo |
| GET | `/api/predictions/recent` | Últimas predições + resultado real |
| POST | `/api/predict` | Prediz um jogo pelo match_id |
| POST | `/api/predict/batch` | Prediz todos os jogos futuros |
| POST | `/api/update-results` | Atualiza resultados reais nas predições |
| POST | `/api/run-etl` | Dispara ETL completo (requer ADMIN_TOKEN) |
| POST | `/api/run-training` | Re-treina o modelo (requer ADMIN_TOKEN) |

---

## Estrutura do projeto

```
brasileirao-ml/
├── api/
│   └── main.py              # FastAPI — todos os endpoints
├── scripts/
│   ├── collect_data.py      # Coleta dados do football-data.org
│   ├── build_features.py    # Calcula features para o modelo
│   └── train_model.py       # Treina XGBoost + Poisson
├── models/                  # Modelos treinados (gerado automaticamente)
├── supabase_schema.sql      # Schema do banco
├── requirements.txt
├── Procfile
├── railway.json
└── .env.example
```

---

## Features do modelo

| Feature | Descrição |
|---------|-----------|
| `home/away_form_pts` | Média de pontos nos últimos 5 jogos |
| `home/away_form_gf/ga` | Média de gols marcados/sofridos |
| `home_home_pts / away_away_pts` | Form específico por mando de campo |
| `h2h_*` | Histórico dos últimos 5 confrontos diretos |
| `home/away_table_pos` | Posição na tabela na época do jogo |
| `pos_diff / pts_diff` | Diferença de posição e pontos |
| `matchday` | Rodada do campeonato |

---

## Benchmark esperado

- **Acurácia resultado (1X2):** 52–57% (baseline aleatório = 33%)
- **MAE gols:** ~0.9–1.1 gols por partida
- **Over 2.5 calibração:** probabilidades calibradas via Isotonic Regression
