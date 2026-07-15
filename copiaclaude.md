# CLAUDE.md — Desafio Quant AI Itaú
# Estratégia V3: Google Trends + Momentum + Logit com Regularização L2

Este arquivo contém as instruções operacionais essenciais do projeto — o que precisa ser
respeitado em toda tarefa. Leia-o integralmente antes de qualquer tarefa.

Para o racional completo por trás de cada decisão, descrição detalhada de cada módulo,
tratamento de cada fonte de dado, cronograma e referências acadêmicas, ver **`docs/STRATEGY.md`**.
Consulte esse arquivo quando for implementar um módulo específico — não precisa estar carregado
o tempo todo.

---

## 1. VISÃO GERAL DA ESTRATÉGIA

O robô seleciona mensalmente até 10 ações da B3 combinando três camadas:

1. **Google Trends (GSVI)** — feature de choque de atenção do investidor de varejo
2. **Momentum** — retorno acumulado t-12 a t-2 meses (skip do último mês), normalizado cross-section
3. **Logit L2** — modelo treinado no **universo amplo** (sem pré-filtro) que aprende o peso ótimo
   de cada sinal (incluindo termo de interação) e produz P(outperform) para cada ação

A carteira final é Top-10 por P(outperform), pesos equal-weight (base), com volatility scaling
**sem alavancagem** (cap 1.0) aplicado sobre os pesos.

**Rebalanceamento:** mensal (primeiro dia útil do mês)
**Universo candidato:** IBrX-100 completo, point-in-time se possível
**Benchmark:** IBOVESPA (`^BVSP`) · **Taxa livre de risco:** Selic (API BACEN)
**Validação:** walk-forward no desenvolvimento + avaliação única em holdout congelado

> Esta é a v3 da estratégia: label cross-section, treino no universo amplo (sem pré-filtro por
> GSVI), filtro de atenção como termo de interação, e holdout congelado contra overfitting de
> backtest. Detalhes de por que cada uma dessas decisões foi tomada estão em `docs/STRATEGY.md` §0.

---

## 2. ESTRUTURA DE ARQUIVOS

```
quant_ai_itau/
│
├── CLAUDE.md                  ← este arquivo (operacional, sempre carregado)
├── docs/
│   └── STRATEGY.md            ← racional completo, módulos, cronograma, referências
├── main.py                    ← orquestrador principal — SEMPRE rodar por aqui
├── config.py                  ← todos os hiperparâmetros do projeto (incl. data de corte do holdout)
├── requirements.txt
│
├── data/
│   ├── raw/                   ← dados brutos baixados (NÃO commitar no git)
│   └── processed/
│       ├── prices.parquet     ← preços diários ajustados (dividendos + splits)
│       ├── gsvi.parquet       ← Google Search Volume Index semanal por ticker
│       ├── selic.parquet      ← taxa Selic diária (API BACEN) — taxa livre de risco e retorno do caixa
│       ├── features.parquet   ← mom_score, gsvi_score, interaction, volatilidade (gerado por features.py)
│       └── labels.parquet     ← labels cross-section gerados por labels.py
│
├── src/
│   ├── data_pipeline.py        ← download de preços (yfinance) e GSVI (pytrends), chamado por main.py --update-data
│   ├── features.py            ← mom_score, gsvi_score, interaction, volatilidade realizada
│   ├── labels.py               ← geração de labels cross-section — CRÍTICO, sem look-ahead
│   ├── model.py                ← LogisticRegression L2: treino walk-forward + predição no universo amplo
│   ├── portfolio.py            ← seleção Top-10 + guarda de sanidade + pesos + buffer zone
│   ├── risk.py                  ← volatility scaling (cap 1.0, sem alavancagem) + cap por ação
│   ├── backtest.py             ← engine walk-forward com custos de transação + separação dev/holdout
│   └── metrics.py              ← Sharpe, Drawdown, Calmar, AUC, ablation, IR
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_signal_analysis.ipynb
│   ├── 03_model_analysis.ipynb
│   └── 04_backtest_results.ipynb
│
└── reports/
    └── results/
```

**Nota:** não existe `universe.py`. O filtro de atenção deixou de ser um estágio de
pré-seleção e virou (a) feature/interação no `features.py` e (b) guarda de sanidade opcional
no `portfolio.py`. Ver `docs/STRATEGY.md` §1 para descrição de cada módulo.

---

## 3. PARÂMETROS CENTRAIS (config.py)

Todos os números do modelo vivem em `config.py`. NUNCA use números mágicos no código.
Se precisar ajustar um parâmetro, altere apenas em `config.py`.

> `MIN_TRAIN_MONTHS` e `HOLDOUT_MONTHS` foram reduzidos em 2026-07-10 (24→18, 15→12) porque o
> GSVI só cobre 5 anos (limite do `pytrends`), encurtando a janela de desenvolvimento
> disponível. Decisão reversível — ver `docs/STRATEGY.md` §7 para o racional completo.

| Parâmetro | Valor padrão | Descrição |
|---|---|---|
| `UNIVERSE` | IBrX-100 | Universo candidato completo — sem pré-filtro |
| `TOP_N_PORTFOLIO` | 10 | Ações finais na carteira |
| `MOM_LOOKBACK_START` | 12 | Início da janela de momentum (meses atrás) |
| `MOM_LOOKBACK_END` | 2 | Fim da janela de momentum (skip do último mês) |
| `GSVI_WINDOW` | 4 | Semanas para calcular média do GSVI delta |
| `GSVI_SEARCH_TERM` | testar `ticker` vs. nome da empresa | Escolher por IC em `02_signal_analysis.ipynb` |
| `LOGIT_C` | 0.1 | Regularização L2 (menor = mais regularização); faixa a testar: 0.01–0.1 |
| `LOGIT_CLASS_WEIGHT` | `None` | Label cross-section já é ~balanceado por construção |
| `VOL_TARGET` | 0.15 | Volatilidade-alvo anual (15%) |
| `VOL_WINDOW` | 21 | Janela de volatilidade realizada (dias úteis) |
| `MAX_VOL_SCALING_FACTOR` | **1.0** | Cap do fator de vol scaling — nunca alavanca acima de 100% |
| `MAX_WEIGHT_PER_STOCK` | 0.20 | Cap máximo por ação na carteira |
| `APPLY_SANITY_FILTER` | `True` | Descartar da Top-10 ações com `mom_score < 0`; testar com/sem via ablation |
| `BUFFER_MARGIN` | 0.02 (sugerido, ajustar na Fase 4) | Margem mínima de P(outperform) para trocar uma ação |
| `WEIGHT_SCHEME` | `'equal'` | `'equal'` (base) ou `'proportional'` (peso ~ P(outperform)) |
| `TRANSACTION_COST` | 0.002 | Custo por trade (0,2% ida+volta; faixa realista B3: 0,1%–0,3%) |
| `MIN_TRAIN_MONTHS` | 18 | Mínimo de meses para primeiro treino do logit (warm-up); reduzido de 24 — ver nota abaixo |
| `HOLDOUT_MONTHS` | 12 | Últimos N meses do dataset viram holdout congelado; reduzido de 15 — ver nota abaixo |
| `GEO` | `'BR'` | Filtro geográfico do Google Trends |

---

## 4. FLUXO DO main.py

```
1. Carregar config.py (params + data de corte do holdout)
2. Verificar/baixar dados (preços do universo completo via yfinance + GSVI via pytrends)
3. features.py  → mom_score, gsvi_score, interaction, volatilidade
4. labels.py    → target cross-section, shift(-1) + mediana contemporânea
5. Loop de backtest no período de DESENVOLVIMENTO (mês a mês, walk-forward):
     a. model.py     → treinar logit com dados até T-1 no universo amplo, predizer P(outperform) em T
     b. portfolio.py → Top-10 por P(outperform), guarda de sanidade, buffer zone, pesos
     c. risk.py      → volatility scaling (cap 1.0) + cap por ação; resto do capital em Selic
     d. backtest.py  → simular retorno do período T com custos de transação
6. metrics.py → métricas do desenvolvimento + ablation (completa vs. sem-logit vs. Ibovespa)
7. (SÓ NO FIM, uma única vez) Avaliar no holdout congelado — nunca antes disso
8. Exportar gráficos e tabelas para reports/results/
```

---

## 5. REGRAS CRÍTICAS — NUNCA VIOLAR

### 5.1 Look-ahead bias — a regra mais importante do projeto

**NUNCA** usar dados do período T ou posteriores para tomar decisão em T.

```python
# CORRETO — label usa retorno futuro, mediana contemporânea do universo
fwd_ret = prices.groupby('ticker')['return_mensal'].shift(-1)
median_fwd = fwd_ret.groupby('date').transform('median')
labels['target'] = (fwd_ret > median_fwd).astype(int)

# ERRADO — isso é look-ahead bias, contamina tudo
labels['target'] = (prices['return_mensal'] > prices.groupby('date')['return_mensal'].transform('median')).astype(int)
```

O label do mês T só é conhecido depois que T+1 acontece.
O modelo treinado com esse label só pode ser usado a partir de T+2.

### 5.2 Walk-forward obrigatório — nunca cross-validation clássica

```python
# CORRETO — treino sempre usa apenas dados passados, janela expande a cada rebalanceamento
for t in rebalance_dates:
    train_data = full_data[full_data['date'] < t]
    model.fit(train_data[features], train_data['target'])
    preds[t] = model.predict_proba(data_at_t[features])[:, 1]

# ERRADO — cross-validation embaralha o tempo
from sklearn.model_selection import cross_val_score  # NÃO USAR em série temporal
```

### 5.3 Normalização cross-section — não time-series

Normalizar momentum e GSVI **entre as ações do universo em cada data**, não ao longo do tempo
para cada ação. Normalização time-series vaza informação futura.

```python
# CORRETO — z-score entre ações na mesma data
features['mom_score'] = features.groupby('date')['momentum_raw'].transform(
    lambda x: (x - x.mean()) / x.std()
)

# ERRADO — z-score ao longo do tempo por ação
features['mom_score'] = features.groupby('ticker')['momentum_raw'].transform(
    lambda x: (x - x.mean()) / x.std()  # usa dados futuros para calcular média
)
```

### 5.4 Universo amplo no treino — nunca pré-filtrar antes do logit

O logit treina sobre **todas** as ações do universo candidato, com dados disponíveis em cada
data. Não existe estágio de pré-seleção por GSVI (ou qualquer outro sinal) antes do treino.
Filtros de atenção entram como feature (`interaction`) ou como guarda de sanidade **depois** do
modelo já ter gerado P(outperform), nunca antes.

### 5.5 Carteira vazia é decisão válida

Se sobrarem menos de 10 ações elegíveis, aloca nas disponíveis e o restante do capital fica em
caixa rendendo Selic. Não forçar posições ruins.

### 5.6 Volatility scaling sem alavancagem — cap sempre em 1.0

```python
fator = min(config.VOL_TARGET / vol_realizada_carteira, config.MAX_VOL_SCALING_FACTOR)  # MAX = 1.0
pesos_escalados = pesos * fator
```

Estratégia é long-only e sem alavancagem: o fator **nunca** multiplica a exposição acima de
100%, só reduz quando a volatilidade sobe. Capital desalocado vai para caixa rendendo Selic.

### 5.7 Holdout congelado — nunca olhar antes da entrega final

Os últimos `HOLDOUT_MONTHS` do dataset são um cofre. Toda escolha de hiperparâmetro é feita
olhando **apenas** o período de desenvolvimento. O holdout é avaliado uma única vez, no fim do
projeto.

### 5.8 Re-treinar o logit a cada rebalanceamento

O modelo deve ser re-treinado mensalmente com a janela expandida de dados disponíveis. Não usar
coeficientes fixos do treino inicial.

---

## 6. CONVENÇÕES DE CÓDIGO

- Nomes de colunas: `snake_case` em português/inglês misturado é permitido, mas seja consistente dentro de cada arquivo
- Datas: sempre `pd.Timestamp`, nunca string
- Tickers da B3: sempre com sufixo `.SA` para yfinance (ex: `PETR4.SA`)
- Parquet: usar `pyarrow` como engine
- Logs: usar `print()` com prefixo de data/hora para rastrear o pipeline
- Commits: um commit por fase concluída (cronograma completo em `docs/STRATEGY.md` §5)

---

## 7. ARMADILHAS RÁPIDAS

| Armadilha | Como evitar |
|---|---|
| Look-ahead no label | `shift(-1)` + mediana contemporânea (regra 5.1) |
| Pré-filtro antes do treino | Universo amplo sempre (regra 5.4) |
| Overfitting de backtest | Holdout congelado, avaliado uma vez (regra 5.7) |
| Vol scaling alavancando | Cap em 1.0 (regra 5.6) |
| Cross-validation em série temporal | Proibido — só walk-forward (regra 5.2) |
| Normalização time-series | Proibido — só cross-section por data (regra 5.3) |
| Pytrends bloqueado | `time.sleep(2-5s)` + cache local |

Lista completa de armadilhas com causa raiz em `docs/STRATEGY.md` §4.

---

## 8. COMO RODAR O PROJETO

```bash
# 1. Ativar ambiente
conda activate quant_itau   # ou: source venv/bin/activate

# 2. Baixar/atualizar dados (apenas quando necessário)
python main.py --update-data

# 3. Rodar backtest completo (só período de desenvolvimento)
python main.py --backtest

# 4. Rodar apenas métricas sobre backtest já salvo
python main.py --metrics-only

# 5. Modo debug (roda apenas últimos 6 meses, mais rápido)
python main.py --backtest --debug

# 6. Avaliação final no holdout congelado — rodar UMA ÚNICA VEZ, na Semana 8
python main.py --holdout-eval
```
