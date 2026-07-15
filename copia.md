# STRATEGY.md — Desafio Quant AI Itaú — Estratégia V3 Completa

Este documento contém o racional completo por trás de cada decisão da estratégia, a descrição
detalhada de cada módulo, o tratamento de cada fonte de dado, o cronograma e as referências
acadêmicas. Consulte-o quando for implementar um módulo específico ou precisar entender o
"porquê" de uma regra.

Para as regras operacionais que precisam ser respeitadas em toda tarefa, ver **`../CLAUDE.md`**.

---

## 0. O QUE MUDOU DE V2 → V3 (E POR QUÊ)

O v2 estava bem estruturado, mas tinha cinco problemas que comprometiam a validade estatística.
As mudanças abaixo são todas correções de fundo, não cosméticas.

**1. Label agora é cross-section, não absoluto.**
No v2, o target era "a ação subiu (em termos absolutos) no mês seguinte?". O problema: as
features são cross-section (z-score entre ações), mas o label era absoluto. Em bull market
quase tudo sobe; em bear quase tudo cai — ou seja, o label era dominado pela direção do
mercado, justamente o componente que as features cross-section removem. Havia um descasamento
de natureza entre X e y.
**Correção:** o target passa a ser "a ação superou a mediana do universo no mês seguinte?".
Isso (a) alinha label e features na mesma natureza relativa, (b) deixa as classes balanceadas
~50/50 por construção — eliminando o problema de "em bull market o modelo aprende a sempre
dizer 1" — e (c) casa exatamente com o que a carteira faz (selecionar as melhores em termos
relativos).

**2. O modelo treina no universo amplo, não no pré-filtrado.**
No v2, o universo era filtrado para o Top-20 por GSVI *antes* do treino. Isso restringia o
range das features (todas com GSVI alto, momentum truncado em >0), o que atenuava os
coeficientes estimados e pedia ao modelo que discriminasse dentro de um grupo já homogêneo.
Além disso era redundante: usar GSVI para pré-filtrar e depois usar GSVI como feature.
**Correção:** o logit treina sobre todo o universo candidato, vendo a faixa completa dos
sinais. O filtro de atenção deixa de ser um corte rígido no pré-processamento.

**3. O filtro de "atenção boa vs. ruim" virou um termo de interação.**
O corte rígido (GSVI alto E momentum > 0) codificava à mão a lógica "atenção só ajuda quando a
ação já sobe". Em vez de hard-code, deixamos o modelo aprender isso via um terceiro feature:
`mom_score × gsvi_score`. Se a interação for positiva e significativa, o modelo captura
sozinho que atenção com momentum positivo é diferente de atenção com momentum negativo — de
forma mais flexível e defensável que um corte binário.

**4. Holdout final ("lockbox") separado do walk-forward.**
O walk-forward protege contra look-ahead *dentro de uma rodada*, mas não contra o overfitting
que acontece quando você ajusta C, vol-alvo, N etc. olhando o mesmo backtest várias vezes
(problema do *deflated Sharpe*, Bailey & López de Prado). **Correção:** os últimos 12–18 meses
ficam num cofre que não se olha até a entrega final.

**5. Honestidade sobre survivorship bias e dados de Trends.**
O v2 dizia "remover ações que não existiam no período" como solução de survivorship — isso é
quase o oposto da correção. O `yfinance` já entrega só os constituintes *atuais*, então ações
que faliram/saíram já estão ausentes, inflando o retorno. O v3 reconhece a direção do viés em
vez de fingir que o resolveu, e trata os problemas point-in-time do Google Trends
explicitamente.

**Decisão de design importante:** todo o desenvolvimento (escolha de hiperparâmetros, ajustes)
acontece no período de *desenvolvimento*. Os últimos 12–18 meses formam um holdout que só é
avaliado uma vez, na entrega.

---

## 1. DESCRIÇÃO DETALHADA DE CADA MÓDULO

### `features.py`
Calcula os sinais de entrada do logit para **todas** as ações do universo (sem pré-filtro):

- **`mom_score`**: retorno acumulado de t-12 a t-2 meses, normalizado cross-section (z-score
  por data). Skip do mês mais recente para evitar reversão de curtíssimo prazo. Cross-section
  por data não vaza informação — usa só dados contemporâneos entre ações.
- **`gsvi_score`**: `gsvi_delta = gsvi_semana_atual / media_gsvi_4_semanas_anteriores - 1`,
  normalizado cross-section, mesma escala do momentum.
- **`interaction`**: `mom_score × gsvi_score` — substitui o filtro binário rígido do v2.
- **`volatilidade`**: volatilidade realizada de 21 dias, usada por `risk.py`.

Saída: `[date, ticker, mom_score, gsvi_score, interaction, volatilidade]`

### `labels.py` ⚠️ CRÍTICO
Gera o label binário cross-section para treino do logit.

```python
# Retorno futuro de cada ação (shift de -1 no eixo temporal)
fwd_ret = prices.groupby('ticker')['return_mensal'].shift(-1)

# Mediana cross-section do universo no MESMO período futuro
median_fwd = fwd_ret.groupby('date').transform('median')

# Label cross-section: superou a mediana?
labels['target'] = (fwd_ret > median_fwd).astype(int)
```

- Label = 1 se retorno futuro da ação supera a mediana cross-section do mesmo período futuro
- Só calculável **depois** que T+1 acontece; só usado para treinar de T+2 em diante
- A mediana é calculada sobre o universo **contemporâneo** (mesma janela T→T+1) — não há look-ahead
- Vantagem: classes balanceadas ~50/50 por construção e alinhamento perfeito com a seleção da carteira

Saída: `[date, ticker, target]`

### `model.py`
Contém a classe `MomentumLogit`:

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    penalty='l2',       # Ridge — essencial contra overfitting
    C=0.1,               # C pequeno = regularização forte (em sklearn, C é o inverso)
    solver='lbfgs',
    max_iter=1000,
    class_weight=None    # com label cross-section as classes já são ~balanceadas
)
```

- `fit(X_train, y_train)`: treina sobre **todo** o universo disponível (sem pré-filtro)
- `predict_proba(X)`: retorna P(outperform) para cada ação
- `get_coefs()`: retorna coeficientes atuais para monitoramento ao longo do tempo

Features de entrada: `[mom_score, gsvi_score, mom_score × gsvi_score]` — três colunas, continua
parcimonioso. As features já estão em escala z (cross-section), o que cuida da sensibilidade
do L2 à escala.

**Treino no universo amplo:** o modelo treina sobre TODAS as ações com dados disponíveis, não
sobre um Top-N pré-filtrado. Assim ele vê a faixa completa de momentum e GSVI, e os
coeficientes não ficam atenuados por range restrito.

**Walk-forward training (obrigatório):**

```
Dados de DESENVOLVIMENTO: Jan/2018 → ~Jun/2023 (tudo antes do holdout)
Holdout (cofre, não tocar até a entrega): ~Jul/2023 → Dez/2024

Dentro do desenvolvimento:
  Rebalanceamento de Jan/2021 → Treino: Jan/2018 a Dez/2020 → Prediz Jan/2021
  Rebalanceamento de Fev/2021 → Treino: Jan/2018 a Jan/2021 → Prediz Fev/2021
  ... janela expande
```

**Sobre o tamanho de amostra (importante e honesto):** com ~100 ações × 24 meses parece
"muitas amostras", mas dentro de cada mês os retornos são fortemente correlacionados pelo fator
de mercado. O N *efetivo* está muito mais perto do número de **meses** do que do número de
linhas. Isso reforça manter o modelo simples (2 sinais + 1 interação, L2 forte). O label
cross-section ajuda aqui porque remove a parte de "direção do mercado" que era a fonte
principal dessa correlação no label.

**Re-treino mensal:** a cada rebalanceamento, com janela expandida. Permite acompanhar mudança
de regime (o notebook `03_model_analysis` mostra se os coeficientes mudam ao longo do tempo).

### `portfolio.py`

**Seleção:** ordenar todo o universo por P(outperform) e pegar a Top-10.

**Guarda de sanidade (opcional, leve):** das Top-10, opcionalmente descartar as com
`mom_score < 0`. Como o modelo já usa momentum e a interação, isso é uma rede de segurança,
não o mecanismo principal. Testar com e sem para ver se agrega.

**Edge case obrigatório:** definir o comportamento quando sobram menos de 10 ações (ex.: alocar
nas disponíveis e o restante em caixa/Selic).

**Pesos (escolher uma, começar pela B):**

- *Opção A — Proporcional à probabilidade:* `pesos = proba_top10 / proba_top10.sum()`
- *Opção B — Equal-weight (mais robusto, recomendado como base):* `pesos = 1 / len(carteira)`

**Rebalanceamento mensal:** registrar todas as trocas (turnover e custos). Para conter
turnover: *buffer zone* — só trocar se a P da nova ação superar a da atual por margem mínima.

**Nota de implementação (2026-07-10):** o cap de concentração por ação
(`config.MAX_WEIGHT_PER_STOCK`) acabou implementado aqui, junto com o cálculo dos pesos-alvo
(`_cap_and_redistribute`), e não em `risk.py` como o planejamento original previa — faz mais
sentido computar peso e cap no mesmo lugar. `risk.py` fica só com o volatility scaling, que
escala o vetor de pesos inteiro pela mesma constante (nunca viola o cap já aplicado, pois o
fator nunca excede 1.0).

### `risk.py`

**Volatility Scaling (com cap):**
- Vol realizada da carteira nos últimos 21 dias úteis; vol-alvo 15% a.a.
- Fator de escala: `min(vol_alvo / vol_realizada, 1.0)` — **limitado a 1.0**. Numa estratégia
  long-only sem alavancagem, nunca multiplicamos a exposição acima de 100%; só reduzimos quando
  a vol sobe.
- O capital desalocado (quando o fator < 1) vai para caixa rendendo Selic. Especificar isso no
  backtest, senão o retorno fica ambíguo.

### `backtest.py`

**Engine walk-forward:**
```
Para cada mês T no período de DESENVOLVIMENTO:
  1. Calcular features com dados até T-1 (universo completo)
  2. Treinar logit com labels até T-1
  3. Calcular P(outperform) para as ações disponíveis em T
  4. Selecionar Top-10, aplicar guarda de sanidade, calcular pesos
  5. Aplicar volatility scaling (cap 1.0; resto em Selic)
  6. Simular retorno do período T
  7. Registrar e avançar
```

**Custos:** 0,1% a 0,3% por trade (realista para B3). Turnover alto corrói alpha — monitorar.

**Warm-up e Holdout:**
- Primeiros ~2 anos: só treino inicial (sem contar retorno)
- Holdout final (12–18 meses): congelado. Toda a otimização de hiperparâmetros é feita no
  desenvolvimento. O holdout é avaliado **uma única vez**, no fim. Essa é a defesa contra
  overfitting de backtest (ajustar parâmetros olhando o resultado).

### `metrics.py`

**Carteira:** retorno total/anualizado, Sharpe (excesso sobre Selic), Max Drawdown e
recuperação, Calmar, Turnover médio.

**Modelo Logit:**
- AUC-ROC: com label cross-section, mede a habilidade de *ranquear* ações por outperformance
  (>0.55 já é útil em finanças)
- Coeficientes ao longo do tempo: momentum vs. GSVI vs. interação — quem domina e como muda por regime
- Calibração com expectativa realista: com 3 features e L2 forte em dado ruidoso, as
  probabilidades ficam comprimidas perto de 0,5. Não esperar "P=0,7 → 70% de acerto"; o valor
  está no *ranking*, não na calibração absoluta.

**Ablation (a prova de valor do modelo):**
- Comparar: (i) estratégia completa, (ii) sem o logit (combinação fixa 50/50 dos dois
  z-scores), (iii) Ibovespa
- Se a versão completa não bate a combinação fixa, o logit não está agregando — e está tudo bem
  descobrir isso; é resultado honesto.

---

## 2. DATASETS NECESSÁRIOS

### 2.1 Preços de Ações
- Preços diários de fechamento **ajustado** (dividendos e splits) + volume
- Mínimo de **5 anos** — o logit precisa de mais amostras, e ainda reservamos 12–18 meses para holdout
- Fonte: `yfinance` para B3 (suficiente para o desafio)
- **Limitação assumida:** o `yfinance` entrega apenas constituintes *atuais*. Ações
  deslistadas/falidas estão ausentes → os retornos do backtest têm viés **otimista**
  (survivorship). Não há correção fácil aqui; documente o viés e sua direção em vez de
  ignorá-lo. Se conseguir a composição point-in-time do índice (mesmo que manualmente para
  alguns rebalanceamentos), melhor ainda.

### 2.2 Google Trends (GSVI)
- Volume de buscas semanal por ação, `pytrends` com `geo='BR'`
- Usar **variação** do GSVI (choque de atenção), nunca o valor absoluto
- Três armadilhas a tratar explicitamente:
  - **Normalização da janela:** o pytrends normaliza pelo máximo da janela consultada (=100).
    Ao costurar séries longas com janelas sobrepostas, garanta que a renormalização **não** use
    pontos futuros da janela — caso contrário você cria um vazamento sutil. Normalize sempre
    com base só no que estaria disponível em T.
  - **Amostragem/reprodutibilidade:** o Trends é uma amostra; a mesma query em dias diferentes
    retorna valores ligeiramente distintos. O GSVI que você "teria visto" em tempo real não é
    exatamente o que o pytrends devolve hoje. Documente isso como limitação do backtest e fixe
    um cache local para reprodutibilidade.
  - **Termo de busca:** decida e justifique — ticker (ex. "PETR4") tende a ser ruído
    quase-zero; nome da empresa ("Petrobras") vem contaminado por buscas não-financeiras. O
    Da-Engelberg-Gao (2011) usou ticker de propósito. Para o Brasil, teste ambos no
    `02_signal_analysis.ipynb` e escolha pelo IC.
- Cache local + `time.sleep(2-5s)` entre requisições para evitar bloqueio.

### 2.3 Labels de Treino — cross-section
Ver `labels.py` acima e regra 5.1 do `CLAUDE.md`.

### 2.4 Benchmark e Taxa Livre de Risco
- IBOVESPA (`^BVSP` via yfinance)
- Selic via API do BACEN (Sharpe em excesso e destino do caixa quando a vol-scaling desaloca)

---

## 3. O QUE OS NOTEBOOKS DEVEM MOSTRAR

### `01_eda.ipynb`
- Distribuição dos retornos mensais (fat tails?)
- Correlação entre momentum e GSVI (se muito correlacionados, a combinação ganha pouco)
- Histórico por ação; cobertura do GSVI

### `02_signal_analysis.ipynb`
- IC do momentum isolado e do GSVI delta isolado
- Teste ticker vs. nome da empresa como termo de busca — qual tem IC melhor
- Se ambos têm IC positivo, a combinação via logit faz sentido

### `03_model_analysis.ipynb`
- Coeficientes ao longo do tempo: momentum, GSVI e a interação — a interação é significativa? Positiva?
- Curva ROC / AUC mensal
- Calibração (com expectativa realista de compressão perto de 0,5)

### `04_backtest_results.ipynb` (apenas desenvolvimento até a Semana 7)
- Curva de capital: completa vs. sem logit (50/50 fixo) vs. Ibovespa
- Underwater chart (drawdown)
- Tabela de métricas; análise por período (2020? 2022?)
- **Na Semana 8:** uma seção final, separada, com a avaliação única no holdout

---

## 4. PROBLEMAS E SOLUÇÕES (referência completa)

| Problema | Por que acontece | Como resolver |
|---|---|---|
| Descasamento feature × label | Features cross-section, label absoluto → label dominado pela direção do mercado | Label cross-section (superar a mediana) |
| Classes desbalanceadas | Em bull market quase tudo sobe | Resolvido de graça pelo label cross-section (~50/50) |
| Range restrito no treino | Pré-filtrar o universo antes de treinar atenua coeficientes | Treinar no universo amplo; filtrar só na carteira |
| Redundância filtro/modelo | GSVI usado para pré-filtrar e como feature | Filtro vira feature + interação, sem pré-corte |
| Look-ahead no label | Usar retorno do período atual | `shift(-1)` + mediana contemporânea; revisão obrigatória |
| Overfitting de backtest | Ajustar C/vol/N olhando o mesmo backtest | Holdout congelado avaliado uma vez |
| Overfitting do logit | Poucos dados independentes | L2 forte (C 0,01–0,1); só 3 features |
| N efetivo pequeno | Retornos intra-mês correlacionados | Modelo simples; não confiar no nº de linhas como N |
| Survivorship bias | yfinance só tem constituintes atuais | Documentar direção (otimista); buscar composição point-in-time se possível |
| GSVI normalização | Trends normaliza pela janela (pode vazar futuro) | Renormalizar só com dados até T; cache fixo |
| GSVI não-reprodutível | Trends é amostra, varia entre dias | Cache local; documentar como limitação |
| Termo de busca ambíguo | Ticker = ruído; nome = contaminado | Testar ambos no notebook, escolher por IC |
| Vol scaling alavanca | Fator > 1 quando vol baixa | Cap em 1.0; caixa não-alocado em Selic |
| Crash de momentum | Rebounds destroem momentum long-only | Vol scaling reduz exposição quando vol sobe |
| Alto turnover | Muitas trocas → custos corroem alpha | Buffer zone com margem mínima |
| API Trends bloqueada | Muitas requisições | `time.sleep(2-5s)` + cache |
| Menos de 10 ações no filtro | Universo pequeno em certos meses | Alocar nas disponíveis; resto em Selic |

---

## 5. CRONOGRAMA SUGERIDO (8 SEMANAS)

| Semana | Foco | Arquivo principal | Risco da semana |
|---|---|---|---|
| 1 | Reunião + setup + definir data do holdout | `config.py`, `requirements.txt` | — |
| 2 | Pipeline de dados: preços (universo completo) e GSVI | `data/` | Bloqueio/normalização Trends |
| 3 | Sinais + interação + label cross-section | `features.py`, `labels.py` | Look-ahead no label |
| 4 | Logit no universo amplo, walk-forward | `model.py` | Overfitting |
| 5 | Carteira + guarda de sanidade + pesos | `portfolio.py` | Edge cases (<10 ações) |
| 6 | Risco (cap 1.0) + backtest completo | `risk.py`, `backtest.py` | Custos subestimados |
| 7 | Métricas + ablation (sem tocar no holdout) | `metrics.py`, notebooks | Overfitting pós-análise |
| 8 | Avaliação única no holdout + documentação | Todos os notebooks | — |

---

## 6. STACK E DEPENDÊNCIAS

```
Python        3.11+
pandas        2.x
numpy         1.26+
scikit-learn  1.4+
yfinance      0.2+
pytrends      4.9+
matplotlib    3.x
seaborn       0.13+
scipy         1.12+
pyarrow       15+     (para .parquet)
requests      2.x     (para API BACEN)
```

---

## 7. AJUSTE DE JANELA DEV/HOLDOUT (2026-07-10)

Ao rodar o pipeline de dados pela primeira vez, descobrimos uma restrição real não prevista no
planejamento original: o `pytrends` só entrega granularidade **semanal** para janelas de até
5 anos (`timeframe="today 5-y"`). Acima disso a resolução cai para mensal, que não serve para
calcular `gsvi_score` (delta sobre média de 4 semanas).

Isso significa que, mesmo com 7 anos de preços disponíveis (2019-07 a 2026-07), o painel de
features (que depende de GSVI) só começa em **2021-08**. A interseção útil entre
`features.parquet` e `labels.parquet` é de **59 meses** (2021-08 a 2026-06), bem menor que o
exemplo do PDF v3 original (~66 meses só de desenvolvimento, Jan/2018–Jun/2023).

**Impacto nos parâmetros originais (`MIN_TRAIN_MONTHS=24`, `HOLDOUT_MONTHS=15`):** sobrariam
apenas ~20 meses de walk-forward real no desenvolvimento (59 − 15 holdout − 24 warm-up). N
efetivo pequeno para ajustar hiperparâmetros com segurança.

**Opções avaliadas:**

| Opção | `MIN_TRAIN_MONTHS` | `HOLDOUT_MONTHS` | Meses de walk-forward no dev |
|---|---|---|---|
| A (original) | 24 | 15 | 20 |
| B | 24 | 12 | 23 |
| C | 18 | 15 | 26 |
| D (escolhida) | 18 | 12 | 29 |
| E (não implementada) | — | — | ~53 (esticaria GSVI ate 2019 via stitching de janelas do pytrends) |

**Decisão:** Opção D. Como o treino é walk-forward com janela expandindo, reduzir
`MIN_TRAIN_MONTHS` só afeta a estabilidade das primeiras previsões do período de
desenvolvimento — a maior parte do período já treina com bem mais que 18 meses de histórico.
`HOLDOUT_MONTHS=12` continua dentro da faixa 12–18 que o próprio PDF v3 aceita. Decisão
totalmente reversível (é só alterar `config.py`, sem mudança de lógica); voltar para a Opção A
se os resultados do desenvolvimento parecerem instáveis por causa do N pequeno.

**Opção E (esticar o GSVI para trás de 2021 até 2019)** foi cogitada e descartada por ora: exigiria
consultar múltiplas janelas de 5 anos do `pytrends` ancoradas em pontos diferentes do passado e
costurá-las com uma técnica de "scaling bridge" (usar o trecho de sobreposição entre duas
janelas para reescalar uma para a base da outra, sem usar pontos futuros). Triplicaria o volume
de chamadas ao `pytrends`, agravando o rate-limiting já observado (ver histórico do projeto —
36 de 99 tickers falharam com erro 429 na primeira tentativa de download do GSVI, recuperados
com retry e backoff exponencial). Fica como possível melhoria futura se a Opção D se mostrar
insuficiente.

**Atualização (2026-07-10, mesmo dia):** os 3 tickers pendentes (`CXSE3`, `CEAB3`, `DIRR3`)
foram recuperados numa nova tentativa — o bloqueio do Google era temporário, como esperado.
Universo de features completo em 99 tickers (98 após o filtro de warm-up de 12 meses; `AUAU3`
continua fora por ter menos de 12 meses de histórico de preço desde o reaproveitamento do
ticker após a fusão com a Cobasi — não relacionado ao GSVI). As métricas da seção 8 abaixo já
refletem esse universo completo.

---

## 8. ABLATION E VARREDURA DE HIPERPARÂMETROS (2026-07-10)

A Fase 7 (`metrics.py`) implementou o teste de ablation previsto na Parte 0 da estratégia:
comparar (i) estratégia completa (logit), (ii) combinação fixa 50/50 dos z-scores sem o logit,
(iii) Ibovespa. Primeira rodada, com 95 tickers (3 ainda sem GSVI por rate-limit — ver seção 7):

| Estratégia | Retorno a.a. | Sharpe (exc. Selic) |
|---|---|---|
| Sem logit (50/50 fixo) | 12,4% | 0,08 |
| Completa (logit) | 11,1% | 0,01 |
| Ibovespa | 10,3% | -0,04 |

Após recuperar os 3 tickers pendentes (seção 7), a rodada final com o universo completo
(98 tickers) melhorou os números em bloco, mas **manteve a mesma conclusão**:

| Estratégia | Retorno a.a. | Sharpe (exc. Selic) |
|---|---|---|
| Sem logit (50/50 fixo) | 17,7% | 0,39 |
| Completa (logit) | 16,1% | 0,30 |
| Ibovespa | 10,3% | -0,04 |

(Max Drawdown da estratégia completa também melhorou: -11,1% → -7,2%, recuperado em 1 mês.)

**A estratégia completa não supera a combinação fixa, com ou sem os 3 tickers.** Antes de aceitar
esse resultado, rodamos
uma varredura de hiperparâmetros (20 combinações: `LOGIT_C` ∈ {0,01, 0,03, 0,1, 0,3, 1,0} ×
`WEIGHT_SCHEME` ∈ {equal, proportional} × `APPLY_SANITY_FILTER` ∈ {True, False}), toda dentro do
período de desenvolvimento (regra 5.7 — holdout não tocado):

- **Em nenhuma das 20 combinações a estratégia completa superou a combinação fixa 50/50.** O
  AUC-ROC fica travado em ~0,50–0,501 independente de `LOGIT_C` — não é um problema de
  regularização mal ajustada, é ausência de sinal explorável nessa janela (consistente com os
  ICs individuais fracos documentados nos notebooks 02/03).
- `APPLY_SANITY_FILTER=True` é confirmado benéfico em todas as combinações — mantido como
  default.
- **Achado colateral:** `WEIGHT_SCHEME='proportional'` melhora consistentemente os dois cenários
  (o baseline sem logit sobe de 12,4% para 16,8% a.a., Sharpe de 0,08 para 0,32). Ainda assim, o
  logit não supera essa versão mais forte do baseline (11,4% vs. 16,8% a.a.) — a lacuna fica
  ainda mais evidente. Durante essa varredura foi encontrado e corrigido um bug latente em
  `portfolio.py`: peso proporcional podia ficar negativo quando o "score" de entrada é uma
  combinação de z-scores (não necessariamente ≥ 0), diferente da probabilidade do logit (sempre
  em [0,1]). Nunca se manifestou nos 29 meses reais (a guarda de sanidade já filtra a maior parte
  do risco), mas foi corrigido por precaução.

**Decisão (2026-07-10):** manter `WEIGHT_SCHEME='equal'` como default — mais simples e robusto
de defender, mesmo com retorno um pouco menor no backtest de desenvolvimento — e **aceitar o
resultado do ablation como conclusão do desenvolvimento**: o logit não está agregando valor
sobre uma combinação ingênua nesta janela, e isso não é um bug nem um problema de tuning. O
ablation cumpriu exatamente o papel para o qual foi desenhado (Parte 0 do planejamento v3):
expor isso honestamente em vez de mascarar. Próximo passo: avaliação única no holdout congelado
(Semana 8).

---

## 9. REFERÊNCIAS ACADÊMICAS

- Jegadeesh, N. & Titman, S. (1993). *Returns to Buying Winners and Losers*. Journal of Finance, 48(1), 65-91.
- Novy-Marx, R. (2012). *Is momentum really momentum?* Journal of Financial Economics, 103(3), 429-453.
- Daniel, K. & Moskowitz, T. J. (2016). *Momentum Crashes*. Journal of Financial Economics, 122(2), 221-247.
- Barroso, P. & Santa-Clara, P. (2015). *Momentum has its moments*. Journal of Financial Economics, 116(1), 111-120.
- Da, Z., Engelberg, J. & Gao, P. (2011). *In Search of Attention*. Journal of Finance, 66(5), 1461-1499. (Justifica o uso de ticker como termo de busca.)
- Bailey, D. & López de Prado, M. (2014). *The Deflated Sharpe Ratio*. Journal of Portfolio Management, 40(5), 94-107. (Fundamenta o holdout congelado contra overfitting de backtest.)
- Wolff, D. et al. (2024). *Stock picking with machine learning*. Journal of Forecasting — momentum (1, 6 e 12 meses) como feature mais importante em modelos logit de seleção de ações.
