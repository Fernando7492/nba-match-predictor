# 🏀 NBA Match Predictor

Sistema preditivo de alta precisão para resultados de partidas da NBA utilizando **Redes Neurais Artificiais (RNA)**, **Aprendizado Profundo (*Deep Learning*)**, **Ratings Dinâmicos (Elo MOV)** e **Métricas Avançadas de Posse de Bola**.

Desenvolvido para a disciplina de Redes Neurais da Universidade Federal do Agreste de Pernambuco (UFAPE).

---

## 📌 Destaques do Projeto

* **Base de Dados Completa (10 Temporadas / 2014–2024):** 23.958 registros de equipe e 11.979 partidas reais coletadas via `nba_api`.
* **Zero Data Leakage:** Defasagem temporal estrita (*Lag-1*) e janelas móveis (3, 7 e 14 jogos) impedindo qualquer vazamento de dados futuros.
* **Engenharia de Atributos de Domínio:**
  * *Elo Rating Dinâmico (Margem de Vitória - MOV / FiveThirtyEight)*.
  * *Quatro Fatores de Dean Oliver ($eFG\%$, $TOV\%$, $OREB\%$, $FT\text{ Rate}$)*.
  * *Eficiência de Arremesso Real ($True\text{ }Shooting\text{ }\%$ e $AST/TOV$)*.
  * *Qualidade de Defesa de Arremesso Cedida e Densidade de Calendário / Fadiga*.
* **9 Modelos no Benchmark:** Baselines Estatísticos, RNA Clássica (MLP), Deep ResNet (Residual), Dual-Branch LSTM, Matchup Transformer, Rede Híbrida de Atenção Cruzada (*Cross-Attention*) e Deep Ensemble Ponderado.
* **Suíte de Testes Automatizados:** 52 testes no Pytest com 100% de aprovação.

---

## 📊 Resultados no Teste Cego (Temporada 2023-24 / 1.230 Jogos)

| Modelo | Família | Acurácia | F1-Score | ROC-AUC | Brier Score $\downarrow$ | Log-Loss $\downarrow$ | ECE $\downarrow$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Mando de Campo)** | Baseline | 54,31% | 0,7039 | 0,5000 | 0,2489 | 0,6909 | 0,0271 |
| **Regressão Logística** | Baseline | 64,96% | 0,6870 | 0,7037 | 0,2177 | 0,6256 | 0,0340 |
| **Random Forest** | Baseline | 63,33% | 0,6917 | 0,7117 | 0,2149 | 0,6185 | 0,0675 |
| **RNA Clássica (MLP)** | RNA | 64,47% | **0,7001** | 0,6950 | 0,2196 | 0,6285 | **0,0140** |
| **Deep ResNet MLP** | Deep Learning | 63,90% | 0,6838 | 0,7067 | 0,2156 | 0,6197 | 0,0391 |
| **Dual-Branch LSTM** | Deep Learning | 64,15% | 0,6764 | 0,6843 | 0,2236 | 0,6383 | 0,0281 |
| **Matchup Transformer** | Deep Learning | 62,68% | 0,6792 | 0,6668 | 0,2283 | 0,6488 | 0,0237 |
| **Hybrid Cross-Attention** | Deep Learning | **65,12%** | 0,6994 | 0,6933 | 0,2214 | 0,6335 | 0,0410 |
| **Deep Ensemble** | Ensemble | 64,31% | 0,6954 | **0,7143** | **0,2140** | **0,6161** | 0,0439 |

> 🏆 **Destaques:** A rede **Hybrid Cross-Attention** liderou em acurácia (**65,12%**), a **RNA Clássica** obteve a melhor calibração de probabilidades (**ECE de 0,0140** e F1 de **0,7001**), e o **Deep Ensemble** alcançou o maior poder discriminativo (**ROC-AUC de 0,7143** e menor Brier Score **0,2140**).

---

## 🚀 Como Executar

### 1. Configurar o Ambiente Virtual
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Rodar a Suíte de Testes Automatizados (52 Testes)
```bash
pytest tests/ -v
```

### 3. Rodar o Treinamento e Benchmark Completo
```bash
PYTHONPATH=. python -m src.evaluation.benchmark
```

Os modelos (.pt) e gráficos comparativos em alta resolução (.png) são salvos na pasta `outputs/`.

---

## 📂 Estrutura do Repositório

```
nba-match-predictor/
├── src/
│   ├── data/           # Coletor (10 anos), pré-processador, Elo MOV, Oliver e sequências
│   ├── models/         # Baselines, MLP, ResNet, Bi-LSTM, Transformer, Cross-Attention, Ensemble
│   ├── training/       # BaseTrainer unificado com Mixed Precision (AMP / FP16) e Early Stopping
│   ├── evaluation/     # Métricas (ECE, Brier, ROC-AUC), Threshold Tuner e Visualizador
│   └── utils/          # Configurações dinâmicas e sementes determinísticas
├── tests/              # 52 testes automatizados (unitários, integração e testes negativos)
├── outputs/            # Checkpoints (.pt), métricas (.csv, .json) e gráficos (.png)
├── requirements.txt    # Dependências do projeto
└── README.md
```
