# NBA Match Predictor

Sistema preditivo para desfechos de partidas da NBA utilizando Redes Neurais Artificiais (RNA), Aprendizado Profundo (Deep Learning), Métricas Avançadas de Posse e Elo Rating, desenvolvido para a disciplina de Redes Neurais da UFAPE.

---

## 🏀 Visão Geral

O projeto avalia três famílias de modelos sobre dados reais de **10 temporadas da NBA (2014–2024)**, totalizando **23.958 registros de equipe e 11.979 partidas reais**:
1. **Engenharia de Atributos Avançada:**
   - **Elo Rating Dinâmico com Margem de Vitória (MOV):** Formulação *FiveThirtyEight* com regressão à média sazonal.
   - **Quatro Fatores de Dean Oliver:** $eFG\%$, $TOV\%$, $OREB\%$, $FT\text{ Rate}$.
   - **Métricas de Posse:** $Pace$, *Offensive & Defensive Rating* (por 100 posses) e *Net Rating*.
   - **Histórico de Confronto Direto (*Head-to-Head*):** Taxa de vitórias e saldo de pontos entre os dois times específicos.
2. **Modelos Avaliados (9 Arquiteturas):**
   - **Linhas de Base (Baselines):** Mando de Campo (*Home Court*), Regressão Logística $L_2$ e *Random Forest*.
   - **RNA Clássica:** *Multi-Layer Perceptron* (MLP) de 2 camadas ocultas com *Dropout* e inicialização Kaiming.
   - **Deep Learning:**
     - *Deep ResNet MLP:* Arquitetura profunda com blocos residuais (*Skip Connections*), *Batch Normalization*, ativação *Mish* e precisão mista AMP.
     - *Dual-Branch Bidirectional LSTM:* Rede neural recorrente para séries temporais dos últimos 10 confrontos de cada time, com concatenação bidirecional correta e inicialização ortogonal.
     - *Temporal Matchup Transformer:* Rede baseada em mecanismo de auto-atenção multicabeça (*Multi-Head Self-Attention*) com projeção escalada por $\sqrt{d_{model}}$.
     - *Hybrid Cross-Attention Fusion Net:* Rede neural multimodal fim-a-fim onde o vetor tabular atua como Consulta (*Query*) e as sequências temporais atuam como Chave/Valor (*Key/Value*).
   - **Deep Ensemble:** Comitê probabilístico ponderado por otimização simplex de minimização de Brier Score e limiar ótimo (*Optimal Threshold Tuning*).

---

## 📊 Resultados Experimentais (Teste Cego - Temporada 2023-24 / 1.230 Jogos)

| Modelo | Família | Acurácia | F1-Score | ROC-AUC | Brier Score $\downarrow$ | Log-Loss $\downarrow$ | ECE $\downarrow$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Mando de Campo)** | Baseline | 54,31% | 0,7039 | 0,5000 | 0,2489 | 0,6909 | 0,0271 |
| **Regressão Logística** | Baseline | 64,55% | 0,6841 | 0,7059 | 0,2164 | 0,6223 | 0,0261 |
| **Random Forest** | Baseline | 64,15% | 0,6973 | 0,7072 | 0,2166 | 0,6226 | 0,0496 |
| **RNA Clássica (MLP)** | RNA | **65,04%** | **0,6993** | 0,6850 | 0,2230 | 0,6381 | 0,0371 |
| **Deep ResNet MLP** | Deep Learning | 63,33% | 0,6979 | 0,6960 | 0,2201 | 0,6304 | 0,0386 |
| **Dual-Branch LSTM** | Deep Learning | 64,15% | 0,6961 | 0,6847 | 0,2240 | 0,6390 | 0,0284 |
| **Matchup Transformer** | Deep Learning | 62,44% | 0,6866 | 0,6689 | 0,2273 | 0,6463 | **0,0215** |
| **Hybrid Cross-Attention** | Deep Learning | 61,71% | 0,6879 | 0,6701 | 0,2272 | 0,6464 | 0,0268 |
| **Deep Ensemble** | Ensemble | 63,98% | 0,6913 | **0,7098** | **0,2153** | **0,6197** | 0,0484 |

---

## 🛠️ Instalação e Execução

### 1. Criar e Ativar Ambiente Virtual
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Executar a Suíte Completa de Testes Automatizados (51 Testes)
```bash
pytest tests/ -v
```

### 3. Executar o Benchmark Completo com GPU
```bash
PYTHONPATH=. python -m src.evaluation.benchmark
```

Os checkpoints dos modelos treinados (.pt) e figuras em alta resolução (.png) são salvos em `outputs/`.

---

## 📁 Estrutura do Repositório

```
nba-match-predictor/
├── src/
│   ├── data/           # Coletor (10 temporadas), pré-processador, Elo MOV, Oliver 4 Factors e sequências
│   ├── models/         # Baselines, MLP, ResNet, Bi-LSTM, Transformer, Cross-Attention, Ensemble
│   ├── training/       # BaseTrainer unificado com AMP, Schedulers e Losses
│   ├── evaluation/     # Métricas (ECE, Brier, ROC-AUC), Threshold Tuner, visualizador e benchmark modular
│   └── utils/          # Configurações de caminhos dinâmicos e sementes fixas
├── tests/              # 51 testes unitários, integração e testes de borda
├── outputs/            # Checkpoints dos modelos (.pt) e figuras (.png)
├── requirements.txt    # Dependências do projeto
└── README.md
```
