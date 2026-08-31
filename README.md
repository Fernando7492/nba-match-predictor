# NBA Match Predictor

Sistema preditivo para desfechos de partidas da NBA utilizando Redes Neurais Artificiais (RNA) e Aprendizado Profundo (Deep Learning), desenvolvido para a disciplina de Redes Neurais da UFAPE.

---

## 🏀 Visão Geral

O projeto avalia três famílias de modelos sobre dados reais de **10 temporadas da NBA (2014–2024)**, totalizando **23.958 registros de equipe e 11.979 partidas reais**:
1. **Linhas de Base (Baselines):** Mando de Campo (*Home Court*), Regressão Logística $L_2$ e *Random Forest*.
2. **RNA Clássica:** *Multi-Layer Perceptron* (MLP) de 2 camadas ocultas com *Dropout*.
3. **Deep Learning:**
   - *Deep ResNet MLP:* Arquitetura profunda com blocos residuais (*Skip Connections*), *Batch Normalization* e ativação *Mish*.
   - *Dual-Branch Bidirectional LSTM:* Rede neural recorrente para séries temporais dos últimos 10 confrontos de cada time.
   - *Temporal Matchup Transformer:* Rede baseada em mecanismo de auto-atenção multicabeça (*Multi-Head Self-Attention*).
4. **Deep Ensemble:** Comitê probabilístico ponderado por otimização simplex de minimização de Brier Score.

---

## 📊 Resultados Experimentais (Teste Cego - Temporada 2023-24 / 1.230 Jogos)

| Modelo | Família | Acurácia | F1-Score | ROC-AUC | Brier Score $\downarrow$ | Log-Loss $\downarrow$ | ECE $\downarrow$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Mando de Campo)** | Baseline | 54,31% | 0,7039 | 0,5000 | 0,2489 | 0,6909 | 0,0271 |
| **Regressão Logística** | Baseline | 63,82% | 0,6810 | 0,6896 | 0,2215 | 0,6335 | **0,0195** |
| **Random Forest** | Baseline | 63,33% | 0,6938 | 0,6890 | 0,2221 | 0,6348 | 0,0259 |
| **RNA Clássica (MLP)** | RNA | 64,31% | 0,7016 | 0,6945 | **0,2198** | **0,6294** | 0,0211 |
| **Deep ResNet MLP** | Deep Learning | 61,71% | 0,6963 | 0,6909 | 0,2222 | 0,6344 | 0,0557 |
| **Dual-Branch LSTM** | Deep Learning | 63,58% | 0,6805 | 0,6941 | 0,2216 | 0,6330 | 0,0281 |
| **Matchup Transformer** | Deep Learning | **64,39%** | 0,6975 | 0,6886 | 0,2233 | 0,6377 | 0,0372 |
| **Deep Ensemble** | Ensemble | 63,66% | **0,7018** | **0,6967** | 0,2200 | 0,6300 | 0,0316 |

---

## 🛠️ Instalação e Execução

### 1. Criar e Ativar Ambiente Virtual
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Executar a Suíte Completa de Testes Automatizados (36 Testes)
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
│   ├── data/           # Coletor (10 temporadas), pré-processador e sequências
│   ├── models/         # Baselines, MLP, ResNet, Bi-LSTM, Transformer, Ensemble
│   ├── training/       # Trainers com Early Stopping, Schedulers e Losses
│   ├── evaluation/     # Métricas (ECE, Brier, ROC-AUC), visualizador e benchmark
│   └── utils/          # Configurações de caminhos e sementes fixas
├── tests/              # 36 testes unitários e de integração
├── outputs/            # Checkpoints dos modelos (.pt) e figuras (.png)
├── requirements.txt    # Dependências do projeto
└── README.md
```
