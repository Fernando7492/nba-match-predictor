# NBA Match Predictor

Sistema preditivo para desfechos de partidas da NBA utilizando Redes Neurais Artificiais (RNA) e Aprendizado Profundo (Deep Learning), desenvolvido para a disciplina de Redes Neurais da UFAPE.

---

## 🏀 Visão Geral

O projeto avalia três famílias de modelos sobre dados reais de **10 temporadas da NBA (2014–2024)**, totalizando **23.958 registros de equipe e 11.979 partidas reais**:
1. **Linhas de Base (Baselines):** Mando de Campo (*Home Court*), Regressão Logística $L_2$ e *Random Forest*.
2. **RNA Clássica:** *Multi-Layer Perceptron* (MLP) de 2 camadas ocultas com *Dropout* e inicialização Kaiming.
3. **Deep Learning:**
   - *Deep ResNet MLP:* Arquitetura profunda com blocos residuais (*Skip Connections*), *Batch Normalization*, ativação *Mish* e precisão mista AMP.
   - *Dual-Branch Bidirectional LSTM:* Rede neural recorrente para séries temporais dos últimos 10 confrontos de cada time, com concatenação bidirecional correta e inicialização ortogonal.
   - *Temporal Matchup Transformer:* Rede baseada em mecanismo de auto-atenção multicabeça (*Multi-Head Self-Attention*) com projeção escalada por $\sqrt{d_{model}}$.
4. **Deep Ensemble:** Comitê probabilístico ponderado por otimização simplex de minimização de Brier Score com fallback de segurança.

---

## 📊 Resultados Experimentais (Teste Cego - Temporada 2023-24 / 1.230 Jogos)

| Modelo | Família | Acurácia | F1-Score | ROC-AUC | Brier Score $\downarrow$ | Log-Loss $\downarrow$ | ECE $\downarrow$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Mando de Campo)** | Baseline | 54,31% | 0,7039 | 0,5000 | 0,2489 | 0,6909 | 0,0271 |
| **Regressão Logística** | Baseline | 63,82% | 0,6810 | 0,6896 | 0,2215 | 0,6335 | **0,0195** |
| **Random Forest** | Baseline | 63,33% | 0,6938 | 0,6890 | 0,2221 | 0,6348 | 0,0259 |
| **RNA Clássica (MLP)** | RNA | 63,25% | 0,6929 | 0,6905 | 0,2213 | 0,6322 | 0,0248 |
| **Deep ResNet MLP** | Deep Learning | **64,47%** | **0,6988** | **0,6980** | **0,2189** | **0,6274** | 0,0235 |
| **Dual-Branch LSTM** | Deep Learning | 63,50% | 0,6758 | 0,6731 | 0,2267 | 0,6454 | 0,0298 |
| **Matchup Transformer** | Deep Learning | 63,82% | 0,6828 | 0,6780 | 0,2245 | 0,6404 | 0,0234 |
| **Deep Ensemble** | Ensemble | 63,58% | 0,6932 | 0,6925 | 0,2210 | 0,6324 | 0,0273 |

---

## 🛠️ Instalação e Execução

### 1. Criar e Ativar Ambiente Virtual
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Executar a Suíte Completa de Testes Automatizados (41 Testes)
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
│   ├── data/           # Coletor (10 temporadas), pré-processador e sequências vetorizadas
│   ├── models/         # Baselines, MLP, ResNet, Bi-LSTM, Transformer, Ensemble
│   ├── training/       # BaseTrainer unificado com AMP, Schedulers e Losses
│   ├── evaluation/     # Métricas (ECE, Brier, ROC-AUC), visualizador e benchmark modular
│   └── utils/          # Configurações de caminhos dinâmicos e sementes fixas
├── tests/              # 41 testes unitários, integração e testes negativos
├── outputs/            # Checkpoints dos modelos (.pt) e figuras (.png)
├── requirements.txt    # Dependências do projeto
└── README.md
```
