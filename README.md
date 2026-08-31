# NBA Match Predictor

Sistema preditivo para desfechos de partidas da NBA utilizando Redes Neurais Artificiais (RNA) e Aprendizado Profundo (Deep Learning), desenvolvido para a disciplina de Redes Neurais da UFAPE.

---

## 🏀 Visão Geral

O projeto avalia três famílias de modelos sobre dados reais de 6 temporadas da NBA (2018–2024, totalizando 14.118 registros de equipe e 7.059 partidas):
1. **Linhas de Base (Baselines):** Mando de Campo (*Home Court*), Regressão Logística $L_2$ e *Random Forest*.
2. **RNA Clássica:** *Multi-Layer Perceptron* (MLP) de 2 camadas ocultas com *Dropout*.
3. **Deep Learning:** *Deep ResNet MLP* (blocos residuais com *Batch Normalization* e ativação *Mish*) e *Dual-Branch Bidirectional LSTM* (processamento de séries temporais dos últimos 10 confrontos de cada time).

---

## 📊 Resultados Experimentais (Teste Cego - Temporada 2023-24)

| Modelo | Família | Acurácia | F1-Score | ROC-AUC | Brier Score $\downarrow$ | Log-Loss $\downarrow$ | ECE $\downarrow$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Mando de Campo)** | Baseline | 54,31% | 0,7039 | 0,5000 | 0,2484 | 0,6899 | 0,0151 |
| **Regressão Logística** | Baseline | 63,58% | 0,6739 | 0,6830 | 0,2233 | 0,6371 | 0,0265 |
| **Random Forest** | Baseline | 63,90% | 0,6963 | 0,6919 | 0,2216 | 0,6339 | 0,0305 |
| **RNA Clássica (MLP)** | RNA | 63,90% | 0,6938 | **0,6926** | **0,2208** | **0,6319** | **0,0235** |
| **Deep ResNet MLP** | Deep Learning | 61,38% | 0,6835 | 0,6737 | 0,2269 | 0,6449 | 0,0377 |
| **Dual-Branch LSTM** | Deep Learning | **64,47%** | **0,6997** | 0,6905 | 0,2230 | 0,6369 | 0,0338 |

---

## 🛠️ Instalação e Execução

### 1. Criar e Ativar Ambiente Virtual
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Executar a Suíte de Testes Automatizados
```bash
pytest tests/ -v
```

### 3. Executar o Benchmark Completo com GPU
```bash
PYTHONPATH=. python -m src.evaluation.benchmark
```

Os resultados tabulares e figuras serão salvos em `outputs/`.

---

## 📁 Estrutura do Repositório

```
nba-match-predictor/
├── src/
│   ├── data/           # Coletor, pré-processador e datasets PyTorch
│   ├── models/         # Baselines, Classical MLP, Deep ResNet, Dual LSTM
│   ├── training/       # Loops de treino com Early Stopping e Schedulers
│   ├── evaluation/     # Métricas (ECE, Brier, ROC-AUC) e visualizadores
│   └── utils/          # Configurações de caminhos e sementes fixas
├── tests/              # 24 testes unitários e de integração
├── outputs/            # Checkpoints dos modelos (.pt) e figuras (.png)
├── requirements.txt    # Dependências do projeto
└── README.md
```
