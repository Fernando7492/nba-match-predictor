from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.utils.config import ProjectPaths, ModelConfig
from src.utils.seed import set_seed
from src.data.collector import NBADataCollector
from src.data.preprocessor import NBAPreprocessor
from src.data.dataset import get_tabular_loaders
from src.data.sequence_pipeline import NBASequencePipeline
from src.models.baseline import HomeCourtBaseline, LogisticRegressionBaseline, RandomForestBaseline
from src.models.mlp import ClassicalMLP
from src.models.deep_mlp import DeepResNetMLP
from src.models.recurrent import DualBranchLSTM
from src.models.transformer import MatchupTransformer
from src.models.ensemble import DeepEnsemblePredictor
from src.training.trainer import ModelTrainer
from src.training.sequence_trainer import SequenceModelTrainer
from src.evaluation.metrics import compute_all_metrics
from src.evaluation.visualizer import (
    plot_learning_curves,
    plot_confusion_matrices,
    plot_roc_curves,
    plot_calibration_curves,
    plot_model_comparison
)

def run_full_benchmark(paths: ProjectPaths | None = None, config: ModelConfig | None = None) -> pd.DataFrame:
    paths = paths or ProjectPaths()
    config = config or ModelConfig()
    set_seed(config.seed)

    collector = NBADataCollector(paths=paths)
    raw_df = collector.collect_all_seasons()

    preprocessor = NBAPreprocessor(paths=paths, rolling_windows=config.rolling_windows)
    train_df, val_df, test_df = preprocessor.process_and_split(raw_df)

    valid_game_ids = set(train_df["GAME_ID"]).union(set(val_df["GAME_ID"])).union(set(test_df["GAME_ID"]))

    feature_cols = preprocessor.feature_columns
    train_loader, val_loader, test_loader = get_tabular_loaders(
        train_df, val_df, test_df, feature_cols, batch_size=config.batch_size
    )

    eval_train_loader, eval_val_loader, eval_test_loader = get_tabular_loaders(
        train_df, val_df, test_df, feature_cols, batch_size=config.batch_size
    )

    seq_pipeline = NBASequencePipeline(paths=paths, sequence_length=config.sequence_length)
    seq_train_ds, seq_val_ds, seq_test_ds = seq_pipeline.build_sequences(raw_df, valid_game_ids=valid_game_ids)

    seq_train_loader = DataLoader(seq_train_ds, batch_size=config.batch_size, shuffle=True)
    seq_val_loader = DataLoader(seq_val_ds, batch_size=config.batch_size, shuffle=False)
    seq_test_loader = DataLoader(seq_test_ds, batch_size=config.batch_size, shuffle=False)

    x_train_tab = train_df[feature_cols].values
    y_train_tab = train_df["TARGET_HOME_W"].values
    x_val_tab = val_df[feature_cols].values
    y_val_tab = val_df["TARGET_HOME_W"].values
    x_test_tab = test_df[feature_cols].values
    y_test_tab = test_df["TARGET_HOME_W"].values

    models_val_probas: dict[str, np.ndarray] = {}
    models_test_probas: dict[str, np.ndarray] = {}
    histories: dict[str, dict[str, list[float]]] = {}
    results: list[dict] = []

    criterion = nn.BCEWithLogitsLoss()

    home_baseline = HomeCourtBaseline().fit(x_train_tab, y_train_tab)
    p_home_val = home_baseline.predict_proba(x_val_tab)[:, 1]
    p_home_test = home_baseline.predict_proba(x_test_tab)[:, 1]
    models_val_probas["Baseline (Mando)"] = p_home_val
    models_test_probas["Baseline (Mando)"] = p_home_test
    m_home = compute_all_metrics(y_test_tab, p_home_test)
    m_home["model"] = "Baseline (Mando)"
    m_home["family"] = "Baseline"
    results.append(m_home)

    lr_baseline = LogisticRegressionBaseline(random_state=config.seed).fit(x_train_tab, y_train_tab)
    p_lr_val = lr_baseline.predict_proba(x_val_tab)[:, 1]
    p_lr_test = lr_baseline.predict_proba(x_test_tab)[:, 1]
    models_val_probas["Regressão Logística"] = p_lr_val
    models_test_probas["Regressão Logística"] = p_lr_test
    m_lr = compute_all_metrics(y_test_tab, p_lr_test)
    m_lr["model"] = "Regressão Logística"
    m_lr["family"] = "Baseline"
    results.append(m_lr)

    rf_baseline = RandomForestBaseline(random_state=config.seed).fit(x_train_tab, y_train_tab)
    p_rf_val = rf_baseline.predict_proba(x_val_tab)[:, 1]
    p_rf_test = rf_baseline.predict_proba(x_test_tab)[:, 1]
    models_val_probas["Random Forest"] = p_rf_val
    models_test_probas["Random Forest"] = p_rf_test
    m_rf = compute_all_metrics(y_test_tab, p_rf_test)
    m_rf["model"] = "Random Forest"
    m_rf["family"] = "Baseline"
    results.append(m_rf)

    input_dim = len(feature_cols)
    mlp_model = ClassicalMLP(input_dim=input_dim, hidden_dim_1=64, hidden_dim_2=32, dropout=0.2)
    mlp_opt = torch.optim.AdamW(mlp_model.parameters(), lr=1e-3, weight_decay=1e-4)
    mlp_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(mlp_opt, mode="min", factor=0.5, patience=4)
    mlp_trainer = ModelTrainer(
        model=mlp_model,
        optimizer=mlp_opt,
        criterion=criterion,
        scheduler=mlp_sched,
        patience=config.early_stopping_patience,
        save_path=paths.outputs_models / "classical_mlp.pt",
        device=config.device
    )
    hist_mlp = mlp_trainer.fit(train_loader, val_loader, epochs=config.epochs)
    histories["RNA Clássica (MLP)"] = hist_mlp
    p_mlp_val = mlp_trainer.predict_proba(eval_val_loader).numpy().ravel()
    p_mlp_test = mlp_trainer.predict_proba(eval_test_loader).numpy().ravel()
    models_val_probas["RNA Clássica (MLP)"] = p_mlp_val
    models_test_probas["RNA Clássica (MLP)"] = p_mlp_test
    m_mlp = compute_all_metrics(y_test_tab, p_mlp_test)
    m_mlp["model"] = "RNA Clássica (MLP)"
    m_mlp["family"] = "RNA"
    results.append(m_mlp)

    resnet_model = DeepResNetMLP(input_dim=input_dim, hidden_dim=128, num_blocks=3, dropout=0.25)
    resnet_opt = torch.optim.AdamW(resnet_model.parameters(), lr=1e-3, weight_decay=1e-4)
    resnet_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(resnet_opt, mode="min", factor=0.5, patience=4)
    resnet_trainer = ModelTrainer(
        model=resnet_model,
        optimizer=resnet_opt,
        criterion=criterion,
        scheduler=resnet_sched,
        patience=config.early_stopping_patience,
        save_path=paths.outputs_models / "deep_resnet.pt",
        device=config.device
    )
    hist_resnet = resnet_trainer.fit(train_loader, val_loader, epochs=config.epochs)
    histories["Deep ResNet MLP"] = hist_resnet
    p_resnet_val = resnet_trainer.predict_proba(eval_val_loader).numpy().ravel()
    p_resnet_test = resnet_trainer.predict_proba(eval_test_loader).numpy().ravel()
    models_val_probas["Deep ResNet MLP"] = p_resnet_val
    models_test_probas["Deep ResNet MLP"] = p_resnet_test
    m_resnet = compute_all_metrics(y_test_tab, p_resnet_test)
    m_resnet["model"] = "Deep ResNet MLP"
    m_resnet["family"] = "Deep Learning"
    results.append(m_resnet)

    seq_stat_dim = len(seq_pipeline.stat_cols)
    lstm_model = DualBranchLSTM(input_dim=seq_stat_dim, hidden_dim=64, num_layers=2, dropout=0.2)
    lstm_opt = torch.optim.AdamW(lstm_model.parameters(), lr=1e-3, weight_decay=1e-4)
    lstm_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(lstm_opt, mode="min", factor=0.5, patience=4)
    lstm_trainer = SequenceModelTrainer(
        model=lstm_model,
        optimizer=lstm_opt,
        criterion=criterion,
        scheduler=lstm_sched,
        patience=config.early_stopping_patience,
        save_path=paths.outputs_models / "dual_branch_lstm.pt",
        device=config.device
    )
    hist_lstm = lstm_trainer.fit(seq_train_loader, seq_val_loader, epochs=config.epochs)
    histories["Dual-Branch LSTM"] = hist_lstm
    p_lstm_val = lstm_trainer.predict_proba(seq_val_loader).numpy().ravel()
    p_lstm_test = lstm_trainer.predict_proba(seq_test_loader).numpy().ravel()
    models_val_probas["Dual-Branch LSTM"] = p_lstm_val
    models_test_probas["Dual-Branch LSTM"] = p_lstm_test
    m_lstm = compute_all_metrics(y_test_tab, p_lstm_test)
    m_lstm["model"] = "Dual-Branch LSTM"
    m_lstm["family"] = "Deep Learning"
    results.append(m_lstm)

    trans_model = MatchupTransformer(input_dim=seq_stat_dim, d_model=64, nhead=4, num_layers=2, dropout=0.2)
    trans_opt = torch.optim.AdamW(trans_model.parameters(), lr=5e-4, weight_decay=1e-4)
    trans_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(trans_opt, mode="min", factor=0.5, patience=4)
    trans_trainer = SequenceModelTrainer(
        model=trans_model,
        optimizer=trans_opt,
        criterion=criterion,
        scheduler=trans_sched,
        patience=config.early_stopping_patience,
        save_path=paths.outputs_models / "matchup_transformer.pt",
        device=config.device
    )
    hist_trans = trans_trainer.fit(seq_train_loader, seq_val_loader, epochs=config.epochs)
    histories["Matchup Transformer"] = hist_trans
    p_trans_val = trans_trainer.predict_proba(seq_val_loader).numpy().ravel()
    p_trans_test = trans_trainer.predict_proba(seq_test_loader).numpy().ravel()
    models_val_probas["Matchup Transformer"] = p_trans_val
    models_test_probas["Matchup Transformer"] = p_trans_test
    m_trans = compute_all_metrics(y_test_tab, p_trans_test)
    m_trans["model"] = "Matchup Transformer"
    m_trans["family"] = "Deep Learning"
    results.append(m_trans)

    ensemble_candidate_val = {
        "RNA Clássica (MLP)": p_mlp_val,
        "Deep ResNet MLP": p_resnet_val,
        "Dual-Branch LSTM": p_lstm_val,
        "Matchup Transformer": p_trans_val,
        "Regressão Logística": p_lr_val,
        "Random Forest": p_rf_val
    }
    ensemble_candidate_test = {
        "RNA Clássica (MLP)": p_mlp_test,
        "Deep ResNet MLP": p_resnet_test,
        "Dual-Branch LSTM": p_lstm_test,
        "Matchup Transformer": p_trans_test,
        "Regressão Logística": p_lr_test,
        "Random Forest": p_rf_test
    }

    ensemble = DeepEnsemblePredictor().fit_weights(ensemble_candidate_val, y_val_tab)
    p_ens_test = ensemble.predict_proba(ensemble_candidate_test)
    models_test_probas["Deep Ensemble"] = p_ens_test
    m_ens = compute_all_metrics(y_test_tab, p_ens_test)
    m_ens["model"] = "Deep Ensemble"
    m_ens["family"] = "Ensemble"
    results.append(m_ens)

    results_df = pd.DataFrame(results)
    
    plot_learning_curves(histories, paths.outputs_figures / "learning_curves.png")
    plot_confusion_matrices(y_test_tab, models_test_probas, paths.outputs_figures / "confusion_matrices.png")
    plot_roc_curves(y_test_tab, models_test_probas, paths.outputs_figures / "roc_curves.png")
    plot_calibration_curves(y_test_tab, models_test_probas, paths.outputs_figures / "calibration_curves.png")
    plot_model_comparison(results_df, paths.outputs_figures / "model_comparison.png")

    paths.outputs_models.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(paths.root / "outputs" / "evaluation_results.csv", index=False)
    with open(paths.root / "outputs" / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(paths.outputs_models / "ensemble_weights.json", "w") as f:
        json.dump(ensemble.weights, f, indent=2)

    return results_df

if __name__ == "__main__":
    df = run_full_benchmark()
    print(df.to_string())
