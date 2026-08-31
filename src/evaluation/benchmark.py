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

def evaluate_and_record(
    name: str,
    family: str,
    y_test: np.ndarray,
    p_test: np.ndarray
) -> dict:
    metrics = compute_all_metrics(y_test, p_test)
    metrics["model"] = name
    metrics["family"] = family
    return metrics

def run_baselines(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    seed: int
) -> tuple[list[dict], dict[str, np.ndarray], dict[str, np.ndarray]]:
    results = []
    val_probas = {}
    test_probas = {}

    home = HomeCourtBaseline().fit(x_train, y_train)
    p_h_val = home.predict_proba(x_val)[:, 1]
    p_h_test = home.predict_proba(x_test)[:, 1]
    val_probas["Baseline (Mando)"] = p_h_val
    test_probas["Baseline (Mando)"] = p_h_test
    results.append(evaluate_and_record("Baseline (Mando)", "Baseline", y_test, p_h_test))

    lr = LogisticRegressionBaseline(random_state=seed).fit(x_train, y_train)
    p_lr_val = lr.predict_proba(x_val)[:, 1]
    p_lr_test = lr.predict_proba(x_test)[:, 1]
    val_probas["Regressão Logística"] = p_lr_val
    test_probas["Regressão Logística"] = p_lr_test
    results.append(evaluate_and_record("Regressão Logística", "Baseline", y_test, p_lr_test))

    rf = RandomForestBaseline(random_state=seed).fit(x_train, y_train)
    p_rf_val = rf.predict_proba(x_val)[:, 1]
    p_rf_test = rf.predict_proba(x_test)[:, 1]
    val_probas["Random Forest"] = p_rf_val
    test_probas["Random Forest"] = p_rf_test
    results.append(evaluate_and_record("Random Forest", "Baseline", y_test, p_rf_test))

    return results, val_probas, test_probas

def run_tabular_deep_models(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    eval_val_loader: DataLoader,
    eval_test_loader: DataLoader,
    input_dim: int,
    y_test: np.ndarray,
    paths: ProjectPaths,
    config: ModelConfig
) -> tuple[list[dict], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, dict]]:
    results = []
    val_probas = {}
    test_probas = {}
    histories = {}

    mlp = ClassicalMLP(input_dim=input_dim, hidden_dim_1=64, hidden_dim_2=32, dropout=0.2)
    mlp_opt = torch.optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=config.weight_decay)
    mlp_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(mlp_opt, mode="min", factor=0.5, patience=4)
    mlp_trainer = ModelTrainer(
        model=mlp,
        optimizer=mlp_opt,
        scheduler=mlp_sched,
        patience=config.early_stopping_patience,
        save_path=paths.outputs_models / "classical_mlp.pt",
        device=config.device,
        use_amp=True
    )
    histories["RNA Clássica (MLP)"] = mlp_trainer.fit(train_loader, val_loader, epochs=config.epochs)
    p_mlp_val = mlp_trainer.predict_proba(eval_val_loader).numpy().ravel()
    p_mlp_test = mlp_trainer.predict_proba(eval_test_loader).numpy().ravel()
    val_probas["RNA Clássica (MLP)"] = p_mlp_val
    test_probas["RNA Clássica (MLP)"] = p_mlp_test
    results.append(evaluate_and_record("RNA Clássica (MLP)", "RNA", y_test, p_mlp_test))

    resnet = DeepResNetMLP(input_dim=input_dim, hidden_dim=128, num_blocks=3, dropout=0.25)
    resnet_opt = torch.optim.AdamW(resnet.parameters(), lr=1e-3, weight_decay=config.weight_decay)
    resnet_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(resnet_opt, mode="min", factor=0.5, patience=4)
    resnet_trainer = ModelTrainer(
        model=resnet,
        optimizer=resnet_opt,
        scheduler=resnet_sched,
        patience=config.early_stopping_patience,
        save_path=paths.outputs_models / "deep_resnet.pt",
        device=config.device,
        use_amp=True
    )
    histories["Deep ResNet MLP"] = resnet_trainer.fit(train_loader, val_loader, epochs=config.epochs)
    p_resnet_val = resnet_trainer.predict_proba(eval_val_loader).numpy().ravel()
    p_resnet_test = resnet_trainer.predict_proba(eval_test_loader).numpy().ravel()
    val_probas["Deep ResNet MLP"] = p_resnet_val
    test_probas["Deep ResNet MLP"] = p_resnet_test
    results.append(evaluate_and_record("Deep ResNet MLP", "Deep Learning", y_test, p_resnet_test))

    return results, val_probas, test_probas, histories

def run_sequence_deep_models(
    seq_train_loader: DataLoader,
    seq_val_loader: DataLoader,
    seq_test_loader: DataLoader,
    stat_dim: int,
    y_test: np.ndarray,
    paths: ProjectPaths,
    config: ModelConfig
) -> tuple[list[dict], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, dict]]:
    results = []
    val_probas = {}
    test_probas = {}
    histories = {}

    lstm = DualBranchLSTM(input_dim=stat_dim, hidden_dim=64, num_layers=2, dropout=0.2)
    lstm_opt = torch.optim.AdamW(lstm.parameters(), lr=1e-3, weight_decay=config.weight_decay)
    lstm_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(lstm_opt, mode="min", factor=0.5, patience=4)
    lstm_trainer = SequenceModelTrainer(
        model=lstm,
        optimizer=lstm_opt,
        scheduler=lstm_sched,
        patience=config.early_stopping_patience,
        save_path=paths.outputs_models / "dual_branch_lstm.pt",
        device=config.device,
        use_amp=True
    )
    histories["Dual-Branch LSTM"] = lstm_trainer.fit(seq_train_loader, seq_val_loader, epochs=config.epochs)
    p_lstm_val = lstm_trainer.predict_proba(seq_val_loader).numpy().ravel()
    p_lstm_test = lstm_trainer.predict_proba(seq_test_loader).numpy().ravel()
    val_probas["Dual-Branch LSTM"] = p_lstm_val
    test_probas["Dual-Branch LSTM"] = p_lstm_test
    results.append(evaluate_and_record("Dual-Branch LSTM", "Deep Learning", y_test, p_lstm_test))

    trans = MatchupTransformer(input_dim=stat_dim, d_model=64, nhead=4, num_layers=2, dropout=0.2)
    trans_opt = torch.optim.AdamW(trans.parameters(), lr=5e-4, weight_decay=config.weight_decay)
    trans_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(trans_opt, mode="min", factor=0.5, patience=4)
    trans_trainer = SequenceModelTrainer(
        model=trans,
        optimizer=trans_opt,
        scheduler=trans_sched,
        patience=config.early_stopping_patience,
        save_path=paths.outputs_models / "matchup_transformer.pt",
        device=config.device,
        use_amp=True
    )
    histories["Matchup Transformer"] = trans_trainer.fit(seq_train_loader, seq_val_loader, epochs=config.epochs)
    p_trans_val = trans_trainer.predict_proba(seq_val_loader).numpy().ravel()
    p_trans_test = trans_trainer.predict_proba(seq_test_loader).numpy().ravel()
    val_probas["Matchup Transformer"] = p_trans_val
    test_probas["Matchup Transformer"] = p_trans_test
    results.append(evaluate_and_record("Matchup Transformer", "Deep Learning", y_test, p_trans_test))

    return results, val_probas, test_probas, histories

def run_full_benchmark(
    paths: ProjectPaths | None = None,
    config: ModelConfig | None = None
) -> pd.DataFrame:
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
    _, eval_val_loader, eval_test_loader = get_tabular_loaders(
        train_df, val_df, test_df, feature_cols, batch_size=config.batch_size
    )

    seq_pipeline = NBASequencePipeline(paths=paths, sequence_length=config.sequence_length)
    seq_train_ds, seq_val_ds, seq_test_ds = seq_pipeline.build_sequences(raw_df, valid_game_ids=valid_game_ids)

    seq_train_loader = DataLoader(seq_train_ds, batch_size=config.batch_size, shuffle=True)
    seq_val_loader = DataLoader(seq_val_ds, batch_size=config.batch_size, shuffle=False)
    seq_test_loader = DataLoader(seq_test_ds, batch_size=config.batch_size, shuffle=False)

    x_train = train_df[feature_cols].values
    y_train = train_df["TARGET_HOME_W"].values
    x_val = val_df[feature_cols].values
    y_val = val_df["TARGET_HOME_W"].values
    x_test = test_df[feature_cols].values
    y_test = test_df["TARGET_HOME_W"].values

    all_results = []
    all_val_probas = {}
    all_test_probas = {}
    all_histories = {}

    b_res, b_val, b_test = run_baselines(x_train, y_train, x_val, x_test, y_test, config.seed)
    all_results.extend(b_res)
    all_val_probas.update(b_val)
    all_test_probas.update(b_test)

    t_res, t_val, t_test, t_hist = run_tabular_deep_models(
        train_loader, val_loader, test_loader, eval_val_loader, eval_test_loader,
        len(feature_cols), y_test, paths, config
    )
    all_results.extend(t_res)
    all_val_probas.update(t_val)
    all_test_probas.update(t_test)
    all_histories.update(t_hist)

    s_res, s_val, s_test, s_hist = run_sequence_deep_models(
        seq_train_loader, seq_val_loader, seq_test_loader,
        len(seq_pipeline.stat_cols), y_test, paths, config
    )
    all_results.extend(s_res)
    all_val_probas.update(s_val)
    all_test_probas.update(s_test)
    all_histories.update(s_hist)

    ensemble_candidate_val = {
        "RNA Clássica (MLP)": all_val_probas["RNA Clássica (MLP)"],
        "Deep ResNet MLP": all_val_probas["Deep ResNet MLP"],
        "Dual-Branch LSTM": all_val_probas["Dual-Branch LSTM"],
        "Matchup Transformer": all_val_probas["Matchup Transformer"],
        "Regressão Logística": all_val_probas["Regressão Logística"],
        "Random Forest": all_val_probas["Random Forest"]
    }
    ensemble_candidate_test = {
        "RNA Clássica (MLP)": all_test_probas["RNA Clássica (MLP)"],
        "Deep ResNet MLP": all_test_probas["Deep ResNet MLP"],
        "Dual-Branch LSTM": all_test_probas["Dual-Branch LSTM"],
        "Matchup Transformer": all_test_probas["Matchup Transformer"],
        "Regressão Logística": all_test_probas["Regressão Logística"],
        "Random Forest": all_test_probas["Random Forest"]
    }

    ensemble = DeepEnsemblePredictor().fit_weights(ensemble_candidate_val, y_val)
    p_ens_test = ensemble.predict_proba(ensemble_candidate_test)
    all_test_probas["Deep Ensemble"] = p_ens_test
    all_results.append(evaluate_and_record("Deep Ensemble", "Ensemble", y_test, p_ens_test))

    results_df = pd.DataFrame(all_results)

    plot_learning_curves(all_histories, paths.outputs_figures / "learning_curves.png")
    plot_confusion_matrices(y_test, all_test_probas, paths.outputs_figures / "confusion_matrices.png")
    plot_roc_curves(y_test, all_test_probas, paths.outputs_figures / "roc_curves.png")
    plot_calibration_curves(y_test, all_test_probas, paths.outputs_figures / "calibration_curves.png")
    plot_model_comparison(results_df, paths.outputs_figures / "model_comparison.png")

    paths.outputs_models.mkdir(parents=True, exist_ok=True)
    paths.outputs_figures.mkdir(parents=True, exist_ok=True)
    
    results_df.to_csv(paths.root / "outputs" / "evaluation_results.csv", index=False)
    with open(paths.root / "outputs" / "evaluation_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    with open(paths.outputs_models / "ensemble_weights.json", "w") as f:
        json.dump(ensemble.weights, f, indent=2)

    return results_df

if __name__ == "__main__":
    df = run_full_benchmark()
    print(df.to_string())
