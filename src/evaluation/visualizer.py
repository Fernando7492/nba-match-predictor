from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.calibration import calibration_curve

plt.rcParams.update({
    "figure.autolayout": False,
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "axes.edgecolor": "#cccccc",
    "axes.linewidth": 0.8
})
sns.set_theme(style="whitegrid", palette="muted")

def plot_learning_curves(
    histories: dict[str, dict[str, list[float]]],
    output_path: Path
) -> None:
    if not histories:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for name, hist in histories.items():
        if "train_loss" in hist and hist["train_loss"]:
            axes[0].plot(hist["train_loss"], label=f"{name} (Treino)", linestyle="--", alpha=0.7)
            axes[0].plot(hist["val_loss"], label=f"{name} (Validação)", linewidth=2.0)
        if "train_acc" in hist and hist["train_acc"]:
            axes[1].plot(hist["train_acc"], label=f"{name} (Treino)", linestyle="--", alpha=0.7)
            axes[1].plot(hist["val_acc"], label=f"{name} (Validação)", linewidth=2.0)

    axes[0].set_title("Curva de Perda (Loss)", fontsize=13, fontweight="bold", pad=10)
    axes[0].set_xlabel("Época", fontsize=11)
    axes[0].set_ylabel("Loss", fontsize=11)
    axes[0].legend(loc="upper right", fontsize=8)

    axes[1].set_title("Curva de Acurácia", fontsize=13, fontweight="bold", pad=10)
    axes[1].set_xlabel("Época", fontsize=11)
    axes[1].set_ylabel("Acurácia", fontsize=11)
    axes[1].legend(loc="lower right", fontsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_confusion_matrices(
    y_true: np.ndarray,
    model_probas: dict[str, np.ndarray],
    output_path: Path
) -> None:
    if not model_probas:
        return
    n_models = len(model_probas)
    cols = min(4, n_models)
    rows = (n_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4 * rows), squeeze=False)
    flat_axes = axes.ravel()
    
    i = -1
    for i, (name, prob) in enumerate(model_probas.items()):
        y_pred = (prob >= 0.5).astype(int)
        cm = confusion_matrix(y_true, y_pred, normalize="true")
        sns.heatmap(
            cm,
            annot=True,
            fmt=".2%",
            cmap="Blues",
            cbar=False,
            ax=flat_axes[i],
            xticklabels=["Derrota Mandante", "Vitória Mandante"],
            yticklabels=["Derrota Mandante", "Vitória Mandante"]
        )
        flat_axes[i].set_title(name, fontsize=11, fontweight="bold")
        flat_axes[i].set_xlabel("Predito", fontsize=9)
        flat_axes[i].set_ylabel("Real", fontsize=9)

    for j in range(i + 1, len(flat_axes)):
        fig.delaxes(flat_axes[j])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_roc_curves(
    y_true: np.ndarray,
    model_probas: dict[str, np.ndarray],
    output_path: Path
) -> None:
    if not model_probas:
        return
    plt.figure(figsize=(8, 6))
    
    for name, prob in model_probas.items():
        fpr, tpr, _ = roc_curve(y_true, prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})", linewidth=1.8)

    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Aleatório (AUC = 0.500)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Taxa de Falsos Positivos (1 - Especificidade)", fontsize=11)
    plt.ylabel("Taxa de Verdadeiros Positivos (Sensibilidade)", fontsize=11)
    plt.title("Curvas ROC - Comparativo de Modelos", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="lower right", fontsize=9)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_calibration_curves(
    y_true: np.ndarray,
    model_probas: dict[str, np.ndarray],
    output_path: Path,
    n_bins: int = 10
) -> None:
    if not model_probas:
        return
    plt.figure(figsize=(8, 6))
    
    for name, prob in model_probas.items():
        prob_true, prob_pred = calibration_curve(y_true, prob, n_bins=n_bins)
        plt.plot(prob_pred, prob_true, marker="o", linewidth=1.5, label=name)

    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfeitamente Calibrado")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("Probabilidade Média Predita", fontsize=11)
    plt.ylabel("Fração de Positivos", fontsize=11)
    plt.title("Curvas de Calibração de Probabilidades", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="upper left", fontsize=9)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_model_comparison(
    results_df: pd.DataFrame,
    output_path: Path
) -> None:
    if results_df.empty:
        return
    metrics_to_plot = ["accuracy", "f1_score", "roc_auc", "brier_score"]
    metric_labels = ["Acurácia", "F1-Score", "ROC-AUC", "Brier Score (Menor = Melhor)"]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
        ax = axes[idx]
        sorted_df = results_df.sort_values(by=metric, ascending=(metric == "brier_score"))
        bars = ax.barh(sorted_df["model"], sorted_df[metric], color="#3182ce", alpha=0.85)
        ax.set_title(label, fontsize=12, fontweight="bold", pad=8)
        ax.set_xlim(0, max(sorted_df[metric].max() * 1.15, 0.8))

        for bar in bars:
            val = bar.get_width()
            ax.text(
                val + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}",
                va="center",
                ha="left",
                fontsize=9
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
