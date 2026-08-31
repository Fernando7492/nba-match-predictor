from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.calibration import calibration_curve

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

def plot_learning_curves(
    histories: dict[str, dict[str, list[float]]],
    save_path: Path | str
) -> None:
    n_models = len(histories)
    fig, axes = plt.subplots(n_models, 2, figsize=(12, 4 * n_models), squeeze=False)

    for i, (model_name, hist) in enumerate(histories.items()):
        epochs = range(1, len(hist["train_loss"]) + 1)

        axes[i, 0].plot(epochs, hist["train_loss"], label="Treino", color="#1f77b4", lw=2)
        axes[i, 0].plot(epochs, hist["val_loss"], label="Validação", color="#d62728", lw=2, linestyle="--")
        axes[i, 0].set_title(f"{model_name} - Perda (Binary Cross-Entropy)")
        axes[i, 0].set_xlabel("Época")
        axes[i, 0].set_ylabel("Loss")
        axes[i, 0].legend()

        axes[i, 1].plot(epochs, hist["train_acc"], label="Treino", color="#2ca02c", lw=2)
        axes[i, 1].plot(epochs, hist["val_acc"], label="Validação", color="#ff7f0e", lw=2, linestyle="--")
        axes[i, 1].set_title(f"{model_name} - Acurácia")
        axes[i, 1].set_xlabel("Época")
        axes[i, 1].set_ylabel("Acurácia")
        axes[i, 1].legend()

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_confusion_matrices(
    y_true: np.ndarray,
    model_probas: dict[str, np.ndarray],
    save_path: Path | str
) -> None:
    n_models = len(model_probas)
    cols = 3
    rows = (n_models + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
    axes = np.array(axes).reshape(-1)

    for i, (name, prob) in enumerate(model_probas.items()):
        pred = (np.asarray(prob) >= 0.5).astype(int)
        cm = confusion_matrix(y_true, pred, normalize="true")
        sns.heatmap(
            cm,
            annot=True,
            fmt=".2%",
            cmap="Blues",
            cbar=False,
            ax=axes[i],
            xticklabels=["Derrota Mandante", "Vitória Mandante"],
            yticklabels=["Derrota Mandante", "Vitória Mandante"]
        )
        axes[i].set_title(f"Matriz de Confusão: {name}")
        axes[i].set_xlabel("Predito")
        axes[i].set_ylabel("Real")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_roc_curves(
    y_true: np.ndarray,
    model_probas: dict[str, np.ndarray],
    save_path: Path | str
) -> None:
    plt.figure(figsize=(8, 7))

    for name, prob in model_probas.items():
        fpr, tpr, _ = roc_curve(y_true, prob)
        roc_score = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_score:.3f})")

    plt.plot([0, 1], [0, 1], color="grey", lw=1.5, linestyle="--", label="Aleatório (AUC = 0.500)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Taxa de Falsos Positivos (FPR)")
    plt.ylabel("Taxa de Verdadeiros Positivos (TPR)")
    plt.title("Curvas ROC - Comparação de Modelos")
    plt.legend(loc="lower right")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_calibration_curves(
    y_true: np.ndarray,
    model_probas: dict[str, np.ndarray],
    save_path: Path | str
) -> None:
    plt.figure(figsize=(8, 7))

    for name, prob in model_probas.items():
        prob_true, prob_pred = calibration_curve(y_true, prob, n_bins=10)
        plt.plot(prob_pred, prob_true, marker="o", lw=2, label=name)

    plt.plot([0, 1], [0, 1], color="black", lw=1.5, linestyle="--", label="Perfeitamente Calibrado")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("Probabilidade Média Predita")
    plt.ylabel("Fração de Positivos Observada")
    plt.title("Diagrama de Confiabilidade (Calibração de Probabilidades)")
    plt.legend(loc="upper left")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_model_comparison(
    results_df: pd.DataFrame,
    save_path: Path | str
) -> None:
    metrics = ["accuracy", "f1_score", "roc_auc", "brier_score"]
    metric_labels = ["Acurácia", "F1-Score", "ROC-AUC", "Brier Score (Menor é Melhor)"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.flatten()

    for idx, (m, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx]
        sorted_df = results_df.sort_values(m, ascending=(m == "brier_score"))
        palette = "viridis_r" if m == "brier_score" else "viridis"
        bars = sns.barplot(
            data=sorted_df,
            x="model",
            y=m,
            hue="model",
            legend=False,
            palette=palette,
            ax=ax
        )
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=25)
        
        for p in ax.patches:
            height = p.get_height()
            if not np.isnan(height):
                ax.annotate(
                    f"{height:.3f}",
                    (p.get_x() + p.get_width() / 2.0, height),
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    xytext=(0, 3),
                    textcoords="offset points"
                )

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
