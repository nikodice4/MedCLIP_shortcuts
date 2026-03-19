# plotting.py: script for plotting the confidence curves (and calibration curves, to be continued)

import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from . import config
from sklearn.calibration import calibration_curve

def plot_confidence():
    # loading the saved mean probabilities
    df = pd.read_csv(config.REPORTS_DIR / "mean_probability_per_layer_test.csv")

    # confidence defined at 0.5
    df["confidence"] = (df["mean_probability"] - 0.5).abs()

    train_df = df[df["split"] == "train"]
    test_df  = df[df["split"] == "test"]

    # plotting
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(train_df["layer"], train_df["confidence"], label="Train", marker="o")
    ax.plot(test_df["layer"],  test_df["confidence"],  label="Test",  marker="o")

    ax.set_xlabel("Layer")
    ax.set_ylabel("Confidence")
    ax.set_title("Confidence per layer")
    ax.set_xticks(df["layer"].unique())
    ax.legend()
    ax.grid(True)

    plt.tight_layout()

    out_path = config.FIGURES_DIR / "confidence_per_layer.png"
    plt.savefig(out_path, dpi=150)
    print(f"saved confidence -> {out_path}")

# TODO: fix the function
def generate_calibration_curves(models, subgroup=True):
    colors = list(mcolors.TABLEAU_COLORS)
    plt.figure()
    plt.plot([0, 1], 
         [0, 1], 
         linestyle='dotted', 
         label='Perfectly Calibrated')
    for i,model_name in enumerate(models):
        df_probas = pd.read_csv(config.REPORTS_DIR / "calibration.csv")
        y_true = df_probas["Pneumothorax"]
        y_proba = df_probas["proba_Pneumothorax"]
        prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)
        
        if subgroup:
            plt.plot(prob_pred,prob_true,linestyle='solid',marker='o',linewidth=1,color=colors[i],label=f'{models[model_name]}, all')
            
            y_true = df_probas[df_probas["Drain"]==1]["Pneumothorax"]
            y_proba = df_probas[df_probas["Drain"]==1]["proba_Pneumothorax"]
            prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)
            plt.plot(prob_pred,prob_true,linestyle='dashed',marker='^',linewidth=1,color=colors[i],label=f'{models[model_name]}, only drain')

            y_true = df_probas[df_probas["Drain"]==0]["Pneumothorax"]
            y_proba = df_probas[df_probas["Drain"]==0]["proba_Pneumothorax"]
            prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)
            plt.plot(prob_pred,prob_true,linestyle='dashdot',marker='s',linewidth=1,color=colors[i],label=f'{models[model_name]}, no drain')
        else:
            plt.plot(prob_pred,prob_true,linestyle='solid',marker='o',linewidth=1,color=colors[i],label=f'{models[model_name]}')

    # plt.title('Calibration curves for all CLIP-based models')
    plt.xlabel('Mean predicted probability',fontdict = {'fontsize' : 20})
    plt.ylabel('Fraction of positives',fontdict = {'fontsize' : 20})
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=15)
    plt.savefig(f"./reports/figures/calibration_cxr14.png", bbox_inches='tight', dpi=300)
    plt.close()


if __name__ == "__main__":
    plot_confidence()