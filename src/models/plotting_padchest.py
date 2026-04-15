# plotting.py: script for plotting the confidence curves (and calibration curves, to be continued)

import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from . import config
from sklearn.calibration import calibration_curve

def plot_confidence():
    config.FIGURES_DIR_PADCHEST.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR_PADCHEST_SEX.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR_PADCHEST_SCANNER.mkdir(parents=True, exist_ok=True)
    # loading the saved mean probabilities
    # train_df = pd.read_csv(config.REPORTS_DIR / "mean_probability_per_layer_train.csv")
    # test_df = pd.read_csv(config.REPORTS_DIR / "mean_probability_per_layer_test.csv")
    # drain_df = pd.read_csv(config.REPORTS_DIR / "mean_probability_per_layer_drain.csv")
    # no_drain_df = pd.read_csv(config.REPORTS_DIR / "mean_probability_per_layer_nodrain.csv")

    confidence_data = {
        "Train_data": ("mean_probability_per_layer_train.csv", config.REPORTS_DIR_PADCHEST, config.FIGURES_DIR_PADCHEST),
        "Test_data": ("mean_probability_per_layer_test.csv", config.REPORTS_DIR_PADCHEST, config.FIGURES_DIR_PADCHEST),
        "Test_data_scanner_IDC": ("mean_probability_per_layer_ImagingDynamicsCompanyLtd.csv", config.REPORTS_DIR_PADCHEST_SCANNER, config.FIGURES_DIR_PADCHEST_SCANNER),
        "Test_data_scanner_PMS": ("mean_probability_per_layer_PhilipsMedicalSystems.csv", config.REPORTS_DIR_PADCHEST_SCANNER, config.FIGURES_DIR_PADCHEST_SCANNER),
        "Test_data_female": ("mean_probability_per_layer_female.csv", config.REPORTS_DIR_PADCHEST_SEX, config.FIGURES_DIR_PADCHEST_SEX),
        "Test_data_male": ("mean_probability_per_layer_male.csv", config.REPORTS_DIR_PADCHEST_SEX, config.FIGURES_DIR_PADCHEST_SEX),
    }

    for name, (filename, reports_dir, figures_dir) in confidence_data.items():
        df = pd.read_csv(reports_dir / filename)

        # confidence defined at 0.5
        df["confidence"] = (df["mean_probability"] - 0.5).abs()

        # plotting
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(df["layer"], df["confidence"], label=name, marker="o")
        # ax.plot(test_df["layer"],  test_df["confidence"],  label="Test",  marker="o")
        # ax.plot(drain_df["layer"],  drain_df["confidence"],  label="Drains",  marker="o")
        # ax.plot(no_drain_df["layer"],  no_drain_df["confidence"],  label="No drains",  marker="o")

        ax.set_xlabel("Layer")
        ax.set_ylabel("Confidence")
        ax.set_title("Confidence per layer")
        ax.set_xticks(df["layer"].unique())
        ax.legend()
        ax.grid(True)
        ax.set_ylim(0, 0.5)

        plt.tight_layout()

        out_path = figures_dir / f"confidence_per_layer_{name}.png"
        plt.savefig(out_path, dpi=150)
        print(f"saved confidence -> {out_path}")

# TODO: fix the function
def generate_calibration_curves_drains():
    config.FIGURES_DIR_PADCHEST_SCANNER.mkdir(parents=True, exist_ok=True)
    # colors = list(mcolors.TABLEAU_COLORS)
    plt.figure()
    plt.plot([0, 1], 
         [0, 1], 
         linestyle='dotted', 
         label='Perfectly Calibrated')
    # for i, model_name in enumerate(models):
    df_probas = pd.read_csv(config.REPORTS_DIR_PADCHEST_SCANNER / "calibration_scanner.csv")
    y_true = df_probas["y_true"] # actual pneumothorax
    y_proba = df_probas["y_prob"] # probability pneumothorax
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)
    
    # if subgroup:
    plt.plot(prob_pred, prob_true,linestyle='solid',marker='o',linewidth=1,label='MedCLIP, all')
    
    y_true = df_probas[df_probas["scanner"]=="ImagingDynamicsCompanyLtd"]["y_true"]
    y_proba = df_probas[df_probas["scanner"]=="ImagingDynamicsCompanyLtd"]["y_prob"]
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)
    plt.plot(prob_pred, prob_true,linestyle='dashed',marker='^',linewidth=1,label='MedCLIP, ImagingDynamicsCompanyLtd')

    y_true = df_probas[df_probas["scanner"]=="PhilipsMedicalSystems"]["y_true"]
    y_proba = df_probas[df_probas["scanner"]=="PhilipsMedicalSystems"]["y_prob"]
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)
    plt.plot(prob_pred, prob_true,linestyle='dashdot',marker='s',linewidth=1,label='MedCLIP, PhilipsMedicalSystems')
    # else:
        # plt.plot(prob_pred,prob_true,linestyle='solid',marker='o',linewidth=1,color=colors[i],label=f'{models[model_name]}')

    # plt.title('Calibration curves for all CLIP-based models')
    plt.xlabel('Mean predicted probability',fontdict = {'fontsize' : 20})
    plt.ylabel('Fraction of positives',fontdict = {'fontsize' : 20})
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=15)
    plt.savefig(config.FIGURES_DIR_PADCHEST_SCANNER/ "calibration_curve_scanner.png", bbox_inches='tight', dpi=300)
    plt.close()


def generate_calibration_curves_sex():
    config.FIGURES_DIR_PADCHEST_SEX.mkdir(parents=True, exist_ok=True)
    # colors = list(mcolors.TABLEAU_COLORS)
    plt.figure()
    plt.plot([0, 1], 
         [0, 1], 
         linestyle='dotted', 
         label='Perfectly Calibrated')
    # for i, model_name in enumerate(models):
    df_probas = pd.read_csv(config.REPORTS_DIR_PADCHEST_SEX / "calibration_sex.csv")
    y_true = df_probas["y_true"] # actual pneumothorax
    y_proba = df_probas["y_prob"] # probability pneumothorax
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)
    
    # if subgroup:
    plt.plot(prob_pred, prob_true,linestyle='solid',marker='o',linewidth=1,label='MedCLIP, all')
    
    y_true = df_probas[df_probas["sex"]=="F"]["y_true"]
    y_proba = df_probas[df_probas["sex"]=="F"]["y_prob"]
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)
    plt.plot(prob_pred, prob_true,linestyle='dashed',marker='^',linewidth=1,label='MedCLIP, Female patients')

    y_true = df_probas[df_probas["sex"]=="M"]["y_true"]
    y_proba = df_probas[df_probas["sex"]=="M"]["y_prob"]
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)
    plt.plot(prob_pred, prob_true,linestyle='dashdot',marker='s',linewidth=1,label='MedCLIP, Male patients')
    # else:
        # plt.plot(prob_pred,prob_true,linestyle='solid',marker='o',linewidth=1,color=colors[i],label=f'{models[model_name]}')

    # plt.title('Calibration curves for all CLIP-based models')
    plt.xlabel('Mean predicted probability',fontdict = {'fontsize' : 20})
    plt.ylabel('Fraction of positives',fontdict = {'fontsize' : 20})
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=15)
    plt.savefig(config.FIGURES_DIR_PADCHEST_SEX / "calibration_curve_sex.png", bbox_inches='tight', dpi=300)
    plt.close()


if __name__ == "__main__":
    plot_confidence()
    generate_calibration_curves_drains()
    generate_calibration_curves_sex()