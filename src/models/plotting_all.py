import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from . import config
from sklearn.calibration import calibration_curve

# if the plot title doesn't say train or test, it is by default test data
############################################# CONFIDENCE CURVES #############################################
############### CHESTX PNEUMOTHORAX
def plot_confidence():
    config.FIGURES_DIR_DRAIN.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR_SEX.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    overall_data = {
        "Train data for NIH-CXR14, pneumothorax": ("mean_probability_per_layer_train.csv", config.REPORTS_DIR, config.FIGURES_DIR),
        "Test data for NIH-CXR14, pneumothorax": ("mean_probability_per_layer_test.csv", config.REPORTS_DIR, config.FIGURES_DIR)
        }

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, (filename, reports_dir, figures_dir) in overall_data.items():
        df = pd.read_csv(reports_dir / filename)
        # confidence defined at 0.5
        df["confidence"] = (df["mean_probability"] - 0.5).abs()
        ax.plot(df["layer"], df["confidence"], label=name, marker="o")
        # ax.plot(test_df["layer"],  test_df["confidence"],  label="Test",  marker="o")
        # ax.plot(drain_df["layer"],  drain_df["confidence"],  label="Drains",  marker="o")
        # ax.plot(no_drain_df["layer"],  no_drain_df["confidence"],  label="No drains",  marker="o")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Confidence")
    ax.set_title("Confidence per layer for overall ChestX data, pneumothorax")
    ax.set_xticks(df["layer"].unique())
    ax.legend()
    ax.grid(True)
    ax.set_ylim(0, 0.5)

    plt.tight_layout()

    out_path = figures_dir / f"confidence_per_layer_{name}.png"
    plt.savefig(out_path, dpi=150)
    print(f"saved confidence overall (chestx) -> {out_path}")

    ############### SEX PLOTS COMBINED
    sex_data = {
        "Female patients NIH-CXR14, pneumothorax": ("mean_probability_per_layer_female.csv", config.REPORTS_DIR_SEX, config.FIGURES_DIR_SEX),
        "Male patients NIH-CXR14, pneumothorax": ("mean_probability_per_layer_male.csv", config.REPORTS_DIR_SEX, config.FIGURES_DIR_SEX),
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, (filename, reports_dir, figures_dir) in sex_data.items():
        df = pd.read_csv(reports_dir / filename)
        df["confidence"] = (df["mean_probability"] - 0.5).abs()
        ax.plot(df["layer"], df["confidence"], label=label, marker="o")

    ax.set_xlabel("Layer")
    ax.set_ylabel("Confidence")
    ax.set_title("Confidence per layer for patient sex ChestX data, pneumothorax")
    ax.set_xticks(df["layer"].unique())
    ax.legend()
    ax.grid(True)
    ax.set_ylim(0, 0.5)

    plt.tight_layout()

    out_path = figures_dir / f"confidence_per_layer_{name}.png"
    plt.savefig(out_path, dpi=150)
    print(f"saved confidence sex (chestx) -> {out_path}")

    ############### DRAIN PLOTS COMBINED
    drain_data = {
        "Drains NIH-CXR14, pneumothorax": ("mean_probability_per_layer_drain.csv", config.REPORTS_DIR_DRAIN, config.FIGURES_DIR_DRAIN),
        "No drains NIH-CXR14, pneumothorax": ("mean_probability_per_layer_nodrain.csv", config.REPORTS_DIR_DRAIN, config.FIGURES_DIR_DRAIN),
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, (filename, reports_dir, figures_dir) in drain_data.items():
        df = pd.read_csv(reports_dir / filename)
        df["confidence"] = (df["mean_probability"] - 0.5).abs()
        ax.plot(df["layer"], df["confidence"], label=label, marker="o")

    ax.set_xlabel("Layer")
    ax.set_ylabel("Confidence")
    ax.set_title("Confidence per layer for drains ChestX data, pneumothorax")
    ax.set_xticks(df["layer"].unique())
    ax.legend()
    ax.grid(True)
    ax.set_ylim(0, 0.5)

    plt.tight_layout()

    out_path = figures_dir / f"confidence_per_layer_{name}.png"
    plt.savefig(out_path, dpi=150)
    print(f"saved confidence drains (chestx) -> {out_path}")

############### PADCHEST CARDIOMGELY
def plot_confidence_padchest():
    config.FIGURES_DIR_PADCHEST.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR_PADCHEST_SEX.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR_PADCHEST_SCANNER.mkdir(parents=True, exist_ok=True)

    overall_data = {
        "Train data PadChest, cardiomegaly": ("mean_probability_per_layer_train.csv", config.REPORTS_DIR_PADCHEST, config.FIGURES_DIR_PADCHEST),
        "Test data PadChest, cardiomegaly": ("mean_probability_per_layer_test.csv", config.REPORTS_DIR_PADCHEST, config.FIGURES_DIR_PADCHEST),
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, (filename, reports_dir, figures_dir) in overall_data.items():
        df = pd.read_csv(reports_dir / filename)
        # confidence defined at 0.5
        df["confidence"] = (df["mean_probability"] - 0.5).abs()
        ax.plot(df["layer"], df["confidence"], label=name, marker="o")
        # ax.plot(test_df["layer"],  test_df["confidence"],  label="Test",  marker="o")
        # ax.plot(drain_df["layer"],  drain_df["confidence"],  label="Drains",  marker="o")
        # ax.plot(no_drain_df["layer"],  no_drain_df["confidence"],  label="No drains",  marker="o")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Confidence")
    ax.set_title("Confidence per layer for overall PadChest data, cardiomegaly")
    ax.set_xticks(df["layer"].unique())
    ax.legend()
    ax.grid(True)
    ax.set_ylim(0, 0.5)

    plt.tight_layout()

    out_path = figures_dir / f"confidence_per_layer_{name}.png"
    plt.savefig(out_path, dpi=150)
    print(f"saved confidence overall (padchest) -> {out_path}")

    ############### SEX PLOTS COMBINED
    sex_data = {
        "Female patients PadChest, cardiomegaly": ("mean_probability_per_layer_female.csv", config.REPORTS_DIR_PADCHEST_SEX, config.FIGURES_DIR_PADCHEST_SEX),
        "Male patients PadChest, cardiomegaly": ("mean_probability_per_layer_male.csv", config.REPORTS_DIR_PADCHEST_SEX, config.FIGURES_DIR_PADCHEST_SEX),
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, (filename, reports_dir, figures_dir) in sex_data.items():
        df = pd.read_csv(reports_dir / filename)
        df["confidence"] = (df["mean_probability"] - 0.5).abs()
        ax.plot(df["layer"], df["confidence"], label=label, marker="o")

    ax.set_xlabel("Layer")
    ax.set_ylabel("Confidence")
    ax.set_title("Confidence per layer for patient sex PadChest data, cardiomegaly")
    ax.set_xticks(df["layer"].unique())
    ax.legend()
    ax.grid(True)
    ax.set_ylim(0, 0.5)

    plt.tight_layout()

    out_path = figures_dir / f"confidence_per_layer_{name}.png"
    plt.savefig(out_path, dpi=150)
    print(f"saved confidence sex (padchest) -> {out_path}")

    ############### SCANNER PLOTS COMBINED
    scanner_data = {
        "ImagingDynamicsCompanyLtd PadChest, cardiomegaly": ("mean_probability_per_layer_ImagingDynamicsCompanyLtd.csv", config.REPORTS_DIR_PADCHEST_SCANNER, config.FIGURES_DIR_PADCHEST_SCANNER),
        "PhilipsMedicalSystems PadChest, cardiomegaly": ("mean_probability_per_layer_PhilipsMedicalSystems.csv", config.REPORTS_DIR_PADCHEST_SCANNER, config.FIGURES_DIR_PADCHEST_SCANNER),
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, (filename, reports_dir, figures_dir) in scanner_data.items():
        df = pd.read_csv(reports_dir / filename)
        df["confidence"] = (df["mean_probability"] - 0.5).abs()
        ax.plot(df["layer"], df["confidence"], label=label, marker="o")

    ax.set_xlabel("Layer")
    ax.set_ylabel("Confidence")
    ax.set_title("Confidence per layer for scanner machines PadChest data, cardiomegaly")
    ax.set_xticks(df["layer"].unique())
    ax.legend()
    ax.grid(True)
    ax.set_ylim(0, 0.5)

    plt.tight_layout()

    out_path = figures_dir / f"confidence_per_layer_{name}.png"
    plt.savefig(out_path, dpi=150)
    print(f"saved confidence scanner (padchest) -> {out_path}")

############### PADCHEST PNEUMOTHORAX
def plot_confidence_padchest_px():
    config.FIGURES_DIR_PADCHEST_PX.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR_PADCHEST_SEX_PX.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR_PADCHEST_SCANNER_PX.mkdir(parents=True, exist_ok=True)

    overall_data = {
        "Train data PadChest, pneumothorax": ("mean_probability_per_layer_train.csv", config.REPORTS_DIR_PADCHEST_PX, config.FIGURES_DIR_PADCHEST_PX),
        "Test data PadChest, pneumothorax": ("mean_probability_per_layer_test.csv", config.REPORTS_DIR_PADCHEST_PX, config.FIGURES_DIR_PADCHEST_PX),
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, (filename, reports_dir, figures_dir) in overall_data.items():
        df = pd.read_csv(reports_dir / filename)
        # confidence defined at 0.5
        df["confidence"] = (df["mean_probability"] - 0.5).abs()
        ax.plot(df["layer"], df["confidence"], label=name, marker="o")
        # ax.plot(test_df["layer"],  test_df["confidence"],  label="Test",  marker="o")
        # ax.plot(drain_df["layer"],  drain_df["confidence"],  label="Drains",  marker="o")
        # ax.plot(no_drain_df["layer"],  no_drain_df["confidence"],  label="No drains",  marker="o")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Confidence")
    ax.set_title("Confidence per layer for overall PadChest data, pneumothorax")
    ax.set_xticks(df["layer"].unique())
    ax.legend()
    ax.grid(True)
    ax.set_ylim(0, 0.5)

    plt.tight_layout()

    out_path = figures_dir / f"confidence_per_layer_{name}.png"
    plt.savefig(out_path, dpi=150)
    print(f"saved confidence overall (padchest_px) -> {out_path}")

    ############### SEX PLOTS COMBINED
    sex_data = {
        "Female patients PadChest, pneumothorax": ("mean_probability_per_layer_female.csv", config.REPORTS_DIR_PADCHEST_SEX_PX, config.FIGURES_DIR_PADCHEST_SEX_PX),
        "Male patients PadChest, pneumothorax": ("mean_probability_per_layer_male.csv", config.REPORTS_DIR_PADCHEST_SEX_PX, config.FIGURES_DIR_PADCHEST_SEX_PX),
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, (filename, reports_dir, figures_dir) in sex_data.items():
        df = pd.read_csv(reports_dir / filename)
        df["confidence"] = (df["mean_probability"] - 0.5).abs()
        ax.plot(df["layer"], df["confidence"], label=label, marker="o")

    ax.set_xlabel("Layer")
    ax.set_ylabel("Confidence")
    ax.set_title("Confidence per layer for patient sex PadChest data, pneumothorax")
    ax.set_xticks(df["layer"].unique())
    ax.legend()
    ax.grid(True)
    ax.set_ylim(0, 0.5)

    plt.tight_layout()

    out_path = figures_dir / f"confidence_per_layer_{name}.png"
    plt.savefig(out_path, dpi=150)
    print(f"saved confidence sex (padchest_px) -> {out_path}")

    ############### SCANNER PLOTS COMBINED
    scanner_data = {
        "ImagingDynamicsCompanyLtd PadChest, pneumothorax": ("mean_probability_per_layer_ImagingDynamicsCompanyLtd.csv", config.REPORTS_DIR_PADCHEST_SCANNER_PX, config.FIGURES_DIR_PADCHEST_SCANNER_PX),
        "PhilipsMedicalSystems PadChest, pneumothorax": ("mean_probability_per_layer_PhilipsMedicalSystems.csv", config.REPORTS_DIR_PADCHEST_SCANNER_PX, config.FIGURES_DIR_PADCHEST_SCANNER_PX),
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, (filename, reports_dir, figures_dir) in scanner_data.items():
        df = pd.read_csv(reports_dir / filename)
        df["confidence"] = (df["mean_probability"] - 0.5).abs()
        ax.plot(df["layer"], df["confidence"], label=label, marker="o")

    ax.set_xlabel("Layer")
    ax.set_ylabel("Confidence")
    ax.set_title("Confidence per layer for scanner machines PadChest data, pneumothorax")
    ax.set_xticks(df["layer"].unique())
    ax.legend()
    ax.grid(True)
    ax.set_ylim(0, 0.5)

    plt.tight_layout()

    out_path = figures_dir / f"confidence_per_layer_{name}.png"
    plt.savefig(out_path, dpi=150)
    print(f"saved confidence scanner (padchest_px) -> {out_path}")


############################################# CALIBRATION CURVES #############################################
############### CHESTX PNEUMOTHORAX
############### DRAINS
def calibration_curves_drains():
    config.FIGURES_DIR_DRAIN.mkdir(parents=True, exist_ok=True)
    # colors = list(mcolors.TABLEAU_COLORS)
    # plt.figure()
    plt.figure(figsize=(10, 7))
    plt.plot([0, 1], 
         [0, 1], 
         linestyle='dotted', 
         label='Perfectly Calibrated')
    # for i, model_name in enumerate(models):
    df_probas = pd.read_csv(config.REPORTS_DIR_DRAIN / "calibration_drain.csv")
    y_true = df_probas["y_true"] # actual pneumothorax
    y_proba = df_probas["y_prob"] # probability pneumothorax
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)
    
    # if subgroup:
    plt.plot(prob_pred, prob_true,linestyle='solid',marker='o',linewidth=1,label='MedCLIP, all')
    
    y_true = df_probas[df_probas["drain"]==1]["y_true"]
    y_proba = df_probas[df_probas["drain"]==1]["y_prob"]
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)
    plt.plot(prob_pred, prob_true,linestyle='dashed',marker='^',linewidth=1,label='MedCLIP, only drain')

    y_true = df_probas[df_probas["drain"]==0]["y_true"]
    y_proba = df_probas[df_probas["drain"]==0]["y_prob"]
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)
    plt.plot(prob_pred, prob_true,linestyle='dashdot',marker='s',linewidth=1,label='MedCLIP, no drain')
    # else:
        # plt.plot(prob_pred,prob_true,linestyle='solid',marker='o',linewidth=1,color=colors[i],label=f'{models[model_name]}')

    # plt.title('Calibration curves for all CLIP-based models')
    plt.xlabel('Mean predicted probability',fontdict = {'fontsize' : 16})
    plt.ylabel('Fraction of positives',fontdict = {'fontsize' : 16})
    plt.title("Calibration curve for drains ChestX, pneumothorax", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=15)
    plt.savefig(config.FIGURES_DIR_DRAIN / "calibration_curve_drains.png", bbox_inches='tight', dpi=300)
    plt.close()

############### CHESTX PNEUMOTHORAX
############### SEX
def calibration_curves_sex():
    config.FIGURES_DIR_SEX.mkdir(parents=True, exist_ok=True)
    # colors = list(mcolors.TABLEAU_COLORS)
    # plt.figure()
    plt.figure(figsize=(10, 7))
    plt.plot([0, 1], 
         [0, 1], 
         linestyle='dotted', 
         label='Perfectly Calibrated')
    # for i, model_name in enumerate(models):
    df_probas = pd.read_csv(config.REPORTS_DIR_SEX / "calibration_sex.csv")
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
    plt.xlabel('Mean predicted probability',fontdict = {'fontsize' : 16})
    plt.ylabel('Fraction of positives',fontdict = {'fontsize' : 16})
    plt.title("Calibration curve for patient sex ChestX, pneumothorax", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=15)
    plt.savefig(config.FIGURES_DIR_SEX / "calibration_curve_sex.png", bbox_inches='tight', dpi=300)
    plt.close()

############### PADCHEST CARDIOMEGALY
############### SCANNER
def calibration_curves_scanner_padchest():
    config.FIGURES_DIR_PADCHEST_SCANNER.mkdir(parents=True, exist_ok=True)
    # colors = list(mcolors.TABLEAU_COLORS)
    # plt.figure()
    plt.figure(figsize=(10, 7))
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
    plt.xlabel('Mean predicted probability',fontdict = {'fontsize' : 16})
    plt.ylabel('Fraction of positives',fontdict = {'fontsize' : 16})
    plt.title("Calibration curve for scanner machines PadChest, cardiomegaly", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=15)
    plt.savefig(config.FIGURES_DIR_PADCHEST_SCANNER/ "calibration_curve_scanner.png", bbox_inches='tight', dpi=300)
    plt.close()

############### PADCHEST CARDIOMEGALY
############### SEX
def calibration_curves_sex_padchest():
    config.FIGURES_DIR_PADCHEST_SEX.mkdir(parents=True, exist_ok=True)
    # colors = list(mcolors.TABLEAU_COLORS)
    # plt.figure()
    plt.figure(figsize=(10, 7))
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
    plt.xlabel('Mean predicted probability',fontdict = {'fontsize' : 16})
    plt.ylabel('Fraction of positives',fontdict = {'fontsize' : 16})
    plt.title("Calibration curve for patient sex PadChest, cardiomegaly", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=15)
    plt.savefig(config.FIGURES_DIR_PADCHEST_SEX / "calibration_curve_sex.png", bbox_inches='tight', dpi=300)
    plt.close()

############### PADCHEST PNEUMOTHORAX
############### SCANNER
def calibration_curves_scanner_padchest_px():
    config.FIGURES_DIR_PADCHEST_SCANNER_PX.mkdir(parents=True, exist_ok=True)
    # colors = list(mcolors.TABLEAU_COLORS)
    # plt.figure()
    plt.figure(figsize=(10, 7))
    plt.plot([0, 1], 
         [0, 1], 
         linestyle='dotted', 
         label='Perfectly Calibrated')
    # for i, model_name in enumerate(models):
    df_probas = pd.read_csv(config.REPORTS_DIR_PADCHEST_SCANNER_PX / "calibration_scanner.csv")
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
    plt.xlabel('Mean predicted probability',fontdict = {'fontsize' : 16})
    plt.ylabel('Fraction of positives',fontdict = {'fontsize' : 16})
    plt.title("Calibration curve for scanner machines PadChest, pneumothorax", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=15)
    plt.savefig(config.FIGURES_DIR_PADCHEST_SCANNER_PX/ "calibration_curve_scanner.png", bbox_inches='tight', dpi=300)
    plt.close()

############### PADCHEST PNEUMOTHORAX
############### SEX
def calibration_curves_sex_padchest_px():
    config.FIGURES_DIR_PADCHEST_SEX_PX.mkdir(parents=True, exist_ok=True)
    # colors = list(mcolors.TABLEAU_COLORS)
    # plt.figure()
    plt.figure(figsize=(10, 7))
    plt.plot([0, 1], 
         [0, 1], 
         linestyle='dotted', 
         label='Perfectly Calibrated')
    # for i, model_name in enumerate(models):
    df_probas = pd.read_csv(config.REPORTS_DIR_PADCHEST_SEX_PX / "calibration_sex.csv")
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
    plt.xlabel('Mean predicted probability',fontdict = {'fontsize' : 16})
    plt.ylabel('Fraction of positives',fontdict = {'fontsize' : 16})
    plt.title("Calibration curve for patient sex PadChest, pneumothorax", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=15)
    plt.savefig(config.FIGURES_DIR_PADCHEST_SEX_PX / "calibration_curve_sex.png", bbox_inches='tight', dpi=300)
    plt.close()


if __name__ == "__main__":
    # plotting all confidence curves for all datasets
    plot_confidence()
    plot_confidence_padchest()
    # plot_confidence_padchest_px()
    
    # calibration curves for chestx
    calibration_curves_drains()
    calibration_curves_sex()

    # calibration curves for padchest (cardiomegaly)
    calibration_curves_scanner_padchest()
    calibration_curves_sex_padchest()

    # calibration curves for padchest (pneumothorax)
    # calibration_curves_scanner_padchest_px()
    # calibration_curves_sex_padchest_px()