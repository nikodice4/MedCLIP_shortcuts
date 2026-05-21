import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from . import config
from sklearn.calibration import calibration_curve
import numpy as np
from scipy.stats import bootstrap



############################################# Bootstrap helper function ###################################
def bootstrap_band(predictions_path, subgroup_col=None, subgroup_value=None, filters=None):
    df = pd.read_csv(predictions_path)
    if subgroup_col is not None:
        df = df[df[subgroup_col] == subgroup_value].reset_index(drop=True) #we make a new df with the subgroup and the subgroup value (M fx)
    if filters is not None:
        for col, val in filters.items():
            df = df[df[col] == val]
        df = df.reset_index(drop=True)

    layer_cols = [c for c in df.columns if c.startswith("layer_")] #make list with all layers
    #print(layer_cols)
    layer_cols.sort(key=lambda c: int(c.split("_")[1])) #sort to be in correct order

    
    # Boland confidence per image per layer
    conf = np.array([[abs(p - 0.5) for p in probs] for probs in df[layer_cols].to_numpy()])

    data = tuple(conf[:, i] for i in range(conf.shape[1])) #make tuple for scipy

    def layerwise_means(*layer_samples):
        means = []
        for layer in layer_samples:
            means.append(layer.mean())
        return np.array(means)

    res = bootstrap(
        data,
        statistic=layerwise_means,
        n_resamples=config.N_BOOT,
        confidence_level=0.95,
        method="percentile",
        paired=True,
        random_state=np.random.default_rng(config.BOOT_SEED),
    )

    layers = [int(col.split("_")[1]) for col in layer_cols]
    mean = np.array([layer.mean() for layer in data])

    ci_low = res.confidence_interval.low
    ci_high = res.confidence_interval.high
    return layers, mean, ci_low, ci_high

# if the plot title doesn't say train or test, it is by default test data
############################################# CONFIDENCE CURVES #############################################
############### CHESTX PNEUMOTHORAX
def plot_confidence():
    config.FIGURES_DIR_DRAIN.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR_SEX.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    predictions_path = config.REPORTS_DIR_LAYER / "predictions_per_layer_test.csv"

    overall_data = {
        "Train data for NIH-CXR14, pneumothorax": ("mean_probability_per_layer_train.csv", config.REPORTS_DIR, config.FIGURES_DIR),
        "Test data for NIH-CXR14, pneumothorax": ("mean_probability_per_layer_test.csv", config.REPORTS_DIR, config.FIGURES_DIR)
        }

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, (filename, reports_dir, figures_dir) in overall_data.items():
        df = pd.read_csv(reports_dir / filename)
        # confidence defined at 0.5
        # df["confidence"] = (df["mean_probability"] - 0.5).abs()
        line, = ax.plot(df["layer"], df["mean_probability"], label=name, marker="o") # having to rename to mean_probability (actuallu confidence)
        # ax.plot(test_df["layer"],  test_df["confidence"],  label="Test",  marker="o")
        # ax.plot(drain_df["layer"],  drain_df["confidence"],  label="Drains",  marker="o")
        # ax.plot(no_drain_df["layer"],  no_drain_df["confidence"],  label="No drains",  marker="o")
        if "Test" in name:
            layers, mean, lo, hi = bootstrap_band(predictions_path)
            ax.fill_between(layers, lo, hi, alpha=0.2, color=line.get_color())

    ax.set_xlabel("Layer", fontdict = {'fontsize' : 16})
    ax.set_ylabel("Confidence", fontdict = {'fontsize' : 16})
    ax.set_title("Confidence per layer overall NIH-CXR14, pneumothorax", fontsize=18)
    ax.set_xticks(df["layer"].unique())
    ax.legend()
    ax.grid(axis='y')
    ax.set_ylim(0, 0.5)

    plt.tight_layout()

    out_path = figures_dir / f"confidence_chestX_pneumothorax_overall.png"
    plt.savefig(out_path, dpi=300)
    print(f"saved confidence overall (chestx) -> {out_path}")

    ############### SEX PLOTS COMBINED
    sex_data = {
        "Female patients NIH-CXR14, pneumothorax": ("mean_probability_per_layer_female.csv", config.REPORTS_DIR_SEX, config.FIGURES_DIR_SEX, "F"),
        "Male patients NIH-CXR14, pneumothorax": ("mean_probability_per_layer_male.csv", config.REPORTS_DIR_SEX, config.FIGURES_DIR_SEX, "M"),
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, (filename, reports_dir, figures_dir, sex_val) in sex_data.items():
        df = pd.read_csv(reports_dir / filename)
        # df["confidence"] = (df["mean_probability"] - 0.5).abs()
        line, = ax.plot(df["layer"], df["mean_probability"], label=label, marker="o") # having to rename to mean_probability (actuallu confidence)
        layers, mean, lo, hi = bootstrap_band(predictions_path, "sex", sex_val)
        ax.fill_between(layers, lo, hi, alpha=0.2, color=line.get_color())

    ax.set_xlabel("Layer", fontdict = {'fontsize' : 16})
    ax.set_ylabel("Confidence", fontdict = {'fontsize' : 16})
    ax.set_title("Confidence per layer patient sex NIH-CXR14, pneumothorax", fontsize=18)
    ax.set_xticks(df["layer"].unique())
    ax.legend()
    ax.grid(axis='y')
    ax.set_ylim(0, 0.5)

    plt.tight_layout()

    out_path = figures_dir / f"confidence_chestX_pneumothorax_sex.png"
    plt.savefig(out_path, dpi=300)
    print(f"saved confidence sex (chestx) -> {out_path}")

    ############### DRAIN PLOTS COMBINED
    drain_data = {
        "Drains NIH-CXR14, pneumothorax": ("mean_probability_per_layer_drain.csv", config.REPORTS_DIR_DRAIN, config.FIGURES_DIR_DRAIN, 1),
        "No drains NIH-CXR14, pneumothorax": ("mean_probability_per_layer_nodrain.csv", config.REPORTS_DIR_DRAIN, config.FIGURES_DIR_DRAIN, 0),
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, (filename, reports_dir, figures_dir, drain_val) in drain_data.items():
        df = pd.read_csv(reports_dir / filename)
        # df["confidence"] = (df["mean_probability"] - 0.5).abs()
        line, = ax.plot(df["layer"], df["mean_probability"], label=label, marker="o") # having to rename to mean_probability (actuallu confidence)
        layers, mean, lo, hi = bootstrap_band(predictions_path, "drain", drain_val)
        ax.fill_between(layers, lo, hi, alpha=0.2, color=line.get_color())

    ax.set_xlabel("Layer", fontdict = {'fontsize' : 16})
    ax.set_ylabel("Confidence", fontdict = {'fontsize' : 16})
    ax.set_title("Confidence per layer drains NIH-CXR14, pneumothorax", fontsize=18)
    ax.set_xticks(df["layer"].unique())
    ax.legend()
    ax.grid(axis='y')
    ax.set_ylim(0, 0.5)

    plt.tight_layout()

    out_path = figures_dir / f"confidence_chestX_pneumothorax_drain.png"
    plt.savefig(out_path, dpi=300)
    print(f"saved confidence drains (chestx) -> {out_path}")

############### GOLD LABEL x DRAIN COMBINED
    df_pred = pd.read_csv(predictions_path)
    layer_cols = sorted([c for c in df_pred.columns if c.startswith("layer_")],
                        key=lambda c: int(c.split("_")[1]))
    layers_x = [int(c.split("_")[1]) for c in layer_cols]

    conf_df = (df_pred[layer_cols] - 0.5).abs()

    conf_df["y_true"] = df_pred["y_true"]
    conf_df["drain"] = df_pred["drain"]

    label_drain_data = {
        "Pneumothorax positive, drain": {"y_true": 1, "drain": 1},
        "Pneumothorax positive, no drain": {"y_true": 1, "drain": 0},
        "Pneumothorax negative, drain": {"y_true": 0, "drain": 1},
        "Pneumothorax negative, no drain": {"y_true": 0, "drain": 0},
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, filters in label_drain_data.items():
        subset = conf_df[(conf_df["y_true"] == filters["y_true"]) & (conf_df["drain"] == filters["drain"])]
        mean_per_layer = subset[layer_cols].mean()
        line, = ax.plot(layers_x, mean_per_layer.values, label=f"{label} (n={len(subset)})", marker="o")
        layers, mean, lo, hi = bootstrap_band(predictions_path, filters=filters)
        ax.fill_between(layers, lo, hi, alpha=0.2, color=line.get_color())

    ax.set_xlabel("Layer", fontdict = {'fontsize' : 16})
    ax.set_ylabel("Confidence", fontdict = {'fontsize' : 16})
    ax.set_title("Confidence per layer drains NIH-CXR14, pneumothorax", fontsize=18)
    ax.set_xticks(layers_x)
    ax.legend()
    ax.grid(axis='y')
    ax.set_ylim(0, 0.5)

    plt.tight_layout()

    out_path = config.FIGURES_DIR_DRAIN / "confidence_chestX_pneumothorax_drain_gold_label.png"
    plt.savefig(out_path, dpi=300)
    print(f"saved confidence gold label x drain (chestx) -> {out_path}")

    ############### GOLD LABEL x SEX COMBINED
    df_pred = pd.read_csv(predictions_path)
    layer_cols = sorted([c for c in df_pred.columns if c.startswith("layer_")],
                        key=lambda c: int(c.split("_")[1]))
    layers_x = [int(c.split("_")[1]) for c in layer_cols]

    conf_df = (df_pred[layer_cols] - 0.5).abs()

    conf_df["y_true"] = df_pred["y_true"]
    conf_df["sex"] = df_pred["sex"]

    label_sex_data = {
        "Pneumothorax positive, female": {"y_true": 1, "sex": "F"},
        "Pneumothorax positive, male": {"y_true": 1, "sex": "M"},
        "Pneumothorax negative, female": {"y_true": 0, "sex": "F"},
        "Pneumothorax negative, male":   {"y_true": 0, "sex": "M"},
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, filters in label_sex_data.items():
        subset = conf_df[(conf_df["y_true"] == filters["y_true"]) & (conf_df["sex"] == filters["sex"])]
        mean_per_layer = subset[layer_cols].mean()
        line, = ax.plot(layers_x, mean_per_layer.values, label=f"{label} (n={len(subset)})", marker="o")
        layers, mean, lo, hi = bootstrap_band(predictions_path, filters=filters)
        ax.fill_between(layers, lo, hi, alpha=0.2, color=line.get_color())

    ax.set_xlabel("Layer", fontdict = {'fontsize' : 16})
    ax.set_ylabel("Confidence", fontdict = {'fontsize' : 16})
    ax.set_title("Confidence per layer patient sex NIH-CXR14, pneumothorax", fontsize=18)
    ax.set_xticks(layers_x)
    ax.legend()
    ax.grid(axis='y')
    ax.set_ylim(0, 0.5)

    plt.tight_layout()

    out_path = config.FIGURES_DIR_SEX / "confidence_chestX_pneumothorax_sex_gold_label.png"
    plt.savefig(out_path, dpi=300)
    print(f"saved confidence gold label x sex (chestx) -> {out_path}")

############### PADCHEST CARDIOMGELY
def plot_confidence_padchest():
    config.FIGURES_DIR_PADCHEST.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR_PADCHEST_SEX.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR_PADCHEST_SCANNER.mkdir(parents=True, exist_ok=True)

    predictions_path = config.REPORTS_DIR_PADCHEST_LAYER / "predictions_per_layer_test.csv"

    overall_data = {
        "Train data PadChest, cardiomegaly": ("mean_probability_per_layer_train.csv", config.REPORTS_DIR_PADCHEST, config.FIGURES_DIR_PADCHEST),
        "Test data PadChest, cardiomegaly": ("mean_probability_per_layer_test.csv", config.REPORTS_DIR_PADCHEST, config.FIGURES_DIR_PADCHEST),
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, (filename, reports_dir, figures_dir) in overall_data.items():
        df = pd.read_csv(reports_dir / filename)
        # confidence defined at 0.5
        # df["confidence"] = (df["mean_probability"] - 0.5).abs()
        line, = ax.plot(df["layer"], df["mean_probability"], label=name, marker="o") # having to rename to mean_probability (actuallu confidence)
        # ax.plot(test_df["layer"],  test_df["confidence"],  label="Test",  marker="o")
        # ax.plot(drain_df["layer"],  drain_df["confidence"],  label="Drains",  marker="o")
        # ax.plot(no_drain_df["layer"],  no_drain_df["confidence"],  label="No drains",  marker="o")
        if "Test" in name:
            layers, mean, lo, hi = bootstrap_band(predictions_path)
            ax.fill_between(layers, lo, hi, alpha=0.2, color=line.get_color())
    ax.set_xlabel("Layer", fontdict = {'fontsize' : 16})
    ax.set_ylabel("Confidence", fontdict = {'fontsize' : 16})
    ax.set_title("Confidence per layer overall PadChest data, cardiomegaly", fontsize=18)
    ax.set_xticks(df["layer"].unique())
    ax.legend()
    ax.grid(axis='y')
    ax.set_ylim(0, 0.5)

    plt.tight_layout()

    out_path = figures_dir / f"confidence_padchest_cardiomegaly_overall.png"
    plt.savefig(out_path, dpi=300)
    print(f"saved confidence overall (padchest) -> {out_path}")

    ############### SEX PLOTS COMBINED
    sex_data = {
        "Female patients PadChest, cardiomegaly": ("mean_probability_per_layer_female.csv", config.REPORTS_DIR_PADCHEST_SEX, config.FIGURES_DIR_PADCHEST_SEX, "F"),
        "Male patients PadChest, cardiomegaly": ("mean_probability_per_layer_male.csv", config.REPORTS_DIR_PADCHEST_SEX, config.FIGURES_DIR_PADCHEST_SEX, "M"),
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, (filename, reports_dir, figures_dir, sex_val) in sex_data.items():
        df = pd.read_csv(reports_dir / filename)
        # df["confidence"] = (df["mean_probability"] - 0.5).abs()
        line, = ax.plot(df["layer"], df["mean_probability"], label=label, marker="o") # having to rename to mean_probability (actuallu confidence)
        layers, mean, lo, hi = bootstrap_band(predictions_path, "sex", sex_val)
        ax.fill_between(layers, lo, hi, alpha=0.2, color=line.get_color())

    ax.set_xlabel("Layer", fontdict = {'fontsize' : 16})
    ax.set_ylabel("Confidence", fontdict = {'fontsize' : 16})
    ax.set_title("Confidence per layer patient sex PadChest data, cardiomegaly", fontsize=18)
    ax.set_xticks(df["layer"].unique())
    ax.legend()
    ax.grid(axis='y')
    ax.set_ylim(0, 0.5)

    plt.tight_layout()

    out_path = figures_dir / f"confidence_padchest_cardiomegaly_sex.png"
    plt.savefig(out_path, dpi=300)
    print(f"saved confidence sex (padchest) -> {out_path}")

    ############### SCANNER PLOTS COMBINED
    scanner_data = {
        "ImagingDynamicsCompanyLtd PadChest, cardiomegaly": ("mean_probability_per_layer_ImagingDynamicsCompanyLtd.csv", config.REPORTS_DIR_PADCHEST_SCANNER, config.FIGURES_DIR_PADCHEST_SCANNER, "ImagingDynamicsCompanyLtd"),
        "PhilipsMedicalSystems PadChest, cardiomegaly": ("mean_probability_per_layer_PhilipsMedicalSystems.csv", config.REPORTS_DIR_PADCHEST_SCANNER, config.FIGURES_DIR_PADCHEST_SCANNER, "PhilipsMedicalSystems"),
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, (filename, reports_dir, figures_dir, scanner_val) in scanner_data.items():
        df = pd.read_csv(reports_dir / filename)
        # df["confidence"] = (df["mean_probability"] - 0.5).abs()
        line, = ax.plot(df["layer"], df["mean_probability"], label=label, marker="o") # having to rename to mean_probability (actuallu confidence)
        layers, mean, lo, hi = bootstrap_band(predictions_path, "scanner", scanner_val)
        ax.fill_between(layers, lo, hi, alpha=0.2, color=line.get_color())

    ax.set_xlabel("Layer", fontdict = {'fontsize' : 16})
    ax.set_ylabel("Confidence", fontdict = {'fontsize' : 16})
    ax.set_title("Confidence per layer scanner machines PadChest, cardiomegaly", fontsize=18)
    ax.set_xticks(df["layer"].unique())
    ax.legend()
    ax.grid(axis='y')
    ax.set_ylim(0, 0.5)

    plt.tight_layout()

    out_path = figures_dir / f"confidence_padchest_cardiomegaly_scanner.png"
    plt.savefig(out_path, dpi=300)
    print(f"saved confidence scanner (padchest) -> {out_path}")

############### GOLD LABEL x SCANNER COMBINED
    df_pred = pd.read_csv(predictions_path)
    layer_cols = sorted([c for c in df_pred.columns if c.startswith("layer_")],
                        key=lambda c: int(c.split("_")[1]))
    layers_x = [int(c.split("_")[1]) for c in layer_cols]

    conf_df = (df_pred[layer_cols] - 0.5).abs()

    conf_df["y_true"] = df_pred["y_true"]
    conf_df["scanner"] = df_pred["scanner"]

    label_scanner_data = {
        "Cardiomegaly positive, ImagingDynamicsCompanyLtd": {"y_true": 1, "scanner": "ImagingDynamicsCompanyLtd"},
        "Cardiomegaly positive, PhilipsMedicalSystems": {"y_true": 1, "scanner": "PhilipsMedicalSystems"},
        "Cardiomegaly negative, ImagingDynamicsCompanyLtd": {"y_true": 0, "scanner": "ImagingDynamicsCompanyLtd"},
        "Cardiomegaly negative, PhilipsMedicalSystems": {"y_true": 0, "scanner": "PhilipsMedicalSystems"},
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, filters in label_scanner_data.items():
        subset = conf_df[(conf_df["y_true"] == filters["y_true"]) & (conf_df["scanner"] == filters["scanner"])]
        mean_per_layer = subset[layer_cols].mean()
        line, = ax.plot(layers_x, mean_per_layer.values, label=f"{label} (n={len(subset)})", marker="o")
        layers, mean, lo, hi = bootstrap_band(predictions_path, filters=filters)
        ax.fill_between(layers, lo, hi, alpha=0.2, color=line.get_color())

    ax.set_xlabel("Layer", fontdict = {'fontsize' : 16})
    ax.set_ylabel("Confidence", fontdict = {'fontsize' : 16})
    ax.set_title("Confidence per layer scanner machines PadChest, cardiomegaly", fontsize=18)
    ax.set_xticks(layers_x)
    ax.legend()
    ax.grid(axis='y')
    ax.set_ylim(0, 0.5)

    plt.tight_layout()

    out_path = config.FIGURES_DIR_PADCHEST_SCANNER / "confidence_padchest_cardiomegaly_scanner_gold_label.png"
    plt.savefig(out_path, dpi=300)
    print(f"saved confidence gold label x scanner (padchest) -> {out_path}")

    ############### GOLD LABEL x SEX COMBINED
    df_pred = pd.read_csv(predictions_path)
    layer_cols = sorted([c for c in df_pred.columns if c.startswith("layer_")],
                        key=lambda c: int(c.split("_")[1]))
    layers_x = [int(c.split("_")[1]) for c in layer_cols]
    conf_df = (df_pred[layer_cols] - 0.5).abs() # Boland confidence: |p - 0.5|
    conf_df["y_true"] = df_pred["y_true"]
    conf_df["sex"] = df_pred["sex"]

    label_sex_data = {
        "Cardiomegaly positive, female": {"y_true": 1, "sex": "F"},
        "Cardiomegaly positive, male": {"y_true": 1, "sex": "M"},
        "Cardiomegaly negative, female": {"y_true": 0, "sex": "F"},
        "Cardiomegaly negative, male": {"y_true": 0, "sex": "M"},
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, filters in label_sex_data.items():
        subset = conf_df[(conf_df["y_true"] == filters["y_true"]) & (conf_df["sex"] == filters["sex"])]
        mean_per_layer = subset[layer_cols].mean()
        line, = ax.plot(layers_x, mean_per_layer.values, label=f"{label} (n={len(subset)})", marker="o")
        layers, mean, lo, hi = bootstrap_band(predictions_path, filters=filters)
        ax.fill_between(layers, lo, hi, alpha=0.2, color=line.get_color())

    ax.set_xlabel("Layer", fontdict = {'fontsize' : 16})
    ax.set_ylabel("Confidence", fontdict = {'fontsize' : 16})
    ax.set_title("Confidence per layer patient sex PadChest, cardiomegaly", fontsize=18)
    ax.set_xticks(layers_x)
    ax.legend()
    ax.grid(axis='y')
    ax.set_ylim(0, 0.5)

    plt.tight_layout()

    out_path = config.FIGURES_DIR_PADCHEST_SEX / "confidence_padchest_pneumothorax_sex_gold_label.png"
    plt.savefig(out_path, dpi=300)
    print(f"saved confidence gold label x sex (padchest) -> {out_path}")

############### PADCHEST PNEUMOTHORAX
def plot_confidence_padchest_px():
    config.FIGURES_DIR_PADCHEST_PX.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR_PADCHEST_SEX_PX.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR_PADCHEST_SCANNER_PX.mkdir(parents=True, exist_ok=True)

    overall_data = {
        "Train data PadChest, pneumothorax": ("mean_probability_per_layer_train.csv", config.REPORTS_DIR_PADCHEST_PX, config.FIGURES_DIR_PADCHEST_PX),
        "Test data PadChest, pneumothorax": ("mean_probability_per_layer_test.csv", config.REPORTS_DIR_PADCHEST_PX, config.FIGURES_DIR_PADCHEST_PX),
    }

    predictions_path = config.REPORTS_DIR_PADCHEST_LAYER_PX / "predictions_per_layer_test.csv"

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, (filename, reports_dir, figures_dir) in overall_data.items():
        df = pd.read_csv(reports_dir / filename)
        # confidence defined at 0.5
        # df["confidence"] = (df["mean_probability"] - 0.5).abs()
        line, = ax.plot(df["layer"], df["mean_probability"], label=name, marker="o") # having to rename to mean_probability (actuallu confidence)
        # ax.plot(test_df["layer"],  test_df["confidence"],  label="Test",  marker="o")
        # ax.plot(drain_df["layer"],  drain_df["confidence"],  label="Drains",  marker="o")
        # ax.plot(no_drain_df["layer"],  no_drain_df["confidence"],  label="No drains",  marker="o")
        if "Test" in name:
            layers, mean, lo, hi = bootstrap_band(predictions_path)
            ax.fill_between(layers, lo, hi, alpha=0.2, color=line.get_color())
    ax.set_xlabel("Layer", fontdict = {'fontsize' : 16})
    ax.set_ylabel("Confidence", fontdict = {'fontsize' : 16})
    ax.set_title("Confidence per layer overall PadChest, pneumothorax", fontsize=18)
    ax.set_xticks(df["layer"].unique())
    ax.legend()
    ax.grid(axis='y')
    ax.set_ylim(0, 0.5)

    plt.tight_layout()

    out_path = figures_dir / f"confidence_padchest_pneumothorax_overall.png"
    plt.savefig(out_path, dpi=300)
    print(f"saved confidence overall (padchest_px) -> {out_path}")

    ############### SEX PLOTS COMBINED
    sex_data = {
        "Female patients PadChest, pneumothorax": ("mean_probability_per_layer_female.csv", config.REPORTS_DIR_PADCHEST_SEX_PX, config.FIGURES_DIR_PADCHEST_SEX_PX, "F"),
        "Male patients PadChest, pneumothorax": ("mean_probability_per_layer_male.csv", config.REPORTS_DIR_PADCHEST_SEX_PX, config.FIGURES_DIR_PADCHEST_SEX_PX, "M"),
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, (filename, reports_dir, figures_dir, sex_val) in sex_data.items():
        df = pd.read_csv(reports_dir / filename)
        # df["confidence"] = (df["mean_probability"] - 0.5).abs()
        line, = ax.plot(df["layer"], df["mean_probability"], label=label, marker="o") # having to rename to mean_probability (actuallu confidence)
        layers, mean, lo, hi = bootstrap_band(predictions_path, "sex", sex_val)
        ax.fill_between(layers, lo, hi, alpha=0.2, color=line.get_color())

    ax.set_xlabel("Layer", fontdict = {'fontsize' : 16})
    ax.set_ylabel("Confidence", fontdict = {'fontsize' : 16})
    ax.set_title("Confidence per layer patient sex PadChest, pneumothorax", fontsize=18)
    ax.set_xticks(df["layer"].unique())
    ax.legend()
    ax.grid(axis='y')
    ax.set_ylim(0, 0.5)

    plt.tight_layout()

    out_path = figures_dir / f"confidence_padchest_pneumothorax_sex.png"
    plt.savefig(out_path, dpi=300)
    print(f"saved confidence sex (padchest_px) -> {out_path}")

    ############### SCANNER PLOTS COMBINED
    scanner_data = {
        "ImagingDynamicsCompanyLtd PadChest, pneumothorax": ("mean_probability_per_layer_ImagingDynamicsCompanyLtd.csv", config.REPORTS_DIR_PADCHEST_SCANNER_PX, config.FIGURES_DIR_PADCHEST_SCANNER_PX, "ImagingDynamicsCompanyLtd"),
        "PhilipsMedicalSystems PadChest, pneumothorax": ("mean_probability_per_layer_PhilipsMedicalSystems.csv", config.REPORTS_DIR_PADCHEST_SCANNER_PX, config.FIGURES_DIR_PADCHEST_SCANNER_PX, "PhilipsMedicalSystems"),
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, (filename, reports_dir, figures_dir, scanner_val) in scanner_data.items():
        df = pd.read_csv(reports_dir / filename)
        # df["confidence"] = (df["mean_probability"] - 0.5).abs()
        line, = ax.plot(df["layer"], df["mean_probability"], label=label, marker="o") # having to rename to mean_probability (actuallu confidence)
        layers, mean, lo, hi = bootstrap_band(predictions_path, "scanner", scanner_val)
        ax.fill_between(layers, lo, hi, alpha=0.2, color=line.get_color())

    ax.set_xlabel("Layer", fontdict = {'fontsize' : 16})
    ax.set_ylabel("Confidence", fontdict = {'fontsize' : 16})
    ax.set_title("Confidence per layer scanner machines PadChest, pneumothorax", fontsize=18)
    ax.set_xticks(df["layer"].unique())
    ax.legend()
    ax.grid(axis='y')
    ax.set_ylim(0, 0.5)

    plt.tight_layout()

    out_path = figures_dir / f"confidence_padchest_pneumothorax_sex_scanner.png"
    plt.savefig(out_path, dpi=300)
    print(f"saved confidence scanner (padchest_px) -> {out_path}")

############### GOLD LABEL x SCANNER COMBINED
    df_pred = pd.read_csv(predictions_path)
    layer_cols = sorted([c for c in df_pred.columns if c.startswith("layer_")],
                        key=lambda c: int(c.split("_")[1]))
    layers_x = [int(c.split("_")[1]) for c in layer_cols]

    conf_df = (df_pred[layer_cols] - 0.5).abs()

    conf_df["y_true"] = df_pred["y_true"]
    conf_df["scanner"] = df_pred["scanner"]

    label_scanner_data = {
        "Pneumothorax positive, ImagingDynamicsCompanyLtd": {"y_true": 1, "scanner": "ImagingDynamicsCompanyLtd"},
        "Pneumothorax positive, PhilipsMedicalSystems": {"y_true": 1, "scanner": "PhilipsMedicalSystems"},
        "Pneumothorax negative, ImagingDynamicsCompanyLtd": {"y_true": 0, "scanner": "ImagingDynamicsCompanyLtd"},
        "Pneumothorax negative, PhilipsMedicalSystems": {"y_true": 0, "scanner": "PhilipsMedicalSystems"},
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, filters in label_scanner_data.items():
        subset = conf_df[(conf_df["y_true"] == filters["y_true"]) & (conf_df["scanner"] == filters["scanner"])]
        mean_per_layer = subset[layer_cols].mean()
        line, = ax.plot(layers_x, mean_per_layer.values, label=f"{label} (n={len(subset)})", marker="o")
        layers, mean, lo, hi = bootstrap_band(predictions_path, filters=filters)
        ax.fill_between(layers, lo, hi, alpha=0.2, color=line.get_color())

    ax.set_xlabel("Layer", fontdict = {'fontsize' : 16})
    ax.set_ylabel("Confidence", fontdict = {'fontsize' : 16})
    ax.set_title("Confidence per layer scanner machines PadChest, pneumothorax", fontsize=18)
    ax.set_xticks(layers_x)
    ax.legend()
    ax.grid(axis='y')
    ax.set_ylim(0, 0.5)

    plt.tight_layout()

    out_path = config.FIGURES_DIR_PADCHEST_SCANNER_PX / "confidence_padchest_pneumothorax_scanner_gold_label.png"
    plt.savefig(out_path, dpi=300)
    print(f"saved confidence gold label x scanner (padchest_px) -> {out_path}")

    ############### GOLD LABEL x SEX COMBINED
    df_pred = pd.read_csv(predictions_path)
    layer_cols = sorted([c for c in df_pred.columns if c.startswith("layer_")],
                        key=lambda c: int(c.split("_")[1]))
    layers_x = [int(c.split("_")[1]) for c in layer_cols]

    conf_df = (df_pred[layer_cols] - 0.5).abs()

    conf_df["y_true"] = df_pred["y_true"]
    conf_df["sex"] = df_pred["sex"]

    label_sex_data = {
        "Pneumothorax positive, female": {"y_true": 1, "sex": "F"},
        "Pneumothorax positive, male": {"y_true": 1, "sex": "M"},
        "Pneumothorax negative, female": {"y_true": 0, "sex": "F"},
        "Pneumothorax negative, male": {"y_true": 0, "sex": "M"},
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, filters in label_sex_data.items():
        subset = conf_df[(conf_df["y_true"] == filters["y_true"]) & (conf_df["sex"] == filters["sex"])]
        mean_per_layer = subset[layer_cols].mean()
        line, = ax.plot(layers_x, mean_per_layer.values, label=f"{label} (n={len(subset)})", marker="o")
        layers, mean, lo, hi = bootstrap_band(predictions_path, filters=filters)
        ax.fill_between(layers, lo, hi, alpha=0.2, color=line.get_color())

    ax.set_xlabel("Layer", fontdict = {'fontsize' : 16})
    ax.set_ylabel("Confidence", fontdict = {'fontsize' : 16})
    ax.set_title("Confidence per layer patient sex PadChest, pneumothorax", fontsize=18)
    ax.set_xticks(layers_x)
    ax.legend()
    ax.grid(axis='y')
    ax.set_ylim(0, 0.5)

    plt.tight_layout()

    out_path = config.FIGURES_DIR_PADCHEST_SEX_PX / "confidence_padchest_pneumothorax_sex_gold_label.png"
    plt.savefig(out_path, dpi=300)
    print(f"saved confidence gold label x sex (padchest_px) -> {out_path}")


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
    plt.grid(axis='y')
    plt.xlabel('Mean predicted probability',fontdict = {'fontsize' : 16})
    plt.ylabel('Fraction of positives',fontdict = {'fontsize' : 16})
    plt.title("Calibration curve for drains NIH-CXR14, pneumothorax", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=15)
    plt.savefig(config.FIGURES_DIR_DRAIN / "calibration_curve_drains.png", bbox_inches='tight', dpi=300)
    #plt.close()

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
    plt.grid(axis='y')
    plt.xlabel('Mean predicted probability',fontdict = {'fontsize' : 16})
    plt.ylabel('Fraction of positives',fontdict = {'fontsize' : 16})
    plt.title("Calibration curve for patient sex NIH-CXR14, pneumothorax", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=15)
    plt.savefig(config.FIGURES_DIR_SEX / "calibration_curve_sex.png", bbox_inches='tight', dpi=300)
    #plt.close()

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
    plt.grid(axis='y')
    plt.xlabel('Mean predicted probability',fontdict = {'fontsize' : 16})
    plt.ylabel('Fraction of positives',fontdict = {'fontsize' : 16})
    plt.title("Calibration curve for scanner machines PadChest, cardiomegaly", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=15)
    plt.savefig(config.FIGURES_DIR_PADCHEST_SCANNER/ "calibration_curve_scanner.png", bbox_inches='tight', dpi=300)
    #plt.close()

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
    plt.grid(axis='y')
    plt.xlabel('Mean predicted probability',fontdict = {'fontsize' : 16})
    plt.ylabel('Fraction of positives',fontdict = {'fontsize' : 16})
    plt.title("Calibration curve for patient sex PadChest, cardiomegaly", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=15)
    plt.savefig(config.FIGURES_DIR_PADCHEST_SEX / "calibration_curve_sex.png", bbox_inches='tight', dpi=300)
    #plt.close()

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
    plt.grid(axis='y')
    plt.xlabel('Mean predicted probability',fontdict = {'fontsize' : 16})
    plt.ylabel('Fraction of positives',fontdict = {'fontsize' : 16})
    plt.title("Calibration curve for scanner machines PadChest, pneumothorax", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=15)
    plt.savefig(config.FIGURES_DIR_PADCHEST_SCANNER_PX/ "calibration_curve_scanner.png", bbox_inches='tight', dpi=300)
    #plt.close()

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
    plt.grid(axis='y')
    plt.xlabel("Mean predicted probability",fontdict = {'fontsize' : 16})
    plt.ylabel("Fraction of positives",fontdict = {'fontsize' : 16})
    plt.title("Calibration curve for patient sex PadChest, pneumothorax", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=15)
    plt.savefig(config.FIGURES_DIR_PADCHEST_SEX_PX / "calibration_curve_sex.png", bbox_inches='tight', dpi=300)
    #plt.close()


if __name__ == "__main__":
    # plotting all confidence curves for all datasets
    plot_confidence()
    plot_confidence_padchest()
    plot_confidence_padchest_px()
    
    # calibration curves for chestx
    calibration_curves_drains()
    calibration_curves_sex()

    # calibration curves for padchest (cardiomegaly)
    calibration_curves_scanner_padchest()
    calibration_curves_sex_padchest()

    # calibration curves for padchest (pneumothorax)
    calibration_curves_scanner_padchest_px()
    calibration_curves_sex_padchest_px()