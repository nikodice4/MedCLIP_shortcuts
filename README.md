# MedCLIP subgroup project

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

MedCLIP model trained on medical imaging data to uncover subgroups for robustness and fairness assessment

## Project Organization

```              
├── data               <- Data folder is in git.ignore
│   ├── processed
|   |   ├── ChestX-ray14
|   |   |   ├── test
|   |   |   |   ├── files
|   |   |   |   |   └── processed_labels_drains.csv
|   |   |   |   └── images
|   |   |   └── train
|   |   |       ├── files
|   |   |       |   └── processed_labels_train.csv
|   |   |       └── images
|   |   └── PadChest
|   |       ├── processed_labels_px.csv
|   |       ├── processed_labels.csv
|   |       └── images
│   |   
│   └── raw
|       ├── ChestX-ray14
|       |   ├── images
|       |   ├── Data_Entry_2017.csv
|       |   ├── NIH-CX14_TubeAnnotations_NonExperts_aggregated.csv
|       |   ├── test_list.txt
|       |   ├── train_val_list.txt
|       |   └── images
|       └── PadChest
|           ├── images
|           ├── Invalid_images.csv
|           └── PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv
│
├── docs               
│
├── models
│   ├── checkpoint
|   |   └── Path for when the model stopped running on the HPC
│   │
│   ├── medclip
|   |   └── The whole MedCLIP GitHub: https://github.com/RyanWangZf/MedCLIP
│   │
│   └── probes
|       └── Weights for the linear probes' dataset configurations
│
├── notebooks      
│   ├── eda_pre.ipynb
│   │ 
│   ├── look_into_stats.ipynb
│   │
│   ├── manual_analysis.ipynb
│   │
│   └── metric_notebook.ipynb
│
├── padchest_splits     
│   ├── padchest
|   |   ├── train_split.csv
|   |   ├── val_split.csv
|   |   └── test_split.csv
│   │
│   └── padchest_px
|       ├── train_split.csv
|       ├── val_split.csv
|       └── test_split.csv
|
├── pretrained
│   └── medclip-resnet
|       └── MedCLIP weights you can downloads from Wang et al.'s GitHub
│
├── references
│
├── reports
│   ├── chestx
│   │   ├── confidence_images
│   │   |   ├── disease
|   |   |   |   └── 5 images with the highest predicted probability of disease
│   │   |   ├── no_disease
|   |   |   |   └── 5 images with the lowest predicted probability of disease
│   │   |   └── uncertain
|   |   |       └── 5 images with the most uncertain predicted probability of disease (closest to 0.5)
│   │   │
│   │   ├── drain
│   │   |   ├── calibration_drain.csv
│   │   |   ├── mean_probability_per_layer_drain.csv
│   │   |   └── mean_probability_per_layer_nodrain.csv
│   │   │
│   │   ├── layer
│   │   |   └── predictions_per_layer_test.csv
│   │   │
│   │   └── sex
│   │       ├── calibration_sex.csv
│   │       ├── mean_probability_per_layer_female.csv
│   │       └── mean_probability_per_layer_male.csv
│   │ 
│   ├── figures
│   │   ├── chestx
│   │   │   ├── drain_plots
|   |   |   |   └── Calibration and confidence curve images
│   │   │   │ 
│   │   │   └── sex_plots
|   |   |       └── Calibration and confidence curve images
│   │   │ 
│   │   ├── padchest
│   │   │   ├── scanner_plots
|   |   |   |   └── Calibration and confidence curve images
│   │   │   │  
│   │   │   └── sex_plots
|   |   |       └── Calibration and confidence curve images
│   │   │ 
│   │   └── padchest_px
│   │       ├── scanner_plots
|   |       |   └── Calibration and confidence curve images
│   │       │  
│   │       └── sex_plots
|   |           └── Calibration and confidence curve images
│   │
│   ├── padchest
│   │   ├── confidence_images
│   │   |   ├── disease
|   |   |   |   └── 5 images with the highest predicted probability of disease
│   │   |   ├── no_disease
|   |   |   |   └── 5 images with the lowest predicted probability of disease
│   │   |   └── uncertain
|   |   |       └── 5 images with the most uncertain predicted probability of disease (closest to 0.5)
│   │   │
│   │   ├── padchest_scanner
│   │   |   ├── calibration_scanner.csv
│   │   |   ├── mean_probability_per_layer_ImagingDynamicsCompanyLtd.csv
│   │   |   └── mean_probability_per_layer_PhilipsMedicalSystems.csv
│   │   │
│   │   ├── layer
│   │   |   └── predictions_per_layer_test.csv
│   │   │
│   │   └── padchest_sex
│   │       ├── calibration_sex.csv
│   │       ├── mean_probability_per_layer_female.csv
│   │       └── mean_probability_per_layer_male.csv
│   │      
│   └── padchest_px
│   │   ├── confidence_images
│   │   |   ├── disease
|   |   |   |   └── 5 images with the highest predicted probability of disease
│   │   |   ├── no_disease
|   |   |   |   └── 5 images with the lowest predicted probability of disease
│   │   |   └── uncertain
|   |   |       └── 5 images with the most uncertain predicted probability of disease (closest to 0.5)
│   │   │
│   │   ├── padchest_scanner_px
│   │   |   ├── calibration_scanner.csv
│   │   |   ├── mean_probability_per_layer_ImagingDynamicsCompanyLtd.csv
│   │   |   └── mean_probability_per_layer_PhilipsMedicalSystems.csv
│   │   │
│   │   ├── layer
│   │   |   └── predictions_per_layer_test.csv
│   │   │
│   │   └── padchest_sex_px
│   │       ├── calibration_sex.csv
│   │       ├── mean_probability_per_layer_female.csv
│   │       └── mean_probability_per_layer_male.csv
│   |
│   └── src
│       ├── data
|       │   ├── data_split.py
|       │   │
|       │   ├── datasets.py
|       │   │
|       │   ├── process_px_padchest.py
|       │   │
|       │   ├── processing_padchest.py
|       │   │
|       │   ├── run_data_split.py
|       │   │
|       │   ├── test_processing_chestX.py
|       │   │
|       │   └── train_processing_chestX.py
│       |
|       ├── models
|       │   ├── checkpointing.py
|       │   │ 
|       │   ├── config.py
|       │   │
|       │   ├── evaluation_padchest_px.py
|       │   │
|       │   ├── evaluation_padchest.py
|       │   │
|       │   ├── evaluation.py
|       │   │
|       │   ├── main.py
|       │   │
|       │   ├── plotting_all.py
|       │   │
|       │   ├── resnet_probes.py
|       │   │
|       │   └── train.py
│       |
│       └── slurm
│           └── slurm jobs to run the code on HPC
│
├── .gitignore
│
├── LICENSE
|   └── MIT license
│
├── Makefile
│
├── README.md
│
├── notion.txt
│
├── pyproject.toml 
│
└── requirements.txt

```

## Install and run
```
https://github.itu.dk/nizp/MedCLIP_subgroup/edit/master/README.md
cd datamaps_2025fallresearch
```

## Create the environment and install dependencies:
```
make create_environment
make requirements
```

## Get the data
```
To run this project, you will need both the NIH-CXR14 (refered to as chestX in this repository dataset) and the PadChest data.
NIH-CXR14: https://www.kaggle.com/datasets/nih-chest-xrays/data
PadChest: http://bimcv.cipf.es/bimcv-projects/padchest/
```

## Process the data, first test and train for chestX separately, and then PadChest for both cardiomegaly and pneumothorax  
```
make train_preprocess_chestX
make test_preprocess_chestX
make preprocess_padchest
make process_px_padchest
```

## Train probes
```
make train_probes
make train_probes_padchest label=cardiomegaly
make train_probes_padchest label=pneumothorax
```

## Do evaluation and plot
```
make evaluation_current
make plotting_all
```

## For EDA, AUROC scores and manual analysis, run the notebooks: 
```
eda_pre.ipynb
manual_analysis.ipynb
metric_notebook.ipynb
```

## Folder for example images 
```
Go to confidence_images for each of the datasets

```
