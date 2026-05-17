# MedCLIP subgroup project

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

MedCLIP model trained on medical imaging data to uncover subgroups for robustness and fairness assessment

## Project Organization

```              
├── data               <- Data folder is in git.ignore
│   ├── processed  
│   │   
│   └── raw          
│
├── docs               
│
│
├── models
│   ├── checkpoint  
│   │
│   ├── medclip 
│   │
│   └── probes    
│
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
│
├── padchest_splits     
│   ├── padchest
│   ├── padchest_px
│
│
├── references
│
│
├── reports
│   ├── chestx
│   │   ├── confidence_images
│   │   │
│   │   ├── drain
│   │   │
│   │   ├── layer
│   │   │
│   │   └── sex
│   │ 
│   ├── figures
│   │   ├── chestx
│   │   │   ├── drain_plots
│   │   │   │ 
│   │   │   └── sex_plots
│   │   │ 
│   │   ├── padchest
│   │   │   ├── scanner_plots
│   │   │   │  
│   │   │   └── sex_plots
│   │   │ 
│   │   └── padchest_px
│   │       ├── scanner_plots
│   │       │  
│   │       └── sex_plots
│   │
│   ├── padchest
│   │   ├── confidence_images
│   │   ├── layer
│   │   │
│   │   ├── padchest_scanner
│   │   │
│   │   └── padchest_sex
│   │      
│   └── padchest_px
│       ├── confidence_images
│       │ 
│       ├── layer
│       │ 
│       ├── padchest_scanner_px
│       │ 
│       └── padchest_sex_px
│
│├── src
│   ├── data
│   │   ├── data_split.py
│   │   │
│   │   ├── datasets.py
│   │   │
│   │   ├── process_px_padchest.py
│   │   │
│   │   ├── processing_padchest.py
│   │   │
│   │   ├── run_data_split.py
│   │   │
│   │   ├── test_processing_chestX.py
│   │   │
│   │   └── test_processing_chestX.py
│   │ 
│   ├── models
│   │   ├── checkpointing.oy
│   │   │ 
│   │   ├── config.py
│   │   │
│   │   ├── evaluation_padchest.py
│   │   │
│   │   ├── evaluation_padchest_px.py
│   │   │
│   │   ├── main.py
│   │   │
│   │   ├── plotting_all.py
│   │   │
│   │   ├── resnet_probes.py
│   │   │
│   │   └── train.py
│   │
│   └── slurm
│       └──slurm jobs to run the code on HPC
│
│
├── .gitignore
│
│
├── LICENSE
│
│
├── Makefile
│
│
├── README.md
│
│
├── notion.txt
│
│
├── pyproject.toml 
│
│
└── requirements.txt

--------

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
To run this project, you will need both the NIH-ChestX14 (refered to as chestX in this repository dataset) and the PadChest data.
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
