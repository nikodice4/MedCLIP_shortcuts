.PHONY: clean data lint requirements

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROJECT_NAME = MedCLIP_subgroup
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

preprocess:
	$(PYTHON_INTERPRETER) src/data/processing.py

## Make Train_None
train_None:
	NB_EPOCHS=100 \
	BATCH_SIZE=32 \
	IMBALANCE_RATIO=None \
	LEARNING_RATE=0.00001 \
	ES_DELTA=0.001 \
	ES_PATIENCE=5 \
	MODEL_NAME=resnet50 \
	NB_FOLDS=3 \
	$(PYTHON_INTERPRETER) src/models/cnn_run.py

## Make Train_0
train_0:
	NB_EPOCHS=100 \
	BATCH_SIZE=32 \
	IMBALANCE_RATIO=0 \
	LEARNING_RATE=0.00001 \
	ES_DELTA=0.001 \
	ES_PATIENCE=5 \
	MODEL_NAME=resnet50 \
	NB_FOLDS=3 \
	$(PYTHON_INTERPRETER) src/models/cnn_run.py

## Make Train_1
train_1:
	NB_EPOCHS=100 \
	BATCH_SIZE=32 \
	IMBALANCE_RATIO=1 \
	LEARNING_RATE=0.00001 \
	ES_DELTA=0.001 \
	ES_PATIENCE=5 \
	MODEL_NAME=resnet50 \
	NB_FOLDS=3 \
	$(PYTHON_INTERPRETER) src/models/cnn_run.py

## Make Mapping_&_Plots_None
mapping_None:
	$(PYTHON_INTERPRETER) -m src.mapping.dynamics_filtering \
		-o data/processed/testing_dynamics/ \
		--ratio None \
		--plot \
		--filter \
		--metric confidence

## Make Mapping_&_Plots_0
mapping_0:
	$(PYTHON_INTERPRETER) -m src.mapping.dynamics_filtering \
		-o data/processed/testing_dynamics/ \
		--ratio 0 \
		--plot \
		--filter \
		--metric confidence

## Make Mapping_&_Plots_1
mapping_1:
	$(PYTHON_INTERPRETER) -m src.mapping.dynamics_filtering \
		-o data/processed/testing_dynamics/ \
		--ratio 1 \
		--plot \
		--filter \
		--metric confidence

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	
	conda create --name $(PROJECT_NAME) python=$(PYTHON_VERSION) -y
	
	@echo ">>> conda env created. Activate with:\nconda activate $(PROJECT_NAME)"
	

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


# ## Make dataset
# .PHONY: data
# data: requirements
# 	$(PYTHON_INTERPRETER) cookie_cutter_data_maps/dataset.py


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)