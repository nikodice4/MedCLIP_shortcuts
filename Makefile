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

train_preprocess_chestX:
	$(PYTHON_INTERPRETER) src/data/train_processing_chestX.py

test_preprocess_chestX:
	$(PYTHON_INTERPRETER) src/data/test_processing_chestX.py

train_probes:
	$(PYTHON_INTERPRETER) -m src.models.main

train_probes_padchest:
	$(PYTHON_INTERPRETER) -m src.models.main --dataset padchest

evaluation:
	$(PYTHON_INTERPRETER) -m src.models.evaluation

evaluation_padchest:
	$(PYTHON_INTERPRETER) -m src.models.evaluation_padchest

plotting:
	$(PYTHON_INTERPRETER) -m src.models.plotting

preprocess_padchest:
	$(PYTHON_INTERPRETER) src/data/processing_padchest.py \
    data/raw/padchest \
    data/raw/padchest/images \
    data/processed/padchest \
    ""

# run_medclip:
# 	$(PYTHON_INTERPRETER) src/models/model.py

run_medclip:
	/home/nizp/.conda/envs/MedCLIP_subgroup/bin/python src/models/model.py

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


.PHONY: remove_environment
create_environment:
	conda remove --name MedCLIP_subgroup --all -y


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