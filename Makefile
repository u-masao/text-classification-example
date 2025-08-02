#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = text-classification-example
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python
PACKAGE_NAME = text_classification_example

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python dependencies
.PHONY: requirements
requirements:
	uv sync

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	uv run ruff check --fix tests $(PACKAGE_NAME)
	uv run ruff format tests $(PACKAGE_NAME)

## Run tests
.PHONY: test
test:
	uv run python -m pytest tests

## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	uv venv --python $(PYTHON_VERSION)
	@echo ">>> New uv virtual environment created. Activate with:"
	@echo ">>> Windows: .\\\\.venv\\\\Scripts\\\\activate"
	@echo ">>> Unix/macOS: source ./.venv/bin/activate"

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## run dvc repro
.PHONY: repro
repro: check_commit PIPELINE.md
	uv run dvc repro
	git commit dvc.lock -m 'run dvc repro' || true

## check commit
.PHONY: check_commit
check_commit:
	git status
	git diff --exit-code
	git diff --exit-code --staged

## make pipeline
PIPELINE.md: dvc.yaml params.yaml
	echo -n '# DVC pipeline\n' > $@
	echo -n '\n## summary\n\n' >> $@
	uv run dvc dag --md >> $@
	echo -n '\n## detail\n\n' >> $@
	uv run dvc dag --md --outs >> $@
	git commit $@ -m 'update dvc pipeline' || true

## run mlflow ui
.PHONY: mlflow_ui
mlflow_ui:
	uv run mlflow ui -h 0.0.0.0 -p 5000 --backend-store-uri sqlite:///mlruns.db

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
