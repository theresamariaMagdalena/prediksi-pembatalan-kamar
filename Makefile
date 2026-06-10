.PHONY: help install dev train app test lint format clean

help:
	@echo "Targets:"
	@echo "  install  Install runtime dependencies + package"
	@echo "  dev      Install with dev extras and pre-commit hooks"
	@echo "  train    Train the model and write artifacts to models/"
	@echo "  app      Launch the Streamlit app"
	@echo "  test     Run the test suite"
	@echo "  lint     Run ruff"
	@echo "  format   Format with black + ruff --fix"
	@echo "  clean    Remove caches and generated artifacts"

install:
	pip install -e .

dev:
	pip install -e ".[dev]"
	pre-commit install

train:
	python -m hotel_cancel.train

app:
	streamlit run app/streamlit_app.py

test:
	pytest

lint:
	ruff check .

format:
	ruff check --fix .
	black .

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__ *.egg-info src/*.egg-info
	rm -f models/*.pkl reports/figures/*.png
