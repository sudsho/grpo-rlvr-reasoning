.PHONY: install fmt lint test sft grpo eval clean

install:
	pip install -e ".[dev]"

fmt:
	ruff format src tests

lint:
	ruff check src tests

test:
	pytest -q

sft:
	bash scripts/train_sft_warmup.sh

grpo:
	bash scripts/train_grpo.sh

eval:
	bash scripts/eval_all.sh

clean:
	rm -rf build dist *.egg-info .pytest_cache
	find . -name __pycache__ -type d -exec rm -rf {} +
