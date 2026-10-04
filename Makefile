VENV ?= .venv
PY := $(VENV)/bin/python

.PHONY: help install ingest ask eval test lint fmt report clean

help:
	@grep -E '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/' | column -t -s "$$(printf '\t')"

$(PY):  ## create the virtualenv (prefers uv, falls back to venv)
	@command -v uv >/dev/null 2>&1 && uv venv $(VENV) --python 3.11 || python3 -m venv $(VENV)

install: $(PY)  ## editable install with dev extras
	@command -v uv >/dev/null 2>&1 && uv pip install --python $(PY) -e ".[dev]" || $(PY) -m pip install -e ".[dev]"

ingest:  ## chunk and index data/docs
	$(PY) -m raglab ingest

ask:  ## ask a single question: make ask Q="..."
	@test -n "$(Q)" || (echo 'usage: make ask Q="your question"' && exit 1)
	$(PY) -m raglab ask "$(Q)"

eval:  ## run the golden set, write reports/ and docs/index.html
	$(PY) -m raglab eval

test:  ## run the test suite
	$(PY) -m pytest

lint:  ## ruff check + format check
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

fmt:  ## apply formatting
	$(PY) -m ruff check --fix .
	$(PY) -m ruff format .

report: eval  ## rebuild and describe the HTML report
	@echo "open docs/index.html ($(shell wc -c < docs/index.html) bytes, no external assets)"

clean:  ## remove caches
	rm -rf $(VENV) .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
