# One-shot style setup (closest Python equivalent to "mvn install" for this repo).
# macOS/Linux: run `make install` then `make run`
# Windows: use `setup_and_run.bat` or follow README venv steps.

.PHONY: help venv install spacy run cli doctor pull-models

PYTHON ?= python3
VENV   ?= venv
PY      = $(VENV)/bin/python
PIP     = $(VENV)/bin/pip

help:
	@echo "Targets:"
	@echo "  make install  Create venv (if missing), upgrade pip, install requirements, spaCy model"
	@echo "  make run      Run Streamlit UI (uses $(VENV)/bin/python -m streamlit)"
	@echo "  make cli      Run python run.py"
	@echo "  make doctor      Check ffmpeg and combined model file size (LFS pointer hint)"
	@echo "  make pull-models Fetch combined video model via Git LFS (needs git-lfs installed)"
	@echo "  make spacy       Download en_core_web_lg into the venv only"

venv:
	@test -d $(VENV) || $(PYTHON) -m venv $(VENV)

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(MAKE) spacy

spacy:
	$(PIP) install "https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.5.0/en_core_web_lg-3.5.0-py3-none-any.whl"

run:
	$(PY) -m streamlit run app.py

cli:
	$(PY) run.py

pull-models:
	@command -v git-lfs >/dev/null 2>&1 || { echo "Install Git LFS first: brew install git-lfs (macOS)"; exit 1; }
	git lfs install
	git lfs pull --include="models/audio_face_combined/audio_face_combined_model.pth"

doctor:
	@command -v ffmpeg >/dev/null 2>&1 && echo "ffmpeg: ok" || echo "ffmpeg: missing — install FFmpeg and ensure it is on PATH"
	@if [ -f models/audio_face_combined/audio_face_combined_model.pth ]; then \
	  sz=$$(wc -c < models/audio_face_combined/audio_face_combined_model.pth); \
	  if [ "$$sz" -lt 1048576 ]; then \
	    echo "combined model: tiny file ($$sz bytes) — likely Git LFS pointer; run: git lfs install && git lfs pull"; \
	  else \
	    echo "combined model: present ($$sz bytes)"; \
	  fi; \
	else \
	  echo "combined model: missing"; \
	fi
