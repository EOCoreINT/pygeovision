# Build the documentation:  make -C docs html     (output in docs/_build/html)
SPHINXOPTS ?= -W --keep-going
SPHINXBUILD ?= python -m sphinx
SOURCEDIR = .
BUILDDIR = _build

.PHONY: html html-en translations clean linkcheck regenerate check-generated
# English plus every translation (the English build starts the others; see conf.py)
html:
	$(SPHINXBUILD) -b html $(SPHINXOPTS) $(SOURCEDIR) $(BUILDDIR)/html
# English only: quicker while you write
html-en:
	DOCS_SKIP_OTHER_LANGS=1 $(SPHINXBUILD) -b html $(SPHINXOPTS) $(SOURCEDIR) $(BUILDDIR)/html
# Check the translation files
translations:
	python tools/check_translations.py
linkcheck:
	$(SPHINXBUILD) -b linkcheck $(SOURCEDIR) $(BUILDDIR)/linkcheck
clean:
	rm -rf $(BUILDDIR)
# Pages written by tools/generate_*.py and the flow figures: rewrite them, or fail if they are out of date
regenerate:
	cd .. && for g in generate_model_pages generate_pipeline_pages generate_reference_docs make_flow_figures; do PYTHONPATH=. python tools/$$g.py; done
check-generated:
	cd .. && for g in generate_model_pages generate_pipeline_pages generate_reference_docs make_flow_figures; do PYTHONPATH=. python tools/$$g.py --check || exit 1; done
