# Build the documentation:  make -C docs html     (output in docs/_build/html)
SPHINXOPTS ?= -W --keep-going
SPHINXBUILD ?= python -m sphinx
SOURCEDIR = .
BUILDDIR = _build

.PHONY: html clean linkcheck
html:
	$(SPHINXBUILD) -b html $(SPHINXOPTS) $(SOURCEDIR) $(BUILDDIR)/html
linkcheck:
	$(SPHINXBUILD) -b linkcheck $(SOURCEDIR) $(BUILDDIR)/linkcheck
clean:
	rm -rf $(BUILDDIR)
