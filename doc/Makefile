# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

docs:
	make clean
	make html

build-lang:
	rm -rf build/html/$(lang)
	$(SPHINXBUILD) -D language=$(lang) source build/html/$(lang)

update-text:
	rm -rf build/gettext
	make gettext

update-lang:
	sphinx-intl update -p build/gettext -l $(lang)

build-deploy:
	rm -rf build/html
	make update-text
	make build-lang lang=en SPHINXBUILD=sphinx-multiversion
	make build-lang lang=fr
	cp -r build/html/en/$(version)/* build/html/

serve:
	make docs
	python -m http.server --directory build/html

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
