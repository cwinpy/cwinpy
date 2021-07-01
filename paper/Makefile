default: pdf

MDFILE=paper.md
BIBFILE=paper.bib
PDFFILE=paper.pdf

# some variables for the paper
REPOSITORY=https://github.com/cwinpy/cwinpy
ARCHIVE=https://dx.doi.org/XX.XXXX/zenodo.XXXXXXX
REVIEW=https://github.com/openjournals/joss-reviews/issues/XXX
ARCHIVEDOI=http://dx.doi.org/XX.XXXX/zenodo.XXXXXXX
FORMATTEDDOI=XX.XXXXX/joss.XXXXX
YEAR=2021
TITLE="cwinpy: a Python package for inference with continuous gravitational-wave signals from pulsars"
AUTHOR="Pitkin"
ISSUE=XX
VOLUME=X
PAGE=XXX
JOURNALNAME="Journal of Open Source Software"
PUBLISHED="XX August 2021"
SUBMITTED="XX August 2021"
PAPERURL=https://www.theoj.org/joss-papers/joss.XXXXX/XX.XXXXX.joss.XXXXX.pdf

latex: $(MDFILE) $(BIBFILE)
	# get JOSS template
	curl https://raw.githubusercontent.com/openjournals/whedon/master/resources/latex.template > latex.template
	# get JOSS logo
	curl https://raw.githubusercontent.com/openjournals/whedon/master/resources/joss/logo.png > joss-logo.png
	# get citation style file
	curl https://raw.githubusercontent.com/openjournals/whedon/master/resources/joss/apa.csl > apa.csl
	# use pandoc and pandoc-citeproc to create LaTeX source
	pandoc \
-V repository="$(REPOSITORY)" \
-V journal_name=$(JOURNALNAME) \
-V paper_url="$(PAPERURL)" \
-V archive_doi="$(ARCHIVEDOI)" \
-V review_issue_url="$(REVIEW)" \
-V formatted_doi="$(FORMATTEDDOI)" \
-V logo_path="joss-logo.png" \
-V year="$(YEAR)" \
-V paper_title=$(TITLE) \
-V citation_author=$(AUTHOR) \
-V issue="$(ISSUE)" \
-V volume="$(VOLUME)" \
-V page="$(PAGE)" \
-V submitted=$(SUBMITTED) \
-V published=$(PUBLISHED) \
-V graphics="true" \
-V papersize=a4paper \
-V geometry:margin=1in \
-s $(MDFILE) \
-o $(PDFFILE) \
--pdf-engine=xelatex \
--citeproc \
--csl apa.csl \
--template latex.template

pdf: latex

clean:
	rm paper.bcf paper.aux paper.log paper.out paper.run.xml

clean_downloads:
	rm latex.template joss-logo.png
