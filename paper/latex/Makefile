MAIN = main
BIBFILE = references

.PHONY: all clean

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex $(BIBFILE).bib
	pdflatex -interaction=nonstopmode $(MAIN).tex
	bibtex $(MAIN)
	pdflatex -interaction=nonstopmode $(MAIN).tex
	pdflatex -interaction=nonstopmode $(MAIN).tex

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.fls *.fdb_latexmk

# Upload to Overleaf: zip the directory and import at overleaf.com
zip:
	zip -r ../diffvaxpp_submission.zip . --exclude "*.pdf" --exclude "*.log" --exclude "*.aux"
