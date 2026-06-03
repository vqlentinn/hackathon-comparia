.PHONY: all clean tectonic pdflatex

MAIN = main

# Par défaut : tectonic (pas besoin de MacTeX / sudo)
all: tectonic

tectonic:
	tectonic $(MAIN).tex

# Alternative si MacTeX est installé (/Library/TeX/texbin dans le PATH)
pdflatex:
	pdflatex $(MAIN).tex
	bibtex $(MAIN)
	pdflatex $(MAIN).tex
	pdflatex $(MAIN).tex

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out $(MAIN).toc $(MAIN).pdf
