PYTHON ?= python3
PDFLATEX ?= pdflatex

.PHONY: pdf pptx plots clean

pdf: plots
	$(PDFLATEX) -interaction=nonstopmode -halt-on-error -jobname=team_6 team_6_report.tex
	$(PDFLATEX) -interaction=nonstopmode -halt-on-error -jobname=team_6 team_6_report.tex

pptx: plots
	$(PYTHON) create_presentation.py

plots:
	$(PYTHON) generate_preprocessing_plots.py

clean:
	rm -f team_6.aux team_6.fdb_latexmk team_6.fls team_6.log team_6.out team_6.toc team_6.pdf
