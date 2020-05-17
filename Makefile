install-conda:
	bash install-conda.sh

install-conda-env:
	conda env create -f environment.yml
	conda activate mestrado-env

run:
	cd src
	python filter_analysis.py

