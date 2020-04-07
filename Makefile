install:
	conda env create -f environment.yml
	conda activate mestrado-env

run:
	python src/filter_analysis.py

