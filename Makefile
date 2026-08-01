.PHONY: install notebooks

install:
	python -m pip install -r requirements.txt

notebooks:
	jupyter lab notebooks/ demo/
