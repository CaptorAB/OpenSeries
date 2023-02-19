venv:
	python3 -m venv ./venv
	venv/bin/python --version
	venv/bin/pip install --upgrade pip
	venv/bin/pip install poetry==1.3.2
	poetry install --with test

test:
	PYTHONPATH=${PWD} poetry run coverage run -m pytest --verbose --durations=20 --durations-min=2.0 ./
	PYTHONPATH=${PWD} poetry run coverage report -m

.PHONY: test
