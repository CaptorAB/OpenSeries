
venv:
	python3 -m venv ./venv
	venv/bin/python --version
	venv/bin/pip install --upgrade pip
	venv/bin/pip install --upgrade wheel
	venv/bin/pip install --upgrade .[dev]

test:
	PYTHONPATH=${PWD} venv/bin/coverage run -m pytest --verbose --durations=20 --durations-min=2.0 ./
	PYTHONPATH=${PWD} venv/bin/coverage report -m

upgrade:
	PYTHONPATH=${PWD} venv/bin/pip install --upgrade pip
	PYTHONPATH=${PWD} venv/bin/pip install --upgrade .

clean:
	rm -rf dist
	rm -rf openseries.egg-info
	rm -rf venv

.PHONY: test
