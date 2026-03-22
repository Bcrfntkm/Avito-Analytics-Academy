# How to run tests and doctests

## Requirements
Python 3.8+
pip install pytest coverage flake8

## Run doctest for morse.encode
python3 -m doctest -v morse.py

## Run pytest for pytest tests
pytest -v

## Run unittest tests
python3 -m unittest discover -v -s tests -p "test_*.py"

## Coverage report (html)
coverage run -m pytest
coverage html
# report at htmlcov/index.html

## flake8 check
flake8 .
