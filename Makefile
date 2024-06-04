lint:
	poetry run flake8
	poetry run black --check .
	poetry run isort --check .

format:
	poetry run black .
	poetry run isort .
