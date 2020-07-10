#!/bin/bash

rm -r .pytest_cache
black .
python -m pytest --pylint --pylint-rcfile=pylintrc --mypy --mypy-ignore-missing-imports --durations=5
coverage-badge -f -o coverage.svg
RET_VALUE=$?
exit $RET_VALUE
