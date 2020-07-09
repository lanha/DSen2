# DSen2
## Setup
Create a new virtual environment e.g. using [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/):
```bash
mkvirtualenv --python=$(which python3.7) dsen2
```
Install requirements:
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Testing
To run tests:
```bash
bash test.sh
```
