# tfomics

## Install

Install this package from the GitHub page. Install TensorFlow as well.

```
python -m pip install --no-cache-dir https://github.com/p-koo/tfomics/tarball/master
python -m pip install --no-cache-dir tensorflow
```

### Developers

Developers should clone the repository and install in editable mode with development dependencies. The code below also creates a virtual environment for the package.

```
git clone https://github.com/p-koo/tfomics.git
cd tfomics
python -m venv venv
source ./venv/bin/activate
python -m pip install --no-cache-dir --editable .[dev]
```
