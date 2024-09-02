# gpt-2: python package
This repository contains code of [applyllm](https://pypi.org/project/applyllm/) python PyPI package, for loading and training open source llm models e.g. LlaMA2, Mixtral 8x7B, etc.

## Creat VENV
use the `create_env.sh` script to create a venv on macosx

```shell
VERSION=3.12;
PREFIX=gpt;
ENV_NAME="${PREFIX}${VERSION}";
ENV_ROOT="$HOME/VENV";
source ./envtools/create_env.sh -p ${ENV_ROOT}/${ENV_NAME} -v $VERSION
```

## Setup a local venv on Macosx Apple Silicon
```shell
source $HOME/VENV/gpt3.12/bin/activate
# PROJECT=gpt-2
python3 -m pip install --upgrade pip
python3 -m pip install --no-cache-dir -r ./requirements_arm64.txt
# python3 -m pip install --no-cache-dir -r ./requirements_mac.txt
# python3 -m pip install --no-cache-dir -r ./$PROJECT/requirements_arm64.txt
```

## Add a jupyter notebook kernel to VENV
```shell
VENV_NAME="gpt3.12"
VENV_DIR="$HOME/VENV"
source ${VENV_DIR}/${VENV_NAME}/bin/activate;
python3 -m pip install --upgrade pip
python3 -m pip install ipykernel
deactivate
```

We need to reactivate the venv so that the ipython kernel is available after installation.
```shell
VENV_NAME="gpt3.12"
VENV_DIR="$HOME/VENV"
source ${VENV_DIR}/${VENV_NAME}/bin/activate;
python3 -m ipykernel install --user --name=${VENV_NAME} --display-name ${VENV_NAME}
```
Note: 
* restart the vs code, to select the venv as jupyter notebook kernel

Reference:
* https://ipython.readthedocs.io/en/stable/install/kernel_install.html
* https://anbasile.github.io/posts/2017-06-25-jupyter-venv/

## Remove ipykernel
```shell
VENV_NAME="gpt3.12"
jupyter kernelspec uninstall -y ${VENV_NAME}
```

## Remove all package from venv
```shell 
python3 -m pip freeze | xargs pip uninstall -y
python3 -m pip list
```

## Issues

### TqdmWarning: IProgress not found. Please update jupyter and ipywidgets
```
pip install ipywidgets
```
which will update the ipywidgets and also the widgetsnbextension.

* https://stackoverflow.com/questions/53247985/tqdm-4-28-1-in-jupyter-notebook-intprogress-not-found-please-update-jupyter-an