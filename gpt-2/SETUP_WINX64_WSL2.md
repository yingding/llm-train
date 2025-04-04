# gpt-2: python package
This repository contains code of [applyllm](https://pypi.org/project/applyllm/) python PyPI package, for loading and training open source llm models e.g. LlaMA2, Mixtral 8x7B, etc.

## Activate WSL2 on windows 11 (Not working with NPU)

Install WSL from powershell with
```powershell
wsl --install
```
Reference:
* [Install WSL | Microsoft Learn](https://learn.microsoft.com/en-us/windows/wsl/install)

Base commands
```powershell
wsl --status
```

login as specific user
```powershell
wsl --user <Username>
```
this will change to the bash shell.

the current home dir is:
`/mnt/c/Users/<Username>` 

Reference:
* [Basic commands for WSL | Microsoft Learn](https://learn.microsoft.com/en-us/windows/wsl/basic-commands)

Install default python3.12 
```shell
sudo apt-get update
sudo apt-get install libpython3-dev
sudo apt-get install python3-venv
```

Update the WSL2 Ubuntu OS
```shell
which python3;
# /usr/bin/python3
lsb_release -a;
# Description:    Ubuntu 24.04.1 LTS
echo $SHELL
# /bin/bash

sudo apt update;
sudo apt upgrade;
```

## Install python3.12 on WSL2

Python3.12
```shell
sudo apt install -y python3.12 python3.12-venv
```

## Install cmake
```
sudo apt install -y cmake build-essential libtbb-dev
```
this will be used to compile the intel lib in ubuntu

* Reference: https://github.com/intel/intel-npu-acceleration-library/issues/13#issuecomment-2147679352
<!--
## Install python3.11 on WSL2
Add universal repository
```shell
sudo add-apt-repository universe;
sudo apt update
```

Python3.11
```shell
sudo apt install -y python3.11 python3.11-venv
```

Reference:
* https://askubuntu.com/a/1398569
* https://rothoma2.com/2023/06/03/how-to-install-python-3-11-on-ubuntu-wsl/
-->

## Creat VENV
```shell
VERSION="3.12";
PREFIX="gpt";
FLAVOUR="wsl";
ENV_NAME="${PREFIX}${VERSION}${FLAVOUR}";
ENV_ROOT="/mnt/c/Users/yingdingwang/Documents/VENV";
ENV_PATH="${ENV_ROOT}/${ENV_NAME}";
which python$VERSION;
python${VERSION} -m venv $ENV_PATH;
```

Activate the VENV from WSL bash terminal:
```shell
VERSION="3.12";
PREFIX="gpt";
FLAVOUR="wsl";
ENV_NAME="${PREFIX}${VERSION}${FLAVOUR}";
ENV_ROOT="/mnt/c/Users/yingdingwang/Documents/VENV";
ENV_PATH="${ENV_ROOT}/${ENV_NAME}";

source ${ENV_PATH}/bin/activate;
# which python$VERSION;
# python${VERSION} -m pip install --upgrade pip;

which python;
python -m pip install --upgrade pip;
```

## Install packages for WSL python venv 
```shell
# file location from Windows directory
PROJ_ROOT="/mnt/c/Users/yingdingwang/Documents/VCS";
PROJ_NAME="llm-train";
PROJ_PATH="${PROJ_ROOT}/${PROJ_NAME}";
cd ${PROJ_PATH};

which python;
python -m pip install --no-cache-dir -r ./requirements_winx64_wsl.txt;
# DISPLAY= python -m pip install --no-cache-dir -r ./requirements_winx64_wsl.txt;
```
Note:
* use `python` instead of `python3`, since it is linked to wrong python SDK 
* encounter run issue, use `Set-ExecutionPolicy RemoteSigned` as admin to set the run privilege from powershell7 and restart powershell session.
* it takes a long time to install all the packages

<!-- 
Reference:
* WSL2 connect to xserver, use `DISPLAY= <cmd>` to tell not to connect to xserver
* https://github.com/microsoft/WSL/issues/6643#issuecomment-1033864007
--> 

## Connect from native windows VS code to WSL venv

1. Start VS code
2. press `><` sign on the left bottom
3. Choose WSL 
4. Open Folder, show local to choose your local windows folder

5. Create WSL workdir:
```shell
mkdir -p $HOME/VCS/
```
6. Create a Workdir in the WSL linux space, you can not use the windows workspace.
7. Choose the python interpreter for the WSL linux version of VS code workspace

Reference:
* https://code.visualstudio.com/docs/remote/wsl 


## Add a jupyter notebook kernel to VENV
```shell
VERSION="3.12";
PREFIX="gpt";
FLAVOUR="wsl";
ENV_NAME="${PREFIX}${VERSION}${FLAVOUR}";
ENV_ROOT="/mnt/c/Users/yingdingwang/Documents/VENV";
ENV_PATH="${ENV_ROOT}/${ENV_NAME}";
source ${ENV_PATH}/bin/activate;

which python;
python -m pip install --upgrade pip;
python -m pip install --no-cache-dir ipykernel;
deactivate
```

We need to reactivate the venv so that the ipython kernel is available after installation.
```shell
VERSION="3.12";
PREFIX="gpt";
FLAVOUR="wsl";
ENV_NAME="${PREFIX}${VERSION}${FLAVOUR}";
ENV_ROOT="/mnt/c/Users/yingdingwang/Documents/VENV";
ENV_PATH="${ENV_ROOT}/${ENV_NAME}";
source ${ENV_PATH}/bin/activate;

which python;
python -m ipykernel install --user --name=${ENV_NAME} --display-name ${ENV_NAME};
```
Note: 
* restart the vs code, to select the venv as jupyter notebook kernel

Reference:
* https://ipython.readthedocs.io/en/stable/install/kernel_install.html
* https://anbasile.github.io/posts/2017-06-25-jupyter-venv/

## Remove ipykernel
```shell
VERSION="3.12";
PREFIX="gpt";
FLAVOUR="wsl";
ENV_NAME="${PREFIX}${VERSION}${FLAVOUR}";
ENV_ROOT="/mnt/c/Users/yingdingwang/Documents/VENV";
ENV_PATH="${ENV_ROOT}/${ENV_NAME}";
source ${ENV_PATH}/bin/activate;

which python;
jupyter kernelspec uninstall -y ${ENV_NAME}
```

## (Optional) Remove all package from venv
```shell 
python3 -m pip freeze | xargs pip uninstall -y
python3 -m pip list
```

# Install the Intel Driver on WSL2
```shell
wget -qO - https://repositories.intel.com/gpu/intel-graphics.key |
    sudo gpg --yes --dearmor --output /usr/share/keyrings/intel-graphics.gpg
```

Reference:
* https://dgpu-docs.intel.com/driver/installation.html#ubuntu
* https://medium.com/intel-analytics-software/stable-diffusion-with-intel-arc-gpus-f2986bba8365

## Issues

### Warning: IProgress not found. Please update jupyter and ipywidgets
```shell
pip install ipywidgets
```
which will update the ipywidgets and also the widgetsnbextension.

* https://stackoverflow.com/questions/53247985/tqdm-4-28-1-in-jupyter-notebook-intprogress-not-found-please-update-jupyter-an

### Install intel npu acceleration library in WSL2
The driver and kernel is not available on WSL2

need `cmake`, ``c++ compile`

* https://github.com/intel/intel-npu-acceleration-library

intel npu acceleration library is not working in WSL2, there is no npu driver for WSL2
https://github.com/intel/intel-npu-acceleration-library/issues/13#issuecomment-2150497424