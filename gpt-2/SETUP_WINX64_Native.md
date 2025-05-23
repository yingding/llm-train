# gpt-2: python package
This repository contains code of [applyllm](https://pypi.org/project/applyllm/) python PyPI package, for loading and training open source llm models e.g. LlaMA2, Mixtral 8x7B, etc.

## Creat VENV
use the `create_env.sh` script to create a venv on windows 11

```powershell
cd $env:USERPROFILE\Documents\VCS\llm-train;

$VERSION="3.12";
$ENV_NAME="gpt";
$ENV_SURFIX="pip";
$PM="pip";
$WORK_DIR="$env:USERPROFILE\Documents\VENV\";
.\envtools\create_env.ps1 -VERSION $VERSION -ENV_NAME $ENV_NAME -ENV_SURFIX $ENV_SURFIX -PM $PM -WORK_DIR $WORK_DIR;
```

## (optional) Manually setup a local venv on windows 11
```powershell
$env:VERSION = "3.12";
$env:PREFIX = "gpt";
$env:ENV_NAME = "$env:PREFIX$env:VERSION";
$env:ENV_ROOT="$HOME\Documents\VENV";
& "$env:ENV_ROOT\$env:ENV_NAME\Scripts\Activate.ps1";
# see the right python sdk is activated
which python
# PROJECT=gpt-2
# important to use python, but not python3
python -m pip install --upgrade pip
# cd the project from power shell
python -m pip install --no-cache-dir -r ./requirements_winx64.txt
```
Note:
* use `python` instead of `python3`, since it is linked to wrong python SDK 
* encounter run issue, use `Set-ExecutionPolicy RemoteSigned` as admin to set the run privilege from powershell7 and restart powershell session.

## (optional) Add a jupyter notebook kernel to VENV
```powershell
$env:VERSION = "3.12";
$env:PREFIX = "gpt";
$env:ENV_NAME = "$env:PREFIX$env:VERSION";
$env:ENV_ROOT="$HOME\Documents\VENV";
& "$env:ENV_ROOT\$env:ENV_NAME\Scripts\Activate.ps1";
python -m pip install --upgrade pip
python -m pip install ipykernel
deactivate
```

We need to reactivate the venv so that the ipython kernel is available after installation.
```powershell
$env:VERSION = "3.12";
$env:PREFIX = "gpt";
$env:ENV_NAME = "$env:PREFIX$env:VERSION";
$env:ENV_ROOT="$HOME\Documents\VENV";
& "$env:ENV_ROOT\$env:ENV_NAME\Scripts\Activate.ps1";
python -m ipykernel install --user --name=$env:ENV_NAME --display-name $env:ENV_NAME;
```
Note: 
* restart the vs code, to select the venv as jupyter notebook kernel

Reference:
* https://ipython.readthedocs.io/en/stable/install/kernel_install.html
* https://anbasile.github.io/posts/2017-06-25-jupyter-venv/

## Remove ipykernel
```powershell
$env:VERSION = "3.12";
$env:PREFIX = "gpt";
$env:ENV_NAME = "$env:PREFIX$env:VERSION";
jupyter kernelspec uninstall -y $env:ENV_NAME
```

## (Optional) Remove all package from venv
For the venv python
```powershell
which python
python -m pip freeze | %{$_.split('==')} | %{python -m pip uninstall -y $_}
python -m pip list
```

Note: `which` cmd can be installed from powershell with `winget install which`

For the system python3
```powershell
which python3
python3 -m pip freeze | %{$_.split('==')} | %{python3 -m pip uninstall -y $_}
python3 -m pip list
```

## Issues

### Warning: IProgress not found. Please update jupyter and ipywidgets
```powershell
pip install ipywidgets
```
which will update the ipywidgets and also the widgetsnbextension.

* https://stackoverflow.com/questions/53247985/tqdm-4-28-1-in-jupyter-notebook-intprogress-not-found-please-update-jupyter-an