# Introduction
Use Powershell 7x to run the cmds to create python venv with pip. Conda is initialized in Powershell

## Create a windows native XPU Training env
Install the Intell extension
```Powershell
cd $env:USERPROFILE\Documents\VCS\llm-train;
$env:VERSION="3.13";
$env:ENV_NAME="train"; #prefix for the full env
$env:ENV_SURFIX="winconda";
$env:PM="conda";
.\envtools\create_env.ps1 -VERSION $env:VERSION -ENV_NAME $env:ENV_NAME -ENV_SURFIX $env:ENV_SURFIX -PM $env:PM;

# env full path to be used in conda env update
$env:ENV_FULL_NAME="${env:ENV_NAME}${env:VERSION}${env:ENV_SURFIX}";

# set the progress bar
# $env:PIP_PROGRESS_BAR="on"
# $env:PYTHONUNBUFFERED="1"
# conda env update --prefix "${env:USERPROFILE}\Documents\VENV\${env:ENV_FULL_NAME}" --file environment_winx64.yml -vv

# take 10min to run
conda env update --prefix "${env:USERPROFILE}\Documents\VENV\${env:ENV_FULL_NAME}" --file environment_winx64.yml 
```

<!--
Alternative split conda and pip
```Powershell
conda env update --prefix "$env:USERPROFILE\Documents\VENV\$env:ENV_FULL_NAME" --file environment_winx64.yml --prune
conda run --prefix "$env:USERPROFILE\Documents\VENV\$env:ENV_FULL_NAME" python -m pip install -r requirements_winx64.txt --progress-bar on -v
```
-->

## List conda env
```Powershell
conda env list
```

## Activate conda env
```Powershell
# activate environment
$env:VERSION="3.13";
$env:ENV_NAME="train"; #prefix for the full env
$env:ENV_SURFIX="winconda";
$env:ENV_FULL_NAME="${env:ENV_NAME}${env:VERSION}${env:ENV_SURFIX}";
conda activate "$HOME\Documents\VENV\${env:ENV_FULL_NAME}";
```

## Deactivate conda env
# To deactivate an active environment, use
```Powershell
conda deactivate
```