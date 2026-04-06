# Introduction
Use Powershell 7x to run the cmds to create python venv with pip. Conda is initialized in Powershell

## Create a windows native XPU Training env
Install the Intell extension
```Powershell
cd $env:USERPROFILE\Documents\VCS\llm-train;
$env:VERSION="3.13";
$env:ENV_NAME="train"; #prefix for the full env
$env:ENV_SURFIX="pip";
$env:PM="pip";
.\envtools\create_env.ps1 -VERSION $env:VERSION -ENV_NAME $env:ENV_NAME -ENV_SURFIX $env:ENV_SURFIX -PM $env:PM;
```

## Install the packages
```powershell
which python;
& "python" -m pip install -r requirements_winx64_pip_torch.txt;
& "python" -m pip install -r requirements_winx64_pip.txt;
```

## Activate pip env
```Powershell
# activate environment
$env:VERSION="3.13";
$env:ENV_NAME="train"; #prefix for the full env
$env:ENV_SURFIX="pip";
$env:ENV_FULL_NAME="${env:ENV_NAME}${env:VERSION}${env:ENV_SURFIX}";
$env:ENV_ROOT="${env:USERPROFILE}\Documents\VENV\${env:ENV_FULL_NAME}";
& "$env:ENV_ROOT\Scripts\Activate.ps1";
```

## Deactivate pip env
To deactivate an active environment, use
```Powershell
deactivate
```

