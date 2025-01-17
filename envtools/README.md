## Create a windows native python venv
```powershell
$env:VERSION="3.12";
$env:ENV_NAME="gpt";
$env:ENV_SURFIX="";
.\envtools\create_env.ps1 -VERSION $env:VERSION -ENV_NAME $env:ENV_NAME -ENV_SURFIX $env:ENV_SURFIX;
```