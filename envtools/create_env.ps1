param (
    [string]$VERSION = "",    # python version
    [string]$ENV_NAME = "",   # prefix
    [string]$ENV_SURFIX = "",  # the sufix after python version
    [string]$WORK_DIR = "$HOME\Documents\VENV\",  # the work directory
    [string]$PM = "pip" # conda or pip for package manager
)

# test $VERSION and $ENV_NAME is not ""
if ($VERSION -eq "" -or $ENV_NAME -eq "") {
    Write-Error "The parameter 'VERSION' and 'ENV_NAME' must be set."
    exit 1
}

# Validate the $VERSION parameter
$validVersions = @("3.11", "3.12", "3.13")
if ($VERSION -notin $validVersions) {
    Write-Error "Invalid VERSION '$VERSION'. Valid actions are: $($validVersions -join ', ')."
    exit 1
} 

# powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
# uv venv azfdymcp3.12uv --python python3.12
$validPM = @("pip", "conda", "uv")
if ($PM -notin $validPM) {
    Write-Error "Invalid PM (package manager) '$PM'. Valid actions are: $($validPM -join ', ')."
    exit 1
} else {
    # combine the $ENV_NAME and $VERSION and $ENV_SURFIX to ENV_NAME
    $env:ENV_FULL_NAME = "$ENV_NAME$VERSION$ENV_SURFIX";
    $env:ENV_ROOT = "$WORK_DIR$env:ENV_FULL_NAME";
    $env:PY_VERSION = "$VERSION";
    
    # test whether the $env:ENV_ROOT exists
    if (Test-Path $env:ENV_ROOT) {
        Write-Host "The virtual environment '$env:ENV_FULL_NAME' already exists."
        # exit successfully
        exit 0
    }


    if ($PM -eq "pip") {
        # create the virtual environment
        & "python$env:PY_VERSION" -m venv "$env:ENV_ROOT";
        # activate the virtual environment
        & "$env:ENV_ROOT\Scripts\Activate.ps1";
        # pass the instruction "which python" to activated python virtual environment
        # replace & "which" python; since the which is not available in Windows
        Invoke-Expression "(Get-Command python).Source";
        & "python" -m pip install --upgrade pip;
        & "python" -m pip install ipykernel;
        & "python" -m ipykernel install --user --name=$env:ENV_FULL_NAME --display-name $env:ENV_FULL_NAME;
        # & "deactivate"

    } elseif ($PM -eq "uv") {
        # create the virtual environment using uv
        # uv venv $env:ENV_FULL_NAME --python python$env:PY_VERSION;
        # uv venv $env:ENV_FULL_NAME --python python$env:PY_VERSION --work-dir $WORK_DIR;
        # uv venv $env:ENV_FULL_NAME --python python$env:PY_VERSION --work-dir $WORK_DIR --force;
        & "uv" venv "$env:ENV_ROOT" --python python$env:PY_VERSION;
        
        # activate the virtual environment
        & "$env:ENV_ROOT\Scripts\Activate.ps1";
        
        # pass the instruction which python to activated python virtual environment
        Invoke-Expression "(Get-Command python).Source";
        
        # & "uv" pip install --upgrade pip;
        & "uv" pip install ipykernel;
        & "python" -m ipykernel install --user --name=$env:ENV_FULL_NAME --display-name $env:ENV_FULL_NAME;
        # & "python" -m pip install --upgrade pip;
        # & "python" -m pip install ipykernel;
        # & "python" -m ipykernel install --user --name=$env:ENV_FULL_NAME --display-name $env:ENV_FULL_NAME;
    
    } elseif ($PM -eq "conda") {
        # C:\Users\yingdingwang\AppData\Local\miniconda3\Library\bin\conda.BAT
        # if ($env:PY_VERSION -eq "3.11") {
        #     $condaCmd = "$env:USERPROFILE\AppData\Local\miniconda311\Scripts\conda.exe";
        # } elseif ($env:PY_VERSION -eq "3.12") {
        #     $condaCmd = "$env:USERPROFILE\AppData\Local\miniconda312\Scripts\conda.exe";
        # }
        # $condaCmd = "$env:USERPROFILE\AppData\Local\miniconda3\Scripts\conda.exe";
        $condaCmd = "C:\ProgramData\miniconda3\Scripts\conda.exe";

        # accept the terms of the channel
        Invoke-Expression "& $condaCmd tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main";
        Invoke-Expression "& $condaCmd tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r";
        Invoke-Expression "& $condaCmd tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2";
 
        $previousLocation = Get-Location
        # create the virtual environment
        # & "conda" create --prefix "$env:ENV_ROOT" python="$env:PY_VERSION";
        # Start-Process $condaCmd -ArgumentList "create --prefix $env:ENV_ROOT python=$env:PY_VERSION -y" -Wait;
        
        Invoke-Expression "& $condaCmd create --prefix $env:ENV_ROOT python=$env:PY_VERSION -y";
        # Invoke-Expression "& conda create --prefix $env:ENV_ROOT python=$env:PY_VERSION -y";
        

        # activate the virtual environment
        Set-Location -Path $env:ENV_ROOT;
        
        # Initialize conda for PowerShell
        # Invoke-Expression "& $condaCmd init powershell"
        # Invoke-Expression "& $condaCmd config --set auto_activate_base false"
        
        # Invoke-Expression "& $condaCmd init powershell"
        # Invoke-Expression "& $condaCmd config --set auto_activate_base false"


        # back to the previous location
        Set-Location -Path $previousLocation;
        
        Write-Host "$PM venv: $env:ENV_ROOT created successfully.";

        Invoke-Expression "& $condaCmd env list";
        # activate in the Anaconda Prompt

        # Invoke-Expression "& conda activate $env:ENV_ROOT";

        # pass the instruction which python to activated python virtual environment
        # Invoke-Expression "& which python";

        # & "conda" install -y ipykernel;
        # & "python" -m ipykernel install --user --name=$env:ENV_FULL_NAME --display-name $env:ENV_FULL_NAME;
        # & conda deactivate
    }


}