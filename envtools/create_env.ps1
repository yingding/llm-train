param (
    [string]$VERSION = "",    # python version
    [string]$ENV_NAME = "",   # prefix
    [string]$ENV_SURFIX = "",  # the sufix after python version
    [string]$WORK_DIR = "$HOME\Documents\VENV\"  # the work directory
)

# test $VERSION and $ENV_NAME is not ""
if ($VERSION -eq "" -or $ENV_NAME -eq "") {
    Write-Error "The parameter 'VERSION' and 'ENV_NAME' must be set."
    exit 1
}

# Validate the $VERSION parameter
$validVersions = @("3.11", "3.12")
if ($VERSION -notin $validVersions) {
    Write-Error "Invalid VERSION '$VERSION'. Valid actions are: $($validVersions -join ', ')."
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
    } else {
        # create the virtual environment
        & "python$env:PY_VERSION" -m venv "$env:ENV_ROOT";
        # activate the virtual environment
        & "$env:ENV_ROOT\Scripts\Activate.ps1";
        # pass the instruction "which python" to activated python virtual environment
        & "which" python;
        & "python" -m pip install --upgrade pip;
        & "python" -m pip install ipykernel;
        & "python" -m ipykernel install --user --name=$env:ENV_FULL_NAME --display-name $env:ENV_FULL_NAME;
        # & "deactivate"
    }
}