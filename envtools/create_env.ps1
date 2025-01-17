param (
    [string]$VERSION = "",  # python version
    [string]$ENV_NAME = ""   # prefix
    [string]$ENV_SURFIX = ""    # the sufix after python version
)

# test 
if (-Not (Test-Path $VERSION)) {
    Write-Error "The file 'local.env' does not exist in the current directory."
    exit 1
}