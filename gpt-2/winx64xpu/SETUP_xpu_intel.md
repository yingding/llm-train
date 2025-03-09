# Intro

use the `create_env.sh` script to create a venv on windows 11

Python 3.12 venv in `Anacoda Powershell Prompt`
```powershell
$VERSION="3.12";
$ENV_NAME="gpt";
$ENV_SURFIX="winconda";
$PM="conda";
$WORK_DIR="$env:USERPROFILE\Documents\VENV\";
.\envtools\create_env.ps1 -VERSION $VERSION -ENV_NAME $ENV_NAME -ENV_SURFIX $ENV_SURFIX -PM $PM -WORK_DIR $WORK_DIR;
```

### (Optional) Experimenting with DirectML for NPU training
Python 3.12 venv
```Anaconda Powershell Prompt
$env:VERSION="3.12";
$env:ENV_NAME="directml";
$env:ENV_SURFIX="winconda";
$env:PM="conda";
.\envtools\create_env.ps1 -VERSION $env:VERSION -ENV_NAME $env:ENV_NAME -ENV_SURFIX $env:ENV_SURFIX -PM $env:PM;
```

## Install XPU with intel-extension-for-pytorch

Training framework

Intell GPU features using torch `xpu` device
Reference:
* https://intel.github.io/intel-extension-for-pytorch/

In Anaconda Powershell:

```powershell
$env:VERSION = "3.12";
$env:ENV_NAME = "gpt${env:VERSION}winconda";
conda activate "$env:USERPROFILE\Documents\VENV\${env:ENV_NAME}";

# path env
conda env update --prefix "$HOME\Documents\VENV\${env:ENV_NAME}" --file environment_winx64.yml

# which pip
which pip

# install additional pip packages from storage 0 (us storage is fast in eu)
python -m pip install torch==2.5.1+cxx11.abi torchvision==0.20.1+cxx11.abi torchaudio==2.5.1+cxx11.abi intel-extension-for-pytorch==2.5.10+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/mtl/us/

# install additional pip packages from storage 1 (cn storage is too slow in eu)
# python -m pip install torch==2.5.1+cxx11.abi torchvision==0.20.1+cxx11.abi torchaudio==2.5.1+cxx11.abi intel-extension-for-pytorch==2.5.10+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/mtl/cn/
```
Reference:
* https://intel.github.io/intel-extension-for-pytorch/