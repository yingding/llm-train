# Introduction
Use Powershell 7x to run the cmds to create python venv with pip.
Use Anaconda Powershell Prompt to run the cmds to create python venv with conda.

## Create a windows native python venv using pip
Open 
```powershell
cd $env:USERPROFILE\Documents\VCS\llm-train;

$env:VERSION="3.12";
$env:ENV_NAME="gpt";
$env:ENV_SURFIX="winpip";
$env:PM="pip";
.\envtools\create_env.ps1 -VERSION $env:VERSION -ENV_NAME $env:ENV_NAME -ENV_SURFIX $env:ENV_SURFIX -PM $env:PM;
```


## Create a windows native python venv using miniconda

Python 3.11 venv
```Anaconda Powershell Prompt
$env:VERSION="3.11";
$env:ENV_NAME="gpt";
$env:ENV_SURFIX="winconda";
$env:PM="conda";
.\envtools\create_env.ps1 -VERSION $env:VERSION -ENV_NAME $env:ENV_NAME -ENV_SURFIX $env:ENV_SURFIX -PM $env:PM;
```

Python 3.12 venv
```Anaconda Powershell Prompt
$env:VERSION="3.12";
$env:ENV_NAME="gpt";
$env:ENV_SURFIX="winconda";
$env:PM="conda";
.\envtools\create_env.ps1 -VERSION $env:VERSION -ENV_NAME $env:ENV_NAME -ENV_SURFIX $env:ENV_SURFIX -PM $env:PM;
```

## Install XPU with intel-extension-for-pytorch

xpu
```powershell
$env:VERSION = "3.12";
$env:ENV_NAME = "gpt${env:VERSION}winconda";
conda activate "$HOME\Documents\VENV\${env:ENV_NAME}";

# path env
conda env update --prefix "$HOME\Documents\VENV\${env:ENV_NAME}" --file environment_winx64.yml

# which pip
which pip

# install additional pip packages
python -m pip install torch==2.5.1+cxx11.abi torchvision==0.20.1+cxx11.abi torchaudio==2.5.1+cxx11.abi intel-extension-for-pytorch==2.5.10+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/mtl/us/
```

## Environment

* gpt3.11winconda: native windows python 3.11 conda env with (ipex-llm[npu])
* gpt3.12winconda: native windows python 3.12 conda env with intel-extension-for-pytorch (xpu, gpu)