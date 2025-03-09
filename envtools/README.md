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


```powershell
cd $env:USERPROFILE\Documents\VCS\llm-train;

$env:VERSION="3.12";
$env:ENV_NAME="dbs";
$env:ENV_SURFIX="";
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

xpu
```Anaconda Powershell
$env:VERSION = "3.12";
$env:ENV_NAME = "gpt${env:VERSION}winconda";
conda activate "$HOME\Documents\VENV\${env:ENV_NAME}";

# path env
conda env update --prefix "$HOME\Documents\VENV\${env:ENV_NAME}" --file environment_winx64.yml

# which pip
which pip

# install additional pip packages from storage 0 (us storage is fast in eu)
python -m pip install torch==2.5.1+cxx11.abi torchvision==0.20.1+cxx11.abi torchaudio==2.5.1+cxx11.abi intel-extension-for-pytorch==2.5.10+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/mtl/us/

# install additional pip packages from storage 1
python -m pip install torch==2.5.1+cxx11.abi torchvision==0.20.1+cxx11.abi torchaudio==2.5.1+cxx11.abi intel-extension-for-pytorch==2.5.10+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/mtl/cn/
```
Reference:
* https://intel.github.io/intel-extension-for-pytorch/

## Install NPU with directml
```Anaconda Powershell
$env:VERSION = "3.12";
$env:ENV_NAME = "directml${env:VERSION}winconda";
conda activate "$HOME\Documents\VENV\${env:ENV_NAME}";

# path env
conda env update --prefix "$HOME\Documents\VENV\${env:ENV_NAME}" --file environment_winx64.yml

# which pip
which pip

# depends on torch 2.4.1, only gpu
pip install torch-directml==0.2.5.dev240914

pip install intel-npu-acceleration-library==1.4.0
```
pytorch with directml, directml with onnx runtime on intel npu


## Environment

* gpt3.11winconda: native windows python 3.11 conda env with (ipex-llm[npu]): ipex-llm is inference and serving only.
* gpt3.12winconda: native windows python 3.12 conda env with intel-extension-for-pytorch (xpu, gpu)

## Inference FrameWork
both ipex-llm, intel-npu-acceleration-library are inference framework

both intel-extension-for-pytorch, directml with onnex are the training time framework