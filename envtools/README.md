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
cd $env:USERPROFILE\Documents\VCS\llm-train;
$env:VERSION="3.11";
$env:ENV_NAME="gpt";
$env:ENV_SURFIX="winconda";
$env:PM="conda";
.\envtools\create_env.ps1 -VERSION $env:VERSION -ENV_NAME $env:ENV_NAME -ENV_SURFIX $env:ENV_SURFIX -PM $env:PM;
```

Python 3.12 venv
```Anaconda Powershell Prompt
cd $env:USERPROFILE\Documents\VCS\llm-train;

$env:VERSION="3.12";
$env:ENV_NAME="gpt";
$env:ENV_SURFIX="winconda";
$env:PM="conda";
.\envtools\create_env.ps1 -VERSION $env:VERSION -ENV_NAME $env:ENV_NAME -ENV_SURFIX $env:ENV_SURFIX -PM $env:PM;
```

Python 3.12 venv
```Anaconda Powershell Prompt
cd $env:USERPROFILE\Documents\VCS\llm-train;

$env:VERSION="3.12";
$env:ENV_NAME="directml";
$env:ENV_SURFIX="winconda";
$env:PM="conda";
.\envtools\create_env.ps1 -VERSION $env:VERSION -ENV_NAME $env:ENV_NAME -ENV_SURFIX $env:ENV_SURFIX -PM $env:PM;
```

## Create and install xpu torch intel-extension (final one)
Add the path to your user env variables: (installed for all users)
*  C:\ProgramData\miniconda3\Scripts\

Update the conda in default base env, before create the venv prefix for new conda env
```Anaconda Powershell Admin
# update base
conda update -n base -c defaults conda
```

Create VENV
```Anaconda Powershell
cd $env:USERPROFILE\Documents\VCS\llm-train;

$env:VERSION="3.12";
$env:ENV_NAME="gpt";
$env:ENV_SURFIX="winconda";
$env:PM="conda";
.\envtools\create_env.ps1 -VERSION $env:VERSION -ENV_NAME $env:ENV_NAME -ENV_SURFIX $env:ENV_SURFIX -PM $env:PM;
```

<!--
```Anaconda Powershell Admin
# update base
$env:VERSION = "3.12";
$env:ENV_NAME = "gpt${env:VERSION}winconda";
conda activate "$HOME\Documents\VENV\${env:ENV_NAME}";
# used prefix not name, -n will not work
# conda not installed in prefix
# conda update --prefix "$HOME\Documents\VENV\${env:ENV_NAME}" -c defaults conda
```
-->

Install the Intell extension
```Anaconda Powershell
$env:VERSION = "3.12";
$env:ENV_NAME = "gpt${env:VERSION}winconda";
conda activate "$HOME\Documents\VENV\${env:ENV_NAME}";

# path env
cd $env:USERPROFILE\Documents\VCS\llm-train;
conda env update --prefix "$HOME\Documents\VENV\${env:ENV_NAME}" --file environment_winx64.yml

# which pip
which pip
```


## (Optional manually) Install XPU with intel-extension-for-pytorch 2.7.10

Training framework

xpu
```Anaconda Powershell
$env:VERSION = "3.12";
$env:ENV_NAME = "gpt${env:VERSION}winconda";
conda activate "$HOME\Documents\VENV\${env:ENV_NAME}";

# path env
cd $env:USERPROFILE\Documents\VCS\llm-train;
conda env update --prefix "$HOME\Documents\VENV\${env:ENV_NAME}" --file environment_winx64.yml

# which pip
which pip

# the torch and intel-extension are now installed with environment_winx64.yml file
# install python2.7 additional pip packages from storage 0 (us storage is fast in eu)
# python -m pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/xpu
python -m pip install intel-extension-for-pytorch==2.7.10+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
```
Reference:
* https://intel.github.io/intel-extension-for-pytorch/

Sanity Test
```Anaconda Powershell
python -c "import torch; import intel_extension_for_pytorch as ipex; print(torch.__version__); print(ipex.__version__); [print(f'[{i}]: {torch.xpu.get_device_properties(i)}') for i in range(torch.xpu.device_count())];"
```

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