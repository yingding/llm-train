### (option1) Automatically creating NPU Venv on Windows 11 

Run this once before create conda venv
```Anaconda Powershell Prompt
conda init powershell
conda activate base
conda deactivate 
```

Create a conda python 3.11 venv
from an Anaconda powershell prompt, not a powershell prompt.
```Anaconda Powershell Prompt
$env:VERSION="3.11";
$env:ENV_NAME="gpt";
$env:ENV_SURFIX="winconda";
$env:PM="conda";
.\envtools\create_env.ps1 -VERSION $env:VERSION -ENV_NAME $env:ENV_NAME -ENV_SURFIX $env:ENV_SURFIX -PM $env:PM;
```

Activate conda venv
```Anaconda Powershell Prompt
# $condaCmd ="$env:USERPROFILE\AppData\Local\miniconda311\Scripts\conda.exe";
$env:VERSION = "3.11";
$env:ENV_NAME = "gpt${env:VERSION}winconda";
conda activate "$HOME\Documents\VENV\${env:ENV_NAME}";
```
Note:
* You need to add the specific conda version path to the user `Path` environment variable
```console
C:\Users\yingdingwang\AppData\Local\miniconda3
C:\Users\yingdingwang\AppData\Local\miniconda3\Scripts
```
So that you can use `conda activate`, otherwise you will get `conda init before conda activate` error.

You can use conda to create different python venv verison

## Open Anaconda Powershell Prompt
```Anaconda PowerShell Prompt
conda env list
conda activate C:\Users\yingdingwang\Documents\VENV\gpt3.11winconda
pip install --pre --upgrade ipex-llm[npu]

set IPEX_LLM_NPU_MTL=1

pip install -r requirements_winx64_conda.txt
```
Note:
* `set IPEX_LLM_NPU_MTL=1` for Intel Core™ Ultra Processors (Series 1) with processor number 1xxH (code name Meteor Lake)

Reference:
* https://github.com/intel/ipex-llm/blob/main/docs/mddocs/Quickstart/npu_quickstart.md

## Run a model on NPU
In Anaconda PowerShell Prompt
```
cd .\gpt-2\winx64npu\playground

python ./generate.py --repo-id-or-model-path "Qwen/Qwen2.5-1.5B" --save-directory "C:\Users\yingdingwang\MODELS\npu"

python ./generate.py --repo-id-or-model-path "meta-llama/Llama-2-7b-chat-hf" --save-directory "C:\Users\yingdingwang\MODELS\npu"

Qwen/Qwen2.5-72B-Instruct

python qwen.py --repo-id-or-model-path "Qwen/Qwen2-7B-Instruct"

# python qwen.py --repo-id-or-model-path "Qwen/Qwen2-7B-Instruct" --save-directory "C:\Users\yingdingwang\MODELS\npu"
```

Reference:
* https://github.com/intel/ipex-llm/tree/main/python/llm/example/NPU/HF-Transformers-AutoModels/Save-Load
* https://github.com/intel/ipex-llm/tree/main/python/llm/example/NPU/HF-Transformers-AutoModels/LLM


