## NPU Windows Native Venv

### (option1) Automatically creating NPU Venv on Windows 11 
```powershell
$env:VERSION="3.12";
$env:ENV_NAME="gpt";
$env:ENV_SURFIX="";
.\envtools\create_env.ps1 -VERSION $env:VERSION -ENV_NAME $env:ENV_NAME -ENV_SURFIX $env:ENV_SURFIX;
```
this will create a native python3.12 venv, install the python notebook kernel.

In the activated python venv from powsershell install all the dependency:
```powershell
$env:VERSION = "3.12";
$env:ENV_NAME="gpt";
$env:ENV_SURFIX="";
$env:ENV_NAME_GEN = "$env:ENV_NAME$env:VERSION$env:ENV_SURFIX";
$env:ENV_ROOT="$HOME\Documents\VENV";
& "$env:ENV_ROOT\$env:ENV_NAME_GEN\Scripts\Activate.ps1";

cd "C:\Users\yingdingwang\Documents\VCS\llm-train";
python -m pip install -r ./requirements_winx64.txt;
# python -m pip install --no-cache-dir -r ./requirements_winx64.txt;
```
this will take a while.

### Testing NPU Venv
The NPU python lib (`intel-npu-acceleration-library==1.4.0`) works at the time of writing (17/01/2025) only with the native windows venv `gpt3.12`.

Run the notebook ``gpt-2\winx64npu\playground\matmul.ipynb` to test the npu.

## Issues
* NPU example workaround https://github.com/intel/intel-npu-acceleration-library/issues/108#issuecomment-2463827536





