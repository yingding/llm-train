# Create
```shell
# pushd /Users/yingding/Code/VCS/ai/llm-train;
VERSION=3.12;
ENV_NAME="agent${VERSION}";
source ./envtools/create_env.sh -p ~/Code/VENV -e ${ENV_NAME} -v $VERSION;
# popd;
```

# Activate
```shell
VERSION=3.12;
ENV_NAME="agent${VERSION}";
PROJ_PATH="$HOME/Code/VCS/ai/llm-train"
source ~/Code/VENV/${ENV_NAME}/bin/activate;
cd ${PROJ_PATH};
python3 -m pip install -r ${PROJ_PATH}/requirements_arm64.txt --no-cache;
```