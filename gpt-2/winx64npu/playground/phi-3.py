from applyllm.accelerators import (
    AcceleratorHelper,
)
import os, sys

# import windows patch module from the repository to the sys path
# relatively from the current file location
patch_module_path = os.path.join(os.path.dirname(__file__), "../")
sys.path.append(patch_module_path)

# path for windows, to change the model cache directory
from patch.win_patch import (
    DIR_MODE_MAP
)

# TODO: rename the init_mps_torch to init_env_torch(dir_setting: DirectorySetting)
AcceleratorHelper.init_mps_torch(dir_setting=DIR_MODE_MAP["win_local"])
# models are cached at C:\Users\yingdingwang\MODELS\huggingface\hub

import torch
from transformers import AutoTokenizer, pipeline, TextStreamer
from intel_npu_acceleration_library.compiler import CompilerConfig
import intel_npu_acceleration_library as npu_lib
import warnings

torch.random.manual_seed(0)

compiler_conf = CompilerConfig(dtype=npu_lib.int4)
model = npu_lib.NPUModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    config=compiler_conf,
    torch_dtype="auto",
)

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
streamer = TextStreamer(tokenizer, skip_prompt=True)

messages = [
    {
        "role": "system",
        "content": "You are a helpful digital assistant. Please provide safe, ethical and accurate information to the user.",
    },
    {
        "role": "user",
        "content": "Can you provide ways to eat combinations of bananas and dragonfruits?",
    },
]

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

generation_args = {
    "max_new_tokens": 500,
    "return_full_text": False,
    "temperature": 0.7,
    "do_sample": True,
    "streamer": streamer,
}

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    pipe(messages, **generation_args)
