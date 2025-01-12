import torch
from transformers import BertModel

model = BertModel.from_pretrained("bert-base-uncased")
model = model.to("xpu")
model.eval()

vocab_size = model.config.vocab_size
batch_size = 128
seq_length = 512
data = torch.randint(vocab_size, size=[batch_size, seq_length])
data = data.to("xpu")

#################### code changes ####################  # noqa F401
import intel_extension_for_pytorch as ipex

model = ipex.optimize(model)
######################################################  # noqa F401

with torch.no_grad():
    d = torch.randint(vocab_size, size=[batch_size, seq_length])
    model = torch.jit.trace(model, (d,), check_trace=False, strict=False)
    model = torch.jit.freeze(model)

    model(data)

print("Execution finished")