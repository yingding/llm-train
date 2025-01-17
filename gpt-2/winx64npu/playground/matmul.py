from intel_npu_acceleration_library.backend import MatMul
import numpy as np
from applyllm.utils import time_func
from dataclasses import dataclass


# @time_func
# def run_npu_matmul(inC, outC, batch):

#     # Create both inputs
#     X1 = np.random.uniform(-1, 1, (batch, inC)).astype(np.float16)
#     X2 = np.random.uniform(-1, 1, (outC, inC)).astype(np.float16)

#     mm = MatMul(inC, outC, batch, profile=False)

#     return mm.run(X1, X2)

@dataclass
class MatMulConfig:
    inC: int
    outC: int
    batch: int

    def __post_init__(self):
        # Create both inputs
        self.X1 = np.random.uniform(-1, 1, (self.batch, self.inC)).astype(np.float16)
        self.X2 = np.random.uniform(-1, 1, (self.outC, self.inC)).astype(np.float16)

    @time_func
    def run_npu_matmul(self):
        mm = MatMul(self.inC, self.outC, self.batch, profile=False)
        return mm.run(self.X1, self.X2)

    @time_func
    def run_cpu_matmul(self):
        np_result = np.matmul(self.X1, self.X2.T)
        return np_result


if __name__ == "__main__":
    # create a config object with 128, 128, 32
    config = MatMulConfig(128, 128, 32)
    # run the npu matmul
    print(config.run_npu_matmul())
    # run the cpu matmul
    print(config.run_cpu_matmul())