import torch
import ultralytics.nn.modules.block as block
from ultralytics.nn.modules.block import C3

# Dummy replacements (just inherit C3 so torch.load works)
class C3k(C3):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class C3k2(C3):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class C2PSA(C3):  # placeholder, real one likely different
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

# Register them into ultralytics namespace
block.C3k = C3k
block.C3k2 = C3k2
block.C2PSA = C2PSA

# Now load checkpoint
ckpt = torch.load("models/eye.pt", map_location="cpu")

print("Checkpoint keys:", ckpt.keys())
print("Train args:", ckpt.get("train_args", None))
print("YAML:", ckpt.get("yaml", None))
