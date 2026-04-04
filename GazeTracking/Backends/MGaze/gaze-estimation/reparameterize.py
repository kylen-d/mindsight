import torch
from models import (
    mobileone_s0,
    reparameterize_model,
)

state_dict = torch.load("mobileone_s0.pt")
model = mobileone_s0(pretrained=False, num_classes=90)  # 90 bins

model.load_state_dict(state_dict)


model.eval()
model_eval = reparameterize_model(model)

torch.save(model_eval.state_dict(), "s0_fused.pt")
