import torch
try:
    ckpt = torch.load('mace_checkpoint_latest.pt', map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict):
        print("Keys:", ckpt.keys())
        print("Epoch:", ckpt.get('epoch', 'N/A'))
        state_dict = ckpt.get('state_dict', ckpt.get('model_state_dict', {}))
        print("Num layers:", len(state_dict))
        print("First 10 keys:", list(state_dict.keys())[:10])
    else:
        print("Model is not a dictionary state_dict, type:", type(ckpt))
except Exception as e:
    print(f"Error loading: {e}")
