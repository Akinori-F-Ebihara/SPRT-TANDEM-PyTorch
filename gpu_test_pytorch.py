import os
import torch
print(f'{torch.__version__=}')
print(f'{torch.cuda.device_count()=}')
print(f'{torch.cuda.is_available()=}')
# def grab_gpu(gpu=0):
#     # restrict visible device(s)
#     # os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
#     # Get cpu or gpu device for training.
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f'Using {device=}')
#     return torch.device(device)

# device = grab_gpu() 
