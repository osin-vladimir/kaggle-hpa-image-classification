import torch
import time
import numpy  as np
import random
import platform

# set seeds
SEED = int(time.time())
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# set cuda/cudnn
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled   = True
n_devices                      = torch.cuda.device_count()
device                         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('-----------------------------------------------------------------------------------------------------------')
print('random seed        :  {}'.format(SEED))
print('python  version    :  {}'.format(platform.python_version()))
print('pytorch version    : ', torch.__version__)
print('your    device     : ', device)
print('cuda    available  : ', torch.cuda.is_available())
print('cuda    version    : ', torch.version.cuda)
print('cudnn   version    : ', torch.backends.cudnn.version())
print('gpu devices        : ', n_devices)

if n_devices > 1:
    for dev_ind in range(n_devices):
        print('\tgpu name {} : {}'.format(dev_ind, torch.cuda.get_device_name(dev_ind)))
    print('\tcurrent    : {}'.format(torch.cuda.get_device_name(torch.cuda.current_device())))
elif n_devices == 1:
    print('\tgpu name : {}'.format(torch.cuda.get_device_name(0)))
elif n_devices == 0:
    print('\tcpu usage')
print('-----------------------------------------------------------------------------------------------------------')