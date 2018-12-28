import subprocess
from .util_print import str_warning, str_verbose


def set_gpu(gpu, check=True):
    import os
    _check_gpu(gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    import torch
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    if check:
        if not _check_gpu_setting_in_use(gpu):
            print('[Warning] gpu setting overwritten. torch.cuda may be initialized before running this function.')


def _check_gpu_setting_in_use(gpu):
    '''
    check that CUDA_VISIBLE_DEVICES is actually working
    by starting a clean thread with the same CUDA_VISIBLE_DEVICES
    '''
    import subprocess
    output = subprocess.check_output('CUDA_VISIBLE_DEVICES=%s python -c "import torch; print(torch.cuda.device_count())"' % gpu, shell=True)
    output = output.decode().strip()
    import torch
    return torch.cuda.device_count() == int(output)


def _check_gpu(gpu):
    msg = subprocess.check_output('nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,nounits,noheader -i %s' % (gpu,), shell=True)
    msg = msg.decode('utf-8')
    all_ok = True
    for line in msg.split('\n'):
        if line == '':
            break
        stats = [x.strip() for x in line.split(',')]
        gpu = stats[0]
        util = int(stats[1])
        mem_used = int(stats[2])
        if util > 10 or mem_used > 1000:  # util in percentage and mem_used in MiB
            print(str_warning, 'Designated GPU in use: id=%s, util=%d%%, memory in use: %d MiB' % (gpu, util, mem_used))
            all_ok = False
    if all_ok:
        print(str_verbose, 'All designated GPU(s) free to use. ')


def set_manual_seed(seed):
    import random
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError as err:
        print('Numpy not found. Random seed for numpy not set. ')
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError as err:
        print('Pytorch not found. Random seed for pytorch not set. ')
