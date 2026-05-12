import sys
try:
    import torch
    print('PyTorch:', torch.__version__, 'cuda_available=', torch.cuda.is_available())
except Exception as e:
    print('PyTorch not available:', e)
try:
    import tensorflow as tf
    print('TensorFlow:', tf.__version__, 'GPUs=', tf.config.list_physical_devices('GPU'))
except Exception as e:
    print('TensorFlow not available:', e)
import shutil, subprocess
for cmd in ('nvidia-smi','nvcc --version'):
    if shutil.which(cmd.split()[0]):
        try:
            print('\\nOutput of',cmd)
            subprocess.run(cmd, shell=True, check=False)
        except Exception as e:
            print('Error running',cmd, e)
    else:
        print('\\n',cmd.split()[0],'not found')