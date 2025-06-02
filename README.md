## WARNING
This is a work in progress repo under active development. Texturing is not optimal currently. Work in progress. Expect fix in the next commit. Sorry and Thank you. 

## Introduction
The repo makes 2 major modifications to the original TRELLIS library:
- implements gsplat for rendering
- replaces nvdiffrast with pytorch3d for texturing.

## Initial Setup
It is suggested to use Cuda 11.8 due to dependency issues. 
- Download installer:
```wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run```
- Make it executable
```chmod +x cuda_11.8.0_520.61.05_linux.run```
- Install:
```sudo ./cuda_11.8.0_520.61.05_linux.run```

## Conda
- Create a new conda environment:
```conda create --name trellis_refactored```
- Activate:
```conda activate trellis_refactored```
- Need to default to cuda 11.8:
```mkdir -p $CONDA_PREFIX/etc/conda/activate.d```\
```echo 'export CUDA_HOME=/usr/local/cuda-11.8' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh```\
```echo 'export PATH=$CUDA_HOME/bin:$PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh```\
```echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh```
- Deactivate and reactivate to reset
- ```nvcc --version``` to check if cuda 11.8 is currently active

## Add conda forge channel
```conda config --add channels conda-forge```\
```conda config --set channel_priority flexible```

## Pytorch 
Pytorch 2.4.0 is recommended to be used with cuda 11.8. Install with:
```conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0  pytorch-cuda=11.8 -c pytorch -c nvidia```

## Run the setup
```. ./setup.sh --basic --xformers --flash-attn --diffoctreerast --spconv --mipgaussian --gsplat --demo```

## Pytorch 3d installation with Conda
```conda install pytorch3d -c pytorch3d```

## BUG ALERTS !!
1. ```python app.py``` gives gradio error: argument of type 'bool' is not iterable.
Happens due to pydantic version mismatch.
Unistall pydantic with:
```pip uninstall pydantic```
Install version 2.10.6
```pip install pydantic==2.10.6```

2. Torchvision fails after installing pytorch3d.
- Remove torchvision completely:
```conda remove torchvision```
- Reinstall with full dependencies:
```conda install pytorch==2.4.0 torchvision==0.19.0 pytorch-cuda=11.8 -c pytorch -c nvidia```
- After installing tqdm install pytorch3d again:
```conda install pytorch3d -c pytorch3d```

## Run the gradio implementation and check in browser
```python app.py```