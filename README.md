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
```echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh```\
- Deactivate and reactivate to reset
- ```nvcc --version``` to check if cuda 11.8 is currently active

## Pytorch 
Pytorch 2.4.0 is recommended to be used with cuda 11.8. Install with:
```conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0  pytorch-cuda=11.8 -c pytorch -c nvidia```

## Run the setup
```. ./setup.sh --basic --xformers --flash-attn --diffoctreerast --spconv --mipgaussian --kaolin --nvdiffrast``

## Import all dependencies for demo
```. ./setup.sh --demo```

## TEMPORARY WORK IN PROGRESS: Need to install gsplat separately
```pip install gsplat```

## TEMPORARY WORK IN PROGRESS: Need to install TorchMCubes for testing purposes. This works in GPU and is significantly faster
```pip install git+https://github.com/tatsy/torchmcubes.git```

## Check the gradio implementation in browser
```python app.py```