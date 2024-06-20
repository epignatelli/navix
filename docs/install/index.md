## Install JAX
NAVIX depends on JAX. 
Follow the official [JAX installation guide](https://github.com/google/jax#installation.) for your OS and preferred accelerator.

For a quick start, you can install JAX for GPU with the following command:
```bash
pip install -U "jax[cuda12]"
```
which will install JAX with CUDA 12 support.


## Install NAVIX
```bash
pip install navix
```

Or, for the latest version from source:
```bash
pip install git+https://github.com/epignatelli/navix
```


## Installing in a conda environment
We recommend install NAVIX in a conda environment.
To create a new conda environment and install NAVIX, run the following commands:
```bash
conda create -n navix python=3.10
conda activate navix
cd <path/to/navix-repo>
pip install navix
```
