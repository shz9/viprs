The `viprs` software is written in `C/C++` and `Cython/Python3` and is designed to be fast and accurate.
The software is designed to be used in a variety of computing environments, including local workstations, 
shared computing environments, and cloud-based computing environments. Because of the dependencies on `C/C++`, you need 
to ensure that a `C/C++` Compiler (with appropriate flags) is present on your system.

## Requirements

Building the `viprs` package requires the following dependencies:

* `python` (>=3.8)
* `C/C++` Compilers
* `cython`
* `numPy` 
* `sciPy` (>=1.5.4)

To take full advantage of the **parallel processing** capabilities of the package, you will also need to make sure that 
the following packages/libraries are available:

* `OpenMP` 
* `BLAS`

### Setting up the environment with `conda`

If you can use `Anaconda` or `miniconda` to manage your Python environment, we recommend using them to create 
a new environment with the required dependencies as follows:

```bash
python_version=3.11  # Change python version here if needed
conda create --name "viprs_env" -c anaconda -c conda-forge python=$python_version compilers openblas -y
conda activate viprs_env
```

Using `conda` to setup and manage your environment is especially *recommended* if you have trouble compiling 
the `C/C++` extensions on your system.

## Installation

### Using `pip`

The package is available for easy installation via the Python Package Index (`pypi`) can 
be installed using `pip`:

```bash
python -m pip install viprs>=0.1
```

### Building from source

You may also build the package from source, by cloning the repository and 
running the `make install` command:

```bash
git clone https://github.com/shz9/viprs.git
cd viprs
make install
```

### Using a virtual environment

If you wish to use `viprs` on a shared computing environment or cluster, 
it is recommended that you install the package in a virtual environment. Here's a quick 
example of how to install `viprs` on a SLURM-based cluster:

```bash
module load python/3.8
python3 -m venv viprs_env  # Assumes venv is available
source viprs_env/bin/activate
python -m pip install --upgrade pip
python -m pip install viprs>=0.1
```
