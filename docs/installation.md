The `viprs` software is written in `C/C++` and `Cython/Python3` and is designed to be fast and accurate.
The software is designed to be used in a variety of computing environments, including local workstations, 
shared computing environments, and cloud-based computing environments. Because of the dependencies on `C/C++`, you need 
to ensure that a `C/C++` Compiler (with appropriate flags) is present on your system.

## Requirements

Building the `viprs` package requires the following dependencies:

* `python` (3.10, 3.11, 3.12, or 3.13)
* `C/C++` Compilers
* `cython`
* `numPy` 
* `pkg-config`
* `sciPy` (>=1.5.4)

To take full advantage of the **parallel processing** capabilities of the package, you will also need to make sure that 
the following packages/libraries are available:

* `OpenMP` 
* `BLAS`

### Setting up the environment with `conda`

If you can use `Anaconda` or `miniconda` to manage your Python environment, we **recommend** using them to create 
a new environment with the required dependencies as follows:

```bash
python_version=3.11  # Supported versions are 3.10, 3.11, 3.12, and 3.13
conda create --name "viprs_env" -c anaconda -c conda-forge python=$python_version compilers pkg-config openblas -y
conda activate viprs_env
```

Using `conda` to setup and manage your environment is especially *recommended* if you have trouble compiling 
the `C/C++` extensions on your system.

## Installation

### Using `pip`

The package is available for easy installation via the Python Package Index (`pypi`) can 
be installed using `pip`:

```bash
python -m pip install viprs
```

### Using `uv`

If you use [`uv`](https://docs.astral.sh/uv/) to manage Python environments, you can create a virtual environment
with a supported Python version and install `viprs` into it as follows:

```bash
uv venv --python 3.11 viprs_env
source viprs_env/bin/activate
uv pip install viprs
```

For command-line use without manually activating an environment, you can also run the CLI tools through `uvx`:

```bash
uvx --from viprs viprs_fit -h
uvx --from viprs viprs_score -h
uvx --from viprs viprs_evaluate -h
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
module load python/3.11
python3 -m venv viprs_env  # Assumes venv is available
source viprs_env/bin/activate
python -m pip install --upgrade pip
python -m pip install viprs
```

### Using `Docker` containers

If you are using `Docker` containers, you can pull the pre-built CLI image from DockerHub:

```bash
docker pull shadizabad/viprs:latest
docker run --rm shadizabad/viprs:latest viprs_fit -h
```

You can also build the same CLI image locally from the repository checkout:

```bash
docker build --platform linux/amd64 -f containers/cli.Dockerfile -t viprs-cli .
docker run --rm viprs-cli viprs_fit -h
docker run --rm viprs-cli viprs_score -h
docker run --rm viprs-cli viprs_evaluate -h
```

### Using `Apptainer`

On shared computing environments where `Docker` is not available, you can run the DockerHub image with
[`Apptainer`](https://apptainer.org/):

```bash
apptainer pull viprs-cli.sif docker://shadizabad/viprs:latest
apptainer exec viprs-cli.sif viprs_fit -h
```

To run `viprs` commands against files in your current working directory, bind the directory into the container:

```bash
apptainer exec --bind "$PWD:/work" viprs-cli.sif viprs_fit -h
apptainer exec --bind "$PWD:/work" viprs-cli.sif viprs_score -h
apptainer exec --bind "$PWD:/work" viprs-cli.sif viprs_evaluate -h
```
