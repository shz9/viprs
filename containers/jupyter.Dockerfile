# Usage:
# ** Step 1 ** Build the docker image:
# docker build -f ../vemPRS/containers/jupyter.Dockerfile -t viprs-jupyter .
# ** Step 2 ** Run the docker container (pass the appropriate port):
# docker run -p 8888:8888 viprs-jupyter
# ** Step 3 ** Open the link in your browser:
# http://localhost:8888


FROM python:3.11-slim-buster

LABEL authors="Shadi Zabad"
LABEL version="0.1"
LABEL description="Docker image containing all requirements to run the VIPRS package in a Jupyter Notebook"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    unzip \
    wget \
    pkg-config \
    g++ gcc \
    libopenblas-dev \
    libomp-dev

# Download and setup plink2:
RUN mkdir -p /software && \
    wget https://s3.amazonaws.com/plink2-assets/alpha5/plink2_linux_avx2_20240105.zip -O /software/plink2.zip && \
    unzip /software/plink2.zip -d /software && \
    rm /software/plink2.zip

# Download and setup plink1.9:
RUN mkdir -p /software && \
    wget https://s3.amazonaws.com/plink1-assets/plink_linux_x86_64_20231211.zip -O /software/plink.zip && \
    unzip /software/plink.zip -d /software && \
    rm /software/plink.zip

# Add plink1.9 and plink2 to PATH:
RUN echo 'export PATH=$PATH:/software' >> ~/.bashrc

# Install viprs package from PyPI
RUN pip install --upgrade pip viprs jupyterlab

# Expose the port Jupyter Lab will be served on
EXPOSE 8888

# Set the working directory
WORKDIR /viprs_dir

# Copy the current directory contents into the container at /app
COPY . /viprs_dir

# Run Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--NotebookApp.token=''"]
