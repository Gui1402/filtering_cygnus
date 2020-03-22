#FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04
FROM ubuntu:18.04
WORKDIR /home/
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt-get update

RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*
COPY requirements.yml .

RUN wget\
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh\
    && mkdir /root/.conda\
    && bash Miniconda3-latest-Linux-x86_64.sh -b\
    && rm -f Miniconda3-latest-Linux-x86_64.sh\
    && conda install -c conda-forge tensorflow-gpu==1.14.0\
    && conda env create -f requirements.yml

ARG conda_env=mestrado-env
ENV PATH="/root/miniconda3/envs/$conda_env/bin:${PATH}"
ENV CONDA_DEFAULT_ENV="${conda_env}"

COPY ./notebooks ./notebooks
COPY ./data ./data
COPY ./src ./src  







