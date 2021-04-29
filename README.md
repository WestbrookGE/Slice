# Slice

A **flexible** python implementation of paper

>[**Slice: Scalable Linear Extreme Classifiers**](http://manikvarma.org/code/Slice/download.html)

>Himanshu Jain, Venkatesh B., Bhanu Teja Chunduri, and Manik Varma

>WSDM 2019

## Features

Compared to the [original implementation](https://github.com/xmc-aalto/bonsai) written in C++, this code is more flexible in the following parts:

- Sparse/Dense Data Format
- Label Embedding
- Label Classification

This code only provide the naive solution in Slice. You can try anything you like.

## Requirements

- scipy
- sklearn
- numpy
- [xclib](https://github.com/kunaldahiya/pyxclib)
- hnswlib

## Dataset

Please visit the [Extreme Classification Repository](http://manikvarma.org/downloads/XC/XMLRepository.html) to download the benchmark datasets.

## Usage

Please refer to [sample_run.py](sample_run.py)
