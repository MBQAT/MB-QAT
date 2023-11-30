# MBQAT: Mixture Bit Quantization-Aware Training for Speech Enhancement

This repository provides an implementation of "MBQAT: Mixture Bit Quantization-Aware Training for Speech Enhancement"
for complex-structured speech enhancement networks that consist of both convolutional and recurrent neural layers. The
method incorporates a variety of quantization strategies, including standard QAT, learnable scales, structure-specific
bit resolutions, and a blend of static and dynamic quantization, to effectively address the challenges in quantizing
complex neural network architectures.

## Installation

The program is developed using Python 3.9.
Clone this repo, and install the dependencies:

```
git clone https://github.com/MBQAT/MB-QAT.git
cd MB-QAT
pip install -r requirements.txt
```

## Directory Overview

This project includes several directories, each corresponding to different versions and stages of the network:

1. **CRN_original:** Contains the floating-point network with a parameter size of 17.5M.
2. **CRN_100k:** Features our modified floating-point network reduced to 100k parameters.
3. **CRN_100k_StandardQAT:** Houses the network quantized using only Standard Quantization-Aware Training (StandardQAT).
4. **CRN_100k_GeneralQAT:** Represents an enhanced version of the quantized network, built upon StandardQAT. This
   includes specific structure bit resolution, a combination of static and dynamic quantization, and asymmetric
   activation quantization.
5. **CRN_100k_MBQAT:** Implements the full range of methodologies described in our paper, encompassing all GeneralQAT
   strategies and learnable scales.
6. **TCN_100k_MBQAT:** Contains the Temporal Convolutional Network (TCN) quantized using our methods.

## File Description

- In **CRN_original**, ```train.py``` is used for training and testing the network.
- In other directories (**CRN_100k**, **CRN_100k_StandardQAT**, **CRN_100k_GeneralQAT**, **CRN_100k_MBQAT**,
  and **TCN_100k_MBQAT**), ```model_xxx.py``` serves as the training and testing script.
- ```dynamic_quant_xxx.py``` is present in directories where applicable, and is used for dynamic quantization
  post-Quantization-Aware Training (QAT).

## How to run

This guide outlines the steps to train, fine-tune, and evaluate the MBQAT-enhanced networks, using the example of
CRN_100k_MBQAT. The complete process involves training the floating-point network, fine-tuning the quantized network,
evaluating metrics, and dynamically quantizing LSTM with further evaluation.

1. Train the Quantized Network:

- Assuming you have a pre-trained floating-point model, you can train the quantized network with the following command:

```
python model_100k_MBQAT.py -r train -m [Model Directory]
```

- To evaluate metrics, run:

```
python model_100k_MBQAT.py -r test -m [Model Directory]
```

- Both training and testing will automatically output WAV files.

2. Dynamic Quantization of LSTM and Evaluation:

- To dynamically quantize LSTM and evaluate metrics, execute:

```
python dynamic_quant_100k_MBQAT.py
```