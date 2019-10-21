# skipNet

It is a Tensorflow implementation of 'Unpaired Image-to-Speech Synthesis with Multimodal Information Bottleneck' ICCV 2019
https://arxiv.org/abs/1908.07094

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.

Installing Dependencies
1. Install Python 3
2. Install the latest version of TensorFlow for your platform. For better performance, install with GPU support if it's available. This code works with TensorFlow 1.4.
3. Install other denpendencies:
pip install -r requirements.txt

Training
1. Prepare Data
Download Dataset
For Image-Text pair, our model trained on COCO, and can be downloaded from: http://cocodataset.org/#download
For Speech-Text pair, our model trained on MS inner dataset. But it can also be trained by other public dataset, e.g. LibriSpeech
http://www.openslr.org/12

Preprocess the Data
For audio samples, our model trained on mel-spectrogram. 
Write your training list in this format:
'mel-spec-path'|'linear-spec-path'|'lenth'|'text'

2. Train the model 
python3 train.py

Finetune or Train on Your Own Data
You can set your own appropriate hypapameters in hparams.txt

