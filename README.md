# ECG-TIPE-End2End

End-to-end ECG signal processing pipeline developed during my CPGE TIPE project.

The project implements a complete pipeline:

ECG acquisition → signal denoising → feature extraction → machine learning diagnosis.

## Pipeline

ECG electrodes  
↓  
AD8232 analog amplification module  
↓  
ADS1115 Analog-to-Digital Converter  
↓  
Signal denoising (Wavelet transform)  
↓  
Feature extraction  
↓  
KNN classification  
↓  
Automatic diagnosis report

## Project Goal

The objective of this project is to detect cardiac abnormalities automatically
from ECG signals.

The pipeline:

1. acquires ECG signals using AD8232 + ADS1115
2. removes noise using wavelet denoising
3. extracts signal features
4. compares signals with a dataset of annotated ECG recordings
5. predicts the most probable cardiac disease using KNN.

## Dataset

The classification stage uses a dataset of annotated ECG signals
(MIT-BIH arrhythmia dataset and extended ECG records).

Dataset size: **100000+ ECG signals**

## Project Structure
data/ # datasets (not included)
models/ # trained ML models
figures/ # plots and ECG signals
scripts/ # execution scripts
src/ # project source code

## Hardware Setup

ECG acquisition hardware used in the project.
![Hardware setup](figures/Screenshot.png)

## Installation
pip install -r requirements.txt
## Example

Train the KNN model:
python scripts/run_train_knn.py --train data/train.csv --test data/test.csv

## Technologies

Python  
NumPy  
SciPy  
PyWavelets  
Scikit-Learn  
Signal Processing  
Machine Learning

## Author

Mohamed Ali Essadki  
ENSEEIHT – Engineering School  
Artificial Intelligence Track (MOD IA)
