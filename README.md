# Tabular Foundation Model for Performance Prediction in NAS

**Diploma Thesis Project**

This repository contains the source code, experiments, and analysis for my diploma thesis: **"Tabular foundation model for performance prediction in NAS"** (Využití tabulárního základního modelu pro odhad výkonnosti v NAS).

## Abstract

Neural Architecture Search (NAS) automates the design of neural networks but often suffers from prohibitive computational costs due to the need to train thousands of candidate models.

The goal of this thesis is to explore the application of **TabPFN** (Tabular Prior-data Fitted Network) for predicting the performance of neural architectures. TabPFN is a transformer-based foundation model for tabular data that allows for extremely fast predictions via inference, without the need for traditional gradient-based training on the target dataset.

This project focuses on:
1.  **Performance Prediction:** Utilizing TabPFN to predict the final accuracy of architectures based on structural characteristics and **Zero-Cost Proxies**.
2.  **Learning Curve Extrapolation:** Investigating the use of TabPFN to forecast the training trajectory of architectures.
3.  **Benchmarking:** Verifying the suitability of these methods on **NATS-Bench**.

## Key Concepts

* **Foundation Models:** Pre-trained models capable of in-context learning on new tasks.
* **NAS (Neural Architecture Search):** Automating the design of ANN topologies.
* **Zero-Cost Proxies:** Metrics that estimate architecture performance at initialization (e.g., *synflow*, *snip*) to avoid full training.
* **Learning Curves Extrapolation:** Predicting final performance based on early training statistics.

## Installation

To reproduce the experiments, please follow these steps:

### Prerequisites
* Python 3.9+
* CUDA-enabled GPU (recommended for proxy calculation)

### Setup
```bash
# Clone the repository
git clone https://github.com/DominikBernard/TabPFNforNAS.git
cd TabPFNforNAS

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

## Usage

## Project Structure

## References & Literature

This thesis builds upon the following key publications:

* **Hollmann, N., Müller, S., Purucker, L. et al.** (2025). *Accurate predictions on small data with a tabular foundation model.* Nature 637, 319–326. [DOI](https://doi.org/10.1038/s41586-024-08328-6)
* **Mohamed S. Abdelfattah, et al.** (2021). *Zero-Cost Proxies for Lightweight NAS.* ICLR 2021. [DOI](https://doi.org/10.48550/arXiv.2101.08134)
* **Xuanyi Dong, et al.** (2022). *"NATS-Bench: Benchmarking NAS Algorithms for Architecture Topology and Size"* in IEEE Transactions on Pattern Analysis and Machine Intelligence. [DOI](https://doi.org/10.1109/TPAMI.2021.3054824)
* **Shi Bin Hoo, et al.** (2025). *From Tables to Time: How TabPFN-v2 Outperforms Specialized Time Series Forecasting Models.* [arXiv](https://doi.org/10.48550/arXiv.2501.02945)
* **Han-Jia Ye, et al.** (2025). *A Closer Look at TabPFN v2: Strength, Limitation, and Extension.* [arXiv](https://doi.org/10.48550/arXiv.2502.17361)
* **Frank Hutter, et al.** (2019). *Automated Machine Learning - Methods, Systems, Challenges.* Springer. [DOI](https://doi.org/10.1007/978-3-030-05318-5)
* **MINTÁL, Samuel** (2024). *Leveraging lower fidelity proxies for neural network based NAS predictors.* Diploma Thesis, Charles University. [Link](http://hdl.handle.net/20.500.11956/190605)

## Author

**Dominik Bernard**
* **University:** Charles University
* **Faculty:** Faculty of Mathematics and Physics
* **Thesis Supervisor:** Mgr. Roman Neruda, CSc.
