# Tabular Foundation Model for Performance Prediction in NAS

**Diploma Thesis Project**

This repository contains the source code, experiments, and analysis for my diploma thesis: **"Tabular foundation model for performance prediction in NAS"** (Využití tabulárního základního modelu pro odhad výkonnosti v NAS).

## Abstract

**Neural Architecture Search (NAS)** is a~computationally demanding process requiring exhaustive evaluation of candidate architectures.
This thesis investigates the tabular foundation model **TabPFN (Tabular Prior-Data Fitted Network)** as a~performance estimator.
This model predicts target values in a~single forward pass, completely eliminating iterative training. The aim of this work is to propose and evaluate the integration of TabPFN into the NAS process, focusing on learning curve prediction and the direct estimation of final accuracy using structural features and zero-cost proxy metrics.
These approaches are subsequently integrated into comprehensive search algorithms.
Experimental~evaluation on the NATS-Bench benchmark demonstrates that deploying TabPFN~enhances~evaluation efficiency, enabling the reliable discovery of highly performant architectures.

Czech version (česká verze):

**Hledání neuronové architektury (NAS)** je výpočetně velmi náročný proces vyžadující zdlouhavé vyhodnocování mnoha kandidátních architektur.
Tato práce zkoumá potenciál inovativního tabulárního základního modelu **TabPFN (Tabular Prior-Data Fitted Network)** v~roli odhadce výkonnosti.
Klíčovou předností tohoto modelu je jeho schopnost předpovídat cílové hodnoty jediným dopředným průchodem zcela bez nutnosti iterativního trénování.
Cílem práce je navrhnout a~vyhodnotit zapojení TabPFN do procesu NAS se zaměřením na predikci vývoje učících křivek a~na přímý odhad finální přesnosti na základě strukturálních příznaků a~zero-cost proxy metrik.
Tyto přístupy jsou následně integrovány do ucelených prohledávacích algoritmů.
Experimentální vyhodnocení na benchmarku NATS-Bench prokazuje, že nasazení modelu TabPFN zefektivňuje evaluační fázi a~umožňuje spolehlivé nalezení mimořádně výkonných architektur.

## Key Concepts

* **Tabular Foundation Models:** Pre-trained models capable of in-context learning on tabular dataset.
* **NAS (Neural Architecture Search):** Automating the design of ANN topologies.
* **Zero-Cost Proxies:** Metrics that estimate architecture performance at initialization (e.g., *synflow*, *snip*) to avoid full training.
* **Learning Curves Extrapolation:** Predicting the learning curves based on early training statistics.
* **Direct Performance Estimation:** Predicting final performance based on structural characteristics and Zero-Cost Proxies.

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

## Project Structure

## Usage

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
