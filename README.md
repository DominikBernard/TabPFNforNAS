# Tabular Foundation Model for Performance Prediction in NAS

**Diploma Thesis Project**

This repository contains the source code, experiments, and analysis for my diploma thesis: **"Tabular foundation model for performance prediction in NAS"** (Využití tabulárního základního modelu pro odhad výkonnosti v NAS).

---

## Abstract

**Neural Architecture Search (NAS)** is a computationally demanding process requiring exhaustive evaluation of candidate architectures.
This thesis investigates the tabular foundation model **TabPFN (Tabular Prior-Data Fitted Network)** as a performance estimator.
This model predicts target values in a single forward pass, completely eliminating iterative training. The aim of this work is to propose and evaluate the integration of TabPFN into the NAS process, focusing on learning curve prediction and the direct estimation of final accuracy using structural features and zero-cost proxy metrics.
These approaches are subsequently integrated into comprehensive search algorithms.
Experimental evaluation on the NATS-Bench benchmark demonstrates that deploying TabPFN enhances evaluation efficiency, enabling the reliable discovery of highly performant architectures.

*Czech version (česká verze):*
**Hledání neuronové architektury (NAS)** je výpočetně velmi náročný proces vyžadující zdlouhavé vyhodnocování mnoha kandidátních architektur.
Tato práce zkoumá potenciál inovativního tabulárního základního modelu **TabPFN (Tabular Prior-Data Fitted Network)** v roli odhadce výkonnosti.
Klíčovou předností tohoto modelu je jeho schopnost předpovídat cílové hodnoty jediným dopředným průchodem zcela bez nutnosti iterativního trénování.
Cílem práce je navrhnout a vyhodnotit zapojení TabPFN do procesu NAS se zaměřením na predikci vývoje učících křivek a na přímý odhad finální přesnosti na základě strukturálních příznaků a zero-cost proxy metrik.
Tyto přístupy jsou následně integrovány do ucelených prohledávacích algoritmů.
Experimentální vyhodnocení na benchmarku NATS-Bench prokazuje, že nasazení modelu TabPFN zefektivňuje evaluační fázi a umožňuje spolehlivé nalezení mimořádně výkonných architektur.

---

## Key Concepts

* **Tabular Foundation Models:** Pre-trained models capable of in-context learning on tabular datasets.
* **NAS (Neural Architecture Search):** Automating the design of ANN topologies.
* **Zero-Cost Proxies:** Metrics that estimate architecture performance at initialization (e.g., *synflow*, *snip*) to avoid full training.
* **Learning Curves Extrapolation:** Predicting the learning curves based on early training statistics.
* **Direct Performance Estimation:** Predicting final performance based on structural characteristics and Zero-Cost Proxies.

---

## Results

* **Learning Curve Extrapolation:** The most successful approach combined data from multiple training runs with the sorting of auxiliary architecture indices based on their accuracy prior to the splitting epoch. This method successfully mitigated systematic underestimation and allowed TabPFN to better capture learning dynamics.
* **Direct Final Accuracy Prediction:** TabPFN achieved the highest prediction accuracy when combining three distinct feature types: structural characteristics of the architectures, zero-cost proxy metrics, and early training accuracies (with unconnected architectures filtered out). Integrating early training performance led to a demonstrable and significant improvement in prediction precision.
* **Integration into NAS Algorithms:** Two integration strategies were proposed and successfully evaluated on the NATS-Bench (Topology Search Space) benchmark:
  * *Evolutionary Algorithm:* Intensively utilized TabPFN for architecture evaluation and early-stopping of unpromising candidates.
  * *Active Learning:* Applied TabPFN predictions more conservatively, relying primarily on actual trained values.
  * **Overall Efficiency:** Both algorithms reliably discovered highly performant network topologies. The experiments demonstrated that embedding TabPFN into the NAS pipeline is not only functionally viable but also highly computationally efficient, eliminating the need for full iterative training during evaluation phases.

---

## Installation

To reproduce the experiments, please follow these steps:

### Prerequisites
* Python 3.9+
* CUDA-enabled GPU (recommended for proxy calculation)
* Key libraries: `tabpfn`, `nats_bench`, `numpy`, `matplotlib`, `scipy`, `regex`, and `gdown`.

### Setup

**1. Clone the repository and create a virtual environment:**
```bash
# Clone the repository
git clone https://github.com/DominikBernard/TabPFNforNAS.git
cd TabPFNforNAS

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

```

**2. Install dependencies:**

**Note:** The `requirements.txt` is configured to automatically install PyTorch 2.6.0 with CUDA 12.4 support.
```bash
# Install dependencies
pip install -r requirements.txt
pip install -e .
```

---

## Project Structure
The repository is divided into three primary modules:

### 1. tabpfn_final_accuracy/
Focuses on direct performance estimation using structural characteristics, zero-cost proxies, and early training metrics.
* `utils/`: Contains data loaders, architecture feature extraction algorithms (e.g., parsing raw/one-hot edges, connection checks), and results visualization.
* `experiments/`: Contains experimental scripts, which run TabPFN model against multiple features and hyperparameter combinations.
* `analysis/`: Contains analysis scripts to analyse the experimental results.
* **Main file**: `pred_from_archs_with_zero_cost_and_train_acc.py/` runs a single experiment with specified arguments.

### 2. tabpfn_learning_curves/
Implements learning curve extrapolation based on partial training data.
* `utils/`: Contains data loaders, scripts for many artificial features creation, and results visualization.
* `MNIST_example_runs/`: Contains training data of 10 different runs of simple CNN and MLP architectures on MNIST dataset.
* `tabpfn_on_mnist/`: Contains experimental scripts for running tabpfn as learning curve predictor on example runs on MNIST dataset (from `MNIST_example_runs/`). Three approaches are tried: usage of artificial features, learning curve modification (*differences.py* and *sin_modification.py*), and usage of helper runs.
* `tabpfn_on_nats_bench/` Contains experimental scripts focused mainly on prediction based on helper runs.


### 3. tabpfn_in_whole_nas_algorithms/
Integrates TabPFN performance estimation into comprehensive NAS search algorithms.
* `evolution.py`: Runs an evolutionary search, initializing populations, performing mutations on architecture edges, and filtering offspring with TabPFN predictions.
* `active_learning.py`: Selects optimal architectures via active learning approaches.

---

## Usage
Here are a few examples of how to run the core scripts provided in the repository:

### 1. Direct Final Accuracy Prediction
Run a single TabPFN prediction experiment using specific structural and zero-cost features on NATS-Bench architectures:

```bash
python tabpfn_final_accuracy/pred_from_archs_with_zero_cost_and_train_acc.py \
  --train_size 1000 \
  --test_size 200 \
  --dataset cifar10 \
  --edge_features raw \
  --zero_cost_features f g_n sy \
  --epochs_n_features 10 \
  --skip_unconnected
```
> **Note:** Available zero-cost features include `f` (fisher), `g_n` (grad_norm), `g` (grasp), `j_c` (jacob_cov), `p` (plain), `s` (snip), `sy` (synflow), and `sy_bn` (synflow_bn).

### 2. Learning Curve Extrapolation
Extrapolate a learning curve of architecture trained on a MNIST dataset, splitting the train/test threshold at epoch 20 and adding artificial features (like log and seasonality):

```bash
python tabpfn_learning_curves/tabpfn_on_mnist/initial_example_mnist.py \
  --splitting_epoch 20 \
  --add_optimal_log_feature \
  --add_seasonality_features 5 \
  --path_to_save_plot tabpfn_learning_curves/plots/mnist_curve.pdf
```

### 3. Run Comprehensive NAS Algorithms
Execute a complete architecture search utilizing the evolutionary algorithm paired with TabPFN predictions:

```bash
python tabpfn_in_whole_nas_algorithms/evolution.py \
  --mut_prob 0.16 \
  --num_parents 4 \
  --init_population_size 20
```

---

## References & Literature

This thesis builds upon the following key publications:

* **Hollmann, N., Müller, S., Purucker, L. et al.** (2025). *Accurate predictions on small data with a tabular foundation model.* Nature 637, 319–326. [DOI](https://doi.org/10.1038/s41586-024-08328-6)
* **Mohamed S. Abdelfattah, et al.** (2021). *Zero-Cost Proxies for Lightweight NAS.* ICLR 2021. [DOI](https://doi.org/10.48550/arXiv.2101.08134)
* **Xuanyi Dong, et al.** (2022). *"NATS-Bench: Benchmarking NAS Algorithms for Architecture Topology and Size"* in IEEE Transactions on Pattern Analysis and Machine Intelligence. [DOI](https://doi.org/10.1109/TPAMI.2021.3054824)
* **Shi Bin Hoo, et al.** (2025). *From Tables to Time: How TabPFN-v2 Outperforms Specialized Time Series Forecasting Models.* [arXiv](https://doi.org/10.48550/arXiv.2501.02945)
* **Han-Jia Ye, et al.** (2025). *A Closer Look at TabPFN v2: Strength, Limitation, and Extension.* [arXiv](https://doi.org/10.48550/arXiv.2502.17361)
* **Frank Hutter, et al.** (2019). *Automated Machine Learning - Methods, Systems, Challenges.* Springer. [DOI](https://doi.org/10.1007/978-3-030-05318-5)
* **MINTÁL, Samuel** (2024). *Leveraging lower fidelity proxies for neural network based NAS predictors.* Diploma Thesis, Charles University. [Link](http://hdl.handle.net/20.500.11956/190605)

---

## Author

**Dominik Bernard**
* **University:** Charles University
* **Faculty:** Faculty of Mathematics and Physics
* **Thesis Supervisor:** Mgr. Roman Neruda, CSc.

---

## Acknowledgements

**Built with PriorLabs-TabPFN.**

This project also integrates code and concepts from the [Zero-Cost Proxies for Lightweight NAS](https://github.com/mohsaied/zero-cost-nas/) repository.

---

## Citation

If you find this repository useful in your research, please consider citing the diploma thesis:

```bibtex
@mastersthesis{Bernard2026tabPFNforNAS,
  author       = {Bernard, Dominik},
  title        = {Tabular foundation model for performance prediction in NAS},
  school       = {Charles University, Faculty of Mathematics and Physics},
  year         = {2026},
  type         = {Diploma thesis},
  address      = {Prague},
  note         = {Thesis supervisor: Neruda, Roman}
}
```

---

## License

This project is licensed under the Apache 2.0 License - see LICENSE for details.

This project relies on third-party code and models; please refer to the NOTICE file for detailed attribution and specific third-party license terms.

