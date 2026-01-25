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