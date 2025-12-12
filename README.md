# LIAR-GNN-Adversarial

GraphSAGE on the LIAR dataset for fake news detection + Adversarial Attacks (FGSM, PGD, Structural Attacks)

## Overview
This project implements a Graph Neural Network (GNN) using GraphSAGE on the LIAR dataset for multi-class fake news detection (6 classes: true, mostly-true, half-true, barely-true, false, pants-fire).

The graph is constructed based on shared `speaker` and `subject(s)`. Node features include:
- Categorical features (speaker, party affiliation, venue category, job category, state info)
- Numerical features (credibility counts)
- Text features (TF-IDF trigram from statement, reduced with PCA)

The project is structured for easy extension to adversarial attacks and defenses.

## Dataset
The LIAR dataset is used[](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip).

The dataset is automatically split into train/valid/test and saved as TSV files in `data/`.

## Requirements
See `requirements.txt` for dependencies.

Install with:
```bash
pip install -r requirements.txt