<h1 align="center">Medical-Aeronautic RAG Engine</h1>

<p align="center"><i>Defensible AI for medical-fitness decisions — a verdict is never issued without a citable source.</i></p>

<p align="center">
  <img src="https://img.shields.io/badge/license-MIT-C8A24A?style=flat-square" alt="MIT">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Ollama-000000?style=flat-square&logo=ollama&logoColor=white" alt="Ollama">
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/RAG-C8A24A?style=flat-square" alt="RAG">
  <img src="https://img.shields.io/badge/status-flagship-2ea44f?style=flat-square" alt="flagship">
</p>

> **Note** — This is the open technical engine. The user-facing product built on top of it (**AeroFit**, my final-year capstone) lives in a private repo.

## The problem

Aeromedical fitness decisions have to be **traceable**: an examiner must be able to see *why* a pilot was flagged and *which regulation* backs it. A model that just outputs "unfit" with no provenance is useless in a regulated domain — and a language model that hallucinates a rule is worse than useless.

This engine is built around one rule: **it never answers outside its sources.** Every RAG response cites the regulation it came from; every ML verdict is tied to the clinical variables that produced it.

## Features

- **Dual-engine architecture** — a supervised ML classifier for the fitness verdict, plus an air-gapped RAG engine for the regulatory reasoning behind it.
- **Risk-first tuning** — the classifier is optimized for **recall on the risk class**, because a missed at-risk pilot costs far more than a false alarm.
- **Latent-profile segmentation** — unsupervised K-Means (k=3) surfaces risk profiles that aren't explicit in the labels.
- **Air-gapped RAG** — Mistral-7B via Ollama with `nomic-embed-text` embeddings over a public regulatory corpus. Runs fully local; no clinical data leaves the machine.
- **CRISP-DM workflow** — reproducible pipeline from raw data to validation.

## Architecture

```mermaid
flowchart LR
    A[Clinical signals] --> B[Supervised ML<br/>fitness verdict]
    A --> C[K-Means k=3<br/>risk profiles]
    D[Regulatory corpus] --> E[Embeddings<br/>nomic-embed-text]
    E --> F[Air-gapped RAG<br/>Mistral-7B via Ollama]
    B --> G[Traceable verdict<br/>fit / fit-with-restrictions / unfit]
    C --> G
    F --> G
    G --> H["Answer + cited source"]
```

## Quickstart

```bash
git clone https://github.com/akhanER2000/Local-RAG-medical-assistance-aeronautic.git
cd Local-RAG-medical-assistance-aeronautic
pip install -r requirements.txt
ollama pull mistral && ollama pull nomic-embed-text
jupyter notebook   # open the pipeline under notebooks/
```

## Stack

Python · Jupyter · scikit-learn · Ollama (Mistral-7B) · `nomic-embed-text` · Retrieval-Augmented Generation

## Results

- **0.65 recall on the risk class** — deliberately traded precision for recall, because a missed risk is the expensive error.
- Verdicts are **fit / fit-with-restrictions / unfit**, each returned with its supporting regulation.

## Status & roadmap

`status: flagship` · active. Next: expand the regulatory corpus and add per-verdict confidence surfacing.

## License & contact

MIT © 2026 Akhan Lorenzo Espinoza Rojas
[Portfolio](https://cs-portfolio-psi-topaz.vercel.app) · [LinkedIn](https://www.linkedin.com/in/akhan-espinoza)
