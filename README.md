# NeuBacH
Neu (new/neural network) BAckground model for Hxmt

NeuBacH is a novel project aimed at enhancing the accuracy of background modeling for the Hard X-ray Modulation Telescope (HXMT). Leveraging the power of neural networks, specifically the BERT architecture, NeuBacH is designed to train the background model for each telescope on board HXMT and produce a more precise background spectra. The training process incorporates all available information provided on board HXMT, including the status of the telescope (high-voltage settings, particle monitor count rate), the geomagnetic environment (South Atlantic Anomaly (SAA) presence, elevation angle, satellite position, cosmic ray background, etc.), and the spectrum of the blind detector. This new model capitalizes on the state-of-the-art transformer architecture to significantly improve the accuracy and reliability of background noise characterization in X-ray astronomy.

# NeuBacH - Roadmap

This document outlines the current status and the upcoming milestones of the project.

*Updated: Thu, 15 Aug 2024*

## NeuBacH

#### Milestone Summary

| Status | Milestone | Goals | ETA |
| :---: | :--- | :---: | :---: |
| ❌ | **[Implement BERT-based Background Model for LE](#implement-LE-background-model)** | 2 / 5 | Aug 16 2024 |
| ❌ | **[Implement BERT-based Background Model for ME](#implement-ME-background-model)** | 2 / 5 | Aug 17 2024 |
| ✅ | **[Implement BERT-based Background Model for HE](#implement-HE-background-model)** | x / x | TBD |
| 🚀 | **[Data Preprocessing Pipeline](#data-preprocessing-pipeline)** | 0 / 2 | Late Sep 2024 |
| 🚀 | **[Model Validation for Scientific Observations](#model-vaidation-for-scientific-observaations)** | 0 / 3 | Early Oct 2024 |
| 🚀 | **[Integration with HXMT Data Analysis Framework](#integration-hxmt-data-analysis)** | 0 / 3 | TBD |
| 🚀 | **[Final Evaluation and Publishing](#final-evaluation-model-publishing)** | 0 / 2 | Dec 2024 |

#### Implement BERT-based Background Model for LE

> This milestone focuses on developing and implementing a BERT-based model to accurately predict background spectra for the Low Energy (LE) detector on HXMT.

🚀 &nbsp;**OPEN** &nbsp;&nbsp;📉 &nbsp;&nbsp;**2 / 5** goals completed **(40%)** &nbsp;&nbsp;📅 &nbsp;&nbsp;**Tue Sep 30 2024**
