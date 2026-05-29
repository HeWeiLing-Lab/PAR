\# PAR



This repository contains the core model implementation for the manuscript:



\*\*Pathway-aware representation learning reveals histologic correlates of transcriptome-defined immune programs in microsatellite-stable colon adenocarcinoma\*\*



The current release includes the PAR backbone and downstream prediction heads used in the study, including PAR-TA and PAR-MLP variants. Additional documentation, training scripts, evaluation utilities, and model checkpoints will be updated to improve reproducibility.



\## Repository structure



\* `model/par.py`: PAR backbone, including pathway tokenization, cross-attention modules, and WSI-to-pathway decoding.

\* `model/par\_heads.py`: downstream prediction heads built on top of PAR, including pathway-token attention and MLP classifiers.



\## Availability



The repository is publicly available for peer review. Further documentation and trained weights will be added in subsequent updates.



