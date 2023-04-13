## GeometricDP-Histogram

> Folder structure for this project


    .
    ├── probabilities.data             # Dataset of the local client probabilities
    ├── histograms.py                  # Differentially private histogram-related functions
    ├── mechanisms              
    │   ├── base.py                    # Base classes for differential privacy mechanisms
    │   ├── geometric.py               # The geometric mechanism for differential privacy
    │   ├── accountant.py              # Privacy budget accountant for differential privacy
    │   ├── validation.py              # Validation functions for the differential privacy library
    │   └── utils.py                   # Basic functions and other utilities for the differential privacy library
    ├── demo1_concept_probability.py   # The walkthrough of publishing concept-based differentially private histograms
    └── demo2_adult_age.py             # Another walkthrough with the adult age dataset
