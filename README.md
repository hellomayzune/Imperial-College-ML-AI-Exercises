
<p align="center">
• <a href="https://mayzune.com/"><strong>May Zune</strong></a> •
<a href="https://github.com/hellomayzune"><strong>GitHub</strong></a> •
<a href="https://orcid.org/0000-0003-0282-2633"><strong>ORCID</strong></a> •
<a href="https://scholar.google.com/citations?user=LmP8B_4AAAAJ&hl=en"><strong>Google Scholar</strong></a> •
<a href="https://www.researchgate.net/profile/May-Zune"><strong>ResearchGate</strong></a> •
<a href="https://www.linkedin.com/in/mayzune//"><strong>Linkedin</strong></a> •
</p>

<p align="center">
  <a href="https://www.imperial.ac.uk/">
    <img src="https://img.shields.io/badge/Imperial%20College%20London-000025?style=flat" alt="Imperial College London">
  </a>
  <a href="https://github.com/hellomayzune/Imperial-College-Capstone-Black-box-Optimisation/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
  </a>
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-3776AB?style=flat&logo=python&logoColor=white" alt="Python Version">
  </a>
  <a href="https://scikit-learn.org/stable/">
    <img src="https://img.shields.io/badge/scikit--learn-v1.4.0-F7931E?style=flat&logo=scikit-learn&logoColor=white" alt="scikit-learn">
  </a>
</p>


# Unsupervised Machine Learning for Housing Stock Analysis

A collection of practical **unsupervised machine learning projects** developed as part of my learning through the **Imperial College London Professional Certificate in Machine Learning / AI**.

The repository explores how clustering and dimensionality-reduction techniques can be applied to real-world housing, energy, and building-stock datasets. The notebooks focus on identifying **housing archetypes, energy-performance patterns, property segments, and potential fuel-poverty profiles** without relying on predefined target labels.

The primary purpose of this repository is **technical learning and demonstration**: exploring machine learning concepts, understanding their practical applications, comparing algorithms, and evaluating their limitations using realistic datasets, including **Energy Performance Certificate (EPC)** data, **Verisk UKB GIS** housing data, and controlled synthetic examples.

---

## About This Repository

- version: 2 (10 August 2026)

The projects follow an increasingly structured progression from exploratory clustering experiments to more scalable and application-oriented unsupervised learning pipelines.

The work covers:

* **K-Means and Mini-Batch K-Means** for residential building-stock segmentation
* **BIRCH** for scalable clustering of large housing and EPC datasets
* **K-Prototypes** for clustering mixed numerical and categorical data
* **Hierarchical Agglomerative Clustering (HAC)** and hybrid **BIRCH + HAC** approaches
* **Principal Component Analysis (PCA)** for dimensionality reduction and interpretation
* **ColumnTransformer-based preprocessing** for mixed housing datasets
* **Feature engineering, imputation, scaling and categorical encoding**
* **Cluster selection and validation** using inertia, elbow analysis, silhouette score, Calinski-Harabasz and Davies-Bouldin metrics
* **Cluster profiling and visualisation** to interpret discovered housing archetypes
* **Gower distance and t-SNE** for mixed-data similarity and visual exploration
* Comparative analysis of **data-driven clustering versus domain-informed housing classification**

The notebooks deliberately include both successful approaches and methodological limitations. A recurring theme is that producing clusters is only the first step: the resulting groups need to be **interpretable, stable, and relevant to a real housing or energy problem** before they can support practical decision-making.

---

## Core Skills & Tech Stack

### Machine Learning

* Unsupervised Learning
* K-Means Clustering
* Mini-Batch K-Means
* BIRCH
* K-Prototypes
* Hierarchical Agglomerative Clustering
* Hybrid BIRCH + HAC
* Principal Component Analysis (PCA)
* Gower Distance
* t-SNE

### Data Preparation & Feature Engineering

* Exploratory Data Analysis (EDA)
* Missing-value treatment
* Numerical feature transformation
* Ordinal feature engineering
* Standardisation with `StandardScaler`
* Min-Max normalisation with `MinMaxScaler`
* One-hot encoding
* Mixed numerical/categorical preprocessing
* `ColumnTransformer` pipelines
* Log transformations
* Cluster profiling and statistical summaries

### Model Evaluation

* Elbow / inertia analysis
* Silhouette Score
* Calinski-Harabasz Index
* Davies-Bouldin Index
* Cluster cohesion and separation
* PCA visualisation
* Cluster-size analysis
* Feature centroid analysis
* Categorical composition analysis

### Python Ecosystem

* Python
* pandas
* NumPy
* scikit-learn
* matplotlib
* seaborn
* kmodes / K-Prototypes
* Jupyter Notebook

---

# Featured Projects 

## 1. Scalable Housing Clustering

### Mini-Batch K-Means vs BIRCH + HAC

Two related notebooks investigate how unsupervised clustering can be scaled to **large UK residential datasets**.

The housing-stock dataset contains approximately **614,460 residential records** and features including:

* `premise_age`
* `premise_type`
* `premise_floor_count`
* `bungalow`
* `height`
* `building_area`
* `gross_area`

The notebooks demonstrate why conventional Hierarchical Agglomerative Clustering becomes impractical at this scale and investigate more computationally efficient alternatives.

### Methods explored

* K-Means
* Mini-Batch K-Means
* BIRCH
* BIRCH + Hierarchical Agglomerative Clustering
* ColumnTransformer
* One-hot encoding
* Feature scaling
* PCA-based visualisation

A key learning point is the trade-off between **computational scalability, cluster geometry, categorical-data handling and interpretability**.

The analysis also highlights an important practical distinction: scalable clusters do not automatically represent meaningful housing archetypes. Further domain validation is required before using them for retrofit planning, policy or other real-world decisions.

---

## 2. EPC Housing Clustering

### BIRCH, Mini-Batch K-Means and K-Prototypes

This project applies several clustering approaches to a **Sheffield EPC dataset containing 209,230 records and 25 columns**.

The analysis compares:

| Method                 | Main Strength                                             |
| ---------------------- | --------------------------------------------------------- |
| **BIRCH**              | Scalable hierarchical/tree-based clustering               |
| **Mini-Batch K-Means** | Fast clustering for large numerical datasets              |
| **K-Prototypes**       | Directly handles mixed numerical and categorical features |

The workflow covers algorithm selection, data cleaning, feature selection, preprocessing, clustering and cluster profiling.

The project explores applications including:

* Energy-efficiency targeting
* Retrofit prioritisation
* Energy-cost and vulnerability mapping
* Real-estate decarbonisation
* Clean-energy service targeting
* Urban energy and infrastructure planning

A major focus is understanding **why algorithm choice matters when housing data contain different feature types and large numbers of observations**.

---

## 3. UKB Verisk Housing Archetype Analysis

### Mini-Batch K-Means for Large-Scale Property Segmentation

This notebook develops a scalable clustering pipeline using **Verisk UKB GIS residential building-stock data**.

The workflow includes:

1. Data ingestion
2. Feature engineering
3. Log transformations
4. Age mapping
5. Missing-value handling
6. Imputation
7. Standardisation
8. One-hot encoding
9. Mini-Batch K-Means optimisation
10. Cluster profiling
11. PCA visualisation
12. Silhouette analysis
13. Feature-centroid analysis
14. Categorical composition analysis

The number of clusters is evaluated across **K = 2–10**, with the final model using **K = 6** and a batch size of **8,192**.

The resulting groups are profiled as potential residential archetypes, with examples including **Victorian Terraces, Inter-War Semis and Outbuildings**.

This project demonstrates how unsupervised learning can move beyond simply producing cluster labels towards **interpretable building-stock segmentation**.

---

## 4. EPC + PCA: Energy Demand, Carbon & Housing Characteristics

### Principal Component Analysis for Energy-Performance Archetypes

This project uses PCA to investigate the underlying structure of EPC housing data across physical, energy and construction characteristics.

One pipeline uses six standardised variables:

* Floor area
* CO₂ emissions
* Energy consumption
* Current energy efficiency
* Energy rating
* Construction year

The resulting components provide interpretable latent dimensions.

The first three components explain approximately **93.87% of the variance**:

* **PC1 — Energy Inefficiency & Carbon Intensity:** 61.89%
* **PC2 — Property Scale & Unit Intensity:** 21.14%
* **PC3 — Construction Vintage:** 10.84%

A second pipeline incorporates categorical variables including:

* Tenure
* Property type
* Built form

After one-hot encoding and standardisation, PCA is used to create a lower-dimensional representation of the housing stock.

The analysis demonstrates how PCA can help transform a large collection of correlated housing and energy variables into a smaller set of interpretable dimensions suitable for **visualisation, segmentation and downstream clustering**.

---

## 5. EPC + PCA: Archetypes vs Energy Cost

This project explores relationships between **property characteristics, energy consumption and estimated energy costs** using PCA.

The analysis identifies two particularly useful dimensions:

* **PC1 — Overall property size and energy-cost scale**
* **PC2 — Energy consumption / hot-water use relative to property size**

Together, PC1 and PC2 explain approximately **56% of the variance**, while six components are required to reach approximately 80%.

An important learning point from this project is that real-world datasets do not necessarily produce clean, low-dimensional structures. The PCA results demonstrate the difference between analysing independently measured housing variables and working with highly correlated or formula-derived variables.

The notebook therefore focuses not only on PCA implementation but also on **interpreting what the components actually represent and recognising when dimensionality reduction has limited explanatory power**.

---

## 6. Household Fuel Poverty Clustering

### Mixed-Data Clustering with K-Prototypes and Gower Distance

This notebook explores whether unsupervised learning can identify different **household energy-poverty profiles**.

Unlike the property-focused projects, this analysis combines:

* Socio-economic characteristics
* Household characteristics
* Energy costs
* Energy efficiency
* Housing characteristics
* Vulnerability indicators
* Heating-fuel information

The project uses:

* **K-Prototypes** for mixed numerical and categorical data
* **Gower distance** for mixed-data similarity
* Elbow analysis
* Silhouette evaluation
* t-SNE visualisation
* Cluster profiling

Three illustrative profiles emerge from the synthetic dataset:

* **High Vulnerability / Structurally Inefficient**
* **Low Vulnerability / Resilient**
* **Off-Grid / Specific Fuel Risk**

Importantly, this notebook uses **synthetic data**. The resulting clusters are therefore a methodological demonstration rather than evidence about actual households.

This project highlights the potential of unsupervised learning to examine fuel poverty as a **multidimensional problem rather than simply an income-threshold classification task**.

---

## 7. South Yorkshire Residential Building Stock

### K-Means Clustering of Property Characteristics

These notebooks provide a more focused introduction to K-Means clustering using South Yorkshire residential building-stock data.

Two feature sets are explored.

### Residential building characteristics

* Property age
* Floor count
* Gross area

### Alternative housing-stock representation

* Property age
* Premise type
* Gross area

The workflow includes:

**Prepare → Inspect → Normalise → Evaluate K → Cluster → Profile → Visualise → Validate**

Different values of `K` are evaluated using:

* Inertia
* Silhouette Score
* Calinski-Harabasz Score
* Davies-Bouldin Score

Representative sampling is used for computationally expensive validation on the large datasets.

The final structured workflows use **K = 2**, demonstrating how clustering metrics can support—but should not automatically determine—model selection.

---

# Key Highlights

### 🔹 Working with Realistic Housing Data

The projects move beyond toy datasets and explore large, messy datasets representing **real housing and building-stock characteristics**, including EPC and Verisk UKB GIS data.

### 🔹 Scalability Matters

With datasets containing hundreds of thousands of properties, algorithm selection becomes a practical engineering problem.

The repository demonstrates why conventional HAC can become computationally impractical and explores **Mini-Batch K-Means and BIRCH** as scalable alternatives.

### 🔹 Mixed Data Requires Careful Treatment

Housing datasets frequently combine numerical and categorical variables.

The projects compare approaches such as:

* One-hot encoding + K-Means
* K-Prototypes
* ColumnTransformer pipelines
* Gower distance

This highlights the importance of matching the clustering method to the structure of the data.

### 🔹 PCA as an Interpretability Tool

PCA is used not simply as a dimensionality-reduction technique, but as a way to understand the major dimensions underlying housing datasets.

The analysis identifies latent patterns around:

**energy inefficiency → carbon emissions → property scale → energy intensity → construction vintage**

### 🔹 Evaluation Beyond a Single Metric

The notebooks use multiple evaluation techniques rather than relying solely on one clustering score.

These include:

* Elbow analysis
* Silhouette Score
* Calinski-Harabasz
* Davies-Bouldin
* PCA projections
* Cluster-size distributions
* Feature profiles
* Categorical composition

### 🔹 From Clusters to Archetypes

A recurring theme throughout the repository is the distinction between a **mathematical cluster** and a **meaningful housing archetype**.

The projects therefore investigate cluster characteristics and visualise their composition rather than treating cluster labels as self-explanatory.

### 🔹 Understanding Limitations

The notebooks explicitly examine limitations including:

* Sensitivity to outliers
* Assumptions of K-Means about cluster geometry
* High-dimensional effects from one-hot encoding
* Missing-value and imputation bias
* Limitations of PCA for categorical variables
* Computational cost of hierarchical clustering
* Potential overlap between housing archetypes
* The difference between theoretical EPC performance and actual household energy behaviour
* The limitations of synthetic data for fuel-poverty analysis

### 🔹 Technical Learning Over Benchmark Chasing

The repository is intentionally focused on **learning, experimentation and methodological understanding**.

The goal is not to claim that one clustering algorithm is universally superior. Instead, the projects explore the question:

> **Which unsupervised learning approach is appropriate for a particular housing problem, dataset structure and scale?**

---

## Overall Learning Journey

The projects collectively explore an end-to-end unsupervised machine learning workflow:

```text
Housing / Energy Data
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Data Cleaning & Validation
        │
        ▼
Feature Engineering
        │
        ├───────────────┐
        ▼               ▼
Numerical Data      Mixed Data
        │               │
        ▼               ▼
Scaling / PCA      K-Prototypes /
        │           Gower Distance
        ▼               │
K-Means /              ▼
Mini-Batch K-Means   Clustering
        │
        ├───────────────┐
        ▼               ▼
      BIRCH          BIRCH + HAC
        │               │
        └───────┬───────┘
                ▼
       Cluster Evaluation
                │
                ▼
        Cluster Profiling
                │
                ▼
     Housing Archetype / Energy
          Pattern Analysis
```

The overall learning progression is from **basic K-Means experimentation → structured preprocessing → scalable clustering → mixed-data clustering → PCA-based interpretation → domain-oriented housing and energy analysis**.

---

## Important Caveat

These notebooks are primarily **technical learning and demonstration projects**.

The resulting clusters should not automatically be interpreted as validated housing archetypes, retrofit categories or fuel-poverty classifications. Real-world deployment would require additional validation against domain knowledge, observed outcomes and independent datasets.

In particular, the fuel-poverty analysis uses synthetic data, while EPC data represent theoretical building performance and may not capture actual household energy consumption or occupant behaviour.

The repository therefore treats unsupervised learning as a tool for **pattern discovery and hypothesis generation**, rather than as a substitute for domain-informed validation.

---

## What I Learned

Through these projects, I explored not only how to implement clustering and PCA, but also the practical questions surrounding unsupervised machine learning:

* How should features be prepared before clustering?
* When is scaling necessary?
* How should categorical housing variables be represented?
* When does K-Means become computationally impractical?
* When should Mini-Batch K-Means or BIRCH be preferred?
* How can mixed-type data be clustered?
* How should the number of clusters be selected?
* How can clusters be interpreted as meaningful housing profiles?
* What are the limitations of internal clustering metrics?
* When does PCA provide useful insight—and when does it oversimplify the data?
* How can machine-learning outputs be connected to real-world housing and energy applications?

The repository is ultimately a record of that learning process: **from implementing algorithms to critically evaluating when, why and how they should be used.**
