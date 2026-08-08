
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
  <a href="https://pytorch.org/docs/stable/index.html">
    <img src="https://img.shields.io/badge/PyTorch-v2.2.0-EE4C2C?style=flat&logo=pytorch&logoColor=white" alt="PyTorch">
  </a>
</p>


# Machine Learning & AI — Unsupervised Learning Exercises

A collection of practical **machine learning and artificial intelligence projects** developed as part of my learning through the **Imperial College London Professional Certificate in Machine Learning / AI**.

The notebooks demonstrate how different **unsupervised learning techniques** can be applied to housing, residential building stock, and household energy-poverty problems.

The primary purpose of this repository is **technical learning and demonstration**: exploring machine learning concepts, understanding their practical applications, comparing algorithms, and evaluating their limitations using realistic datasets and controlled synthetic examples.

---

## About This Repository

The notebooks cover a range of unsupervised machine learning techniques, including:

- K-Means Clustering
- Mini-Batch K-Means
- K-Prototypes
- BIRCH
- Hierarchical Agglomerative Clustering (HAC)
- Gower Distance
- Silhouette Score
- Calinski-Harabasz Index
- Davies-Bouldin Index
- Elbow Method
- PCA and clustering visualisation
- Feature scaling and normalisation
- Categorical encoding
- `ColumnTransformer` and `Pipeline`
- Cluster profiling and interpretation

The projects use two broad types of datasets:

1. **Real-world datasets** imported from CSV files, particularly residential building-stock data.
2. **Synthetic datasets** generated within notebooks using controlled randomisation, where this provides a useful environment for demonstrating machine learning techniques.

Where synthetic data is used, the purpose is to demonstrate the modelling technique rather than make claims about real-world household or housing populations.

---

# Projects

## 1. Unsupervised K-Means Clustering of South Yorkshire Residential Building Stock

**Technique:** K-Means Clustering

This notebook explores the use of K-Means clustering to identify natural groupings within residential building-stock data for **South Yorkshire**.

The analysis considers characteristics such as:

- Property age
- Number of floors
- Building area
- Gross area
- Property/building type

### Key Learning Objectives

The notebook demonstrates an end-to-end K-Means workflow:

1. Data preparation
2. Missing-value handling
3. Feature selection
4. Data type conversion
5. Feature normalisation using `MinMaxScaler`
6. Exploratory visualisation
7. Testing different values of `K`
8. Elbow analysis
9. Silhouette analysis
10. Cluster profiling
11. 3D visualisation
12. Cluster validity assessment
13. Comparison of raw and normalised clustering results

### Potential Applications

The approach illustrates how unsupervised learning could be used to segment residential building stock into groups with similar physical characteristics.

Potential applications include:

- Building-stock segmentation
- Retrofit planning
- Energy-efficiency analysis
- Housing-stock analysis
- Identification of groups of properties with similar characteristics

### Limitations

The notebook is primarily an exploratory learning exercise. Some exploratory cells contain redundant or legacy code, and the analysis should not be interpreted as a production-ready housing classification system.

In particular, clustering results require careful interpretation and validation before being translated into real-world housing archetypes or policy decisions.

---

## 2. Scalable Unsupervised Clustering for Large Housing Datasets

**Techniques:** Mini-Batch K-Means, BIRCH, Hierarchical Agglomerative Clustering

This notebook investigates the practical challenge of applying clustering algorithms to a large residential housing dataset containing approximately **614,000 records**.

The project focuses particularly on the scalability limitations of conventional **Hierarchical Agglomerative Clustering (HAC)**.

### Key Learning Objectives

The notebook demonstrates:

- Exploratory data analysis
- Data preprocessing
- Limitations of hierarchical clustering
- Mini-Batch K-Means
- BIRCH micro-clustering
- BIRCH + HAC hybrid clustering
- Cluster evaluation
- Cluster profiling
- PCA visualisation
- Categorical composition analysis
- Comparison of clustering approaches

### Why Scalability Matters

Traditional HAC requires extensive pairwise distance calculations. At the scale of this dataset, directly applying HAC becomes computationally impractical.

The notebook therefore explores two alternatives:

#### Mini-Batch K-Means

A scalable variation of K-Means that works with small batches of observations rather than processing the entire dataset during every iteration.

#### BIRCH + HAC

BIRCH first compresses the large dataset into a much smaller set of representative sub-clusters. HAC is then applied to these representative centroids rather than to every original observation.

This demonstrates an important machine learning principle:

> An algorithm that works well on a small dataset may not necessarily be appropriate for a large dataset.

### Potential Applications

The techniques could potentially support:

- Large-scale housing-stock segmentation
- Building archetype exploration
- Residential property analysis
- Scalable exploratory data analysis

However, the resulting clusters should not automatically be interpreted as meaningful housing archetypes without further domain validation.

---

## 3. Scalable Housing Clustering — ColumnTransformer, Mini-Batch K-Means & BIRCH-HAC

**Techniques:** `ColumnTransformer`, `Pipeline`, Mini-Batch K-Means, BIRCH, HAC

This notebook develops the scalable housing-clustering workflow further by incorporating a more structured preprocessing pipeline.

The housing dataset contains both numerical and categorical variables, requiring different preprocessing approaches.

### Data Processing

The workflow uses a `ColumnTransformer` / `Pipeline` approach incorporating techniques such as:

- `SimpleImputer`
- `StandardScaler`
- `OneHotEncoder`

This converts mixed housing characteristics into a form suitable for machine learning algorithms.

### Clustering Approaches

#### Mini-Batch K-Means

Mini-Batch K-Means is used as a scalable clustering baseline, with different values of `K` evaluated using:

- Silhouette Score
- Calinski-Harabasz Index
- Davies-Bouldin Index

Because calculating these metrics across hundreds of thousands of observations can itself be computationally expensive, representative sampling is used for evaluation.

#### BIRCH + HAC

A hybrid approach is also explored:

```text
Large Housing Dataset
        │
        ▼
Data Preprocessing
        │
        ▼
BIRCH
        │
        ▼
Representative Sub-clusters
        │
        ▼
Hierarchical Agglomerative Clustering
        │
        ▼
Final Cluster Assignment
