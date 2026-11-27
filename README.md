**Genome Analysis and AMR Prediction**



**Overview**



This repository contains the computational workflows, analytical scripts, and ML procedures used in the manuscript investigating bacterial genomic diversity, pangenome composition, antimicrobial resistance determinants (ARGs), and ML-based antimicrobial resistance (AMR) prediction.



The project integrates genome quality assessment, genome annotation, pan-genome analysis, resistome characterization, exploratory genomic analysis, feature engineering, ML, external validation, and model interpretability.



Two computational environments were used:



**a) Bash-based bioinformatics environment**: Windows Subsystem for Linux (WSL), Conda, and command-line bioinformatics tools.

**b)Python-based analytical environment**: Jupyter Notebook and Python packages.



**Repository Structure**



**01. Bash-based Bioinformatics Pipeline**

This directory contains Bash scripts for genome processing and antimicrobial resistance characterization.



01.1. QUAST: Assessment of genome assembly quality, including assembly statistics and quality metrics.



01.2.Prokka: Structural and functional annotation of bacterial genome assemblies.



01.3.Panaroo: Identification and characterization of core, accessory, and unique gene content across bacterial genomes.



01.4.RGI-CARD: Identification of antimicrobial resistance genes using Resistance Gene Identifier (RGI) and the Comprehensive Antibiotic Resistance Database (CARD).



01.5. RGI-CARD: Processing, interpretation, and organization of RGI-CARD outputs for downstream resistome characterization and exploratory genomic analyses.



**02. Pangenome, Resistome and Exploratory Genomic Characterization**



Contains Jupyter notebooks and Python scripts for:

a. Pangenome composition and Heaps' law analysis.

b. Distribution of dominant ARGs based on RGI-CARD analysis.

c. Clustered heatmap visualization of ARG profiles with MRSA versus MSSA annotation.

d. Genomic diversity and distribution of ARGs.



**03. Feature Engineering and Machine Learning Workflow**



Contains Jupyter notebooks and Python scripts for:

a. Combining ARG and pangenome features.

b. Feature engineering and preparation using NCBI/BV-BRC feature intersection.

c. ML model development.

d. External validation.

e. SHAP-based model interpretability.



**Computational Environments**

A. Bash and Bioinformatics Environment

The Bash workflows were executed using Windows Subsystem for Linux (WSL), with Conda environments for managing bioinformatics software and dependencies.



The bioinformatics tools include:

i. QUAST

ii. Prokka

iii. Panaroo

iv. RGI

v. CARD (Comprehensive Antibiotic Resistance Database)



B. Python and Jupyter Notebook Environment

The exploratory genomic analyses, feature engineering, machine-learning workflows, external validation, and SHAP interpretability were implemented using Python in Jupyter Notebook.



Python dependencies are listed in:



i. requirements.txt

ii. environment.yml



###### **Workflow Overview**



The analysis is organized into three major stages:



**Stage 1**: Genome processing and annotation



Genome quality assessment, annotation, pangenome reconstruction, and antimicrobial resistance gene identification.



**Stage 2**: Exploratory genomic characterization



Pangenome composition, Heaps' law, ARG distribution, clustered heatmap analysis, and genomic diversity.



**Stage 3**: ML and interpretation



ARG and pangenome feature integration, feature engineering, model development, external validation, and SHAP-based interpretation.



**Installation and Environment Setup**

Bash environment (WSL)

a. Install WSL and an appropriate Linux distribution, such as Ubuntu.

b. Install Miniconda or Anaconda within WSL.

c. Create and activate the required Conda environments for the bioinformatics tools.

d. Install the relevant software and reference databases.

e. Execute the Bash scripts in their respective directories.



**Python environment**



Create the Python environment using:



bash

conda env create -f environment.yml

conda activate genome-amr-ml

Alternatively, install the Python dependencies using:



bash

pip install -r requirements.txt

Launch Jupyter Notebook:



bash

jupyter notebook



**Data Availability**



Input genome assemblies, annotation files, pangenome outputs, RGI-CARD results, and machine-learning datasets should be obtained from their respective public repositories or associated data repositories, where applicable.



Large input datasets and third-party reference databases are not necessarily distributed with this repository.



Refer to the manuscript and associated data availability statement for dataset accession numbers and sources.



**Reproducibility**



The scripts and notebooks are organized according to their corresponding analytical stages.



For reproducibility, users should consider:



i. Using the documented software and package versions.

ii. Using compatible versions of reference databases.

iii. Maintaining the original analytical parameters.

iv. Applying the documented preprocessing and feature-selection procedures.

v. Recording random seeds and model evaluation settings.

vi. Keeping training and external validation datasets separate.



Citation

If you use this repository, please cite the associated manuscript and repository. See CITATION.cff for citation metadata.



License

This repository is distributed under the MIT License unless otherwise specified. See the LICENSE file for details.

