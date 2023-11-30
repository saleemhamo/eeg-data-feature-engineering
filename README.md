# Feature Engineering for Predicting Central Neuropathic Pain

## Project Overview

Predicting Central Neuropathic Pain (CNP) in individuals with Spinal Cord Injury (SCI) is a challenging yet crucial task. This project aims to develop a feature engineering strategy using brain Electroencephalogram (EEG) data to predict the likelihood of patients developing CNP. The dataset consists of EEG recordings from 18 SCI patients, with 8 patients classified as 'negative' (did not develop CNP) and 10 as 'positive' (developed CNP within 6 months).

## Project Structure

The repository is organized into four main files, each focusing on a different aspect of feature engineering:

1. **[Feature_Selection_Filtering_Methods.ipynb](Feature_Selection_Filtering_Methods.ipynb)**: Explore and implement filtering methods for feature selection, such as GridSearchCV, SelectKBest, Recursive Feature Elimination (RFE), Principal Component Analysis (PCA), RandomForest, LASSO, and Support Vector Machine (SVM).

2. **[Feature_Selection_Embedding_Methods.ipynb](Feature_Selection_Embedding_Methods.ipynb)**: Investigate and apply embedding methods for feature selection, specifically using Support Vector Machine (SVM), k-Nearest Neighbors (KNN), and Random Forest classifiers.

3. **[Feature_Selection_Wrapper_Methods.ipynb](Feature_Selection_Wrapper_Methods.ipynb)**: Implement wrapper methods for feature selection, including Backward and Forward Feature Elimination for Random Forest and Support Vector Machine (SVM) classifiers.

4. **[Feature_Extraction_Methods.ipynb](Feature_Extraction_Methods.ipynb)**: Explore various feature extraction techniques, such as standardization with Principal Component Analysis (PCA), Singular Value Decomposition (SVD) with Randomized PCA, t-distributed Stochastic Neighbor Embedding (t-SNE), binarization, and one-hot encoding.

## Dataset

- **Participants**: 18 SCI patients
  - 8 'negative' (did not develop CNP)
  - 10 'positive' (developed CNP within 6 months)
  
- **EEG Data**:
  - 48 electrode EEG recordings at 250 Hz
  - Recorded with eyes closed (EC) and eyes opened (EO)
  - Segments of 5-second length with 10 repetitions per participant
  - Preprocessed data with signal denoising, normalization, temporal segmentation, and frequency band power estimation
  - 180 labeled data points (18 participants with 10 repetitions each) x 432 columns (9 features x 48 electrodes)

## Objective Measure

The evaluation is based on Leave-One-Subject-Out Cross-Validation, considering accuracy, sensitivity, and specificity as performance metrics.

## Usage

1. Open and run the Jupyter notebooks ([Feature_Selection_Filtering_Methods.ipynb](Feature_Selection_Filtering_Methods.ipynb), [Feature_Selection_Embedding_Methods.ipynb](Feature_Selection_Embedding_Methods.ipynb), [Feature_Selection_Wrapper_Methods.ipynb](Feature_Selection_Wrapper_Methods.ipynb), [Feature_Extraction_Methods.ipynb](Feature_Extraction_Methods.ipynb)) in a sequential order to understand and execute the feature engineering strategies.

2. Modify and experiment with the code to adapt it to specific needs or explore additional techniques.

3. Review the results, visualize the impact of each method, and choose the most suitable features for predicting CNP.

Feel free to reach out for any questions or further assistance. Good luck with your feature engineering efforts!
