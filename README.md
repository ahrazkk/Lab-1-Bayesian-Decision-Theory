# Lab 1: Bayesian Decision Theory

## Overview
This laboratory study implements a **Bayesian classifier for binary classification** of iris flowers. Using the famous Iris dataset, the project classifies flowers into two distinct classes:
- **Class 0**: Iris Setosa
- **Class 1**: Iris Versicolor

The implementation demonstrates core concepts of Bayesian decision theory, including posterior probability calculation, risk-based classification, and optimal decision boundary determination.

## Project Goals
- Develop a Bayesian classifier using Gaussian probability distributions
- Implement posterior probability calculations using Bayes' theorem
- Analyze decision boundaries under different risk/penalty scenarios
- Visualize probability distributions and decision thresholds
- Evaluate classification performance on sepal length and sepal width features

## Features
✅ **Gaussian Probability Density Function** - Calculates likelihood for continuous features  
✅ **Bayesian Posterior Calculation** - Implements Bayes' rule for posterior probabilities  
✅ **Risk-Based Classification** - Incorporates misclassification penalties into decision-making  
✅ **Threshold Optimization** - Automatically determines optimal decision boundaries  
✅ **Multi-Penalty Analysis** - Tests multiple penalty values (1.0, 5.0, 10.0)  
✅ **Visualization** - Generates plots showing probability distributions and decision boundaries  

## Implementation Details

### Key Functions

#### `get_iris_data()`
Loads the Iris dataset and extracts the first 100 samples containing only Setosa and Versicolor classes.

#### `get_gaussian_prob(x, mu, sigma)`
Computes the Gaussian probability density function:
P(x) = (1 / √(2πσ²)) * exp(-(x - μ)² / 2σ²)

Code

#### `get_posteriors(x, mu0, std0, mu1, std1)`
Calculates posterior probabilities using Bayes' theorem:
P(wᵢ|x) = P(x|wᵢ) * P(wᵢ) / P(x)

Code

#### `classify_sample(x, mu0, std0, mu1, std1, penalty_val)`
Performs risk-based classification:
- Calculates posterior probabilities
- Computes discriminant function: g(x) = P(w₁|x) - P(w₀|x)
- Evaluates risks for both decision options
- Returns the class with minimum risk

#### `run_analysis(feature_idx, feature_name)`
Main analysis pipeline that:
1. Extracts feature data and computes class statistics
2. Generates probability tables for test values
3. Compares classification under different penalty scenarios
4. Calculates optimal decision thresholds
5. Visualizes probability distributions and boundaries

## Usage

### Requirements
```bash
pip install numpy matplotlib scikit-learn
Running the Analysis
Python
python "ele888_lab1 (1).py"
The script automatically runs analysis on:

Sepal Length (Feature 0)
Sepal Width (Feature 1)
Output
For each feature, the program displays:

Class Statistics - Mean and standard deviation for each class
Table 1: Probabilities - Posterior probabilities and discriminant function values
Table 2: Risks - Risk values under high penalty (10.0) scenario
Calculated Thresholds - Optimal decision boundaries for different penalty values
Visualization - Plot showing Gaussian distributions and decision thresholds
Example Output
Code
==================================================
Results for: Sepal Length
==================================================
Class 0 (Setosa)      -> Mean: 5.006, Std: 0.352
Class 1 (Versicolour)-> Mean: 5.936, Std: 0.516

>>> Table 1: Probabilities (Standard Penalty=1.0)
Val    | P(w0|x)    | P(w1|x)    | g(x)       | Class
-----------------------------------------------------------------
3.3    | 0.9999     | 0.0001     | -0.9998    | 0
4.4    | 0.9325     | 0.0675     | -0.8650    | 0
...

>>> Calculated Thresholds:
Normal (1.0): 5.471
Medium (5.0): 5.258
High  (10.0): 5.144
Methodology
Bayesian Decision Theory
The classifier uses minimum risk decision rule:

Risk(Decide Class 1) = penalty × P(w₀|x)
Risk(Decide Class 0) = 1.0 × P(w₁|x)
Classification decision: Choose class with minimum expected risk

Penalty Analysis
The project tests three penalty scenarios for misclassifying Setosa as Versicolor:

Normal (1.0): Equal cost for both types of misclassification
Medium (5.0): Moderate penalty for misclassifying Class 0
High (10.0): Severe penalty for misclassifying Class 0
Higher penalties shift the decision boundary to be more conservative toward Class 0.

Visualization
The generated plots show:

Blue curve: Class 0 (Setosa) probability distribution
Red curve: Class 1 (Versicolor) probability distribution
Black dashed line: Threshold with penalty = 1.0
Orange dashed line: Threshold with penalty = 5.0
Purple dashed line: Threshold with penalty = 10.0
Key Insights
As penalty increases, the decision threshold shifts toward Class 1 territory
Features with better class separation yield more confident predictions
Risk-based classification allows domain-specific cost considerations
License
This project is part of academic coursework.
