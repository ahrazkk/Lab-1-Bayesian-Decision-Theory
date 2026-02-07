# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris



# getting the data, we only need the first 100 rows
# because thats where class 0 and 1 are
def get_iris_data():
    iris = load_iris()
    X = iris.data[:100]
    y = iris.target[:100]
    return X, y, iris.feature_names




# standard gaussian pdf formula
def get_gaussian_prob(x, mu, sigma):
    const = 1.0 / (np.sqrt(2 * np.pi) * sigma)
    exponent = -((x - mu) ** 2) / (2 * sigma ** 2)
    return const * np.exp(exponent)




# calculating the posteriors using bayes rule
def get_posteriors(x, mu0, std0, mu1, std1):
    # priors are 0.5 for both so they cancel out but
    # keeping them here just in case
    p_w0 = 0.5
    p_w1 = 0.5

    # get likelihoods
    like0 = get_gaussian_prob(x, mu0, std0)
    like1 = get_gaussian_prob(x, mu1, std1)

    # total prob / evidence
    evidence = (like0 * p_w0) + (like1 * p_w1)

    # posterior probs
    post0 = (like0 * p_w0) / evidence
    post1 = (like1 * p_w1) / evidence

    return post0, post1




# does the classification and figures out risk
def classify_sample(x, mu0, std0, mu1, std1, penalty_val=1.0):
    post0, post1 = get_posteriors(x, mu0, std0, mu1, std1)

    # discriminant function g(x)
    g_x = post1 - post0

    # calculating risk here
    # penalty only applies if we pick class 1 when it was actually class 0
    risk_decide_1 = penalty_val * post0

    # standard risk for the other case
    risk_decide_0 = 1.0 * post1

    # pick the lower risk option
    if risk_decide_1 < risk_decide_0:
        prediction = 1
    else:
        prediction = 0

    return post0, post1, g_x, risk_decide_0, risk_decide_1, prediction




# main function to run analysis for a specific feature
def run_analysis(feature_idx, feature_name):
    X, y, _ = get_iris_data()

    # grab the column we want
    data = X[:, feature_idx]

    # split into the two classes
    class0 = data[y == 0] # setosa
    class1 = data[y == 1] # versicolour

    # get stats
    mu0, std0 = np.mean(class0), np.std(class0)
    mu1, std1 = np.mean(class1), np.std(class1)

    print(f"\n{'='*50}")
    print(f"Results for: {feature_name}")
    print(f"{'='*50}")
    print(f"Class 0 (Setosa)      -> Mean: {mu0:.3f}, Std: {std0:.3f}")
    print(f"Class 1 (Versicolour)-> Mean: {mu1:.3f}, Std: {std1:.3f}")

    test_vals = [3.3, 4.4, 1.2, 5.0, 5.7, 6.3, 1.5]

    # --- table 1: probabilities with normal penalty ---
    print(f"\n>>> Table 1: Probabilities (Standard Penalty=1.0)")
    print(f"{'Val':<6} | {'P(w0|x)':<10} | {'P(w1|x)':<10} | {'g(x)':<10} | {'Class'}")
    print("-" * 65)

    for val in test_vals:
        p0, p1, gx, r0, r1, pred = classify_sample(val, mu0, std0, mu1, std1, penalty_val=1.0)
        print(f"{val:<6} | {p0:<10.4f} | {p1:<10.4f} | {gx:<10.4f} | {pred}")

    # --- table 2: risks with high penalty ---
    print(f"\n>>> Table 2: Risks (High Penalty=10.0 for Misclassifying Class 0)")
    print(f"{'Val':<6} | {'Risk(Pick 0)':<12} | {'Risk(Pick 1)':<12} | {'Decision'}")
    print("-" * 65)

    for val in test_vals:
        _, _, _, r0, r1, pred = classify_sample(val, mu0, std0, mu1, std1, penalty_val=10.0)
        print(f"{val:<6} | {r0:<12.4f} | {r1:<12.4f} | {pred}")


    # calculating thresholds
    # scanning a range to find where decision flips
    x_range = np.linspace(min(data)-1, max(data)+1, 2000)

    # 1.0 is normal, 10.0 is high, 5.0 is medium
    penalties_to_test = [1.0, 5.0, 10.0]
    optimal_thresholds = []

    for pen in penalties_to_test:
        best_diff = 99999
        best_x = 0

        # simple linear search for the crossing point
        for x in x_range:
            _, _, _, r0, r1, _ = classify_sample(x, mu0, std0, mu1, std1, penalty_val=pen)

            # find where risks are almost equal
            diff = abs(r0 - r1)
            if diff < best_diff:
                best_diff = diff
                best_x = x

        optimal_thresholds.append(best_x)

    print(f"\n>>> Calculated Thresholds:")
    print(f"Normal (1.0): {optimal_thresholds[0]:.4f}")
    print(f"Medium (5.0): {optimal_thresholds[1]:.4f}")
    print(f"High  (10.0): {optimal_thresholds[2]:.4f}")


    # plotting
    plt.figure(figsize=(10, 6))

    # densities
    plt.plot(x_range, get_gaussian_prob(x_range, mu0, std0), 'b-', label='Class 0 (Setosa)')
    plt.plot(x_range, get_gaussian_prob(x_range, mu1, std1), 'r-', label='Class 1 (Versicolour)')

    # threshold lines
    plt.axvline(optimal_thresholds[0], color='k', linestyle='--', label='Threshold (Risk=1)')
    plt.axvline(optimal_thresholds[1], color='orange', linestyle='--', label='Threshold (Risk=5)')
    plt.axvline(optimal_thresholds[2], color='purple', linestyle='--', label='Threshold (Risk=10)')

    plt.title(f'Bayesian Decision Boundaries: {feature_name}')
    plt.xlabel(f'{feature_name} (cm)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()




# running for sepal length (feature 0)
run_analysis(0, "Sepal Length")

# running for sepal width (feature 1)
run_analysis(1, "Sepal Width")