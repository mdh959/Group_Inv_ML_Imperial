import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


technique_A = np.array([])
technique_B = np.array([])

# Perform paired t-test
t_statistic, p_value = stats.ttest_rel(technique_A, technique_B)

# Print the results
print(f"T-statistic: {t_statistic:.4f}")
print(f"P-value: {p_value:.4f}")

# Interpret the p-value
alpha = 0.05
if p_value < alpha:
    print("Reject null hypothesis: There is a significant difference between the techniques.")
else:
    print("Fail to reject null hypothesis: There is no significant difference between the techniques.")

# Plotting boxplots to visualize the accuracies
plt.figure(figsize=(8, 6))
plt.boxplot([technique_A, technique_B], labels=['Technique A', 'Technique B'])
plt.title('Accuracies Comparison')
plt.ylabel('Accuracy (%)')
plt.grid(True)
plt.show()
