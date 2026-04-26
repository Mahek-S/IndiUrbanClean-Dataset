import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv("results/results.tsv", sep="\t")

# Set augmentation as index
df.set_index("augmentation", inplace=True)

# Transpose so:
# X-axis = corruption types
# Lines = augmentations
df_t = df.T

# Plot
plt.figure()
df_t.plot(marker='o')

plt.title("Augmentation Generalization Across Corruptions")
plt.xlabel("Corruption Type")
plt.ylabel("Accuracy")
plt.grid()

# Save plot
plt.savefig("results/lineplot.png")

# Show plot
plt.show()