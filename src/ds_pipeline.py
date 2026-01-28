#Read in the california dataset
from sklearn.datasets import fetch_california_housing
import pandas as pd
import matplotlib.pyplot as plt

housing = fetch_california_housing(as_frame=True)
df = housing.frame

print(df.head())
print(df.shape)

plt.figure(figsize=(6, 4))
df['MedHouseVal'].plot.box()
plt.title('Box plot of Median House Value')
plt.ylabel('Median House Value')

plt.tight_layout()
plt.savefig('../figures/med_house_val_boxplot.png')
plt.close()
