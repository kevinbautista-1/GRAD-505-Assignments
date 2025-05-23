from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data = { "weight": [4.17, 5.58, 5.18, 6.11, 4.50, 4.61, 5.17, 4.53, 5.33, 5.14, 4.81, 4.17, 4.41, 3.59, 5.87, 3.83, 6.03, 4.89, 4.32, 4.69, 6.31, 5.12, 5.54, 5.50, 5.37, 5.29, 4.92, 6.15, 5.80, 5.26], "group": ["ctrl"] * 10 + ["trt1"] * 10 + ["trt2"] * 10}
PlantGrowth = pd.DataFrame(data)
iris = datasets.load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_target = pd.DataFrame(data=iris.target, columns=['Species/target'])

# Q1a. Histogram of Sepal Width 
plt.hist(iris_df['sepal width (cm)'])
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of Sepal Width')
plt.show()

#Q1c. Mean and Median of Sepal Width
sepal_width = iris_df['sepal width (cm)']
sepal_width_mean = sepal_width.mean()
sepal_width_std = sepal_width.median()
print(f"Mean of Sepal Width: {sepal_width_mean}")
print(f"Median of Sepal Width: {sepal_width_std}")

#Q1d. 73 percentile of Sepal Width
sepal_width_73 = sepal_width.quantile(0.73)
print(f"73rd Percentile of Sepal Width: {sepal_width_73}")

#Q1e. SNS Pairplot of Iris Dataset
iris_df_columns = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
for i in range(len(iris_df_columns)):
   for j in range(i+1, len(iris_df_columns)):
       plt.scatter(data=iris_df, x=iris_df_columns[i], y=iris_df_columns[j])
       plt.title(f'Scatterplot of {iris_df_columns[i]} vs {iris_df_columns[j]}')
       plt.xlabel(iris_df_columns[i])
       plt.ylabel(iris_df_columns[j])
       plt.show()


#Q2a:
# Making bins for histogram
bins = np.arange(3.3, PlantGrowth['weight'].max() + 0.3, 0.3)
plt.hist(PlantGrowth['weight'], bins)
plt.xlabel('Weight')
plt.ylabel('Frequency')
plt.title('Histogram of Plant Growth Weights')
plt.show()

#Q2b: Making boxplots for Plant Growth for each group
sns.boxplot(x='group', y='weight', data=PlantGrowth)
plt.title('Boxplot of Plant Growth Weights by Group')
plt.xlabel('Group')
plt.ylabel('Weight')
plt.show()

#Q2c: Finding minimum and maximum weight for each group using groupby
group_min_max = PlantGrowth.groupby('group')['weight'].agg(['min', 'max', 'count'])
print(group_min_max)

#d.	Find the exact percentage of the "trt1" weights that are below the minimum "trt2" weight.
trt2_min = PlantGrowth[PlantGrowth['group'] == 'trt2']['weight'].min()
trt1_below_trt2_min = PlantGrowth[(PlantGrowth['group'] == 'trt1') & (PlantGrowth['weight'] < trt2_min)]

# Calculate the percentage of 'trt1' weights below the minimum 'trt2' weight
trt1_below_trt2_min_percent = (len(trt1_below_trt2_min) / len(PlantGrowth[PlantGrowth['group'] == 'trt1'])) * 100
print(f"Percentage of 'trt1' weights below the minimum 'trt2' weight: {trt1_below_trt2_min_percent:.2f}%")


#e. Barplot of plants with weight above 5.5 
colors = ['yellow', 'red', 'blue'] 
filtered_plants = PlantGrowth[PlantGrowth['weight'] > 5.5]
freq_table = filtered_plants.groupby('group').size()
plt.bar(freq_table.index, freq_table.values, color=colors)
plt.title('Barplot of Plants with Weight Above 5.5')
plt.xlabel('Group')
plt.ylabel('Frequency')
plt.show()