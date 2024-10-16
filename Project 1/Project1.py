import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

# Step 1: Data Processing
df = pd.read_csv('Project_1_Data.csv')
print(df)

# Step 2: Data Visualization
# Individual Steps
for i in range(1, 14):
    step = df.loc[df['Step'] == i]
    print(step)
    
    X = step.get("X")
    Y = step.get("Y")
    Z = step.get("Z")
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(X, Y, Z)
    # Add same axis for each
    
    plt.show()

# Combined Steps
X = df['X']
Y = df['Y']
Z = df['Z'] 

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(X, Y, Z)

# Step 3: Correlation Analysis
correlation = df.corr()
print(correlation)

plt.figure()
sb.heatmap(correlation)

# Step 4: Classification Model Development/Engineering
coordinates = df[['X', 'Y', 'Z']]
step = df['Step']
print(coordinates, step)