import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Data Processing
df = pd.read_csv('Project_1_Data.csv')
print(df)

# Step 2: Data Visualization
for i in range(1, 14):
    step = df.loc[df['Step'] == i]
    print(step)
    
    X = step.get("X")
    Y = step.get("Y")
    Z = step['Z'].values.reshape(-1, 1) 
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)
    
    plt.show()

X = df['X']
Y = df['Y']
Z = df['Z'] 

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(X, Y, Z)

