import pandas as pd
from matplotlib import pyplot as plt

# Load csv file
file_path = "RMS_acc_match_gp2.csv"
data = pd.read_csv(file_path)

# Read the two columns of the data as x, y
x = data.iloc[:, 0]
y = data.iloc[:, 1]

# Plot the scatter
plt.figure(figsize= (10, 10))
plt.title('RMS(SPL) vs velocity')
plt.scatter(x, y, color='r', marker = 'o', label='RMS velocity')
plt.xlabel('RMS/dB')
plt.ylabel('Velocity/m/s')
plt.legend()
plt.grid()
plt.show()