import matplotlib.pyplot as plt
import numpy as np
# Load the data
plt.rcParams['text.usetex'] = True #for nice plots

vec = np.loadtxt("timing_results_vec.txt", delimiter=',')
nvec = np.loadtxt("timing_results.txt", delimiter=',')

# Split the data into x (lengths) and y (times) values
x_vec, y_vec = vec[:, 1], vec[:, 0]
x_nvec, y_nvec = nvec[:, 1], nvec[:, 0]

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(x_vec, y_vec, '-o', label='vectorized', color = "red")
plt.plot(x_nvec, y_nvec,'-o', label='non vectorized', color = "blue")

# Add labels and title
plt.xlabel('Number of elements')
plt.ylabel('Time taken ($s$)')
plt.title('Timing results of ElementIntegrationLHS')
plt.legend()

# Show the plot

plt.show()