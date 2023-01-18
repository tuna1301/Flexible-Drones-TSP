# Flexible-Drones-TSP
The FDTSP algorithm was developed to solve the work of a truck and multiple drones in tandem.
The folder ‘FDTSP codes” contains 3 sub-folders, i.e., Genetic Algorithm, OR-Tools, and Simulated Annealing. Each folder contains 5 files *.py, i.e., 1drone to 5drones.
Packages:
Or-Tools (install at: https://developers.google.com/optimization/install/python)
Bisecting K-means (The FDTSP algorithm amended the original code from: https://medium.com/@afrizalfir/bisecting-kmeans-clustering-5bc17603b8a2 )
Simulated Annealing (The FDTSP algorithm amended the original code from: https://github.com/chncyhn/simulated-annealing-tsp/blob/master/LICENSE Copyright (c) 2016 Cihan Ceyhan)
Parameter settings Readers could change parameter values corresponding to desired instances in the parameter block. Note: “iter_proposed_model” refers to the number of iterations for the entire FDTSP algorithm. We set it equal to 1 to reduce the computational time process. Readers could set it as large enough for statistical analysis purposes.
The results will be saved in a separate csv file. Please check the file ‘XDRONES-Methods.csv’ in the folder which contains the files. The result file contains several columns, which tell the objective function (tour time of a truck and drones in tandem) and other indicators (TIME TRUCK WAITS DRONES, NUMBER OF CUSTOMERS SERVED BY DRONE, etc..), to provide information for the analysis
