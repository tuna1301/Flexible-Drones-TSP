import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from operator import add
import math, time
import operator
import csv
rnd = np.random
import time
start_time = time.time()
'PARAMETERS'
n = 60                                                                    # number of customers   
square = 20                                                               # square of area
speed_drone = 80                                                          # km/hour
endurance1 = 0.5                                                          # hour
endurance2 = 1                                                            # hour
epochs = 10                                                               # epochs for Bisecting K-mean Algorithm  
iter_proposed_model = 1

'FUNCTIONS'
'Bisecting K-means Module'
def convert_to_2d_array(points):
    points = np.array(points)
    if len(points.shape) == 1:
        points = np.expand_dims(points, -1)
    return points
def SSE(points):
    points = convert_to_2d_array(points)
    centroid = np.mean(points, 0)
    errors = np.linalg.norm(points - centroid, ord=2, axis=1)
    return np.sum(errors)
def kmeans(points, k=2, epochs=epochs, max_iter=100, verbose=False):
    points = convert_to_2d_array(points)
    assert len(points) >= k
    best_sse = np.inf
    last_sse = np.inf
    for ep in range(epochs):
        if ep == 0:
            random_idx = np.random.permutation(points.shape[0])
            centroids = points[random_idx[:k], :]
        for it in range(max_iter):
            # Cluster assignment
            clusters = [None] * k
            for p in points:
                index = np.argmin(np.linalg.norm(centroids-p, 2, 1))
                if clusters[index] is None:
                    clusters[index] = np.expand_dims(p, 0)
                else:
                    clusters[index] = np.vstack((clusters[index], p))
            centroids = [np.mean(c, 0) for c in clusters]
            sse = np.sum([SSE(c) for c in clusters])
            gain = last_sse - sse
            if verbose:
                print((f'Epoch: {ep:3d}, Iter: {it:4d}, '
                       f'SSE: {sse:12.4f}, Gain: {gain:12.4f}'))
            if sse < best_sse:
                best_clusters, best_sse = clusters, sse
            if np.isclose(gain, 0, atol=0.00001):
                break
            last_sse = sse
    return best_clusters, centroids
def bisecting_kmeans(points, kk=2, epochs=epochs, max_iter=100, verbose=False):
    points = convert_to_2d_array(points)
    clusters = [points]
    while len(clusters) < kk:
        max_sse_i = np.argmax([SSE(c) for c in clusters])
        cluster = clusters.pop(max_sse_i)
        kmeans_clusters, centroids = kmeans(cluster, k=kk, epochs=epochs, max_iter=max_iter, verbose=verbose)
        clusters.extend(kmeans_clusters)
    return clusters, centroids
def visualize(new_clusters, new_centroids, trucks, show_coordinates=True):
    plt.figure(figsize=(10, 8))
    for i, new_cluster in enumerate(new_clusters):
        points = convert_to_2d_array(new_cluster)
        if points.shape[1] < 2:
            points = np.hstack([points, np.zeros_like(points)])
        plt.plot(points[:,0], points[:,1], 'o', label='Drone Node')
        plt.scatter(new_centroids[i][0], new_centroids[i][1], marker='*',label='Centroid', s=300)
        plt.scatter(np.array(trucks)[:, 0], np.array(trucks)[:, 1], marker='s', label='Truck Node', color='red', s=100)
        plt.title('Bisecting K-means result',fontsize=16)
        plt.xlabel('service area (mile)',fontsize=16)
        plt.ylabel('service area (mile)',fontsize=16)
        if show_coordinates==True:
            for point in points:
                plt.text(point[0], point[1], '({}, {})'.format(round(point[0], 2), round(point[1], 2)))
            plt.text(new_centroids[i][0], new_centroids[i][1], '({}, {})'.format(round(new_centroids[i][0], 2), round(new_centroids[i][1], 2)))
    resolution_value = 1200
    plt.legend(['Drone Node', 'Centroid', 'Truck Node'])
    plt.savefig("bisecting.png", format="png", dpi=resolution_value)
    plt.show()
'Simulated Annealing module'
class SimAnneal(object):
    def __init__(self, coords, T=-1, alpha=-1, stopping_T=-1, stopping_iter=-1):
        self.coords = coords
        self.N = len(coords)
        self.T = math.sqrt(self.N) if T == -1 else T
        self.T_save = self.T 
        self.alpha = 0.995 if alpha == -1 else alpha
        self.stopping_temperature = 1e-8 if stopping_T == -1 else stopping_T
        self.stopping_iter = 100000 if stopping_iter == -1 else stopping_iter
        self.iteration = 1
        self.nodes = [i for i in range(self.N)]
        self.best_solution = None
        self.best_fitness = float("Inf")
        self.fitness_list = []
    def initial_solution(self):
        cur_node = random.choice(self.nodes)  
        solution = [cur_node]
        free_nodes = set(self.nodes)
        free_nodes.remove(cur_node)
        while free_nodes:
            next_node = min(free_nodes, key=lambda x: self.dist(cur_node, x))  # nearest neighbour
            free_nodes.remove(next_node)
            solution.append(next_node)
            cur_node = next_node
        cur_fit = self.fitness(solution)
        if cur_fit < self.best_fitness:  # If best found so far, update best fitness
            self.best_fitness = cur_fit
            self.best_solution = solution
        self.fitness_list.append(cur_fit)
        return solution, cur_fit
    def dist(self, node_0, node_1):
        coord_0, coord_1 = self.coords[node_0], self.coords[node_1]
        return math.sqrt((coord_0[0] - coord_1[0]) ** 2 + (coord_0[1] - coord_1[1]) ** 2)
    def fitness(self, solution):
        cur_fit = 0
        for i in range(self.N):
            cur_fit += self.dist(solution[i % self.N], solution[(i + 1) % self.N])
        return cur_fit
    def p_accept(self, candidate_fitness):
        return math.exp(-abs(candidate_fitness - self.cur_fitness) / self.T)
    def accept(self, candidate):
        candidate_fitness = self.fitness(candidate)
        if candidate_fitness < self.cur_fitness:
            self.cur_fitness, self.cur_solution = candidate_fitness, candidate
            if candidate_fitness < self.best_fitness:
                self.best_fitness, self.best_solution = candidate_fitness, candidate
        else:
            if random.random() < self.p_accept(candidate_fitness):
                self.cur_fitness, self.cur_solution = candidate_fitness, candidate
    def anneal(self):
        self.cur_solution, self.cur_fitness = self.initial_solution()
        while self.T >= self.stopping_temperature and self.iteration < self.stopping_iter:
            candidate = list(self.cur_solution)
            l = random.randint(2, self.N - 1)
            i = random.randint(0, self.N - l)
            candidate[i : (i + l)] = reversed(candidate[i : (i + l)])
            self.accept(candidate)
            self.T *= self.alpha
            self.iteration += 1
            self.fitness_list.append(self.cur_fitness)
        improvement = 100 * (self.fitness_list[0] - self.best_fitness) / (self.fitness_list[0])
    def batch_anneal(self, times=10):
        for i in range(1, times + 1):
            self.T = self.T_save
            self.iteration = 1
            self.cur_solution, self.cur_fitness = self.initial_solution()
            self.anneal()
    def visualize_routes(self):
        plotTSP([self.best_solution], self.coords) 
    def visualize_routes2(self):
        plotTSP2([self.best_solution], self.coords)
    def get_routes(self):
        return self.best_solution
def plotTSP(paths, points, num_iters=1):
    plt.figure(figsize=(10, 8))
    for i, point_id in enumerate(points):
        plt.scatter(point_id[0], point_id[1], color='g',marker='s',label="Truck Node",s=120)
#         plt.annotate(f"point {i}", xy=(point_id[0]+0.1, point_id[1]+0.1))       
    x = []; y = []
    for i in paths[0]:
        x.append(points[i][0])
        y.append(points[i][1])
    a_scale = float(max(x))/float(100)
    if num_iters > 1:
        for i in range(1, num_iters):
            xi = []; yi = [];
            for j in paths[i]:
                xi.append(points[j][0])
                yi.append(points[j][1])
            plt.arrow(xi[-1], yi[-1], (xi[0] - xi[-1]), (yi[0] - yi[-1]),
                    head_width = a_scale, color = 'r',
                    length_includes_head = True, ls = 'dashed',
                    width = 0.001/float(num_iters))
            for i in range(0, len(x) - 1):
                plt.arrow(xi[i], yi[i], (xi[i+1] - xi[i]), (yi[i+1] - yi[i]),
                        head_width = a_scale, color = 'r', length_includes_head = True,
                        ls = 'dashed', width = 0.001/float(num_iters))
    plt.arrow(x[-1], y[-1], (x[0] - x[-1]), (y[0] - y[-1]), head_width = a_scale,
            color ='red', length_includes_head=True)
    for i in range(0,len(x)-1):
        plt.arrow(x[i], y[i], (x[i+1] - x[i]), (y[i+1] - y[i]), head_width = a_scale,
                color = 'red', length_includes_head = True)
    plt.xlim(min(x), max(x)*1.1)
    plt.ylim(min(y), max(y)*1.1)
    plt.title('Shortest Truck Route after Clustering',fontsize=16)
    plt.xlabel('service area (mile)',fontsize=16)
    plt.ylabel('service area (mile)',fontsize=16)
    plt.legend(['Truck Node'])
    resolution_value = 1200
    plt.savefig("truck nodes.png", format="png", dpi=resolution_value)
    plt.show()     
def plotTSP2(paths, points, num_iters=1):
    plt.figure(figsize=(10, 8))
    for i, point_id in enumerate(points):
        plt.scatter(point_id[0], point_id[1], color='blue',marker='o',label="Customer Node",s=120)
#         plt.annotate(f"point {i}", xy=(point_id[0]+0.1, point_id[1]+0.1))       
    x = []; y = []
    for i in paths[0]:
        x.append(points[i][0])
        y.append(points[i][1])
    a_scale = float(max(x))/float(100)
    if num_iters > 1:
        for i in range(1, num_iters):
            xi = []; yi = [];
            for j in paths[i]:
                xi.append(points[j][0])
                yi.append(points[j][1])
            plt.arrow(xi[-1], yi[-1], (xi[0] - xi[-1]), (yi[0] - yi[-1]),
                    head_width = a_scale, color = 'r',
                    length_includes_head = True, ls = 'dashed',
                    width = 0.001/float(num_iters))
            for i in range(0, len(x) - 1):
                plt.arrow(xi[i], yi[i], (xi[i+1] - xi[i]), (yi[i+1] - yi[i]),
                        head_width = a_scale, color = 'r', length_includes_head = True,
                        ls = 'dashed', width = 0.001/float(num_iters))
    plt.arrow(x[-1], y[-1], (x[0] - x[-1]), (y[0] - y[-1]), head_width = a_scale,
            color ='red', length_includes_head=True)
    for i in range(0,len(x)-1):
        plt.arrow(x[i], y[i], (x[i+1] - x[i]), (y[i+1] - y[i]), head_width = a_scale,
                color = 'red', length_includes_head = True)
    plt.xlim(min(x), max(x)*1.1)
    plt.ylim(min(y), max(y)*1.1)
    plt.title('Shortest Truck Route for all customer nodes (TSP-0D)',fontsize=16)
    plt.xlabel('service area (mile)',fontsize=16)
    plt.ylabel('service area (mile)',fontsize=16)
    plt.legend(['Customer Node'])
    resolution_value = 1200
    plt.savefig("Customer Node.png", format="png", dpi=resolution_value)
    plt.show() 
    
'FDTSP ALGORITHM BEGINS'
dr = 1                                                                        # number of drone
result_final = dict()
for i in range(0,iter_proposed_model):
    xc = rnd.rand(n)*square                                                   #randomly generate points
    yc = rnd.rand(n)*square
    coordinate = {'X':[i for i in xc],
                  'Y':[k for k in yc]}
    df = pd.DataFrame(coordinate, columns= ['X', 'Y'])
    df.to_csv ('export_distance.csv', index = False, header=True)             # saving the X & Y in a file
    df = pd.read_csv('export_distance.csv')                                   # call the file
    df = df[['X','Y']]
    points = np.array(df.values.tolist())
    algorithm = bisecting_kmeans
    nd=2
    K=math.ceil(n/(nd+1))
    verbose = False
    max_iter = 1000
    clusters, centroids = algorithm(points=points, kk=K, verbose=verbose, max_iter=max_iter, epochs=epochs)
    c = 300                             
    d = dr + 1                         
    for j in range(c):
        new_clusters = []
        new_centroids = []
        for i in range(len(clusters)):
            if len(clusters[i])>d:
                K = 2
                verbose = False
                max_iter = 1000
                epochs = epochs
                _clusters, _centroids = bisecting_kmeans(points=clusters[i], kk=K, verbose=verbose, max_iter=max_iter, epochs=epochs)
                new_clusters.extend(_clusters)
                new_centroids.extend(_centroids)
            else:
                new_clusters.append(clusters[i])
                new_centroids.append(centroids[i])
        clusters = new_clusters
        centroids = new_centroids
    trucks = []
    for j in range(len(clusters)):
        error = []
        for i in range(len(clusters[j])):
            k = np.linalg.norm(clusters[j][i]-centroids[j], ord=2, axis=0)
            error.append(k)
        index_min = np.argmin(error)
        truck = clusters[j][index_min]
        trucks.append(truck)
    visualize(new_clusters, new_centroids, trucks, show_coordinates=False)
    trucks = np.array(trucks)
    list_trucks = trucks.copy().tolist()
    sparse_clusters = clusters.copy()
    list_clusters = [cluster.tolist() for cluster in sparse_clusters]
    for i in range(len(list_clusters)):
        list_clusters[i].remove(list_trucks[i])
    list_drones = list_clusters.copy()
    del list_clusters
    list_dr1 = []
    for k in list_drones:
        if len(k)==0:
            list_dr1.append([])
        elif len(k)==1:
            list_dr1.append(k[0]) 
    #TESTING THE DRONE FLIGHT ENDURANCE
    d1_tr_test = []
    for i in range(len(list_dr1)):
        if list_dr1[i]!=[]:
            dr1_test = math.sqrt((list_trucks[i][0]-list_dr1[i][0])**2+(list_trucks[i][1]-list_dr1[i][1])**2)
            d1_tr_test.append(dr1_test)
        else:
            d1_tr_test.append(0)
    #TESTING
    d1_tr_test_note = [random.randrange(1, 1000, 1) for i in range(len(d1_tr_test))]
    for k in range(len(d1_tr_test)):
        if d1_tr_test[k] > (endurance2*speed_drone)/2:
            d1_tr_test_note[k] = 'NO LAUNCHED'
        else:
            d1_tr_test_note[k] = d1_tr_test[k]
    # REPLACING DRONE NODES BY TRUCK NODES (REPLACED NODES)
    noLaunched_indices_d1 = [index for index, element in enumerate(d1_tr_test_note) if element == "NO LAUNCHED"]
    count1 = len(noLaunched_indices_d1)
    # Inserts and modify the length
    list_dr1_new = list_dr1.copy()
    k = 0
    for i, idx in enumerate(noLaunched_indices_d1):
        list_trucks.insert(i, list_dr1_new[idx + k])
        list_dr1_new.insert(i, [])
        k +=1
    for i, idx in enumerate(noLaunched_indices_d1):
        list_dr1_new[idx+count1] = []
        idx +=1
    list_trucks = [i for i in list_trucks if i!=[]]
    trucks_new = list_trucks.copy() 
    trucks_new = np.array(trucks_new)                                              
    pd.DataFrame(trucks_new).to_csv("trucks_new.csv", index=False)                
    data = pd.read_csv("trucks_new.csv")                                          
############################## SIMULATED ANNEALING BEGINS #############################################
    def load_csv():                                                                
        data = pd.read_csv('trucks_new.csv').values
        x, y = list(data[:, 0]), list(data[:, 1])
        return x, y
    x, y = load_csv()
    coords1 = [[x[i], y[i]] for i in range(len(x))]
    sa = SimAnneal(coords1, stopping_iter=5000)
    sa.anneal()
    sa.visualize_routes()
    road_map_sa_truck_nodes = sa.get_routes()
############################## COMBINATION BETWEEN TRUCK AND DRONES BEGINS ###########################
    list_trucks_new = trucks_new.copy().tolist()
    d1_tr = []
    for i in range(len(road_map_sa_truck_nodes)):
        if list_dr1_new[road_map_sa_truck_nodes[i]]!=[]:
            tempo = math.sqrt((list_trucks_new[road_map_sa_truck_nodes[i]][0]-list_dr1_new[road_map_sa_truck_nodes[i]][0])**2+\
                              (list_trucks_new[road_map_sa_truck_nodes[i]][1]-list_dr1_new[road_map_sa_truck_nodes[i]][1])**2)
            d1_tr.append(tempo)
        else:
            d1_tr.append(0)
    #DISTANCE INTER-CLUSTER
    d1_ntr = []
    for i in range(len(road_map_sa_truck_nodes)-1):
        if list_dr1_new[road_map_sa_truck_nodes[i]]!=[]:
            tempo11 = math.sqrt((list_dr1_new[road_map_sa_truck_nodes[i]][0]-list_trucks_new[road_map_sa_truck_nodes[i+1]][0])**2+\
                                (list_dr1_new[road_map_sa_truck_nodes[i]][1]-list_trucks_new[road_map_sa_truck_nodes[i+1]][1])**2)
            d1_ntr.append(tempo11) 
        else:
            d1_ntr.append(0)
    for i in range(len(road_map_sa_truck_nodes)-1,len(road_map_sa_truck_nodes)):
        if list_dr1_new[road_map_sa_truck_nodes[i]]!=[]:
            tempo101 = math.sqrt((list_dr1_new[road_map_sa_truck_nodes[i]][0]-list_trucks_new[road_map_sa_truck_nodes[0]][0])**2+\
                                 (list_dr1_new[road_map_sa_truck_nodes[i]][1]-list_trucks_new[road_map_sa_truck_nodes[0]][1])**2)
            d1_ntr.append(tempo101)
        else:
            d1_ntr.append(0)        
    #TOTAL DISTANCE OF EACH DRONE WITHOUT ENDURANCE
    d1_travel = list(map(add,d1_tr,d1_ntr))
    d1_tr_new = [random.randrange(1, 1000, 1) for i in range(len(d1_tr))]
    d1_ntr_new = [random.randrange(1, 1000, 1) for i in range(len(d1_ntr))]
    for k in range(len(d1_tr)):
        if d1_travel[k] > endurance2*speed_drone and 2*d1_tr[k] < endurance2*speed_drone:
            d1_tr_new[k] = 'COMEBACK'
            d1_ntr_new[k] = 'COMEBACK' #RETURNED NODES
        elif d1_travel[k] > endurance2*speed_drone and 2*d1_tr[k] > endurance2*speed_drone:
            d1_tr_new[k] = 'NO LAUNCHED 2'
            d1_ntr_new[k] = 'NO LAUNCHED 2' # REPLACED NODES
        else:
            d1_tr_new[k] = d1_tr[k]
            d1_ntr_new[k] =  d1_ntr[k]
    d1_tr_new_temp = [random.randrange(1, 1000, 1) for i in range(len(d1_tr_new))]
    d1_ntr_new_temp = [random.randrange(1, 1000, 1) for i in range(len(d1_ntr_new))]
    for k in range(len(d1_tr_new)):
        if d1_tr_new[k] =='COMEBACK' or d1_tr_new[k]=='NO LAUNCHED 2' :
            d1_tr_new_temp[k] = 0
            d1_ntr_new_temp[k] = 0 
        else:
            d1_tr_new_temp[k] = d1_tr_new[k]
            d1_ntr_new_temp[k] = d1_ntr_new[k]
    #TRAVEL TIME 
    travel_d1_tr = [60*within_distance1/speed_drone for within_distance1 in d1_tr_new_temp]
    #SETUP TIME
    setup_d1 = [random.randrange(1, 1000, 1) for i in range(len(d1_tr_new))]
    for k in range(len(d1_tr_new)):
        if d1_tr_new[k] =='NO LAUNCHED 2' or d1_tr_new[k]== 0:
            setup_d1[k] = 0
        else:
            setup_d1[k] = 1 #including ordinal nodes and COMEBACK nodes
    #SERVICE TIME
    service_d1 = [random.randrange(1, 1000, 1) for i in range(len(d1_tr_new))]
 
    for k in range(len(d1_tr_new)):
        if d1_tr_new[k] =='NO LAUNCHED 2' or d1_tr_new[k]== 0:
            service_d1[k] = 0
        else:
            service_d1[k] = 0.5 #including ordinal nodes and COMEBACK nodes
    #TOTAL TIME for each DRONE WITHIN CLUSTER
    within_d1 = [sum(x) for x in zip(travel_d1_tr,setup_d1,service_d1)]
    travel_d1_ntr = [60*within_distance1/speed_drone for within_distance1 in d1_ntr_new_temp]
    #TOTAL TIME FOR EACH DRONE to THE NEXT TRUCK NODES
    time_d1  = [sum(x) for x in zip(within_d1,travel_d1_ntr)]
    time_drone = time_d1
    #TIME FOR TRUCK ROUTE
    #Waiting time at node i
    waiting_distance_dr1 = [random.randrange(1, 1000, 1) for i in range(len(d1_tr_new))]
    for k in range(len(d1_tr_new)):
        if d1_tr_new[k] == 'COMEBACK':
            waiting_distance_dr1[k] = 2*d1_tr[k]
        else:
            waiting_distance_dr1[k] = 0
    waiting_time_dr1 = [60*within_distance1/speed_drone for within_distance1 in waiting_distance_dr1]
    waiting_drone = waiting_time_dr1
    service_truck = [0.5 if i==0 else 0 for i in waiting_drone] #IF the nodes wait drone, then truck service time is 0
    # TRAVEL TIME OF TRUCKS NODES
    distance_trucks = []
    for i in range(len(road_map_sa_truck_nodes)-1):
        temp = math.sqrt((list_trucks_new[road_map_sa_truck_nodes[i]][0]-list_trucks_new[road_map_sa_truck_nodes[i+1]][0])**2+(list_trucks_new[road_map_sa_truck_nodes[i]][1]-list_trucks_new[road_map_sa_truck_nodes[i+1]][1])**2)
        distance_trucks.append(temp)
        if i+1 == len(road_map_sa_truck_nodes)-1: #end node to the start node
            k = math.sqrt((list_trucks_new[road_map_sa_truck_nodes[i+1]][0]-list_trucks_new[road_map_sa_truck_nodes[0]][0])**2+(list_trucks_new[road_map_sa_truck_nodes[i+1]][1]-list_trucks_new[road_map_sa_truck_nodes[0]][1])**2)
            distance_trucks.append(k)
    speed_truck = 35
    travel_truck  = [60*each_distance/speed_truck for each_distance in distance_trucks]
    time_truck = [sum(y) for y in zip(waiting_drone,service_truck,travel_truck)]
    waiting_time_temp = list(map(operator.sub,time_drone,time_truck))
    waiting_time_next_node = [0 if i<0 else i for i in waiting_time_temp]
    total_time = [sum(x) for x in zip(time_truck, waiting_time_next_node)]
    t = sum(total_time) 
    total_time_2 = np.cumsum(total_time)
    m1 = [index for index,value in enumerate(total_time_2) if value > 240] 
    zipped2 = zip(list_trucks_new, list_dr1_new)
    clusters_new = list(zipped2)
    count_clusters = []
    for i in m1:
        for j in range(len(clusters_new[i])):
            if clusters_new[i][j]==[]:
                count_clusters.append(0)
            else:
                count_clusters.append(1)
    SL1 = sum(count_clusters)      
    no_service_by_dr1 = d1_tr_new.count('NO LAUNCHED 2')
    SL2 = no_service_by_dr1
    SL = SL1+SL2
#SA TO FIND TRUCKS-TSP if only truck 
    coords2 = pd.read_csv("export_distance.csv").values.tolist()
    sa = SimAnneal(coords2, stopping_iter=5000)
    sa.anneal()
    sa.visualize_routes2()
    road_map_sa_only_trucks = sa.get_routes()
    distance_trucks2 = []
    for i in range(len(road_map_sa_only_trucks)-1):
        temp2 = math.sqrt((points[road_map_sa_only_trucks[i]][0]-points[road_map_sa_only_trucks[i+1]][0])**2+\
                          (points[road_map_sa_only_trucks[i]][1]-points[road_map_sa_only_trucks[i+1]][1])**2)
        distance_trucks2.append(temp2)
        if i+1 == len(road_map_sa_only_trucks)-1: #end node to the start node
            k2 = math.sqrt((points[road_map_sa_only_trucks[i+1]][0]-points[road_map_sa_only_trucks[0]][0])**2+\
                           (points[road_map_sa_only_trucks[i+1]][1]-points[road_map_sa_only_trucks[0]][1])**2)
            distance_trucks2.append(k2)
    speed_truck2 = 35
    travel_truck2  = [60*each_distance/speed_truck2 for each_distance in distance_trucks2]
    service_truck2 = [0.5 if i>0 else 0 for i in travel_truck2] 
    time_truck2 = [sum(y) for y in zip(travel_truck2,service_truck2)]
    total_time_TSP = np.cumsum(time_truck2)
    m_TSP = [index for index,value in enumerate(total_time_TSP) if value > 240] 
    SL_TSP = len(m_TSP) 
    totaltime_only_truck = sum(time_truck2)
    saving = totaltime_only_truck - t
    saving 
############################## PERFORMING EVALUATION #############################################    
    saving = totaltime_only_truck - t
    total_waiting_time = sum(list( map(add, waiting_drone, waiting_time_next_node)))
    count_deli1 = sum(map(lambda x : x != 'NO LAUNCHED 2' and x !=0, d1_tr_new)) 
    count_deli=count_deli1
    dd = count_deli/n
    drone_should_be_used = len(road_map_sa_truck_nodes)*dr
    count_drone1_used_in_fact = sum([1 if x!=0 else 0 for x in d1_tr_new])
    count_drones_used_in_fact = count_drone1_used_in_fact 
    drones_usage = count_drones_used_in_fact/drone_should_be_used
    for j in d1_tr_test_note:
        count_dr1 = d1_tr_test_note.count('NO LAUNCHED')   
    count_total = count_dr1 
############################## RESULTS RECORDING #############################################    
    headers = ['INSTANCES', 
               'TOTAL TIME OF TRUCK + DRONE',
               'TOTAL TIME OF ONLY TRUCK TSP',
               'SAVING TIME BY PROPOSED MODEL',
               'TOTAL TIME TRUCK WAITS DRONES',
               'NUMBER OF CUSTOMERS SERVED BY DRONE',
               'PERCENTAGE OF DRONE DELIVERIES',
               'DRONES USAGE PERCENTAGE',
               'NUMBER OF DRONE-POINTS  BE REPLACED BY TRUCK',
               'NUMBER OF POINTS CANNOT BE SERVED BY TRUCK & DRONE WITHIN 4 HOURS',
               'NUMBER OF POINTS CANNOT BE SERVED BY TRUCK-TSP ONLY WITHIN 4 HOURS']
    for element in headers:
        if element not in result_final:
            result_final[element] = list()
            if element == 'INSTANCES':
                result_final[element].append('2020_60_60_80')
            elif element == 'TOTAL TIME OF TRUCK + DRONE':
                result_final[element].append(t)
            elif element == 'TOTAL TIME OF ONLY TRUCK TSP':
                result_final[element].append(totaltime_only_truck)
            elif element == 'SAVING TIME BY PROPOSED MODEL':
                result_final[element].append(saving)
            elif element == 'TOTAL TIME TRUCK WAITS DRONES':
                result_final[element].append(total_waiting_time)
            elif element == 'NUMBER OF CUSTOMERS SERVED BY DRONE':
                result_final[element].append(count_deli)
            elif element == 'PERCENTAGE OF DRONE DELIVERIES':
                result_final[element].append(dd)
            elif element == 'DRONES USAGE PERCENTAGE':
                result_final[element].append(drones_usage)
            elif element == 'NUMBER OF DRONE-POINTS  BE REPLACED BY TRUCK':
                result_final[element].append(count_total)
            elif element == 'NUMBER OF POINTS CANNOT BE SERVED BY TRUCK & DRONE WITHIN 4 HOURS':
                result_final[element].append(SL)
            elif element == 'NUMBER OF POINTS CANNOT BE SERVED BY TRUCK-TSP ONLY WITHIN 4 HOURS':
                result_final[element].append(SL_TSP)       
        else:
            if element == 'INSTANCES':
                result_final[element].append('2020_60_60_80')
            elif element == 'TOTAL TIME OF TRUCK + DRONE':
                result_final[element].append(t)
            elif element == 'TOTAL TIME OF ONLY TRUCK TSP':
                result_final[element].append(totaltime_only_truck)
            elif element == 'SAVING TIME BY PROPOSED MODEL':
                result_final[element].append(saving)
            elif element == 'TOTAL TIME TRUCK WAITS DRONES':
                result_final[element].append(total_waiting_time)
            elif element == 'NUMBER OF CUSTOMERS SERVED BY DRONE':
                result_final[element].append(count_deli)
            elif element == 'PERCENTAGE OF DRONE DELIVERIES':
                result_final[element].append(dd)
            elif element == 'DRONES USAGE PERCENTAGE':
                result_final[element].append(drones_usage)
            elif element == 'NUMBER OF DRONE-POINTS  BE REPLACED BY TRUCK':
                result_final[element].append(count_total)
            elif element == 'NUMBER OF POINTS CANNOT BE SERVED BY TRUCK & DRONE WITHIN 4 HOURS':
                result_final[element].append(SL)
            elif element == 'NUMBER OF POINTS CANNOT BE SERVED BY TRUCK-TSP ONLY WITHIN 4 HOURS':
                result_final[element].append(SL_TSP)
print(result_final)
df = pd.DataFrame(result_final, columns = ['INSTANCES', 
                                           'TOTAL TIME OF TRUCK + DRONE',
                                           'TOTAL TIME OF ONLY TRUCK TSP',
                                           'SAVING TIME BY PROPOSED MODEL',
                                           'TOTAL TIME TRUCK WAITS DRONES',
                                           'NUMBER OF CUSTOMERS SERVED BY DRONE',
                                           'PERCENTAGE OF DRONE DELIVERIES',
                                           'DRONES USAGE PERCENTAGE',
                                           'NUMBER OF POINTS CANNOT BE SERVED BY TRUCK & DRONE WITHIN 4 HOURS',
                                           'NUMBER OF POINTS CANNOT BE SERVED BY TRUCK-TSP ONLY WITHIN 4 HOURS',
                                           'NUMBER OF DRONE-POINTS  BE REPLACED BY TRUCK'])
df.to_csv (r'1DRONE-SimulatedAnnealing.csv', index = False, header=True)
print('time consuming for FDTSP:',time.time() - start_time,'seconds')