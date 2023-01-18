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
n = 40                                                                    # number of customers   
square = 40                                                               # square of area
speed_drone = 80                                                          # mph
speed_truck = 35                                                          # mph
endurance1 = 0.5                                                          # hour
endurance2 = 1                                                            # hour
epochs = 10                                                                
epochs_ga = 5                                                             
chromosome = 50                                                          
terminal_condition = 20                                                 
mutation_miltiplier = 5                                                   
iter_proposed_model = 1                                                   # NUMBER OF REPLICATION 

'FUNCTIONS'
'BISECTING K-MEANS MODULE'
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
def kmeans(points, k=2, epochs=10, max_iter=100, verbose=False):
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
def bisecting_kmeans(points, kk=2, epochs=10, max_iter=100, verbose=False):
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
def get_road_map_from_a_point(road_map, start_index):
    index_of_start_point = road_map.index(start_index)
    corr_road_map = road_map[index_of_start_point:]
    corr_road_map.extend(road_map[:index_of_start_point])
    return corr_road_map
def get_road_map_from_a_point(road_map2, start_index2):
    index_of_start_point2 = road_map2.index(start_index2)
    corr_road_map2 = road_map2[index_of_start_point2:]
    corr_road_map2.extend(road_map2[:index_of_start_point2])
    return corr_road_map2
# Visualize road maps
def visualize_function(data, corr_road_map):
    plt.figure(figsize=(10, 8))
    for i in range(len(corr_road_map)):
        plt.scatter(data.values[i][0], data.values[i][1],marker='s',label ='Truck Node',color='green')
#         plt.annotate(i,xy=(data.values[i][0] + 0.3, data.values[i][1] - 0.3),verticalalignment='top',fontsize=17)
    dist = 0
    for i in range(len(corr_road_map)-1):
        start, end = corr_road_map[i], corr_road_map[i+1]
        point1, point2 = data.values[start], data.values[end]
        distance = math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
        dist += distance
        plt.arrow(point1[0], point1[1], 
            point2[0] - point1[0], point2[1] - point1[1],
            head_width=0.2,
            head_length=0.2,
            length_includes_head=True,
            color='red')
        if i+1 == len(corr_road_map)-1:
            start, end = corr_road_map[-1], corr_road_map[0]
            point1, point2 = data.values[start], data.values[end]
            distance = math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
            dist += distance
            plt.arrow(point1[0], point1[1], 
            point2[0] - point1[0], point2[1] - point1[1],
            head_width=0.2,
            head_length=0.2,
            length_includes_head=True,
            color='red')
    plt.title('Shortest Truck Route after Clustering',fontsize=18)
    plt.xlabel('service area (mile)',fontsize=16)
    plt.ylabel('service area (mile)',fontsize=16)
    plt.legend(['Truck Node'])
    resolution_value = 1200
    plt.savefig("truck nodes.png", format="png", dpi=resolution_value)
    plt.show()  
def visualize_function2(data2, corr_road_map2):
    plt.figure(figsize=(10, 8))
    for i in range(len(corr_road_map2)):
        plt.scatter(data2.values[i][0], data2.values[i][1],marker='o',label ='Customer Node',color='blue')
#         plt.annotate(i,xy=(data2.values[i][0] + 0.3, data2.values[i][1] - 0.3), verticalalignment='top',fontsize=17)
    dist = 0
    for i in range(len(corr_road_map2)-1):
        start, end = corr_road_map2[i], corr_road_map2[i+1]
        point1, point2 = data2.values[start], data2.values[end]
        distance = math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
        dist += distance
        plt.arrow(point1[0], point1[1], 
            point2[0] - point1[0], point2[1] - point1[1],
            head_width=0.2,
            head_length=0.2,
            length_includes_head=True,
            color='red')
        if i+1 == len(corr_road_map2)-1:
            start, end = corr_road_map2[-1], corr_road_map2[0]
            point1, point2 = data2.values[start], data2.values[end]
            distance = math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
            dist += distance
            plt.arrow(point1[0], point1[1], 
            point2[0] - point1[0], point2[1] - point1[1],
            head_width=0.2,
            head_length=0.2,
            length_includes_head=True,
            color='red')
    plt.title('Shortest Truck Route for all customer nodes (TSP-0D)',fontsize=18)
    plt.xlabel('service area (mile)',fontsize=16)
    plt.ylabel('service area (mile)',fontsize=16)
    plt.legend(['Customer Node'])
    resolution_value = 1200
    plt.savefig("Customer nodes.png", format="png", dpi=resolution_value)
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
    df.to_csv ('export_distance_drone.csv', index = False, header=True)             
    df = pd.read_csv('export_distance_drone.csv')                                   
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
                epochs = 50
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
    #SEPARATING THE DRONE LIST
    list_dr1 = []
    for k in list_drones:
        if len(k)==0:
            list_dr1.append([])
        else:
            list_dr1.append(k[0])
    #TESTING THE DRONE FLIGHT ENDURANCE
    d1_tr_test = []
    for i in range(len(list_dr1)):
        if list_dr1[i]!=[]:
            dr1_test = math.sqrt((list_trucks[i][0]-list_dr1[i][0])**2+(list_trucks[i][1]-list_dr1[i][1])**2)
            d1_tr_test.append(dr1_test)
        else:
            d1_tr_test.append(0)
    d1_tr_test_note = [random.randrange(1, 1000, 1) for i in range(len(d1_tr_test))]
    for k in range(len(d1_tr_test)):
        if d1_tr_test[k] > (endurance2*speed_drone)/2:
            d1_tr_test_note[k] = 'NO LAUNCHED'
        else:
            d1_tr_test_note[k] = d1_tr_test[k]
    # REPLACING THE DRONE NODES BY TRUCK NODES
    noLaunched_indices_d1 = [index for index, element in enumerate(d1_tr_test_note) if element == "NO LAUNCHED"]
    count1 = len(noLaunched_indices_d1)
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
    pd.DataFrame(trucks_new).to_csv("trucks_new_1drone.csv", index=False)                 
                                        
    #GENETIC ALGORITHM TO FIND SHORTEST ROUTE TSP FOR TRUCK 
    data = pd.read_csv("trucks_new_1drone.csv")       
    def load_csv():                                                           
        data = pd.read_csv('trucks_new_1drone.csv').values
        x, y = list(data[:, 0]), list(data[:, 1])
        return x, y
    def initialization(x,y):
        ini_solution=[]    
        for i in range(chromosome):
            t = random.sample(range(0, numberofcustomer), numberofcustomer) 
            ini_solution.append(t)  
        return ini_solution
    def distance(solution):   
        totaldis = []
        for j in range(len(solution)):
            totaldistance = 0
            for i in range(numberofcustomer - 1):
                k = solution[j][i]
                w = solution[j][i+1]
                temp = math.sqrt((((x[k]-x[w])**2) + ((y[k]-y[w])**2)))
                totaldistance += temp
                if i+1 == numberofcustomer-1:
                    g = solution[j][0]
                    totaldistance += math.sqrt((((x[k]-x[w])**2) + ((y[k]-y[w])**2)))
            totaldis.append(totaldistance)
        return totaldis
    def crossover(parent):  
        offspring=[]
        for i in range(0, chromosome, 2):
            alpha = random.random()
            if alpha < crossover_rate:
                temp_1, temp_2, temp_3, temp_4 = [], [], [], []
                point1 = random.randint(0, numberofcustomer - 1)
                point2 = random.randint(0, numberofcustomer - 1)
                while (point2 == point1):
                    point2 = random.randint(0, numberofcustomer - 1)
                if point1 > point2:
                    temp = point1
                    point1 = point2
                    point2 = temp
                for k in range(point1,point2):
                    temp_1.append(parent[i][k])
                    temp_2.append(parent[i+1][k])
                for j in range(numberofcustomer):                
                    if j in range(point1,point2):
                        temp_3.append(parent[i][j])
                    else:
                        for item in parent[i+1]:
                            if item not in (temp_1 + temp_3):
                                temp_3.append(item)
                                break 
                for j in range(numberofcustomer):                
                    if j in range(point1,point2):
                        temp_4.append(parent[i+1][j])
                    else:
                        for item in parent[i]:
                            if item not in (temp_2 + temp_4):
                                temp_4.append(item) 
                                break                              
                offspring.append(temp_3)
                offspring.append(temp_4)                    
            else:
                offspring.append(parent[i])
                offspring.append(parent[i+1])
        return offspring
    def mutation(parent):
        random.shuffle(parent)
        child = []
        for x in range(mutation_miltiplier):
            for i in range(chromosome):
                alpha = random.random() 
                if alpha < mutation_rate:
                    draw1 = random.randint(0, numberofcustomer - 1)
                    draw2 = random.randint(0, numberofcustomer - 1)
                    while (draw2==draw1):
                        draw2 = random.randint(0, numberofcustomer - 1)
                    temp_7 = []
                    for j in range(numberofcustomer):
                        if j == draw1:
                            temp_7.append(parent[i][draw2])
                        elif j == draw2:
                            temp_7.append(parent[i][draw1])
                        else:
                            temp_7.append(parent[i][j])
                    child.append(temp_7)
                else:
                    child.append(parent[i])
        return child
    def comparsion(parent, child):
        mixparent = []
        for i in range(len(parent)):
            mixparent.append(parent[i])
        for i in range(len(child)):    
            mixparent.append(child[i])
        total_dis1 = distance(parent)
        total_dis2 = distance(child)
        total_dis=[]
        for i in range(len(parent)):
            total_dis.append(total_dis1[i])
        for i in range(len(child)):  
            total_dis.append(total_dis2[i])
        for i in range(len(parent) + len(child)):
            for j in range(len(parent) + len(child) - 1):
                if total_dis[j] >= total_dis[j+ 1]:
                    tem = total_dis[j]
                    total_dis[j] = total_dis[j+ 1]
                    total_dis[j+ 1] = tem
                    tem = mixparent[j]
                    mixparent[j] = mixparent[j+ 1]
                    mixparent[j + 1] = tem
        del mixparent[chromosome:]   
        return mixparent
    def ga_tsp(x, y, parent):
        total_iteration = 1000
        count = 0
        objective=[]
        answer=2000
        for iteration in range(total_iteration):
            offspring = crossover(parent)
            child = mutation(offspring)        
            parent = comparsion(parent, child)
            totaldis = distance(parent)
            answer = min(totaldis)
            z = totaldis.index(answer)
            objective.append(answer)
            if iteration != 0:       
                if objective[iteration-1] == objective[iteration]:
                    count +=1
                else:
                    count = 0
            if count == terminal_condition:         
                break
        return answer, parent[z]
    numberofcustomer = len(trucks_new)                                                     
    best = 600
    tracing_list = []
    for i in range(epochs_ga):
        crossover_rate = 0.9
        mutation_rate = 1
        x, y = load_csv()
        exparent = initialization(x, y)
        start = time.time()
        answer, parent = ga_tsp(x, y, exparent)
        tracing_list.append(parent)
        end = time.time() 
        if answer < best: 
            best = answer
            best_round = i+1     
    road_map = tracing_list[best_round-1]
    start_index = road_map[0]
    corr_road_map = get_road_map_from_a_point(road_map, start_index)
    visualize_function(data, corr_road_map)
    list_trucks_new = trucks_new.copy().tolist()
    d1_tr = []
    for i in range(len(road_map)):
        if list_dr1_new[road_map[i]]!=[]:
            tempo = math.sqrt((list_trucks_new[road_map[i]][0]-list_dr1_new[road_map[i]][0])**2+(list_trucks_new[road_map[i]][1]-list_dr1_new[road_map[i]][1])**2)
            d1_tr.append(tempo)
        else:
            d1_tr.append(0)
    #DISTANCE INTER-CLUSTER
    d1_ntr = []
    for i in range(len(road_map)-1):
        if list_dr1_new[road_map[i]]!=[]:
            tempo11 = math.sqrt((list_dr1_new[road_map[i]][0]-list_trucks_new[road_map[i+1]][0])**2+(list_dr1_new[road_map[i]][1]-list_trucks_new[road_map[i+1]][1])**2)
            d1_ntr.append(tempo11) 
        else:
            d1_ntr.append(0)
    for i in range(len(road_map)-1,len(road_map)):
        if list_dr1_new[road_map[i]]!=[]:
            tempo101 = math.sqrt((list_dr1_new[road_map[i]][0]-list_trucks_new[road_map[0]][0])**2+(list_dr1_new[road_map[i]][1]-list_trucks_new[road_map[0]][1])**2)
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
            d1_ntr_new[k] = 'COMEBACK'  #RETURNED NODES
        elif d1_travel[k] > endurance2*speed_drone and 2*d1_tr[k] > endurance2*speed_drone:
            d1_tr_new[k] = 'NO LAUNCHED 2'
            d1_ntr_new[k] = 'NO LAUNCHED 2' #REPLACED NODES CHECK
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
            setup_d1[k] = 1 # 1 MINUTE
    #SERVICE TIME
    service_d1 = [random.randrange(1, 1000, 1) for i in range(len(d1_tr_new))]
    for k in range(len(d1_tr_new)):
        if d1_tr_new[k] =='NO LAUNCHED 2' or d1_tr_new[k]== 0:
            service_d1[k] = 0
        else:
            service_d1[k] = 0.5 #0.5 MINUTE
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
    service_truck = [0.5 if i==0 else 0 for i in waiting_drone] 
    # TRAVEL TIME OF TRUCKS NODES
    distance_trucks = []
    for i in range(len(road_map)-1):
        temp = math.sqrt((list_trucks_new[road_map[i]][0]-list_trucks_new[road_map[i+1]][0])**2+(list_trucks_new[road_map[i]][1]-list_trucks_new[road_map[i+1]][1])**2)
        distance_trucks.append(temp)
        if i+1 == len(road_map)-1: #end node to the start node
            k = math.sqrt((list_trucks_new[road_map[i+1]][0]-list_trucks_new[road_map[0]][0])**2+(list_trucks_new[road_map[i+1]][1]-list_trucks_new[road_map[0]][1])**2)
            distance_trucks.append(k)
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
# GA FINDS SHORTEST ROUTE FOR TRUCK-ONLY MODE (or TSP-0D)
    numberofcustomer = n                                                   
    data2 = pd.read_csv("export_distance_drone.csv") 
    def load_csv():
        data2 = pd.read_csv('export_distance_drone.csv').values
        x, y = list(data2[:, 0]), list(data2[:, 1])
        return x, y
    best2 = 600
    tracing_list2 = []
    for i in range(epochs_ga):
        crossover_rate = 0.9
        mutation_rate = 1
        x, y = load_csv()
        exparent = initialization(x, y)
        start = time.time()
        answer2, parent = ga_tsp(x, y, exparent)
        tracing_list2.append(parent)
        end = time.time() 
        if answer2 < best2: 
            best2 = answer2
            best_round2 = i+1     
    road_map2 = tracing_list2[best_round2-1]
    start_index2 = road_map2[0]
    corr_road_map2 = get_road_map_from_a_point(road_map2, start_index2)
    visualize_function2(data2, corr_road_map2)
    distance_trucks2 = []
    for i in range(len(road_map2)-1):
        temp2 = math.sqrt((points[road_map2[i]][0]-points[road_map2[i+1]][0])**2+(points[road_map2[i]][1]-points[road_map2[i+1]][1])**2)
        distance_trucks2.append(temp2)
        if i+1 == len(road_map2)-1: #end node to the start node
            k2 = math.sqrt((points[road_map2[i+1]][0]-points[road_map2[0]][0])**2+(points[road_map2[i+1]][1]-points[road_map2[0]][1])**2)
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
################### PERFORMANCE EVALUATION##################
    saving = totaltime_only_truck - t
    total_waiting_time = sum(list( map(add, waiting_drone, waiting_time_next_node)))
    count_deli1 = sum(map(lambda x : x != 'NO LAUNCHED 2' and x !=0, d1_tr_new)) 
    count_deli=count_deli1
    dd = count_deli/n
    drone_should_be_used = len(road_map)*dr
    count_drone1_used_in_fact = sum([1 if x!=0 else 0 for x in d1_tr_new])
    count_drones_used_in_fact = count_drone1_used_in_fact
    drones_usage = count_drones_used_in_fact/drone_should_be_used
    for j in d1_tr_test_note:
        count_dr1 = d1_tr_test_note.count('NO LAUNCHED')   # drone1 nodes replaced by trucks
    count_total = count_dr1
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
                result_final[element].append('1010_80_60_80')  # change the instance (by parameters) if needed
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
                result_final[element].append('1010_80_60_80')  # change the instance (by parameters) if needed
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
df.to_csv (r'1DRONE-GA.xlsx', index = False, header=True)    
print('time consuming for FDTSP:',time.time() - start_time,'seconds')