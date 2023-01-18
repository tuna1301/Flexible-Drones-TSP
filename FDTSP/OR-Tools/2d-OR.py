import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from operator import add
import math, time
import operator
import csv
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
rnd = np.random
import time
start_time = time.time()
'PARAMETERS'
n = 80                                                                   # number of customers   
square = 20                                                               # square of area
speed_drone = 40                                                          # km/hour
endurance1 = 0.5                                                          # hour
endurance2 = 1                                                            # hour
epochs = 10                                                               # epochs for Bisecting K-mean Algorithm  
iter_proposed_model = 1                                                   # entire FDTSP iterations

'FUNCTIONS'
#Bisecting module
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

#OR-tool module
def create_data_model():    
    data = {}
    data['distance_matrix'] = maps_centroids_2 
    data['num_vehicles'] = 1
    data['depot'] = 0
    return data

def print_solution(data, manager, routing, solution):
    max_route_distance = 0
    routes = {}
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        route_distance = 0
        route = []
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route.append(node_index)
            plan_output += ' {} -> '.format(node_index)
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
        node_index = manager.IndexToNode(index)
        route.append(node_index)
        routes[f"Vehicle {vehicle_id}"] = route
        plan_output += '{}\n'.format(node_index)
        plan_output += 'Distance of the route: {}m\n'.format(route_distance)
        max_route_distance = max(route_distance, max_route_distance)
    return routes
def visualize_function(final_centroids, routes_solution): #Hybrid system
    plt.figure(figsize=(10, 8))
    for vehicle_id, (key, value) in enumerate(routes_solution.items()):
        if len(value) == 2: continue
        color = 'green'
        for i, point_id in enumerate(value):
            plt.scatter(final_centroids[point_id][0], final_centroids[point_id][1],marker='s',\
                        label="Truck Node",color=[color],s=100)
#             plt.annotate(f"point {point_id}", xy=(final_centroids[point_id][0]+0.1, final_centroids[point_id][1]+0.1))
        for i in range(len(value)-1):
            start, end = value[i], value[i+1]
            point1, point2 = final_centroids[start], final_centroids[end]
            plt.arrow(point1[0], 
                      point1[1], 
                      point2[0] - point1[0], point2[1] - point1[1],
                      head_width=0.2,
                      head_length=0.2,
                      length_includes_head=True,
                      color='red')
    plt.title('Shortest Truck Route after Clustering',fontsize=16)
    plt.xlabel('service area (mile)',fontsize=16)
    plt.ylabel('service area (mile)',fontsize=16)
    plt.legend(['Truck Node'])
    resolution_value = 1200
    plt.savefig("truck nodes.png", format="png", dpi=resolution_value)
    plt.show()
    
def main(visualize=True):
    data = create_data_model()
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']), 
                                           data['num_vehicles'], 
                                           data['depot'])
    routing = pywrapcp.RoutingModel(manager)
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    dimension_name = 'Distance'
    routing.AddDimension(
        transit_callback_index,
        0,    
        3000,  
        True,  
        dimension_name)
    distance_dimension = routing.GetDimensionOrDie(dimension_name)
    distance_dimension.SetGlobalSpanCostCoefficient(100)
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    solution = routing.SolveWithParameters(search_parameters)
    if solution:
        routes_solution = print_solution(data, manager, routing, solution)
        if visualize == True:
            visualize_function(points_truck_TSP, routes_solution)
        return routes_solution
    else:
        print('No solution found !')

#OR-TOOL for TSP truck only
def load_csv2():                                                                
    data2 = pd.read_csv('export_distance.csv').values
    x2, y2 = list(data2[:, 0]), list(data2[:, 1])
    return x2, y2
def create_data2_model():    
    data2 = {}
    data2['distance_matrix'] = maps_centroids_3 #maps_matrix
    data2['num_vehicles'] = 1
    data2['depot'] = 0
    return data2
def print_solution2(data2, manager2, routing2, solution2):
    max_route_distance2 = 0
    routes2 = {}
    for vehicle_id2 in range(data2['num_vehicles']):
        index2 = routing2.Start(vehicle_id2)
        plan_output2 = 'Route for vehicle {}:\n'.format(vehicle_id2)
        route_distance2 = 0
        route2 = []
        while not routing2.IsEnd(index2):
            node_index2 = manager2.IndexToNode(index2)
            route2.append(node_index2)
            plan_output2 += ' {} -> '.format(node_index2)
            previous_index2 = index2
            index2 = solution2.Value(routing2.NextVar(index2))
            route_distance2 += routing2.GetArcCostForVehicle(previous_index2, index2, vehicle_id2)
        node_index2 = manager2.IndexToNode(index2)
        route2.append(node_index2)
        routes2[f"Vehicle {vehicle_id2}"] = route2
        plan_output2 += '{}\n'.format(node_index2)
        plan_output2 += 'Distance of the route: {}m\n'.format(route_distance2)
        max_route_distance2 = max(route_distance2, max_route_distance2)
    return routes2

def visualize_function2(final_centroids2, routes_solution_2): #TSP only
    plt.figure(figsize=(10, 8))
    for vehicle_id2, (key2, value2) in enumerate(routes_solution_2.items()):
        if len(value2) == 2: continue
        color = 'blue'
        for i, point_id2 in enumerate(value2):
            plt.scatter(final_centroids2[point_id2][0], final_centroids2[point_id2][1],\
                        label="Customer Node", color=[color], s=100)
#             plt.annotate(f"point {point_id2}", xy=(final_centroids2[point_id2][0]+0.1, final_centroids2[point_id2][1]+0.1))
        for i in range(len(value2)-1):
            # get the starting point (A) and the end point (B) in the station_address
            start, end = value2[i], value2[i+1]
            # get the coordinate of the stating point (x1, y1) and the end point (x2, y2)
            point1, point2 = final_centroids2[start], final_centroids2[end]
            plt.arrow(point1[0], 
                      point1[1], 
                      point2[0] - point1[0], point2[1] - point1[1],
                      head_width=0.2,
                      head_length=0.2,
                      length_includes_head=True,
                      color='red')
    plt.title('Shortest Truck Route for all customer nodes (TSP-0D)',fontsize=16)
    plt.xlabel('service area (mile)',fontsize=16)
    plt.ylabel('service area (mile)',fontsize=16)
    plt.legend(['Customer Node'])
    resolution_value = 1200
    plt.savefig("all nodes.png", format="png", dpi=resolution_value)
    plt.show()

def main2(visualize=True):
    data2 = create_data2_model()
    manager2 = pywrapcp.RoutingIndexManager(len(data2['distance_matrix']), 
                                           data2['num_vehicles'], 
                                           data2['depot'])
    routing2 = pywrapcp.RoutingModel(manager2)
    def distance_callback2(from_index2, to_index2):
        from_node2 = manager2.IndexToNode(from_index2)
        to_node2 = manager2.IndexToNode(to_index2)
        return data2['distance_matrix'][from_node2][to_node2]
    transit_callback_index2 = routing2.RegisterTransitCallback(distance_callback2)
    routing2.SetArcCostEvaluatorOfAllVehicles(transit_callback_index2)
    dimension_name2 = 'Distance'
    routing2.AddDimension(
        transit_callback_index2,
        0,    
        3000,  
        True, 
        dimension_name2)
    distance_dimension2 = routing2.GetDimensionOrDie(dimension_name2)
    distance_dimension2.SetGlobalSpanCostCoefficient(100)
    search_parameters2 = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters2.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    solution2 = routing2.SolveWithParameters(search_parameters2)
    if solution2:
        routes_solution_2 = print_solution2(data2, manager2, routing2, solution2)
        if visualize == True:
            visualize_function2(points_truck_only, routes_solution_2)
        return routes_solution_2
    else:
        print('No solution found !')
        
############################### FDTSP ALGORITHM begins ###########################
##################################################################################      
dr = 2                                                                    # number of drone
result_final = dict()
for i in range(0,iter_proposed_model):
    xc = rnd.rand(n)*square                                                 
    yc = rnd.rand(n)*square
    coordinate = {'X':[i for i in xc],
                  'Y':[k for k in yc]}
    df = pd.DataFrame(coordinate, columns= ['X', 'Y'])
    df.to_csv ('export_distance.csv', index = False, header=True)             
    df = pd.read_csv('export_distance.csv')                                   
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
    list_dr2 = []
    for k in list_drones:
        if len(k)==0:
            list_dr1.append([])
            list_dr2.append([])
        elif len(k)==1:
            list_dr1.append(k[0])
            list_dr2.append([])
        elif len(k)==2:
            list_dr1.append(k[0])
            list_dr2.append(k[1])
    #TESTING THE ENDURANCE
    d1_tr_test = []
    for i in range(len(list_dr1)):
        if list_dr1[i]!=[]:
            dr1_test = math.sqrt((list_trucks[i][0]-list_dr1[i][0])**2+(list_trucks[i][1]-list_dr1[i][1])**2)
            d1_tr_test.append(dr1_test)
        else:
            d1_tr_test.append(0)
    d2_tr_test = []
    for i in range(len(list_dr2)):
        if list_dr2[i]!=[]:
            dr2_test = math.sqrt((list_trucks[i][0]-list_dr2[i][0])**2+(list_trucks[i][1]-list_dr2[i][1])**2)
            d2_tr_test.append(dr2_test)
        else:
            d2_tr_test.append(0)
    #TESTING THE REPLACED NODES
    d1_tr_test_note = [random.randrange(1, 1000, 1) for i in range(len(d1_tr_test))]
    d2_tr_test_note = [random.randrange(1, 1000, 1) for i in range(len(d2_tr_test))]
    for k in range(len(d1_tr_test)):
        if d1_tr_test[k] > (endurance1*speed_drone)/2:
            d1_tr_test_note[k] = 'NO LAUNCHED'
        else:
            d1_tr_test_note[k] = d1_tr_test[k]
    for k in range(len(d2_tr_test)):
        if d2_tr_test[k] > (endurance1*speed_drone)/2:
            d2_tr_test_note[k] = 'NO LAUNCHED'
        else:
            d2_tr_test_note[k] = d2_tr_test[k]
    ######REPLACING THE REPLACED NODES BY THE TRUCK NODES############
    noLaunched_indices_d1 = [index for index, element in enumerate(d1_tr_test_note) if element == "NO LAUNCHED"]
    noLaunched_indices_d2 = [index for index, element in enumerate(d2_tr_test_note) if element == "NO LAUNCHED"]
    count1 = len(noLaunched_indices_d1)
    count2 = len(noLaunched_indices_d2)
    list_dr1_new = list_dr1.copy()
    list_dr2_new = list_dr2.copy()
    k = 0
    for i, idx in enumerate(noLaunched_indices_d1):
        list_trucks.insert(i, list_dr1_new[idx + k])
        list_dr1_new.insert(i, [])
        list_dr2_new.insert(i, [])
        k +=1
    k = 0
    for i, idx in enumerate(noLaunched_indices_d2):
        list_trucks.insert(i, list_dr2_new[idx + k])
        list_dr1_new.insert(i, [])
        list_dr2_new.insert(i, [])
        k +=1
    for i, idx in enumerate(noLaunched_indices_d1):
        list_dr1_new[idx+count1] = []
        idx +=1
    for i, idx in enumerate(noLaunched_indices_d2):
        list_dr2_new[idx+count2] = []
        idx +=1
    list_trucks = [i for i in list_trucks if i!=[]]
    trucks_new = list_trucks.copy() 
    trucks_new = np.array(trucks_new)                                              
    pd.DataFrame(trucks_new).to_csv("trucks_new.csv", index=False)                
    data = pd.read_csv("trucks_new.csv")                                          
############################# OR-TOOL BEGINS #############################################
    def load_csv():                                                                
        data = pd.read_csv('trucks_new.csv').values
        x, y = list(data[:, 0]), list(data[:, 1])
        return x, y
    x, y = load_csv()
    points_truck_TSP = [[x[i], y[i]] for i in range(len(x))]
    maps_truck_TSP = []
    for i in range(len(points_truck_TSP)):
        for j in range(len(points_truck_TSP)):
            distance_or = math.sqrt((points_truck_TSP[i][0]-points_truck_TSP[j][0])**2+\
                                 (points_truck_TSP[i][1]-points_truck_TSP[j][1])**2)
            maps_truck_TSP.append(distance_or)
    maps_centroids_2 = np.reshape(maps_truck_TSP, (len(points_truck_TSP), len(points_truck_TSP)))
    try:
        routes_solution = main() 
    except:
        main()
    routes_truck = []
    for i in range(len(routes_solution)):
        xx_vehicles = list(routes_solution.values())[i]
        routes_truck.append(xx_vehicles)
    routes_truck[0].pop(-1)
############################## OR-TOOL FINISHED #############################################
############################### COMBINATION BETWEEN TRUCK AND DRONES BEGINs ###########################
    #DISTANCE WITHIN CLUSTER
    list_trucks_new = trucks_new.copy().tolist()
    d1_tr = []
    for i in range(len(routes_truck[0])):
        if list_dr1_new[routes_truck[0][i]]!=[]:
            tempo = math.sqrt((list_trucks_new[routes_truck[0][i]][0]-list_dr1_new[routes_truck[0][i]][0])**2+(list_trucks_new[routes_truck[0][i]][1]-list_dr1_new[routes_truck[0][i]][1])**2)
            d1_tr.append(tempo)
        else:
            d1_tr.append(0)
    d2_tr = []
    for i in range(len(routes_truck[0])):
        if list_dr2_new[routes_truck[0][i]]!=[]:
            tempo2 = math.sqrt((list_trucks_new[routes_truck[0][i]][0]-list_dr2_new[routes_truck[0][i]][0])**2+(list_trucks_new[routes_truck[0][i]][1]-list_dr2_new[routes_truck[0][i]][1])**2)
            d2_tr.append(tempo2)
        else:
            d2_tr.append(0)
    #DISTANCE INTER-CLUSTER
    d1_ntr = []
    for i in range(len(routes_truck[0])-1):
        if list_dr1_new[routes_truck[0][i]]!=[]:
            tempo11 = math.sqrt((list_dr1_new[routes_truck[0][i]][0]-list_trucks_new[routes_truck[0][i+1]][0])**2+(list_dr1_new[routes_truck[0][i]][1]-list_trucks_new[routes_truck[0][i+1]][1])**2)
            d1_ntr.append(tempo11) 
        else:
            d1_ntr.append(0)
    for i in range(len(routes_truck[0])-1,len(routes_truck[0])):
        if list_dr1_new[routes_truck[0][i]]!=[]:
            tempo101 = math.sqrt((list_dr1_new[routes_truck[0][i]][0]-list_trucks_new[routes_truck[0][0]][0])**2+(list_dr1_new[routes_truck[0][i]][1]-list_trucks_new[routes_truck[0][0]][1])**2)
            d1_ntr.append(tempo101)
        else:
            d1_ntr.append(0)        
    d2_ntr = []
    for i in range(len(routes_truck[0])-1):
        if list_dr2_new[routes_truck[0][i]]!=[]:
            tempo12 = math.sqrt((list_dr2_new[routes_truck[0][i]][0]-list_trucks_new[routes_truck[0][i+1]][0])**2+(list_dr2_new[routes_truck[0][i]][1]-list_trucks_new[routes_truck[0][i+1]][1])**2)
            d2_ntr.append(tempo12) 
        else:
            d2_ntr.append(0)
    for i in range(len(routes_truck[0])-1,len(routes_truck[0])):
        if list_dr2_new[routes_truck[0][i]]!=[]:
            tempo102 = math.sqrt((list_dr2_new[routes_truck[0][i]][0]-list_trucks_new[routes_truck[0][0]][0])**2+(list_dr2_new[routes_truck[0][i]][1]-list_trucks_new[routes_truck[0][0]][1])**2)
            d2_ntr.append(tempo102)
        else:
            d2_ntr.append(0)        
    #TOTAL DISTANCE OF EACH DRONE WITHOUT ENDURANCE
    d1_travel = list(map(add,d1_tr,d1_ntr))
    d2_travel = list(map(add,d2_tr,d2_ntr))
    
    d1_tr_new = [random.randrange(1, 1000, 1) for i in range(len(d1_tr))]
    d1_ntr_new = [random.randrange(1, 1000, 1) for i in range(len(d1_ntr))]
    for k in range(len(d1_tr)):
        if d1_travel[k] > endurance1*speed_drone and 2*d1_tr[k] < endurance1*speed_drone:
            d1_tr_new[k] = 'COMEBACK'
            d1_ntr_new[k] = 'COMEBACK' #RETURNED NODE OF 1ST DRONE
        elif d1_travel[k] > endurance1*speed_drone and 2*d1_tr[k] > endurance1*speed_drone:
            d1_tr_new[k] = 'NO LAUNCHED 2'
            d1_ntr_new[k] = 'NO LAUNCHED 2' # CHECKING REPLACED NODE OF 1ST DRONE
        else:
            d1_tr_new[k] = d1_tr[k]
            d1_ntr_new[k] =  d1_ntr[k]
    d2_tr_new = [random.randrange(1, 1000, 1) for i in range(len(d2_tr))]
    d2_ntr_new = [random.randrange(1, 1000, 1) for i in range(len(d2_ntr))]
    for k in range(len(d2_tr)):
        if d2_travel[k] > endurance1*speed_drone and 2*d2_tr[k] < endurance1*speed_drone:
            d2_tr_new[k] = 'COMEBACK'
            d2_ntr_new[k] = 'COMEBACK' #RETURNED NODE OF 2ND DRONE
        elif d2_travel[k] > endurance1*speed_drone and 2*d2_tr[k] > endurance1*speed_drone:
            d2_tr_new[k] = 'NO LAUNCHED 2'
            d2_ntr_new[k] = 'NO LAUNCHED 2' # CHECKING REPLACED NODE OF 2ND DRONE
        else:
            d2_tr_new[k] = d2_tr[k]
            d2_ntr_new[k] =  d2_ntr[k]
    d1_tr_new_temp = [random.randrange(1, 1000, 1) for i in range(len(d1_tr_new))]
    d1_ntr_new_temp = [random.randrange(1, 1000, 1) for i in range(len(d1_ntr_new))]
    for k in range(len(d1_tr_new)):
        if d1_tr_new[k] =='COMEBACK' or d1_tr_new[k]=='NO LAUNCHED 2' :
            d1_tr_new_temp[k] = 0
            d1_ntr_new_temp[k] = 0 
        else:
            d1_tr_new_temp[k] = d1_tr_new[k]
            d1_ntr_new_temp[k] = d1_ntr_new[k]
    d2_tr_new_temp = [random.randrange(1, 1000, 1) for i in range(len(d2_tr_new))]
    d2_ntr_new_temp = [random.randrange(1, 1000, 1) for i in range(len(d2_ntr_new))]
    for k in range(len(d2_tr_new)):
        if d2_tr_new[k] =='COMEBACK' or d2_tr_new[k]=='NO LAUNCHED 2':
            d2_tr_new_temp[k] = 0
            d2_ntr_new_temp[k] = 0 
        else:
            d2_tr_new_temp[k] = d2_tr_new[k]
            d2_ntr_new_temp[k] = d2_ntr_new[k]
    #TRAVEL TIME 
    travel_d1_tr = [60*within_distance1/speed_drone for within_distance1 in d1_tr_new_temp]
    travel_d2_tr = [60*within_distance2/speed_drone for within_distance2 in d2_tr_new_temp]
    #SETUP TIME
    setup_d1 = [random.randrange(1, 1000, 1) for i in range(len(d1_tr_new))]
    for k in range(len(d1_tr_new)):
        if d1_tr_new[k] =='NO LAUNCHED 2' or d1_tr_new[k]== 0:
            setup_d1[k] = 0
        else:
            setup_d1[k] = 1  # 1 MINUTE
    setup_d2 = [random.randrange(1, 1000, 1) for i in range(len(d2_tr_new))]
    for k in range(len(d2_tr_new)):
        if d2_tr_new[k] =='NO LAUNCHED 2' or d2_tr_new[k]== 0:
            setup_d2[k] = 0
        else:
            setup_d2[k] = 1 # 1 MINUTE 
    #SERVICE
    service_d1 = [random.randrange(1, 1000, 1) for i in range(len(d1_tr_new))]
    service_d2 = [random.randrange(1, 1000, 1) for i in range(len(d2_tr_new))]
    for k in range(len(d1_tr_new)):
        if d1_tr_new[k] =='NO LAUNCHED 2' or d1_tr_new[k]== 0:
            service_d1[k] = 0
        else:
            service_d1[k] = 0.5  # 0.5 MINUTES
    for k in range(len(d2_tr_new)):
        if d2_tr_new[k] =='NO LAUNCHED 2' or d2_tr_new[k]== 0:
            service_d2[k] = 0
        else:
            service_d2[k] = 0.5 # 0.5 MINUTES
    #TOTAL TIME for each DRONE WITHIN CLUSTER
    within_d1 = [sum(x) for x in zip(travel_d1_tr,setup_d1,service_d1)]
    within_d2 = [sum(y) for y in zip(travel_d2_tr,setup_d2,service_d2)]
    travel_d1_ntr = [60*within_distance1/speed_drone for within_distance1 in d1_ntr_new_temp]
    travel_d2_ntr = [60*within_distance2/speed_drone for within_distance2 in d2_ntr_new_temp]
    #TOTAL TIME FOR EACH DRONE to THE NEXT TRUCK NODES
    time_d1  = [sum(x) for x in zip(within_d1,travel_d1_ntr)]
    time_d2  = [sum(y) for y in zip(within_d2,travel_d2_ntr)]
    time_drone = [max(time_d1[i],time_d2[i]) for i in range(len(time_d1))]
    waiting_distance_dr1 = [random.randrange(1, 1000, 1) for i in range(len(d1_tr_new))]
    for k in range(len(d1_tr_new)):
        if d1_tr_new[k] == 'COMEBACK':
            waiting_distance_dr1[k] = 2*d1_tr[k]
        else:
            waiting_distance_dr1[k] = 0
    waiting_time_dr1 = [60*within_distance1/speed_drone for within_distance1 in waiting_distance_dr1]
    waiting_distance_dr2 = [random.randrange(1, 1000, 1) for i in range(len(d2_tr_new))]
    for k in range(len(d2_tr_new)):
        if d2_tr_new[k] == 'COMEBACK':
            waiting_distance_dr2[k] = 2*d2_tr[k]
        else:
            waiting_distance_dr2[k] = 0
    waiting_time_dr2 = [60*within_distance2/speed_drone for within_distance2 in waiting_distance_dr2]
    waiting_drone = [max(waiting_time_dr1[i],waiting_time_dr2[i]) for i in range(len(waiting_time_dr1))]
    service_truck = [0.5 if i==0 else 0 for i in waiting_drone]
    distance_trucks = []
    for i in range(len(routes_truck[0])-2):
        temp = math.sqrt((list_trucks_new[routes_truck[0][i]][0]-list_trucks_new[routes_truck[0][i+1]][0])**2+(list_trucks_new[routes_truck[0][i]][1]-list_trucks_new[routes_truck[0][i+1]][1])**2)
        distance_trucks.append(temp)
        if i+1 == len(routes_truck[0])-2: #end node to the start node
            k = math.sqrt((list_trucks_new[routes_truck[0][i+1]][0]-list_trucks_new[routes_truck[0][0]][0])**2+(list_trucks_new[routes_truck[0][0]][1]-list_trucks_new[routes_truck[0][i+2]][1])**2)
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
    zipped2 = zip(list_trucks_new, list_dr1_new,list_dr2_new)
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
    no_service_by_dr2 = d2_tr_new.count('NO LAUNCHED 2')
    SL2 = no_service_by_dr1 + no_service_by_dr2
    SL = SL1+SL2    
    ############################## OR-Tool FOR TRUCK ONLY TSP #####################################                     
    x2, y2 = load_csv2()
    points_truck_only = [[x2[i], y2[i]] for i in range(len(x2))]
    maps_truck_only = []
    for i in range(len(points_truck_only)):
        for j in range(len(points_truck_only)):
            distance_or2 = math.sqrt((points_truck_only[i][0]-points_truck_only[j][0])**2+\
                                (points_truck_only[i][1]-points_truck_only[j][1])**2)
            maps_truck_only.append(distance_or2)
    maps_centroids_3 = np.reshape(maps_truck_only, (len(points_truck_only), len(points_truck_only)))
    try:
        routes_solution_2 = main2()
    except:
        main2()
    routes_truck2 = []
    for i in range(len(routes_solution_2)):
        xx_vehicles2 = list(routes_solution_2.values())[i]
        routes_truck2.append(xx_vehicles2)
    routes_truck2[0].pop(-1)
    distance_trucks2 = []
    for i in range(len(routes_truck2[0])-2):
        temp2 = math.sqrt((points[routes_truck2[0][i]][0]-points[routes_truck2[0][i+1]][0])**2+(points[routes_truck2[0][i]][1]-points[routes_truck2[0][i+1]][1])**2)
        distance_trucks2.append(temp2)
        if i+1 == len(routes_truck2[0])-2: #end node to the start node
            k2 = math.sqrt((points[routes_truck2[0][i+1]][0]-points[routes_truck2[0][0]][0])**2+(points[routes_truck2[0][i+1]][1]-points[routes_truck2[0][0]][1])**2)
            distance_trucks2.append(k2)
    speed_truck2 = 35
    travel_truck2  = [60*each_distance/speed_truck2 for each_distance in distance_trucks2]
    service_truck2 = [0.5 if i>0 else 0 for i in travel_truck2] 
    time_truck2 = [sum(y) for y in zip(travel_truck2,service_truck2)]
    total_time_TSP = np.cumsum(time_truck2)
    m_TSP = [index for index,value in enumerate(total_time_TSP) if value > 240] 
    SL_TSP = len(m_TSP) #number of points cannot be served due to exceeding 4 hours
    totaltime_only_truck = sum(time_truck2)
    saving = totaltime_only_truck - t
    saving
# ############################## PERFORMING EVALUATION #############################################    
    saving = totaltime_only_truck - t
    total_waiting_time = sum(list( map(add, waiting_drone, waiting_time_next_node)))
    count_deli1 = sum(map(lambda x : x != 'NO LAUNCHED 2' and x !=0, d1_tr_new)) 
    count_deli2 = sum(map(lambda x : x != 'NO LAUNCHED 2' and x !=0, d2_tr_new))
    count_deli=count_deli1+count_deli2
    dd = count_deli/n
    drone_should_be_used = len(routes_truck[0])*dr
    count_drone1_used_in_fact = sum([1 if x!=0 else 0 for x in d1_tr_new])
    count_drone2_used_in_fact = sum([1 if x!=0 else 0 for x in d2_tr_new])
    count_drones_used_in_fact = count_drone1_used_in_fact + count_drone2_used_in_fact 
    drones_usage = count_drones_used_in_fact/drone_should_be_used
    for j in d1_tr_test_note:
        count_dr1 = d1_tr_test_note.count('NO LAUNCHED') 
    for j in d2_tr_test_note:
        count_dr2 = d2_tr_test_note.count('NO LAUNCHED')
    count_total = count_dr1+count_dr2 
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
                result_final[element].append('1010_80_30_40')
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
                result_final[element].append('1010_80_30_40')
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
df.to_csv (r'export_result-Or-tool.csv', index = False, header=True)
print('time consuming for FDTSP:',time.time() - start_time,'seconds')