"""Distance matrix + capacitated VRP (OR-Tools) + greedy baseline."""
import math
import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import config as C


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dl = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def distance_matrix_m(lat, lon, road_factor=C.ROAD_FACTOR):
    """Full pairwise road-distance estimate in metres (int). lat/lon include depot at index 0.
    To use real OSM roads instead, see osm_distance_matrix() below (needs internet + osmnx)."""
    lat = np.asarray(lat)[:, None]; lon = np.asarray(lon)[:, None]
    d = haversine_km(lat, lon, lat.T, lon.T) * road_factor * 1000.0
    return np.rint(d).astype(np.int64)


def osm_distance_matrix(lat, lon, place="Bengaluru, Karnataka, India"):
    """OPTIONAL (run on your laptop): shortest-path distances on the real OSM road graph."""
    import osmnx as ox, networkx as nx  # pip install osmnx
    G = ox.graph_from_place(place, network_type="drive")
    nodes = ox.distance.nearest_nodes(G, X=list(lon), Y=list(lat))
    n = len(nodes)
    D = np.zeros((n, n), dtype=np.int64)
    for i, s in enumerate(nodes):
        lengths = nx.single_source_dijkstra_path_length(G, s, weight="length")
        for j, t in enumerate(nodes):
            D[i, j] = int(lengths.get(t, 1e7))
    return D


def solve_vrp(dist, demand, capacity=C.TRUCK_CAPACITY_L, max_dist_m=int(C.MAX_ROUTE_KM * 1000),
              time_limit_s=0.5, fixed_cost_m=5000):
    """dist: (n+1)x(n+1) int metres, node 0 = depot. demand: length n+1 ints (demand[0]=0).
    Returns dict(routes=[[node,...],...], distance_m=int, dropped=[nodes])."""
    n = len(dist)
    if n <= 1:
        return dict(routes=[], distance_m=0, dropped=[])
    total = int(sum(demand))
    n_veh = int(math.ceil(total / capacity * 1.25)) + 1
    n_veh = min(n_veh, n - 1)
    mgr = pywrapcp.RoutingIndexManager(n, n_veh, 0)
    rt = pywrapcp.RoutingModel(mgr)
    dl = dist.tolist()

    def d_cb(i, j):
        return dl[mgr.IndexToNode(i)][mgr.IndexToNode(j)]
    cb = rt.RegisterTransitCallback(d_cb)
    rt.SetArcCostEvaluatorOfAllVehicles(cb)
    rt.AddDimension(cb, 0, max_dist_m, True, "Dist")
    dem_cb = rt.RegisterUnaryTransitCallback(lambda i: int(demand[mgr.IndexToNode(i)]))
    rt.AddDimensionWithVehicleCapacity(dem_cb, 0, [int(capacity)] * n_veh, True, "Cap")
    rt.SetFixedCostOfAllVehicles(int(fixed_cost_m))
    for node in range(1, n):
        rt.AddDisjunction([mgr.NodeToIndex(node)], 10_000_000)
    p = pywrapcp.DefaultRoutingSearchParameters()
    p.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    p.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    p.time_limit.FromMilliseconds(int(time_limit_s * 1000))
    sol = rt.SolveWithParameters(p)
    if sol is None:
        return solve_greedy(dist, demand, capacity, max_dist_m)
    routes, tot, visited = [], 0, set()
    for v in range(n_veh):
        idx = rt.Start(v)
        r = []
        while not rt.IsEnd(idx):
            node = mgr.IndexToNode(idx)
            if node != 0:
                r.append(node); visited.add(node)
            nxt = sol.Value(rt.NextVar(idx))
            tot += dl[mgr.IndexToNode(idx)][mgr.IndexToNode(nxt)]
            idx = nxt
        if r:
            routes.append(r)
    dropped = [k for k in range(1, n) if k not in visited]
    return dict(routes=routes, distance_m=int(tot), dropped=dropped)


def solve_greedy(dist, demand, capacity=C.TRUCK_CAPACITY_L, max_dist_m=int(C.MAX_ROUTE_KM * 1000)):
    """Nearest-neighbour heuristic with capacity & range limits (baseline for solver comparison)."""
    n = len(dist)
    left = set(range(1, n)); routes = []; tot = 0
    while left:
        cur, load, run, r = 0, 0, 0, []
        while True:
            cand = [k for k in left if load + demand[k] <= capacity and
                    run + dist[cur][k] + dist[k][0] <= max_dist_m]
            if not cand:
                break
            k = min(cand, key=lambda x: dist[cur][x])
            run += dist[cur][k]; load += demand[k]; cur = k; r.append(k); left.remove(k)
        if not r:                      # unreachable bin -> drop
            k = left.pop(); continue
        run += dist[cur][0]; tot += run; routes.append(r)
    return dict(routes=routes, distance_m=int(tot), dropped=[])
