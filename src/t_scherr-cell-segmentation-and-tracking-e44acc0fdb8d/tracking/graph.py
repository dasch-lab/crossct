"""Utilities to construct a graph from vertices and find the best paths through the graph
 using coupled minimum cost flow."""
import re
from itertools import chain
from itertools import combinations, product

import cvxopt
import numpy as np
from cvxopt.glpk import ilp as int_lin_prog


class Edge:
    """Defines a directed edge between a pair of vertices."""
    def __init__(self, start_vertex, end_vertex, cost, edge_capacity):
        """
        Creates a directed edge between two vertices.
        Args:
            start_vertex: the Vertex instance the edge starts from
            end_vertex: the Vertex instance the edge ends at
            cost: cost of the edge
            edge_capacity: capacity of the edge (see min cost flow programming)
        """
        self.check_edge(start_vertex, end_vertex)
        self.cost = cost
        self.capacity = edge_capacity
        self.start_vertex = start_vertex
        self.end_vertex = end_vertex
        self._id = self.start_vertex.id.string_id() + 'x' + self.end_vertex.id.string_id()

    def check_edge(self, start_vertex, end_vertex):
        """Checks that edges are directed forwards in time and no self connections exist."""
        assert start_vertex != end_vertex, 'No self loops allowed'
        assert start_vertex.id.time <= end_vertex.id.time, 'no backwards connections allowed'

    def __eq__(self, other):
        """Defines equality of edges."""
        if isinstance(other, Edge):
            if self.start_vertex.id == other.start_vertex.id and self.end_vertex.id == other.end_vertex.id:
                return True
        return False

    def string_id(self):
        """Returns the edge id as a string."""
        return self._id


class VertexId:
    """Provides a unique id for a vertex."""
    def __init__(self, time, index_id):
        """
        A unique identifier for each vertex in a graph.
        Args:
            time: an integer defining the time point of the vertex in the graph
            index_id: identifier of the vertex within the considered time step
        """
        assert isinstance(time, int), 'time should be an int'
        self.time = time
        self.index_id = index_id
        self._id = str(self.time) + '_' + str(self.index_id)

    def __eq__(self, other):
        """Defines equality of vertex ids."""
        if self.time == other.time and self.index_id == other.index_id:
            return True
        return False

    def string_id(self):
        """Returns the vertex if as a string."""
        return self._id


class Vertex:
    """Defines vertices in a graph."""
    def __init__(self, time, index_id, edge_capacity=1, features=None):
        """
        Initialises a "normal" vertex in a graph.
        Args:
            time: an int defining the time point where the vertex exists
            index_id: index of the vertex within the considered time point
            edge_capacity: an int specifying the maximum capacity of the in/out edges
            features: a np.array of features
        """
        self.id = VertexId(time, index_id)
        self.type = 'normal'
        self.features = np.array(features)
        self.edge_capacity = edge_capacity  # maximum capacity of edges ending at vertex
        self.in_edges = {}
        self.out_edges = {}
        self.neighbours_current = set()
        self.neighbours_next = set()
        self.position = []

    def __eq__(self, other):
        """Defines equality of two vertices."""
        if self.id == other.id:
            if self.type == other.type:
                return True
        return False

    def add_edge(self, edge):
        """Adds incoming/ outgoing edges to vertex."""
        if edge.start_vertex == self:
            assert edge.string_id() not in self.out_edges.keys(), 'edge with same id already added to vertex'
            self.out_edges[edge.string_id()] = edge
        elif edge.end_vertex == self:
            assert edge.string_id() not in self.in_edges, \
                'edge {name} already added to vertex'.format(name=edge.string_id())
            self.in_edges[edge.string_id()] = edge


class SplitVertex(Vertex):
    """Defines a split vertex in a graph (allows modelling of splitting objects(vertices) over time.)"""
    def __init__(self, time, index_id, edge_capacity=1, features=None):
        """Initialises a split vertex."""
        super().__init__(time, index_id, edge_capacity, features)
        self.type = 'split'


class AppearVertex(Vertex):
    """Defines an appear vertex in a graph (allows modelling of appearing objects(vertices) over time.)"""
    def __init__(self, time, index_id, edge_capacity, features=None):
        """Initialises an apper vertex."""
        super().__init__(time, index_id, edge_capacity, features)
        self.type = 'appear'


class DeleteVertex(Vertex):
    """Defines a delete vertex in a graph (allows modelling of disappearing objects(vertices) over time.)"""
    def __init__(self, time, index_id, edge_capacity, features=None):
        """Initialises a disappear vertex."""
        super().__init__(time, index_id, edge_capacity, features)
        self.type = 'delete'


class SourceVertex(Vertex):
    """Defines the source vertex which is the source of all flow."""
    def __init__(self, time, index_id, edge_capacity, features=None):
        """Initialises source vertex."""
        super().__init__(time, index_id, edge_capacity, features)
        self.type = 'source'


class SinkVertex(Vertex):
    """Defines the sink vertex which is the sink of all flow."""
    def __init__(self, time, index_id, edge_capacity, features=None):
        """Initialises sink vertex."""
        super().__init__(time, index_id, edge_capacity, features)
        self.type = 'sink'


class VerticesDict:
    """Contains all vertices."""
    def __init__(self):
        """
        Initialises vertices dict.
        data: a dict storing all vertices by their unique identifier
        data_time_points: a list storing all vertices by the time step they exist in
        """

        self.data = {}
        self.data_time_points = {}

    def __getitem__(self, data_index):
        """Returns a single vertex or all vertices at a time point."""
        if isinstance(data_index, tuple):
            data_key = VertexId(*data_index).string_id()
            assert data_key in self.data.keys(), 'vertex id {vertex_id} not in dictionary'.format(vertex_id=data_index)
            return self.data[data_key]
        if isinstance(data_index, str):
            assert data_index in self.data.keys(), 'vertex id {vertex_id} not in dictionary'.format(vertex_id=data_index)
            return self.data[data_index]
        assert isinstance(data_index, int), '{data_index} is not an integer time point'.format(data_index=data_index)
        assert data_index in self.data_time_points.keys(), \
            'time point {data_index} not in time steps. ' \
            'Available time point: {time_points}'.format(data_index=data_index,
                                                         time_points=self.data_time_points.keys())
        return self.data_time_points[data_index]

    def add_vertex(self, vertex):
        """Adds a vertex to the vertices dict."""
        assert vertex.id.string_id() not in self.data.keys(), 'vertex with same id already exists'
        self.data.update({vertex.id.string_id(): vertex})
        if vertex.id.time not in self.data_time_points.keys():
            self.data_time_points[vertex.id.time] = [vertex]
        else:
            self.data_time_points[vertex.id.time].append(vertex)

    def get_time_steps(self):
        """Returns all time steps vertices have been added to."""
        return sorted(self.data_time_points.keys())

    def get_vertices_by_type(self, vertex_type, time=None):
        """Returns all vertices of a specific type and/or at a time point"""
        if time is None:
            return filter(lambda x: x.type == vertex_type, self.data.values())
        return filter(lambda x: x.type == vertex_type, self.data_time_points[time])

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        self.data_keys = iter(list(self.data.keys()))
        return self

    def __next__(self):
        return self.data[next(self.data_keys)]

    def __contains__(self, item):
        return item in self.data.keys()


class SparseGraph:
    """Defines a graph which is not fully connected. Optimal paths through the graph are found by
    integer linear programming."""
    def __init__(self, cut_off_distance, allow_cell_division=True):
        """
                Constructs a graph from a set of vertices and finds optimal paths between vertices by minimizing
                a coupled minimum cost flow problem
                vertices: contains all vertices of the graph
                edges: contains the edges between the vertices
                valid_edges: defines which kinds of directed edges are allowed in the graph
                distance_threshold: define threshold for minimum cost of appearance/ deletion operations
                """
        self.vertices = VerticesDict()
        self.edges = {}
        # directed edges start_vertex: [possible end vertices]
        self.valid_edges = {'source': ['appear', 'normal'],
                            'sink': [],
                            'split': ['normal'],
                            'appear': ['split', 'normal', 'delete'],
                            'delete': ['sink'],
                            'normal': ['normal', 'split', 'delete', 'sink']}
        self.result = None  # optimization result
        self.distance_threshold = cut_off_distance
        self.cell_division = allow_cell_division

    def get_vertex(self, vertex_id):
        """Returns a vertex instance given its string id."""
        return self.vertices[vertex_id]

    def add_vertex(self, vertex):
        """Adds a vertex to the graph."""
        self.vertices.add_vertex(vertex)

    def get_matching_candidates(self):
        """"""
        for vertex in self.vertices:
            vertex.neighbours_next.update({edge.end_vertex.id.string_id()
                                           for edge in vertex.out_edges.values()})
            for edge in vertex.out_edges.values():
                edge.end_vertex.neighbours_current.update(vertex.neighbours_next)

    def construct_graph(self):
        """Sets up the graph from a set of provided vertices for min cost flow optimization."""
        # add edges between normal vertices
        self.get_matching_candidates()
        all_time_points = self.vertices.get_time_steps()
        assert all_time_points, 'empty vertices list'
        assert len(set(all_time_points)) > 1, 'all vertices in same time step'
        start_time = all_time_points[0]
        end_time = all_time_points[-1]

        print('Add vertices to graph')

        for i, time_point in enumerate(all_time_points[1:]):
            normal_vertices_current_step = list(self.vertices.get_vertices_by_type('normal', all_time_points[i]))
            normal_vertices_next_step = list(self.vertices.get_vertices_by_type('normal', time_point))

            # add appear vertex
            appear_vertex = AppearVertex(all_time_points[i], 'a', len(normal_vertices_next_step))
            appear_vertex.neighbours_next = {vertex.id.string_id()
                                             for vertex in normal_vertices_next_step}
            self.add_vertex(appear_vertex)
            # add delete vertex
            delete_vertex = DeleteVertex(time_point, 'd', len(normal_vertices_current_step))
            for vertex in normal_vertices_current_step:
                vertex.neighbours_next.add(delete_vertex.id.string_id())
            self.add_vertex(delete_vertex)
            appear_vertex.neighbours_next.add(delete_vertex.id.string_id())

            # add split vertices at time step t+1
            if self.cell_division:
                vertex_triplets = chain(*[product([vertex], combinations(vertex.neighbours_next, 2))
                                          for vertex in normal_vertices_current_step])
                for mother_vertex, daughter_vertices_ids in vertex_triplets:
                    daughter_vertex_1 = self.vertices[daughter_vertices_ids[0]]
                    daughter_vertex_2 = self.vertices[daughter_vertices_ids[1]]
                    # daughter vertices are mutual neighbours
                    if daughter_vertices_ids[0] in daughter_vertex_2.neighbours_current \
                            and daughter_vertices_ids[1] in daughter_vertex_1.neighbours_current:
                        out_vertex_ids = sorted([daughter_vertex_1.id.string_id(), daughter_vertex_2.id.string_id()])
                        split_time = min(daughter_vertex_1.id.time, daughter_vertex_2.id.time)
                        split_vertex = SplitVertex(split_time, 's_' + out_vertex_ids[0]
                                                   + '__' + out_vertex_ids[1], 1)
                        if split_vertex.id.string_id() not in self.vertices:
                            # add out edges from split vertex to normal vertices
                            self.add_vertex(split_vertex)
                            self.construct_edge(appear_vertex, split_vertex)
                            self.construct_edge(split_vertex, daughter_vertex_1)
                            self.construct_edge(split_vertex, daughter_vertex_2)
                        else:
                            split_vertex = self.vertices[split_vertex.id.string_id()]

                        self.construct_edge(mother_vertex, split_vertex)
        print('Add remaining edges')
        vertex_pairs = chain(*[product([vertex], vertex.neighbours_next)
                               for vertex in self.vertices])
        for start_vertex, end_vertex_id in vertex_pairs:
            self.construct_edge(start_vertex, self.vertices[end_vertex_id])

        print('Add sink and source vertex to graph')
        normal_vertices = self.vertices.get_vertices_by_type('normal')

        n_normal_vertices = sum(1 for _ in normal_vertices)
        source_vertex = SourceVertex(start_time - 1, 'source', n_normal_vertices)
        sink_vertex = SinkVertex(end_time + 1, 'sink', n_normal_vertices)
        self.add_vertex(source_vertex)
        self.add_vertex(sink_vertex)

        for appear_vertex in self.vertices.get_vertices_by_type('appear'):
            self.construct_edge(source_vertex, appear_vertex)
        for delete_vertex in self.vertices.get_vertices_by_type('delete'):
            self.construct_edge(delete_vertex, sink_vertex)

        # connect all vertices of first time/last time point to source/ sink
        for vertex in self.vertices[start_time]:
            self.construct_edge(source_vertex, vertex)
        for vertex in self.vertices[end_time]:
            self.construct_edge(vertex, sink_vertex)

    def compute_constraints(self):
        """Sets up the equality constraints for optimization."""
        print('Set up constraints')
        flow_variables = {key: i
                          for i, key in enumerate(self.edges.keys())}
        # define equality constraints A_eq*x = b_eq
        A_eq = []
        b_eq = []
        ####################
        # flow conservation constraints
        # ##################
        # flow_in == flow_out for all non source/sink vertices
        A_flow_conservation = []
        b_flow_conservation = []
        for vertex in self.vertices:
            if not (vertex.type == 'sink' or vertex.type == 'source'):
                in_keys = vertex.in_edges.keys()
                out_keys = vertex.out_edges.keys()
                constraint = dict()
                for edge_key in in_keys:
                    constraint[flow_variables[edge_key]] = 1
                for edge_key in out_keys:
                    constraint[flow_variables[edge_key]] = -1
                A_flow_conservation.append(constraint)
                b_flow_conservation.append(0)
        A_eq.extend(A_flow_conservation)
        b_eq.extend(b_flow_conservation)

        # flow out of source == n units
        source_vertex = list(self.vertices.get_vertices_by_type('source'))[0]
        A_source_constraint = dict()
        for edge_key in source_vertex.out_edges.keys():
            A_source_constraint[flow_variables[edge_key]] = 1
        b_source_constraint = source_vertex.edge_capacity
        A_eq.append(A_source_constraint)
        b_eq.append(b_source_constraint)

        ##################
        # constraint input==1, output==1 to all normal vertices
        ##################
        A_input_constraint = []
        b_input_constraint = []
        for vertex in self.vertices.get_vertices_by_type('normal'):
            constraint = dict()
            for edge_key in vertex.in_edges.keys():
                constraint[flow_variables[edge_key]] = 1
            A_input_constraint.append(constraint)
            b_input_constraint.append(vertex.edge_capacity)
            constraint = dict()
            for edge_key in vertex.out_edges.keys():
                constraint[flow_variables[edge_key]] = 1
            A_input_constraint.append(constraint)
            b_input_constraint.append(vertex.edge_capacity)
        A_eq.extend(A_input_constraint)
        b_eq.extend(b_input_constraint)

        ###############################
        # n input/ output units to appear/disappear
        ###############################
        A_input_constraint = []
        b_input_constraint = []
        for vertex in self.vertices.get_vertices_by_type('appear'):
            constraint = dict()
            for edge_key in vertex.out_edges.keys():
                constraint[flow_variables[edge_key]] = 1
            A_input_constraint.append(constraint)
            b_input_constraint.append(vertex.edge_capacity)
        A_eq.extend(A_input_constraint)
        b_eq.extend(b_input_constraint)

        A_input_constraint = []
        b_input_constraint = []
        for vertex in self.vertices.get_vertices_by_type('delete'):
            constraint = dict()
            for edge_key in vertex.in_edges.keys():
                constraint[flow_variables[edge_key]] = 1
            A_input_constraint.append(constraint)
            b_input_constraint.append(vertex.edge_capacity)
        A_eq.extend(A_input_constraint)
        b_eq.extend(b_input_constraint)

        ########################
        # coupling of split vertices with appear vertices
        #########################
        A_split_coupling_constraint = []
        b_split_coupling_constraint = []
        for vertex in self.vertices.get_vertices_by_type('split'):
            in_edges = vertex.in_edges
            out_edges = vertex.out_edges
            constraint = dict()
            appear_vertex_key = None
            for edge_key in in_edges.keys():
                # sum in edges == flow appear vertex
                factor = 1
                if isinstance(in_edges[edge_key].start_vertex, AppearVertex):
                    factor = -1
                    assert appear_vertex_key is None, 'multiple appear vertices mapped with same split vertex'
                    appear_vertex_key = edge_key
                constraint[flow_variables[edge_key]] = factor

            A_split_coupling_constraint.append(constraint)
            b_split_coupling_constraint.append(0)

            for edge_key in out_edges.keys():
                constraint = dict()
                constraint[flow_variables[edge_key]] = 1
                constraint[flow_variables[appear_vertex_key]] = -1
                A_split_coupling_constraint.append(constraint)
                b_split_coupling_constraint.append(0)
        A_eq.extend(A_split_coupling_constraint)
        b_eq.extend(b_split_coupling_constraint)

        return A_eq, b_eq

    def compute_edge_cost(self, start_vertex, end_vertex):
        """Computes the edge cost between two vertices."""

        if start_vertex.type == 'appear' and end_vertex.type == 'normal':
            return self.calc_vertex_appear_cost(start_vertex, end_vertex)

        if start_vertex.type == 'normal' and end_vertex.type == 'delete':
            return self.calc_vertex_delete_cost(start_vertex, end_vertex)

        if start_vertex.type == 'normal' and end_vertex.type == 'split':
            # get other split cell
            return self.calc_vertex_split_cost(start_vertex, end_vertex)

        if start_vertex.type == 'normal' and end_vertex.type == 'normal':
            return compute_distance(start_vertex.features[0], end_vertex.features[0])
        # all other edges have zero cost
        return 0

    def calc_vertex_appear_cost(self, start_vertex, end_vertex):
        return 0  # free adding of vertices at t+1

    def calc_vertex_delete_cost(self, start_vertex, end_vertex):
        return self.distance_threshold  # costly deletion of vertices at t

    def calc_vertex_split_cost(self, start_vertex, end_vertex):
        """Computes cost between mother cell at t and potential daughter cells at t+1."""
        # end vertex is in this case the split vertex with 2 out going edges
        daughter_vertices = [out_edge.end_vertex
                             for out_edge in end_vertex.out_edges.values()
                             if out_edge.end_vertex.type == 'normal']
        assert len(daughter_vertices) == 2, ' each split vertex has exactly 2 output vertices'
        mean_position = 0.5 * (np.array(daughter_vertices[0].features[0]) +
                               np.array(daughter_vertices[1].features[0]))
        total_cost = compute_distance(start_vertex.features[0], mean_position)
        dist_daughters = compute_distance(daughter_vertices[0].features[0], daughter_vertices[1].features[0])
        if daughter_vertices[0].features[1] > daughter_vertices[1].features[1]:
            ratio_daughters = daughter_vertices[1].features[1] / daughter_vertices[0].features[1]
        else:
            ratio_daughters = daughter_vertices[0].features[1] / daughter_vertices[1].features[1]

        ratio_d1_mother = daughter_vertices[0].features[1] / start_vertex.features[1]
        ratio_d2_mother = daughter_vertices[1].features[1] / start_vertex.features[1]

        # split condition: similar size of daughter cells,
        # combined size of daughter cells close to size of mother cell
        # distance of daughter cells reasonable small
        if ratio_daughters > 0.5 and (ratio_d1_mother + ratio_d2_mother) < 1.2 \
                and dist_daughters < 2*np.sqrt(start_vertex.features[1]):
            return total_cost
        return self.distance_threshold*10

    def solve_matching_problem(self):
        eq_constraints, b_eq = self.compute_constraints()
        print('Add Equations')
        flow_variable_names = {}
        for i, edge_key in enumerate(self.edges.keys()):
            flow_variable_names[i] = edge_key
        # total number of flow variables
        n_flow_vars = len(flow_variable_names)

        costs = cvxopt.matrix([self.edges[flow_variable_names[id_flow_var]].cost
                               for id_flow_var in flow_variable_names])

        capacity = [self.edges[flow_variable_names[id_flow_var]].capacity
                    for id_flow_var in flow_variable_names]

        # inequality constraints : h_lower <= G_ineq * x <= h_upper
        upper_border = cvxopt.spmatrix(1, range(n_flow_vars), range(n_flow_vars), (n_flow_vars, n_flow_vars))
        h_upper = capacity
        G_ineq = list()
        G_ineq.append(upper_border)
        lower_border = cvxopt.spmatrix(-1, range(n_flow_vars), range(n_flow_vars), (n_flow_vars, n_flow_vars))
        h_lower = [0] * n_flow_vars
        h_ineq = h_upper[:]
        h_ineq.extend(h_lower)
        G_ineq.append(lower_border)
        G_ineq = cvxopt.sparse(G_ineq)
        h_ineq = cvxopt.matrix(h_ineq, tc='d')

        # equality constraints: A_eq*x = b_eq
        data = list()
        row_index = list()
        col_index = list()
        for i, coefficients in enumerate(eq_constraints):
            index, factor = list(zip(*[(id_flow_var, coeff) for id_flow_var, coeff in coefficients.items()]))
            data.extend(factor)
            col_index.extend(index)
            row_id = [i] * len(index)
            row_index.extend(row_id)
        A_eq = cvxopt.spmatrix(data, row_index, col_index, (len(eq_constraints), n_flow_vars))
        b_eq = cvxopt.matrix(b_eq, tc='d')

        integer_vars = set(flow_variable_names.keys())
        print('Optimization')
        status, x = int_lin_prog(costs, G_ineq, h_ineq, A_eq, b_eq, integer_vars)
        assert np.all(abs(np.array(x) - np.array(x).astype(int)) < 1e-16), 'solution is not an integer solution'
        assert status == 'optimal', 'non optimal status'
        # get optimization result: flow of n units over edges
        self.result = {flow_variable_names[id_flow_var]: int(x_val)
                       for id_flow_var, x_val in enumerate(x)}
        return self.result

    def construct_edge(self, start_vertex, end_vertex):
        """Constructs edges between two vertices."""
        if end_vertex.type not in self.valid_edges[start_vertex.type]:
            return

        if isinstance(start_vertex, SplitVertex):
            # split vertex output: exactly 2 vertices
            # name of the split vertex constructed from the 2 vertices the split vertex outputs to
            if not end_vertex.id.string_id() in start_vertex.id.string_id():
                return

        edge = Edge(start_vertex, end_vertex, self.compute_edge_cost(start_vertex, end_vertex),
                    min(start_vertex.edge_capacity, end_vertex.edge_capacity))
        if edge.string_id() not in self.edges.keys():
            self.edges[edge.string_id()] = edge
            start_vertex.add_edge(edge)
            end_vertex.add_edge(edge)

    def print_graph(self):
        """Prints for each vertex the vertices which provide input to the vertex/ the vertex outputs to."""
        for vertex in self.vertices:
            print('___' * 4)
            print('vertex:', vertex.id.string_id())
            in_vertex = [e.start_vertex.id.string_id() for e in vertex.in_edges.values()]
            out_vertex = [e.end_vertex.id.string_id() for e in vertex.out_edges.values()]
            print('in:', in_vertex)
            print('out:', out_vertex)

    def calc_trajectories(self):
        """Computes the trajectories given by the flow variables of the edges."""
        trajectories = {}
        # find connected vertices: flow over their mutual edge is > 0
        flow_edges = list(filter(lambda x: self.result[x] > 0, self.result.keys()))
        flow_edges.sort(key=lambda x: int(re.match(r'-*\d+', x).group()))
        for edge_name in flow_edges:
            # source -> normal : new vertex
            # appear -> normal : new vertex
            # appear -> delete : ignore
            # delete -> source: ignore
            # appear -> split : ignore
            # normal -> split : create new trajectory (start_vertex: n), add successors
            # normal -> delete: ignore
            # normal -> normal: add to trajectory

            start_vertex = self.edges[edge_name].start_vertex
            end_vertex = self.edges[edge_name].end_vertex
            if start_vertex.type in ['source', 'appear'] and end_vertex.type == 'normal':
                track_id = len(trajectories)
                end_vertex.track_id = track_id
                trajectories[len(trajectories)] = {'predecessor': [start_vertex], 'track': [end_vertex],
                                                   'successor': []}
            elif start_vertex.type == 'split' and end_vertex.type == 'normal':
                # create new trajectory
                mother_vertex = [in_edge.start_vertex
                                 for in_edge in start_vertex.in_edges.values()
                                 if self.result[in_edge.string_id()] > 0 and in_edge.start_vertex.type == 'normal']
                assert len(mother_vertex) == 1, ' split vertex has exactly one normal vertex predecessor'
                end_vertex.track_id = len(trajectories)
                trajectories[len(trajectories)] = {'predecessor': mother_vertex,
                                                   'track': [end_vertex], 'successor': []}
                for m_vertex in mother_vertex:
                    trajectories[m_vertex.track_id]['successor'].append(end_vertex)

            elif start_vertex.type == 'normal' and end_vertex.type == 'normal':
                end_vertex.track_id = start_vertex.track_id
                trajectories[start_vertex.track_id]['track'].append(end_vertex)

        return trajectories


def get_id_representation(trajectories):
    string_trajectories = {}
    for key, track in trajectories.items():
        string_trajectories[key] = {}
        for sub_key, vertices in track.items():
            string_ids = [vertex.id.string_id() for vertex in vertices]
            string_trajectories[key][sub_key] = string_ids
            if sub_key == 'predecessor':
                string_trajectories[key]['pred_track_id'] = sorted([vertex.track_id
                                                                    for vertex in vertices
                                                                    if hasattr(vertex, 'track_id')])
            if sub_key == 'successor':
                string_trajectories[key]['succ_track_id'] = sorted([vertex.track_id
                                                                    for vertex in vertices
                                                                    if hasattr(vertex, 'track_id')])
    return string_trajectories


def compute_distance(vec_a, vec_b):
    # note : for large vector components overflow possible alternative for |x| > |y|: |x|*sqrt(1 + (y/x)**2)
    return np.linalg.norm(vec_a-vec_b)


def graph_tracking(tracks, matching_candidates, cutoff_distance=float('inf'), allow_cell_division=True):
    """
    Computes for a set of tracks with a set of potential matching candidates the best matching
        based on coupled minimum cost flow.
    Args:
        tracks: a dict containing the tracking id of a track and its features {track_id: track_features}
        matching_candidates: a dict containing for each track the set of potential matching candidates
         and their features {track_id: {candidate_id: candidate_features}}}
        cutoff_distance:
        allow_cell_division:

    Returns:

    """
    output = {}
    if matching_candidates:
        # construct graph
        graph = SparseGraph(cutoff_distance, allow_cell_division=allow_cell_division)
        for track_id in matching_candidates.keys():
            track_features = tracks[track_id]
            vertex_id = VertexId(0, track_id).string_id()
            if vertex_id not in graph.vertices:
                vertex = Vertex(0, track_id, features=track_features)
                graph.add_vertex(vertex)
            else:
                vertex = graph.vertices[vertex_id]

            for candidate in matching_candidates[track_id].items():
                candidate_id, candidate_features = candidate
                n_vertex_id = VertexId(1, candidate_id).string_id()
                if n_vertex_id not in graph.vertices:
                    n_vertex = Vertex(1, candidate_id, features=candidate_features)
                    graph.add_vertex(n_vertex)
                else:
                    n_vertex = graph.vertices[n_vertex_id]
                graph.construct_edge(vertex, n_vertex)
                #vertex.neighbours_next.update(n_vertex.id.string_id())
        graph.construct_graph()
        graph.solve_matching_problem()
        track_data = graph.calc_trajectories()
        track_data = get_id_representation(track_data)
        # extract for each track id the best matching candidates
        for _, single_track_data in track_data.items():
            start_id = single_track_data['track'][0].split('_')[-1]
            if 'source' in single_track_data['predecessor'][0]:
                if len(single_track_data['track']) == 2:
                    # no cell division
                    end_id = single_track_data['track'][1].split('_')[-1]
                    output[int(start_id)] = tuple([int(end_id)])

                elif single_track_data['successor']:
                    successors = tuple([int(succ.split('_')[-1]) for succ in single_track_data['successor']])
                    output[int(start_id)] = successors
                else:
                    output[int(start_id)] = ()
    return output


if __name__ == '__main__':
    GRAPH = SparseGraph(20)
    DUMMY_POSITION = {1: [32, 30], 2: [35, 40]}
    DUMMY_NEIGHBORS = {1: {2: [29, 29], 3: [31, 31]}, 2: {2: [29, 29], 4: [40, 50], 5: [50, 50]}}
    VERTICES = list()

    VERTICES.append(Vertex(0, 1, features=(10, 10)))
    VERTICES.append(Vertex(0, 2, features=(18, 18)))
    VERTICES.append(Vertex(1, 1, features=(14, 17)))

    for t_id, position in DUMMY_POSITION.items():
        new_vertex = Vertex(0, t_id, features=position)
        if new_vertex.id.string_id() not in GRAPH.vertices:
            GRAPH.add_vertex(new_vertex)
        for neighbor in DUMMY_NEIGHBORS[t_id].items():
            neighbor_id, neighbor_position = neighbor
            v = Vertex(1, neighbor_id, features=neighbor_position)
            if v.id.string_id() not in GRAPH.vertices:
                GRAPH.add_vertex(v)
            GRAPH.construct_edge(new_vertex, v)

    GRAPH.construct_graph()
    RESULT = GRAPH.solve_matching_problem()
    print(get_id_representation(GRAPH.calc_trajectories()))
