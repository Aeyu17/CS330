from dataclasses import dataclass
from datetime import datetime
from utils import get_length
import math

@dataclass
class Node:
    node_id: int
    connections: dict[int, int] # node id -> connection weight
    location: tuple[float, float]

@dataclass
class NodeRecord:
    node: Node
    connection: int # node_id of previous node, 0 if null
    cost_so_far: float
    est_total_cost: float

def heuristic_estimate(node: Node, goal: Node) -> float:
    current_location = node.location
    goal_location = goal.location
    return get_length((goal_location[0] - current_location[0], goal_location[1] - current_location[1])) # Euclidean distance

def get_lowest_estimated_cost(records: list[NodeRecord]):
    smallest_cost = math.inf
    smallest_record: NodeRecord | None = None
    for i, record in enumerate(records):
        if record.est_total_cost < smallest_cost:
            smallest_record = record
            smallest_cost = record.est_total_cost
            smallest_idx = i
    
    return smallest_idx, smallest_record

def get_node(records: list[NodeRecord], id: int):
    for i, record in enumerate(records):
        if record.node.node_id == id:
            return i, record
        
    return None, None

def main():
    with open('CS 330, Pathfinding, Graph AB Nodes v3.txt', 'r') as f:
        node_txt = f.read()

    with open('CS 330, Pathfinding, Graph AB Connections v3.txt', 'r') as f:
        connections_txt = f.read()

    with open('output.txt', 'w') as f:
        f.write(f'CS 330, Pathfinding, Begin {datetime.now()}\n\n')
        f.write(f'Loaded scenario Adventure Baypath AB\n\n')
        f.write('Nodes\n')

        nodes: dict[int, Node] = {}

        for line in filter(lambda x: len(x) > 0 and x[0] != '#', node_txt.split('\n')):
            # Semi-complicated line, basically split the line by commas, and strip each
            node_fields = list(map(lambda x: x.strip(), line.split(',')))

            id = int(node_fields[1])

            new_node = Node(node_id=id, connections={}, location=(float(node_fields[7]), float(node_fields[8])))
            nodes[id] = new_node

            f.write(f'N {id} {node_fields[2]} {node_fields[3]} {node_fields[4]} {node_fields[5]} {node_fields[6]} {node_fields[7]} {node_fields[8]} {node_fields[9]} {node_fields[10]} {node_fields[11][1:-1]}\n')

        f.write('\nConnections\n')
        for line in filter(lambda x: len(x) > 0 and x[0] != '#', connections_txt.split('\n')):
            # Again, split line by commas, strip each; same as nodes
            connection_fields = list(map(lambda x: x.strip(), line.split(',')))

            from_node_id = int(connection_fields[2])
            to_node_id = int(connection_fields[3])
            weight = int(connection_fields[4])

            nodes[from_node_id].connections[to_node_id] = weight

            f.write(f'C {connection_fields[1]} {connection_fields[2]} {connection_fields[3]} {connection_fields[4]} {connection_fields[5]} {connection_fields[6]}\n')

        f.write('\n')

        # -- CHANGE PATHS HERE --
        paths_to_find = [(1, 29), (1, 38), (11, 1), (33, 66), (58, 43)]

        for path in paths_to_find:
            start_node_id = path[0]
            end_node_id = path[1]

            print(f'Path to find: {start_node_id} -> {end_node_id}')

            start_node = nodes[start_node_id]
            goal_node = nodes[end_node_id]

            node_records: dict[int, NodeRecord] = {} # mainly used for getting the path at the end

            start_node_record = NodeRecord(node=start_node, connection=0, cost_so_far=0.0, est_total_cost=heuristic_estimate(start_node, goal_node))

            open_list: list[NodeRecord] = [start_node_record]
            closed_list: list[NodeRecord] = []

            while len(open_list) > 0:
                curr_idx, current_record = get_lowest_estimated_cost(open_list)
                if current_record is None: # mostly done for type checking
                    raise ValueError('Something went wrong here!')
                
                # Be explicit that we're checking node IDs
                if current_record.node.node_id == goal_node.node_id:
                    break

                connection_dict = current_record.node.connections

                for to_node_id in connection_dict.keys():
                    # print(f'Checking {to_node_id}')
                    to_node = nodes[to_node_id]
                    new_cost = current_record.cost_so_far + connection_dict[to_node_id]

                    closed_idx, closed_record_copy = get_node(closed_list, to_node_id)
                    open_idx, open_record_copy = get_node(open_list, to_node_id)
                    if closed_record_copy is not None: # i.e. closed has this node already
                        to_node_record = closed_record_copy
                        if closed_record_copy.cost_so_far <= new_cost: # older node is better; ignore this connection
                            continue 

                        closed_list.pop(closed_idx) # newer node is better; remove the previously closed one
                        to_heuristic = closed_record_copy.est_total_cost - closed_record_copy.cost_so_far 
                    elif open_record_copy is not None:
                        to_node_record = open_record_copy
                        if open_record_copy.cost_so_far <= new_cost: # older node is better; ignore this connection
                            continue

                        open_list.pop(open_idx)
                        to_heuristic = open_record_copy.est_total_cost - open_record_copy.cost_so_far 
                    else: # to node is unvisited
                        to_heuristic = heuristic_estimate(to_node, goal_node)

                    to_node_record = NodeRecord(node=to_node, 
                                                connection=current_record.node.node_id, 
                                                cost_so_far=new_cost, 
                                                est_total_cost=new_cost + to_heuristic
                    )

                    if get_node(open_list, to_node_id)[1] is None:
                        open_list.append(to_node_record)

                open_list.pop(curr_idx)
                closed_list.append(current_record)
                node_records[current_record.node.node_id] = current_record

            if current_record.node.node_id != goal_node.node_id:
                print(f'No path could be found between {start_node_id} and {end_node_id}')
                return
            
            path: list[Node] = []
            total_cost = current_record.cost_so_far
            while current_record.node.node_id != start_node_id:
                path.append(current_record.node)
                current_record = node_records[current_record.connection]

            path.append(start_node)
            path.reverse()
            print(f'Discovered path: {[item.node_id for item in path]} with cost {total_cost}')
            f.write(f'Path from {start_node_id} to {end_node_id} path= {str([item.node_id for item in path])[1:-1]} cost= {total_cost}\n')

        f.write(f'CS 330, Pathfinding, End {datetime.now()}')

if __name__ == '__main__':
    main()