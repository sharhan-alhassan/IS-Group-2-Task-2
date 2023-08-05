from flask import Flask, render_template, request
import random
import math
import numpy as np

app = Flask(__name__)

class Ant:
    def __init__(self, start_node):
        self.path = [start_node]
        self.distance = 0

    def add_node_to_path(self, node):
        self.path.append(node)

    def calculate_distance(self, graph):
        self.distance = 0
        for i in range(len(self.path) - 1):
            self.distance += graph[self.path[i]][self.path[i+1]]

def choose_next_node(graph, pheromone_matrix, current_node, visited, pheromone_exp_weights):
    total_probabilities = 0
    probabilities = []

    for i in range(len(graph)):
        if i not in visited:
            pheromone = pheromone_matrix[current_node][i]
            distance = graph[current_node][i]
            probabilities.append((i, (pheromone ** pheromone_exp_weights) / (distance + 1e-6)))
            total_probabilities += probabilities[-1][1]

    probabilities = [(node, prob / total_probabilities) for node, prob in probabilities]
    probabilities.sort(key=lambda x: x[1], reverse=True)

    rand_val = random.random()
    cumulative_prob = 0

    for node, prob in probabilities:
        cumulative_prob += prob
        if cumulative_prob >= rand_val:
            return node

def update_pheromone_matrix(pheromone_matrix, ants, pheromone_evaporation_rate):
    for i in range(len(pheromone_matrix)):
        for j in range(len(pheromone_matrix[i])):
            pheromone_matrix[i][j] *= (1 - pheromone_evaporation_rate)

    for ant in ants:
        for i in range(len(ant.path) - 1):
            pheromone_matrix[ant.path[i]][ant.path[i+1]] += 1 / ant.distance

def ant_colony_optimization(graph, num_ants, max_iterations, initial_pheromone, pheromone_exp_weights, pheromone_evaporation_rate, num_nodes):
    pheromone_matrix = [[initial_pheromone] * num_nodes for _ in range(num_nodes)]

    best_distance = float('inf')
    best_path = None

    for iteration in range(max_iterations):
        ants = [Ant(random.randint(0, num_nodes-1)) for _ in range(num_ants)]

        for ant in ants:
            visited = set(ant.path)

            while len(visited) < num_nodes:
                current_node = ant.path[-1]
                next_node = choose_next_node(graph, pheromone_matrix, current_node, visited, pheromone_exp_weights)
                ant.add_node_to_path(next_node)
                visited.add(next_node)

            ant.calculate_distance(graph)

            if ant.distance < best_distance:
                best_distance = ant.distance
                best_path = ant.path[:]

        update_pheromone_matrix(pheromone_matrix, ants, pheromone_evaporation_rate)

    # Convert the best path to a format compatible with Chart.js
    best_path_data = []
    for i in range(len(best_path) - 1):
        best_path_data.append((int(best_path[i]), int(best_path[i+1])))

    return best_path_data, best_distance



# # Function to generate a random graph with random distances between nodes
# def generate_random_graph(num_nodes):
#     graph = np.random.randint(10, 100, size=(num_nodes, num_nodes))
#     np.fill_diagonal(graph, 0)
#     return graph.tolist()
# Function to generate a random graph with random distances between nodes

def generate_random_graph(num_nodes, fuzzy):
    base_distance = np.random.randint(10, 100, size=(num_nodes, num_nodes))
    np.fill_diagonal(base_distance, 0)

    # Create a fuzzy distance matrix based on the base distance
    fuzziness = fuzzy
    # fuzziness = 10  # Adjust this value to control the fuzziness level
    fuzzy_distance = base_distance + np.random.uniform(-fuzziness, fuzziness, size=(num_nodes, num_nodes))
    fuzzy_distance = np.clip(fuzzy_distance, 0, None)  # Ensure distances are non-negative

    return fuzzy_distance.tolist()


@app.route('/', methods=['GET', 'POST'])
def index():
    # Example usage
    # graph = [
    #     [0, 10, 15, 20],
    #     [10, 0, 35, 25],
    #     [15, 35, 0, 30],
    #     [20, 25, 30, 0]
    # ]

    if request.method == 'POST':
        population_size = int(request.form['population_size'])
        max_iterations = int(request.form['max_iterations'])
        initial_pheromone = float(request.form['initial_pheromone'])
        pheromone_exp_weights = float(request.form['pheromone_exp_weights'])
        pheromone_evaporation_rate = float(request.form['pheromone_evaporation_rate'])
        fuzziness = int(request.form['fuzziness'])
        # num_nodes = int(request.form['num_nodes'])
        num_nodes = int(request.form.get('num_nodes', 4))
        try:
            num_nodes = int(request.form.get('num_nodes', 4))
        except KeyError:
            # Debug output to check if 'num_nodes' is present in the form data
            print("Form data:", request.form)
            raise

        # Generate a random graph based on the number of nodes
        graph = generate_random_graph(num_nodes, fuzziness)

        # Call the ACO algorithm with the specified parameters
        best_path, best_distance = ant_colony_optimization(graph, population_size, max_iterations, initial_pheromone, pheromone_exp_weights, pheromone_evaporation_rate, num_nodes)

        print("Best path:", best_path)
        print("Best distance:", best_distance)
        
        # Convert the best path to a format compatible with Chart.js
        best_path_data = []
        for i in range(len(best_path) - 1):
            best_path_data.append((best_path[i], best_path[i+1]))

        return render_template('index.html', best_path=best_path_data, best_distance=best_distance,
                            population_size=population_size, max_iterations=max_iterations,
                            initial_pheromone=initial_pheromone, pheromone_exp_weights=pheromone_exp_weights,
                            pheromone_evaporation_rate=pheromone_evaporation_rate, num_nodes=num_nodes, fuzziness=fuzziness)

    # If the request method is GET, render the template with default values
    return render_template('index.html', population_size=5, max_iterations=100, initial_pheromone=1.0,
                            pheromone_exp_weights=2.0, pheromone_evaporation_rate=0.1, num_nodes=4, fuzziness=10)

    
if __name__ == "__main__":

    app.run(debug=True)
