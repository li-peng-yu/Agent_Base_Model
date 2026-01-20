import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.sparse as sp

class NonNormalNetwork:
    """
    Implementation of the non-normal network generation algorithm described in the Methods section.
    """
    def __init__(self, N=1000, N0=5, m=2, theta=0.5, a=2.552, b=3.668, hierarchical=True):
        """
        Args:
            N (int): Total number of nodes.
            N0 (int): Number of top nodes (opinion leaders).
            m (int): Number of in-edges for each new node.
            theta (float): Asymptotic reciprocity rate (0 to 1). Lower means more non-normal.
            a, b (float): Parameters for the sigmoid reciprocity function.
            hierarchical (bool): If True, enforces rho(1)=0 (deaf leaders). If False, uniform reciprocity.
        """
        self.N = N
        self.N0 = N0
        self.m = m
        self.theta = theta
        self.a = a
        self.b = b
        self.hierarchical = hierarchical
        
        # Determine Reciprocity Function rho(l)
        # Offset ensures rho(1) = 0 for top nodes
        if self.hierarchical:
            self.offset = self.theta / (1 + np.exp(-self.a * (1 - self.b)))
        else:
            self.offset = 0
        
        self.G = None
        self.adjacency_matrix = None

    def reciprocity_prob(self, level):
        """Calculates rho(l) based on Eq. (8) in the paper."""
        if not self.hierarchical:
            return self.theta
            
        val = self.theta / (1 + np.exp(-self.a * (level - self.b))) - self.offset
        return np.maximum(val, 0.0) # Ensure non-negative

    def generate(self):
        """Generates the graph using preferential attachment and hierarchical reciprocity."""
        G = nx.DiGraph()
        
        # 1. Initialize N0 top nodes
        top_nodes = list(range(self.N0))
        G.add_nodes_from(top_nodes)
        # Levels: Top nodes are level 1
        levels = {node: 1 for node in top_nodes}

        # Initialize out-degrees for preferential attachment probability
        # Give small initial weight to allow first attachments if out_degree is 0
        node_weights = np.ones(self.N0) 

        # 2. Add remaining nodes sequentially
        for i in tqdm(range(self.N0, self.N), desc="Building Network"):
            G.add_node(i)
            
            # Select m sources based on out-degree (Preferential Attachment)
            # Probability proportional to out-degree (plus epsilon to avoid zero prob)
            probs = node_weights / node_weights.sum()
            sources = np.random.choice(len(node_weights), size=self.m, replace=False, p=probs)
            
            min_source_level = float('inf')

            for j in sources:
                # Add directed edge j -> i (Influence flows from j to i)
                G.add_edge(j, i)
                
                # Update weights (j just gained an out-link)
                node_weights[j] += 1
                
                # Track level: i's level is min(parent_level) + 1
                if levels[j] < min_source_level:
                    min_source_level = levels[j]
                
                # Check for Reciprocity: Add i -> j
                # Reciprocity depends on the hierarchical level of the SOURCE (j)
                # If j is a leader (level 1), prob is low.
                p_recip = self.reciprocity_prob(levels[j])
                if np.random.random() < p_recip:
                    G.add_edge(i, j)
                    # Note: We do not typically update weights for i here in standard PA 
                    # unless using full degree, but standard PA uses out-degree of existing nodes.
            
            # Assign level to new node i
            levels[i] = min_source_level + 1
            
            # Add i to weights array (initial weight 1)
            node_weights = np.append(node_weights, 1.0)

        self.G = G
        # Store as sparse matrix for efficiency in simulation
        self.adjacency_matrix = nx.to_scipy_sparse_array(G, nodelist=range(self.N), format='csr')
        return self.adjacency_matrix

class IsingSimulation:
    """
    Agent-Based Model Simulation using Ising-like dynamics on the generated network.
    """
    def __init__(self, adjacency_matrix, kappa=0.98, p_pm=0.05):
        """
        Args:
            adjacency_matrix: Sparse adjacency matrix A (a_ij = 1 if j influences i).
            kappa (float): Social coupling strength (inverse temperature). 
                           kappa < 1.0 is sub-critical.
            p_pm (float): Base rate of state change.
        """
        self.A = adjacency_matrix
        self.N = self.A.shape[0]
        self.kappa = kappa
        self.p_pm = p_pm
        
        # Initialize spins: Random +/- 1
        self.spins = np.random.choice([-1, 1], size=self.N).astype(float)
        
        # Precompute in-degrees for normalization (Eq 4 denominator)
        # Sum of A along rows (j->i, so sum over j)
        self.k_in = np.array(self.A.sum(axis=1)).flatten()
        # Avoid division by zero for top nodes or isolated nodes
        self.k_in[self.k_in == 0] = 1.0 

        self.history_m = []

    def step(self):
        """
        Performs one Monte Carlo step (asynchronous update of all agents).
        """
        # Calculate local fields: h_i = sum(a_ij * s_j) / k_in_i
        # Sparse matrix multiplication: A dot s
        field = self.A.dot(self.spins) / self.k_in
        
        # Calculate transition probabilities
        # P(s -> -s) = p_pm/2 * (1 - s * tanh(kappa * field))
        # We process updates in a vectorized way for speed, 
        # though strictly ABMs are often sequential. 
        # For N=1000, vectorized synchronous or random-batch is fine approximation.
        # Here we use synchronous update for vectorization efficiency, 
        # which preserves the macroscopic dynamics described.
        
        argument = self.kappa * field
        tanh_val = np.tanh(argument)
        prob_flip = (self.p_pm / 2.0) * (1.0 - self.spins * tanh_val)
        
        # Determine which agents flip
        random_vals = np.random.random(self.N)
        flip_mask = random_vals < prob_flip
        
        # Apply flips
        self.spins[flip_mask] *= -1
        
        # Record net magnetization
        m = np.mean(self.spins)
        self.history_m.append(m)
        return m

    def run(self, steps=5000):
        for _ in range(steps):
            self.step()
        return np.array(self.history_m)

def price_from_magnetization(magnetization_series, P0=100, c=3.0):
    """
    Maps magnetization to price using the exponential relation P ~ exp(c * m).
    Reference: 'Bubble' section in the paper.
    """
    # Using a simplified mapping to visualize the bubbles
    # P_t = P_0 * exp(c * m_t)
    return P0 * np.exp(c * magnetization_series)

def run_experiment():
    # Parameters from the paper
    N = 1000
    N0 = 5  # Small number of opinion leaders
    m = 2
    kappa_subcritical = 0.5 # Lower value to ensure Normal network is stable (paramagnetic)
    steps = 4000
    
    # 1. Highly Non-Normal Network (theta = 0, almost no reciprocity for leaders)
    print("Generating Non-Normal Network (theta=0.1)...")
    net_non_normal = NonNormalNetwork(N=N, N0=N0, m=m, theta=0.1, hierarchical=True)
    A_nn = net_non_normal.generate()
    
    sim_nn = IsingSimulation(A_nn, kappa=kappa_subcritical)
    print("Simulating Non-Normal Dynamics...")
    m_nn = sim_nn.run(steps)
    p_nn = price_from_magnetization(m_nn)

    # 2. Normal/Symmetric Network (theta = 1.0, high reciprocity)
    print("Generating Normal Network (theta=1.0)...")
    net_normal = NonNormalNetwork(N=N, N0=N0, m=m, theta=1.0, hierarchical=False)
    A_n = net_normal.generate()
    
    sim_n = IsingSimulation(A_n, kappa=kappa_subcritical)
    print("Simulating Normal Dynamics...")
    m_n = sim_n.run(steps)
    p_n = price_from_magnetization(m_n)

    # 3. Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    
    # Plot Magnetization
    axes[0, 0].plot(m_nn, color='red', linewidth=0.8, label='Non-Normal (Asymmetric)')
    axes[0, 0].set_title(f'Net Magnetization (Non-Normal, $\\theta=0.1$)')
    axes[0, 0].set_ylabel('$m_t$')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(m_n, color='blue', linewidth=0.8, label='Normal (Symmetric)')
    axes[0, 1].set_title(f'Net Magnetization (Normal, $\\theta=1.0$)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # Plot Price
    axes[1, 0].plot(p_nn, color='darkred', linewidth=1.0)
    axes[1, 0].set_title('Simulated Asset Price (Bubbles)')
    axes[1, 0].set_ylabel('$P_t$')
    axes[1, 0].set_yscale('log') # Log scale helps identify super-exponential growth
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(p_n, color='darkblue', linewidth=1.0)
    axes[1, 1].set_title('Simulated Asset Price (Baseline)')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('reproduced_bubbles.png', dpi=300)
    print("Simulation complete. Results saved to 'reproduced_bubbles.png'.")

if __name__ == "__main__":
    run_experiment()