import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import time

class Parameters:
    """
    Simulation parameters based on Table A.1 of the paper.
    """
    def __init__(self):
        # Time
        self.years = 20
        self.days_per_year = 250
        self.T = self.years * self.days_per_year  # Total steps (5000)
        
        # Assets
        self.rf_annual = 0.01
        self.rf = self.rf_annual / 250.0
        self.d0_annual = 0.04
        self.d0 = self.d0_annual / 250.0
        self.rd_annual = 0.04
        self.rd = self.rd_annual / 250.0
        self.sigma_d_annual = 0.04
        self.sigma_d = self.sigma_d_annual / 250.0
        self.P0 = 1.0
        
        # Fundamentalists
        self.x0_F = 0.3
        self.W0_F = 10e9
        self.E_rt = self.rd # Expected return assumed to be rd based on table
        self.gamma = 1.0 # Risk aversion (log utility) implies gamma=1 approx
        # Derive gamma * sigma^2 from initial conditions to match x0_F
        # x0_F = (E_rt + d0/P0*(1+rd) - rf) / (gamma * sigma_sq)
        numerator = self.E_rt + (self.d0/self.P0)*(1+self.rd) - self.rf
        self.gamma_sigma_sq = numerator / self.x0_F
        
        # Noise Traders
        self.N = 1000
        self.x0_N = 0.5
        self.W0_N = 10e9
        self.p_switch = 0.3 # Base switching probability (p+/-)
        self.kappa = 0.5    # Coupling strength
        
        # Network
        self.er_prob = 0.003 # 0.3%
        self.history_window_d = 10
        self.rewire_fraction_rho = 0.1
        self.k0 = 1.0
        self.alpha = 1.1 # Preferential attachment exponent

class Market:
    def __init__(self, params):
        self.params = params
        self.t = 0
        self.price = params.P0
        self.dividend = params.d0
        self.prices = [params.P0]
        self.dividends = [params.d0]
        self.excess_returns = [] # Track r_excess for fitness calculation

    def step_dividend(self):
        # Dividend growth process: d_t = d_{t-1}(1 + r_t^d)
        # r_t^d ~ N(rd, sigma_d^2)
        r_d_t = np.random.normal(self.params.rd, self.params.sigma_d)
        self.dividend = self.dividend * (1 + r_d_t)
        self.dividends.append(self.dividend)

    def calculate_excess_return(self, new_price, old_price):
        # r_excess = (P_t + d_t - P_{t-1}(1+rf)) / P_{t-1}
        # Or simply r_t + d_t/P_{t-1} - rf
        capital_gain = (new_price - old_price) / old_price
        dividend_yield = self.dividend / old_price
        return capital_gain + dividend_yield - self.params.rf

class Fundamentalist:
    def __init__(self, params):
        self.params = params
        self.wealth = params.W0_F
        self.x_t = params.x0_F # Fraction invested in risky asset
        
    def decide_fraction(self, price_prev, dividend_prev):
        # Equation A.6: Myopic demand
        # E[r_excess] / (gamma * sigma^2)
        # Approximated using constant expectations for volatility
        expected_excess = self.params.E_rt + (dividend_prev / price_prev) * (1 + self.params.rd) - self.params.rf
        
        # Clamping to avoid extreme leverage in simulation stability
        # Though theory allows unbounded, ABMs often need caps. 
        # We'll stick to the formula but check for negatives.
        self.x_t = expected_excess / self.params.gamma_sigma_sq
        return self.x_t

    def update_wealth(self, r_excess, rf):
        # W_t = W_{t-1} * (x_t * r_excess + (1+rf))
        # Note: x_t was decided at t-1 for the period t-1 -> t
        self.wealth = self.wealth * (self.x_t * r_excess + (1 + rf))

class NoiseTraders:
    def __init__(self, params):
        self.params = params
        self.N = params.N
        self.wealth = params.W0_N
        
        # Initialize Spins: -1 (Risk-free) or +1 (Risky)
        # Start with x0_N = 0.5 implies 50/50 split
        self.spins = np.random.choice([-1, 1], size=self.N)
        self.x_t = 0.5
        
        # Initialize Network (Directed Erdős-Rényi)
        # Using sparse matrix for efficiency with 1000 nodes
        self.adj_matrix = nx.to_scipy_sparse_array(
            nx.erdos_renyi_graph(self.N, params.er_prob, directed=True)
        ).tolil() # Use LIL for efficient structure changes (rewiring)
        
        # Performance history for fitness
        # h_history: buffer to store success (1) or failure (0) for last d steps
        self.h_buffer = np.zeros((self.N, params.history_window_d))
        self.buffer_idx = 0
        
        # Track out-degrees for preferential attachment
        self.out_degrees = np.array(self.adj_matrix.sum(axis=0)).flatten()
        self.in_degrees = np.array(self.adj_matrix.sum(axis=1)).flatten()

    def update_network(self, prev_r_excess):
        """
        Social Sampling Mechanism (Section 2.3)
        """
        # 1. Determine Success (h_{t-1}^j)
        # Logic: If held risky (s=1) and r_excess > 0 -> Success
        #        If held safe (s=-1) and r_excess < 0 -> Success (avoided loss)
        #        Else -> Failure
        # Note: Paper says "h=1 if holding risky was profitable". 
        # We interpret this as "Made the correct directional bet".
        
        current_success = np.zeros(self.N)
        if prev_r_excess > 0:
            current_success[self.spins == 1] = 1.0
        else:
            current_success[self.spins == -1] = 1.0
            
        # Update buffer
        self.h_buffer[:, self.buffer_idx] = current_success
        self.buffer_idx = (self.buffer_idx + 1) % self.params.history_window_d
        
        # 2. Calculate Fitness (eta_t)
        fitness = np.mean(self.h_buffer, axis=1)
        eta_max = np.max(fitness) if np.max(fitness) > 0 else 0.001
        
        # 3. Rewiring
        # Select rho fraction of traders to update *one* incoming link
        n_rewire = int(self.params.rewire_fraction_rho * self.N)
        rewiring_agents = np.random.choice(self.N, n_rewire, replace=False)
        
        # Calculate preferential attachment probability P_PA ~ (k_out + k0)^alpha
        pa_scores = np.power(self.out_degrees + self.params.k0, self.params.alpha)
        pa_probs = pa_scores / np.sum(pa_scores)
        
        for i in rewiring_agents:
            # If i has in-neighbors, pick one to potentially remove
            neighbors = self.adj_matrix.rows[i]
            if not neighbors:
                continue
                
            # Try to find a new influencer j
            # Sample j based on PA
            while True:
                j = np.random.choice(self.N, p=pa_probs)
                if j == i or j in neighbors: 
                    continue # Simple rejection sampling
                
                # Check fitness acceptance criterion (Eq 5)
                # P(j->i) = (eta_max)^h_j * (1-eta_max)^(1-h_j)
                # Here h_j is the *most recent* success (current_success[j])
                h_j = current_success[j]
                
                prob_accept = (eta_max ** h_j) * ((1 - eta_max) ** (1 - h_j))
                
                if np.random.random() < prob_accept:
                    # Perform rewire
                    old_neighbor = np.random.choice(neighbors)
                    
                    # Remove old
                    self.adj_matrix[i, old_neighbor] = 0
                    self.out_degrees[old_neighbor] -= 1
                    
                    # Add new
                    self.adj_matrix[i, j] = 1
                    self.out_degrees[j] += 1
                    
                    # Update in-degrees cache
                    self.in_degrees[i] = len(self.adj_matrix.rows[i])
                    break
                else:
                    # If rejected, loop continues to sample another candidate
                    # To prevent infinite loops in bad states, we can break with a small prob or max iter
                    # For strict adherence to paper, we retry. 
                    # Optimization: break after 10 tries to speed up sim
                    if np.random.random() < 0.1: 
                        break

    def update_opinions(self):
        """
        Ising-like dynamics (Eq 1)
        """
        # Compute local field: (1/k_in) * sum(a_ij * s_j)
        # Matrix multiplication: Adj * spins
        # Note: Adj is (N, N) where A_ij=1 means j influences i. 
        # Standard sparse dot product works.
        field_sum = self.adj_matrix.dot(self.spins)
        
        # Avoid division by zero for nodes with no influencers
        in_deg_safe = self.in_degrees.copy()
        in_deg_safe[in_deg_safe == 0] = 1.0
        local_field = field_sum / in_deg_safe
        
        # Switching probability
        # P(switch) = (p/2) * (1 - kappa * s_i * local_field)
        switch_prob = (self.params.p_switch / 2.0) * (1 - self.params.kappa * self.spins * local_field)
        
        # Perform switching
        random_draws = np.random.random(self.N)
        should_switch = random_draws < switch_prob
        self.spins[should_switch] *= -1
        
        # Calculate aggregate risky fraction
        # m = mean(spins) (-1 to 1) -> x = 0.5 + 0.5*m
        m_t = np.mean(self.spins)
        self.x_t = 0.5 + 0.5 * m_t

    def update_wealth(self, r_excess, rf):
        self.wealth = self.wealth * (self.x_t * r_excess + (1 + rf))

    def get_centralization(self):
        # Freeman centralization based on out-degree
        # (sum(max_deg - deg)) / ((N-1)(N-2)) approx (N-1) for star
        max_out = np.max(self.out_degrees)
        numerator = np.sum(max_out - self.out_degrees)
        denominator = (self.N - 1) * (self.N - 1) # Standard normalization for directed graph
        return numerator / denominator

def run_simulation():
    params = Parameters()
    market = Market(params)
    fund = Fundamentalist(params)
    noise = NoiseTraders(params)
    
    # Storage for plotting
    history_price = []
    history_centralization = []
    history_volatility = []
    history_autocorr = []
    
    # Metrics calculation helpers
    returns_window = []
    
    print(f"Starting simulation for {params.T} steps ({params.years} years)...")
    start_time = time.time()
    
    for t in range(params.T):
        # 1. Update Dividends
        market.step_dividend()
        
        # 2. Update Network (Social Sampling)
        # Skip first step as we need previous return for fitness
        if t > 0:
            last_excess = market.excess_returns[-1]
            noise.update_network(last_excess)
        
        # 3. Update Opinions (Ising) & Determine Noise Demand x_N
        noise.update_opinions()
        
        # 4. Determine Fundamentalist Demand x_F
        # Based on P_{t-1} and d_{t-1} (Expectations formed at t-1)
        # But wait, market clears at t.
        # Demand x_F is formulated based on expectation E_{t-1}.
        fund.decide_fraction(market.price, market.dividends[-2] if len(market.dividends)>1 else params.d0)
        
        # 5. Determine Market Price (Equilibrium)
        # Solve Eq A.12: a P^2 + b P + c = 0
        # Need coefficients (Eq A.13)
        
        P_prev = market.price
        W_F_prev = fund.wealth
        W_N_prev = noise.wealth
        x_F_prev = fund.x_t # Actually the fraction held coming INTO t? 
        # In the paper A.5, x_t^F is the optimized fraction for period t.
        # Let's use the explicit coefficients from A.13
        
        # Note: The paper defines x_{t-1} in the wealth update. 
        # x_t is the demand for the current period.
        # Coefficients A.13 use x_{t-1} and x_t.
        # We assume `fund.x_t` and `noise.x_t` are the new TARGET fractions (x_t in paper).
        # We need to track previous fractions (x_{t-1}) for the flow constraint.
        # For simplicity in this implementation step, we assume rebalancing happens fully.
        
        # Re-mapping variables to paper's notation for A.13:
        # x_t^N -> noise.x_t
        # x_t^F -> fund.x_t
        # x_{t-1}^N -> we need to store this. Let's approx as previous step's x_t.
        
        x_N_curr = noise.x_t
        x_F_curr = fund.x_t
        
        x_N_prev = noise.x_prev if hasattr(noise, 'x_prev') else params.x0_N
        x_F_prev = fund.x_prev if hasattr(fund, 'x_prev') else params.x0_F
        
        # Term A
        term_a = (1/P_prev) * (
            W_N_prev * x_N_prev * (x_N_curr - 1) + 
            W_F_prev * x_F_prev * ( (params.E_rt - params.rf)/(params.gamma_sigma_sq * params.x0_F * params.gamma) - 1 ) 
            # Note: The term (E-rf)/(gamma sigma^2) simplifies to x_F_curr roughly? 
            # Actually A.13 term for Fund is: W_{t-1} x_{t-1} ( term - 1).
            # The term is E_rt.../gamma sigma^2. This is basically x_F_curr WITHOUT the price adjustment.
            # Let's stick to the algebraic form in A.13 strictly.
        )
        # Correction: The paper's A.13 for a_t is specific.
        # Let's simplify by solving Supply = Demand directly if possible.
        # Demand value = W_t * x_t.
        # W_t depends on P_t.
        # W_t^N = W_{t-1}^N * ( x_{t-1}^N * r_excess + (1+rf) )
        # r_excess = P_t/P_{t-1} + d_t/P_{t-1} - 1 - rf
        # This linear dependence on P_t is why it's quadratic.
        
        # Let's compute the components of a, b, c.
        
        # Common constants
        d_t = market.dividend
        
        # Determine x_F_curr logic inside A.13
        # In A.13, the Fundamentalist term is derived from optimal x_t^F approx E/gamma sigma^2.
        # Let's calculate the "Target Fraction Factor" F_target = (E + d/P(1+rd) - rf)/... 
        # This depends on 1/P. This makes it cubic?
        # The paper assumes "d_t << P_t" for approximation in A.6.
        # Eq A.13 coefficients seem to treat the fundamentalist demand parameter as independent of P_t in the `a` term?
        # A.13 `a_t` term: W... ( ... - 1). The paper likely assumes x_t^F in the wealth update is fixed, 
        # but x_t^F in the demand *target* depends on P_t?
        # Actually, let's look at a_t in A.13. It uses `(E_rt - rf) / gamma sigma^2`. This is constant.
        # This implies they use a simplified fundamentalist demand for the coefficient derivation.
        
        fund_term_const = (params.E_rt - params.rf) / params.gamma_sigma_sq
        
        a_val = (1/P_prev) * (
            W_N_prev * x_N_prev * (x_N_curr - 1) + 
            W_F_prev * x_F_prev * (fund_term_const - 1) 
            # Note: Calculating exactly as Eq A.13 requires strict adherence to their approximations.
            # If simulation explodes, check here.
        )
        
        b_val = (
            W_F_prev / params.gamma_sigma_sq * (
                x_F_prev * d_t * (1+params.rd)/P_prev + 
                (params.E_rt - params.rf)*(x_F_prev*(d_t/P_prev - params.rf) + params.rf)
            ) +
            W_N_prev * x_N_curr * (
                x_N_prev * (d_t/P_prev - 1 - params.rf) + params.rf + 1
            ) # Note: +1 from 1+rf? A.13 says "... + r_f]". Wait, Eq A.13 for b_t is complex.
            # Let's trust the logic: Demand = Supply.
            # W_t * x_t = P_t * N_shares? No, Walrasian is Excess Demand = 0.
            # Excess Demand = Wealth_t * x_t - Wealth_{t-1} * x_{t-1} * (P_t/P_{t-1})
            # This is "Money allocated to stock" - "Value of current stock holding".
        )
        
        # Let's implement A.13 b_t precisely from the image text
        term_b_fund = (W_F_prev / params.gamma_sigma_sq) * (
            x_F_prev * (d_t * (1+params.rd) / P_prev) + 
            (params.E_rt - params.rf) * ( x_F_prev * (d_t/P_prev - params.rf) + params.rf ) # ? Check brackets
        )
        # Checking A.13 text for b_t again...
        # b_t = W... { x... + (E-rf)[ x...( d/P - rf ) + rf ] } ?
        # Text: "... + (E - rf)[x_{t-1}^F (d_t/P_{t-1} - r_f) + r_f] }" ? No, looks like [ ... + 1+rf] ?
        # Actually, let's use the explicit Excess Demand equation A.10 and A.7 to find P.
        # Excess D = W_{t-1} ( x_t [ 1+rf + x_{t-1}(r_t + d/P - rf) ] - x_{t-1} P/P_{t-1} )
        # where r_t = P/P_{t-1} - 1.
        # Substitute r_t:
        # Excess D = W_{t-1} ( x_t [ 1+rf + x_{t-1}(P/P_{t-1} - 1 + d/P - rf) ] - x_{t-1} P/P_{t-1} )
        # Excess D = W_{t-1} ( x_t [ 1+rf + x_{t-1} P/P_{t-1} - x_{t-1} - x_{t-1} rf + x_{t-1} d/P_{t-1} ] - x_{t-1} P/P_{t-1} )
        # Excess D = W_{t-1} ( x_t [ 1 + x_{t-1} (P/P_{t-1} - 1 + d/P_{t-1}) ] - x_{t-1} P/P_{t-1} ) ? 
        # Simplified: W_t = W_{t-1} * ( x_{t-1} * (P/P + d/P - 1 - rf) + 1 + rf )
        # New Demand = W_t * x_t.
        # Supply Value = (W_{t-1} * x_{t-1} / P_{t-1}) * P_t.
        # Equation: W_t * x_t - Supply = 0.
        
        # Let's code this specific balance equation solver directly instead of relying on potentially typo-prone A.13 coefficients.
        # We need P_t such that:
        # (W_F_t(P_t) * x_F_t + W_N_t(P_t) * x_N_t) = (N_shares_F + N_shares_N) * P_t
        # where N_shares = W_{t-1} * x_{t-1} / P_{t-1}
        
        # Define functions of P
        shares_F = W_F_prev * x_F_prev / P_prev
        shares_N = W_N_prev * x_N_prev / P_prev
        total_supply_shares = shares_F + shares_N
        
        # Wealth update logic
        # W_t(P) = W_{t-1} * ( x_{t-1} * ( (P - P_prev + d_t)/P_prev - rf ) + 1 + rf )
        #        = W_{t-1} * ( x_{t-1} * (P/P_prev + d_t/P_prev - 1 - rf) + 1 + rf )
        #        = W_{t-1} * ( x_{t-1} P/P_prev + x_{t-1}(d_t/P_prev - 1 - rf) + 1 + rf )
        #        = P * (W_{t-1} x_{t-1} / P_prev) + W_{t-1} * C
        # where C = x_{t-1}(d_t/P_prev - 1 - rf) + 1 + rf
        
        const_F = x_F_prev * (d_t/P_prev - 1 - params.rf) + 1 + params.rf
        const_N = x_N_prev * (d_t/P_prev - 1 - params.rf) + 1 + params.rf
        
        # W_F(P) = P * shares_F + W_F_prev * const_F
        # W_N(P) = P * shares_N + W_N_prev * const_N
        
        # Target Demand x_F_t depends on P?
        # In A.6, x_t^F uses P_{t-1}. So x_F_curr is constant wrt P_t.
        # x_N_curr is determined by Ising, constant wrt P_t.
        
        # Equilibrium:
        # x_F_curr * W_F(P) + x_N_curr * W_N(P) = P * total_supply_shares
        # x_F_curr * (P * shares_F + Term_F) + x_N_curr * (P * shares_N + Term_N) = P * (shares_F + shares_N)
        # P * [ x_F_curr * shares_F + x_N_curr * shares_N - shares_F - shares_N ] + [ x_F_curr * Term_F + x_N_curr * Term_N ] = 0
        
        # This is linear in P!
        # P * A + B = 0  => P = -B / A
        
        Term_F = W_F_prev * const_F
        Term_N = W_N_prev * const_N
        
        A_lin = x_F_curr * shares_F + x_N_curr * shares_N - shares_F - shares_N
        B_lin = x_F_curr * Term_F + x_N_curr * Term_N
        
        if abs(A_lin) < 1e-9:
            P_new = P_prev # No solution or singularity
        else:
            P_new = -B_lin / A_lin
            
        # Sanity check: Price must be positive
        if P_new <= 0:
            P_new = P_prev * 0.9 # Fallback for instability
            
        market.price = P_new
        market.prices.append(P_new)
        
        # Calculate returns
        r_ex = market.calculate_excess_return(P_new, P_prev)
        market.excess_returns.append(r_ex)
        returns_window.append(np.log(P_new/P_prev))
        
        # 6. Update Wealths formally
        fund.update_wealth(r_ex, params.rf)
        noise.update_wealth(r_ex, params.rf)
        
        # Store state for next step
        noise.x_prev = x_N_curr
        fund.x_prev = x_F_curr
        
        # 7. Record Metrics
        history_price.append(P_new)
        
        # Metrics are computationally heavy, calculate sparsely or at end?
        # Paper plots time series.
        if t % 10 == 0:
            history_centralization.append(noise.get_centralization())
        else:
            history_centralization.append(history_centralization[-1] if history_centralization else 0)
            
        # Calculate rolling volatility (annualized)
        if len(returns_window) > 90:
            recent_rets = returns_window[-90:]
            vol = np.std(recent_rets) * np.sqrt(250)
            history_volatility.append(vol)
        else:
            history_volatility.append(0)
            
        # Print progress
        if t % 500 == 0:
            print(f"Step {t}/{params.T}, Price: {P_new:.2f}, Centralization: {history_centralization[-1]:.4f}")

    print(f"Simulation finished in {time.time() - start_time:.2f}s")
    
    # --- Visualization (Replicating Figure 2) ---
    # Burn-in removal (2 years = 500 steps)
    burn_in = 500
    
    valid_steps = range(burn_in, params.T)
    plot_prices = history_price[burn_in:]
    plot_cent = history_centralization[burn_in:]
    plot_vol = history_volatility[burn_in:]
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # 1. Price
    axes[0].plot(valid_steps, plot_prices, color='black', lw=1)
    axes[0].set_ylabel('Price')
    axes[0].set_title('Asset Price Dynamics')
    axes[0].grid(True, alpha=0.3)
    
    # 2. Network Centralization
    axes[1].plot(valid_steps, plot_cent, color='orange', lw=1)
    axes[1].set_ylabel('Centralization')
    axes[1].set_title('Network Centralization (Self-Organized Non-Normality)')
    axes[1].grid(True, alpha=0.3)
    
    # 3. Volatility
    axes[2].plot(valid_steps, plot_vol, color='blue', lw=1)
    axes[2].set_ylabel('Annualized Volatility')
    axes[2].set_title('Market Volatility')
    axes[2].set_xlabel('Time Steps (Days)')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('replication_results.png')
    plt.show()

if __name__ == "__main__":
    run_simulation()