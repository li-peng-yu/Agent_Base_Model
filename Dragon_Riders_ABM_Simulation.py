import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ==========================================
# 1. Configuration & Parameters (Table C.1)
# ==========================================

@dataclass
class SimConfig:
    # Market Parameters
    T: int = 50000          # Simulation length (reduced from 50k for demo performance)
    dt_days: int = 1       # Time step interpretation
    
    # Assets
    Rf: float = 0.01 / 250 # Daily risk-free rate
    d0: float = 0.04 / 250 # Initial dividend
    rd: float = 0.04 / 250 # Expected dividend growth rate
    sigma_d: float = 0.000016 # Std dev of dividend growth
    P0: float = 1.0        # Initial price
    sigma_r: float = 0.02  # Expected volatility of risky asset (approx 1-2% daily)
    gamma: float = 2.0     # Risk aversion (Standard value, not explicitly fixed in C.1 but implied CRRA)

    # Noise Traders
    N_noise: int = 1000
    W0_noise: float = 1e9
    x0_noise: float = 0.3
    cH: float = 1.0        # Momentum weight
    cS: float = 1.0        # Opinion weight
    p_plus: float = 0.2    # Base switching prob (approx)
    p_minus: float = 0.2
    theta: float = 0.95    # Memory parameter for momentum
    
    # Fundamentalists
    W0_fund: float = 1e9
    x0_fund: float = 0.3
    
    # Dragon Riders (DR)
    W0_dr: float = 1e7     # Start small to test impact
    x0_dr: float = 0.3
    
    # Social Coupling (OU Process)
    kappa0: float = 0.98
    mu_k: float = 0.98
    eta_k: float = 0.11    # Mean reversion speed
    sigma_k: float = 0.01  # Volatility of kappa
    
    # LPPLS Configuration
    lppls_enabled: bool = True
    window_sizes: List[int] = field(default_factory=lambda: [100, 200, 400]) # Simplified from paper's "20 to 500"
    dt_lppls: int = 5      # Perform fit every N steps (optimization is slow)

# ==========================================
# 2. LPPLS Model & Fitting Engine
# ==========================================

class LPPLS:
    """
    Implements the Log-Periodic Power Law Singularity model fitting.
    Eq (A.1): ln P(t) = A + B(tc - t)^m + C(tc - t)^m * cos(omega * ln(tc - t) - phi)
    """
    def __init__(self):
        pass

    @staticmethod
    def lppls_func(t, tc, m, omega, A, B, C, phi):
        dt = tc - t
        # Filter invalid values to prevent math errors during optimization
        dt = np.maximum(dt, 1e-8) 
        return A + np.power(dt, m) * (B + C * np.cos(omega * np.log(dt) - phi))

    def fit(self, time_idx, log_prices):
        """
        Performs a fit on a single window. 
        Returns (is_bubble, parameters, confidence_metric)
        """
        t = time_idx
        y = log_prices
        
        # Heuristic initial guesses
        t_current = t[-1]
        tc_init = t_current + 20 # Assume crash is near future
        m_init = 0.5
        omega_init = 6.0
        
        # 1. Simplified Nonlinear Optimization
        # We optimize (tc, m, omega) and solve (A, B, C, phi) linearly or include them.
        # For full reproduction speed, we use a reduced parameter search or full minimization.
        # Here we use 'minimize' for general robustness.
        
        def objective(params):
            tc, m, omega, A, B, C, phi = params
            if tc <= t_current: return 1e9
            if not (0.1 <= m <= 0.9): return 1e9
            if not (2 <= omega <= 25): return 1e9
            
            y_pred = self.lppls_func(t, tc, m, omega, A, B, C, phi)
            resid = y - y_pred
            return np.sum(resid**2)

        x0 = [tc_init, m_init, omega_init, y[-1], -0.1, 0.0, 0.0]
        # Bounds help convergence
        bounds = [
            (t_current + 1, t_current + 100), # tc
            (0.1, 0.9),       # m
            (2.0, 25.0),      # omega
            (None, None), (None, None), (None, None), (0, 2*np.pi)
        ]
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, tol=1e-4)
            params = res.x
            success = res.success
            
            # Filtering conditions (Table A.1 in paper)
            tc, m, omega, A, B, C, phi = params
            
            is_bubble = False
            # Condition 1: Fast oscillation constraint (damping)
            # Paper defines Damping = (m * |B|) / (m * |C|) approx? 
            # Actually Table A.1 says D = (m * |B|) / (omega * |C|) > 1 (typographic error in paper? usually beta/omega)
            # We follow the standard definition or simple bounds.
            # Here we implement basic bounds check as "is_bubble"
            if success and (0.1 < m < 0.9) and (2 < omega < 25):
                is_bubble = True
                
            return is_bubble, params
            
        except Exception:
            return False, []

# ==========================================
# 3. Agents
# ==========================================

class Agent:
    def __init__(self, W0, x0):
        self.W = W0          # Wealth
        self.x = x0          # Fraction in risky asset
        self.n = 0.0         # Number of shares (calculated dynamically)

class Fundamentalist(Agent):
    def get_demand_params(self, E_R, R_f, gamma, sigma_r):
        """
        Returns the demand function parameters for Eq (11).
        Demand is derived from maximizing CRRA utility.
        """
        # Simplification: The paper derives specific demand evolution.
        # In the market clearing step (Appendix B), fundamentalists contribute to a, b, c.
        pass

class NoiseTraderGroup(Agent):
    def __init__(self, cfg: SimConfig):
        super().__init__(cfg.W0_noise, cfg.x0_noise)
        self.N_plus = int(cfg.N_noise * cfg.x0_noise) # Traders in risky
        self.N_minus = cfg.N_noise - self.N_plus      # Traders in risk-free
        self.S = (self.N_plus - self.N_minus) / cfg.N_noise # Opinion Index
        self.H = 0.0 # Momentum
        self.kappa = cfg.kappa0
        
    def update_state(self, R_t, cfg: SimConfig):
        # 1. Update Momentum (Eq 13)
        self.H = cfg.theta * self.H + (1 - cfg.theta) * R_t
        
        # 2. Update Kappa (OU Process - Eq 17)
        # kappa(t) = kappa(t-1) + eta*(mu - kappa) + sigma * noise
        noise = np.random.normal(0, 1)
        self.kappa += cfg.eta_k * (cfg.mu_k - self.kappa) + cfg.sigma_k * noise
        
        # 3. Transition Probabilities (Eq 16)
        # S_t = 2*x_n - 1
        term = self.kappa * (self.S + self.H)
        p_plus_trans = (cfg.p_plus / 2.0) * (1.0 - term)  # + to -
        p_minus_trans = (cfg.p_minus / 2.0) * (1.0 + term) # - to +
        
        # Clamp probabilities
        p_plus_trans = np.clip(p_plus_trans, 0, 1)
        p_minus_trans = np.clip(p_minus_trans, 0, 1)
        
        # 4. Stochastic switching
        # Number of switchers from + to -
        k_plus_minus = np.random.binomial(self.N_plus, p_plus_trans)
        # Number of switchers from - to +
        k_minus_plus = np.random.binomial(self.N_minus, p_minus_trans)
        
        self.N_plus += (k_minus_plus - k_plus_minus)
        self.N_minus = cfg.N_noise - self.N_plus
        
        # Update fractions and opinion
        self.x = self.N_plus / cfg.N_noise  # Eq 14
        self.S = (self.N_plus - self.N_minus) / cfg.N_noise

class DragonRider(Agent):
    def __init__(self, cfg: SimConfig):
        super().__init__(cfg.W0_dr, cfg.x0_dr)
        self.lppls_model = LPPLS()
        self.in_bubble = False
        self.target_x = cfg.x0_dr
        self.strategy_state = 'neutral' # neutral, riding, exit
        
    def decide_strategy(self, price_history, t, cfg: SimConfig):
        """
        Implements the Flowchart in Fig 2.
        """
        # Fallback to fundamentalist if not enough data
        if len(price_history) < 50:
            return cfg.x0_dr # Default fundamentalist behavior (simplification)
            
        # Run LPPLS fits
        # In a full simulation, we scan multiple windows. 
        # Here we scan the predefined windows in cfg.
        
        log_P = np.log(np.array(price_history))
        votes = 0
        total_fits = 0
        
        for w in cfg.window_sizes:
            if t > w:
                segment = log_P[-w:]
                time_segment = np.arange(t-w, t)
                is_bub, _ = self.lppls_model.fit(time_segment, segment)
                if is_bub:
                    votes += 1
                total_fits += 1
        
        confidence = votes / total_fits if total_fits > 0 else 0
        
        # Decision Logic (Simplified from Fig 2)
        # If high confidence -> Buy (Ride) -> x = 1.0
        # If very high confidence & damping check -> Sell (Crash imminent) -> x = 0.0
        # Else -> Fundamentalist behavior
        
        # Note: The paper says DR acts as fundamentalist (x^f) if not in bubble
        # Here we approximate x^f as the current x of fundamentalist
        # but DR calculates its own optimal x based on bubble.
        
        if confidence > 0.5: # Thresholds from Table 1
            if confidence > 0.8: # Late stage
                self.target_x = 0.0
            else: # Early stage
                self.target_x = 1.0
        else:
            self.target_x = -1 # Signal to use fundamentalist strategy

        return self.target_x

# ==========================================
# 4. Market & Simulation Engine
# ==========================================

class Market:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.fund = Fundamentalist(cfg.W0_fund, cfg.x0_fund)
        self.noise = NoiseTraderGroup(cfg)
        self.dr = DragonRider(cfg)
        
        self.P = [cfg.P0]
        self.D = [cfg.d0]
        self.t = 0
        
    def step(self):
        t_curr = self.t
        P_old = self.P[-1]
        d_old = self.D[-1]
        
        # 1. Evolve Dividend
        # d_t = d_{t-1} * (1 + r_d + sigma * N(0,1))
        shock = np.random.normal(0, 1)
        r_dt = self.cfg.rd + self.cfg.sigma_d * shock
        d_new = d_old * (1 + r_dt)
        self.D.append(d_new)
        
        # 2. Update Agents Strategies (for next step demand)
        # Noise traders update state based on previous return
        if t_curr > 0:
            R_prev = (self.P[-1] / self.P[-2]) - 1
        else:
            R_prev = 0
            
        self.noise.update_state(R_prev, self.cfg)
        
        # DR Strategy Update (Periodically)
        dr_target_x = -1
        if self.cfg.lppls_enabled and t_curr % self.cfg.dt_lppls == 0 and t_curr > 100:
            dr_target_x = self.dr.decide_strategy(self.P, t_curr, self.cfg)
        
        # 3. Solve for Price P_t (Appendix B Quadratic Equation)
        # a*P^2 + b*P + c = 0
        # We need to map the "v" variables from Appendix B Eq (B.5)
        
        # Common variables
        # Expected return for Fundamentalists (assumed = r_d)
        E_R = self.cfg.rd 
        Rf = self.cfg.Rf
        gamma = self.cfg.gamma
        sigma2 = self.cfg.sigma_r ** 2
        
        # v parameters (Eq B.5)
        # Note: In Eq B.5, v2 depends on P_t-1 (which is P_old)
        v1 = (E_R - Rf) / (gamma * sigma2)
        v2 = (d_new / P_old) - Rf - 1
        v3 = Rf + 1
        v4 = d_new * (1 + self.cfg.rd) / (gamma * sigma2) # Approx (1+r) as (1+rd)
        
        # Agent specific weights
        # Fundamentalist
        W_f = self.fund.W
        x_f_prev = self.fund.x
        
        # Noise
        W_n = self.noise.W
        x_n_curr = self.noise.x # Noise trader x is determined BEFORE price (Eq 14)
        x_n_prev = x_n_curr # Approximation: Paper Eq 22 implies x_t^n is used
        
        # DR
        W_dr = self.dr.W
        x_dr_prev = self.dr.x
        
        # Determine DR mode (Fundamentalist-like vs Fixed)
        # k_t = 1 if using fundamentalist strategy, 0 if fixed x
        # l_t = fixed x value if k_t = 0
        
        if dr_target_x == -1:
            # Acts as fundamentalist
            k_dr = 1.0
            l_dr = 0.0
        else:
            # Acts as chartist/timer (fixed demand fraction)
            k_dr = 0.0
            l_dr = dr_target_x
            
        # Coefficients (Eq B.8, B.9, B.10)
        # a_t
        term_dr_a = W_dr * x_dr_prev * (k_dr * v1 + l_dr - 1)
        term_f_a = W_f * x_f_prev * (v1 - 1)
        term_n_a = W_n * x_n_prev * (x_n_curr - 1)
        a_t = (term_dr_a + term_f_a + term_n_a) / P_old
        
        # b_t
        # Need to handle the complex structure of b_t
        # b_t = Sum [ W * ( ... ) ]
        
        # Helper for inner bracket: (x_prev * v2 + v3)
        inner_dr = x_dr_prev * v2 + v3
        inner_f = x_f_prev * v2 + v3
        inner_n = x_n_prev * v2 + v3
        
        # DR part of b
        b_dr = W_dr * (x_dr_prev * k_dr * (v4 / P_old) + (k_dr * v1 + l_dr) * inner_dr)
        # Fund part of b
        b_f = W_f * (x_f_prev * (v4 / P_old) + v1 * inner_f)
        # Noise part of b
        b_n = W_n * x_n_curr * inner_n
        
        b_t = b_dr + b_f + b_n
        
        # c_t
        c_dr = W_dr * k_dr * v4 * inner_dr
        c_f = W_f * v4 * inner_f
        # Noise doesn't contribute to c_t directly in the same form (see Eq B.10)
        # Wait, Eq B.10 only has DR and F terms. Noise term is 0 in c_t?
        # Looking at Eq B.6 derivation:
        # The term with P^0 comes from supply/demand constants.
        # Check B.10: c_t = Sum(W_dr...) + W_f... (Yes, no W_n term in c_t)
        c_t = c_dr + c_f
        
        # Solve Quadratic
        # P = (-b +/- sqrt(b^2 - 4ac)) / 2a
        delta = b_t**2 - 4 * a_t * c_t
        
        if delta < 0:
            # Fallback if no real solution (should not happen in stable region)
            # print(f"Warning: Delta < 0 at t={t_curr}. Using P_old.")
            P_new = P_old
        else:
            # We need the positive price
            sol1 = (-b_t + np.sqrt(delta)) / (2 * a_t)
            sol2 = (-b_t - np.sqrt(delta)) / (2 * a_t)
            
            if sol1 > 0 and sol2 > 0:
                # Usually the one closer to P_old, or the larger one? 
                # ABM literature usually takes the positive, stable root.
                # Given a_t is likely negative (demand slope), and c_t positive?
                # Let's take the one closest to P_old
                if abs(sol1 - P_old) < abs(sol2 - P_old):
                    P_new = sol1
                else:
                    P_new = sol2
            elif sol1 > 0:
                P_new = sol1
            else:
                P_new = sol2
                
        self.P.append(P_new)
        
        # 4. Update Wealth (Post-trading)
        # Calculate Return R_t
        R_t = (P_new / P_old) - 1
        
        # Needed for Wealth update Eq (10)
        # W_t = (R_t + d_t/P_t-1 - R_f) * x_t-1 * W_t-1 + W_t-1 * (1 + R_f)
        excess_return = R_t + (d_new / P_old) - Rf
        
        # Update Fund
        self.fund.W = excess_return * self.fund.x * self.fund.W + self.fund.W * (1 + Rf)
        # Update Fund x for NEXT step (Eq 8)
        # x_t^f approx (E_R + d_t/P_t (1+rd) - Rf) / (gamma * sigma^2)
        # Note: Denominator assumes price P_t.
        denom = gamma * (sigma2 + (d_new * self.cfg.sigma_d / P_new)**2) # Full variance (Eq 8)
        num = E_R + (d_new/P_new)*(1+self.cfg.rd) - Rf
        self.fund.x = num / denom
        
        # Update Noise
        self.noise.W = excess_return * self.noise.x * self.noise.W + self.noise.W * (1 + Rf)
        # Noise x is already updated in step 2
        
        # Update DR
        self.dr.W = excess_return * self.dr.x * self.dr.W + self.dr.W * (1 + Rf)
        # Update DR x for NEXT step
        if dr_target_x == -1:
            self.dr.x = self.fund.x # Copy fundamentalist
        else:
            # Smooth transition (Eq 23)
            # x_t = x_t-1 + (V_t - x_t-1)/tau * s(V)
            # Here simplified to immediate update or simple smoothing
            tau = 15.0
            self.dr.x = self.dr.x + (dr_target_x - self.dr.x) / tau

        self.t += 1

# ==========================================
# 5. Main Execution
# ==========================================

if __name__ == "__main__":
    np.random.seed(42)
    config = SimConfig()
    
    # Run Simulation
    market = Market(config)
    print(f"Starting Simulation for {config.T} steps...")
    
    history_P = []
    history_Kappa = []
    
    for _ in range(config.T):
        market.step()
        history_P.append(market.P[-1])
        history_Kappa.append(market.noise.kappa)
        if _ % 100 == 0:
            print(f"Step {_}/{config.T} | Price: {market.P[-1]:.2f} | Noise Kappa: {market.noise.kappa:.2f}")

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    ax1.plot(history_P, color='blue', label='Price')
    ax1.set_ylabel('Price')
    ax1.set_title('Asset Price Dynamics with Dragon Riders')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(history_Kappa, color='orange', label='Social Coupling (Kappa)')
    ax2.axhline(y=1.0, color='r', linestyle='--', label='Critical Kappa ~ 1.0')
    ax2.set_ylabel('Kappa')
    ax2.set_xlabel('Time Steps')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    # Save plot instead of show() for file generation workflow compatibility
    plt.savefig('simulation_results.png')
    print("Simulation complete. Results saved to simulation_results.png")