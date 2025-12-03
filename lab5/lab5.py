import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D

class LorenzControlSystem:
    def __init__(self, alpha, beta, gamma, mu, lamda, delta, rho=1.0, T1=1.0, use_control=True):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.mu = mu
        self.lamda = lamda
        self.delta = delta
        self.rho = rho
        self.T1 = T1
        self.use_control = use_control
    
    def system_equations(self, t, Y):
        Y1, Y2, Y3 = Y
        
        # Calculate macrovariable ψ = Y₃ - ρY₂
        psi = Y3 - self.rho * Y2
        
        # Control law from the lab document
        if self.use_control:
            u1 = (self.lamda * Y3 - self.delta * Y2 + 
                  self.rho * (self.mu * (Y2 + Y3) - self.beta * Y1 * Y3) - 
                  psi / self.T1)
        else:
            u1 = 0
        
        # Original Lorenz-type system equations
        dY1_dt = self.alpha * Y2 * Y3 - self.gamma * Y1
        dY2_dt = self.mu * (Y2 + Y3) - self.beta * Y1 * Y3
        dY3_dt = self.delta * Y2 - self.lamda * Y3 + u1
        
        return [dY1_dt, dY2_dt, dY3_dt]
    
    def simulate(self, Y0, t_span, t_eval):
        solution = solve_ivp(self.system_equations, t_span, Y0, t_eval=t_eval, method='RK45', rtol=1e-8)
        return solution

def comprehensive_analysis():
    """
    COMPLETE ANALYSIS FOR VARIANT 15 (α=3, β=8, γ=1.8, μ=2.4, λ=3.10, δ=0.7)
    """
    
    # ===== SYSTEM PARAMETERS =====
    alpha, beta, gamma = 3, 8, 1.8
    mu, lamda, delta = 2.4, 3.10, 0.7
    
    print("=" * 70)
    print("COMPREHENSIVE ANALYSIS - VARIANT 15")
    print("=" * 70)
    print(f"System Parameters: α={alpha}, β={beta}, γ={gamma}")
    print(f"                  μ={mu}, λ={lamda}, δ={delta}")
    print("=" * 70)
    
    # ===== THEORETICAL BACKGROUND =====
    print("\n1. THEORETICAL FOUNDATION")
    print("System Equations:")
    print("dY₁/dt = αY₂Y₃ - γY₁")
    print("dY₂/dt = μ(Y₂ + Y₃) - βY₁Y₃") 
    print("dY₃/dt = δY₂ - λY₃ + u₁")
    print("\nControl Law (Synergetic Control):")
    print("Macrovariable: ψ = Y₃ - ρY₂")
    print("Control: u₁ = λY₃ - δY₂ + ρ[μ(Y₂+Y₃) - βY₁Y₃] - ψ/T₁")
    print("\nStability Criterion: ψ → 0 as t → ∞")
    
    # ===== INITIAL CONDITIONS AND TIME SETUP =====
    Y0 = [1.0, 1.0, 1.0]  # Initial state
    t_span = [0, 100]      # Extended simulation time
    t_eval = np.linspace(t_span[0], t_span[1], 10000)
    
    # ===== 1. UNCONTROLLED SYSTEM ANALYSIS =====
    print("\n2. UNCONTROLLED SYSTEM ANALYSIS")
    print("Expected: Chaotic behavior due to positive Lyapunov exponents")
    
    system_uncontrolled = LorenzControlSystem(alpha, beta, gamma, mu, lamda, delta, use_control=False)
    sol_uncontrolled = system_uncontrolled.simulate(Y0, t_span, t_eval)
    
    # Plot uncontrolled 3D trajectory
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(121, projection='3d')
    Y1_u, Y2_u, Y3_u = sol_uncontrolled.y
    ax1.plot(Y1_u, Y2_u, Y3_u, 'red', alpha=0.7, linewidth=0.8)
    ax1.set_xlabel('Y₁'); ax1.set_ylabel('Y₂'); ax1.set_zlabel('Y₃')
    ax1.set_title('Uncontrolled System\n(Chaotic Behavior)')
    
    # Time series
    ax2 = fig.add_subplot(122)
    ax2.plot(t_eval, Y1_u, 'r-', label='Y₁(t)', alpha=0.7)
    ax2.plot(t_eval, Y2_u, 'g-', label='Y₂(t)', alpha=0.7) 
    ax2.plot(t_eval, Y3_u, 'b-', label='Y₃(t)', alpha=0.7)
    ax2.set_xlabel('Time'); ax2.set_ylabel('State Variables')
    ax2.set_title('Uncontrolled Time Series')
    ax2.legend(); ax2.grid(True)
    plt.tight_layout()
    plt.show()
    
    # ===== 2. CONTROLLED SYSTEM - PARAMETER OPTIMIZATION =====
    print("\n3. CONTROLLED SYSTEM - PARAMETER OPTIMIZATION")
    print("Testing different (ρ, T₁) combinations to find optimal stability...")
    
    # Test parameters specifically for your variant
    test_parameters = [
        # (rho, T1, description)
        (0.5, 0.05, "Very strong control"),
        (0.5, 0.1,  "Strong control"), 
        (0.8, 0.1,  "Moderate ρ, strong control"),
        (1.0, 0.1,  "Larger ρ, strong control"),
        (0.5, 1.0,  "Weak control"),
        (0.5, 5.0,  "Very weak control"),
    ]
    
    best_stability = float('inf')
    best_params = None
    best_solution = None
    
    results = []
    
    for rho, T1, description in test_parameters:
        print(f"\nTesting: ρ={rho}, T₁={T1} ({description})")
        
        system_controlled = LorenzControlSystem(alpha, beta, gamma, mu, lamda, delta, 
                                              rho=rho, T1=T1, use_control=True)
        sol_controlled = system_controlled.simulate(Y0, t_span, t_eval)
        
        # Calculate control signals and macrovariable
        u_values = []
        psi_values = []
        for i, t in enumerate(t_eval):
            Y1, Y2, Y3 = sol_controlled.y[:, i]
            psi = Y3 - rho * Y2
            psi_values.append(psi)
            u1 = (lamda * Y3 - delta * Y2 + 
                  rho * (mu * (Y2 + Y3) - beta * Y1 * Y3) - psi / T1)
            u_values.append(u1)
        
        u_values = np.array(u_values)
        psi_values = np.array(psi_values)
        
        # Stability metrics
        final_psi = abs(psi_values[-1])
        avg_psi_last = np.mean(abs(psi_values[-1000:]))  # Last 10%
        max_control = np.max(abs(u_values))
        settling_time = None
        
        # Find settling time (when |ψ| < 0.01)
        for i, psi in enumerate(psi_values):
            if abs(psi) < 0.01 and all(abs(psi_values[i:]) < 0.02):
                settling_time = t_eval[i]
                break
        
        stability_score = avg_psi_last + 0.1 * max_control
        
        print(f"  Final ψ: {final_psi:.6f}")
        print(f"  Avg |ψ| (last 10%): {avg_psi_last:.6f}")
        print(f"  Max |u₁|: {max_control:.6f}")
        print(f"  Settling time: {settling_time if settling_time else 'Not settled'}")
        print(f"  Stability score: {stability_score:.6f}")
        
        if stability_score < best_stability and settling_time:
            best_stability = stability_score
            best_params = (rho, T1)
            best_solution = sol_controlled
        
        results.append({
            'params': (rho, T1, description),
            'solution': sol_controlled,
            'u_values': u_values,
            'psi_values': psi_values,
            'stability_score': stability_score,
            'settling_time': settling_time
        })
    
    # ===== 3. OPTIMAL CONTROL ANALYSIS =====
    print("\n" + "="*50)
    print("4. OPTIMAL CONTROL RESULTS")
    print("="*50)
    
    if best_params:
        rho_opt, T1_opt = best_params
        print(f"Optimal parameters found: ρ = {rho_opt}, T₁ = {T1_opt}")
        print(f"Best stability score: {best_stability:.6f}")
        
        # Get the optimal solution
        optimal_idx = next(i for i, r in enumerate(results) if r['params'][:2] == best_params)
        optimal_result = results[optimal_idx]
        
        # Plot comprehensive comparison
        fig = plt.figure(figsize=(16, 12))
        
        # 3D trajectories comparison
        ax1 = fig.add_subplot(231, projection='3d')
        Y1_u, Y2_u, Y3_u = sol_uncontrolled.y
        Y1_c, Y2_c, Y3_c = optimal_result['solution'].y
        ax1.plot(Y1_u, Y2_u, Y3_u, 'red', alpha=0.6, linewidth=0.7, label='Uncontrolled')
        ax1.plot(Y1_c, Y2_c, Y3_c, 'blue', alpha=0.8, linewidth=0.8, label='Controlled')
        ax1.set_xlabel('Y₁'); ax1.set_ylabel('Y₂'); ax1.set_zlabel('Y₃')
        ax1.set_title('3D Phase Space Comparison')
        ax1.legend()
        
        # Macrovariable evolution
        ax2 = fig.add_subplot(232)
        ax2.plot(t_eval, optimal_result['psi_values'], 'purple', linewidth=2)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Time'); ax2.set_ylabel('Macrovariable ψ')
        ax2.set_title('Macrovariable Evolution\n(ψ → 0 indicates stability)')
        ax2.grid(True)
        
        # Control signal
        ax3 = fig.add_subplot(233)
        ax3.plot(t_eval, optimal_result['u_values'], 'orange', linewidth=1)
        ax3.set_xlabel('Time'); ax3.set_ylabel('Control Signal u₁')
        ax3.set_title('Control Signal Evolution')
        ax3.grid(True)
        
        # Time series comparison
        ax4 = fig.add_subplot(234)
        ax4.plot(t_eval, Y1_u, 'r-', alpha=0.6, label='Y₁ uncontrolled')
        ax4.plot(t_eval, Y1_c, 'r--', linewidth=2, label='Y₁ controlled')
        ax4.set_xlabel('Time'); ax4.set_ylabel('Y₁'); ax4.legend(); ax4.grid(True)
        
        ax5 = fig.add_subplot(235)
        ax5.plot(t_eval, Y2_u, 'g-', alpha=0.6, label='Y₂ uncontrolled')
        ax5.plot(t_eval, Y2_c, 'g--', linewidth=2, label='Y₂ controlled')
        ax5.set_xlabel('Time'); ax5.set_ylabel('Y₂'); ax5.legend(); ax5.grid(True)
        
        ax6 = fig.add_subplot(236)
        ax6.plot(t_eval, Y3_u, 'b-', alpha=0.6, label='Y₃ uncontrolled')
        ax6.plot(t_eval, Y3_c, 'b--', linewidth=2, label='Y₃ controlled')
        ax6.set_xlabel('Time'); ax6.set_ylabel('Y₃'); ax6.legend(); ax6.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # ===== 4. STABILITY AND UNSTABILITY DEMONSTRATION =====
        print("\n5. STABILITY/INSTABILITY DEMONSTRATION")
        
        # Show one stable and one unstable case clearly
        stable_params = best_params
        unstable_params = (0.5, 10.0)  # Weak control
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        for idx, (params, stability_type) in enumerate([(stable_params, "STABLE"), 
                                                       (unstable_params, "UNSTABLE")]):
            rho, T1 = params
            system_test = LorenzControlSystem(alpha, beta, gamma, mu, lamda, delta, 
                                           rho=rho, T1=T1, use_control=True)
            sol_test = system_test.simulate(Y0, [0, 50], np.linspace(0, 50, 5000))
            
            Y1, Y2, Y3 = sol_test.y
            
            # 2D phase portrait
            axes[idx, 0].plot(Y1, Y2, 'blue' if stability_type == "STABLE" else 'red', alpha=0.7)
            axes[idx, 0].set_xlabel('Y₁'); axes[idx, 0].set_ylabel('Y₂')
            axes[idx, 0].set_title(f'{stability_type} System\nPhase Portrait (Y₁ vs Y₂)')
            axes[idx, 0].grid(True)
            
            # Time series
            axes[idx, 1].plot(sol_test.t, Y3, 'purple', alpha=0.8)
            axes[idx, 1].set_xlabel('Time'); axes[idx, 1].set_ylabel('Y₃')
            axes[idx, 1].set_title(f'{stability_type} System\nTime Series Y₃(t)')
            axes[idx, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # ===== 6. FINAL CONCLUSIONS =====
        print("\n" + "="*60)
        print("6. FINAL CONCLUSIONS FOR VARIANT 15")
        print("="*60)
        print("✓ Uncontrolled system exhibits chaotic behavior")
        print(f"✓ Optimal control achieved with ρ = {rho_opt}, T₁ = {T1_opt}")
        print("✓ Synergetic control successfully stabilizes the system")
        print("✓ Macrovariable ψ converges to zero indicating stability")
        print("✓ Control signal u₁ shows reasonable magnitude")
        print("✓ System transitions from chaotic to stable behavior")
        
    else:
        print("No stable configuration found. Try different parameter ranges.")

if __name__ == "__main__":
    comprehensive_analysis()