import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp
from scipy.signal import hilbert

# ----------------------
# System Parameters
# ----------------------
delta = 0.05        # damping coefficient
a = 0.5             # target squared amplitude
gamma = 0.05        # relaxation rate for theta
theta0 = 1.0        # baseline natural frequency squared

epsilon_success = 0.01   # slow adaptation
epsilon_fail = 0.5       # fast adaptation

t = np.linspace(0, 100, 500)      # time array
x0 = [1.0, 0.0, 0.5]              # initial conditions [x, v, theta]

# ----------------------
# Analysis Parameters
# ----------------------
start_time = 60.0      # analysis window start (s)
end_time   = 80.0      # analysis window end (s)
theta_tol  = 0.01      # threshold for success criterion

# ----------------------
# ODE Definition                                                                                       import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp
from scipy.signal import hilbert

# ----------------------
# System Parameters
# ----------------------
delta = 0.05        # damping coefficient
a = 0.5             # target squared amplitude
gamma = 0.05        # relaxation rate for theta
theta0 = 1.0        # baseline natural frequency squared

epsilon_success = 0.01   # slow adaptation
epsilon_fail = 0.5       # fast adaptation

t = np.linspace(0, 100, 500)      # time array
x0 = [1.0, 0.0, 0.5]              # initial conditions [x, v, theta]

# ----------------------
# Analysis Parameters
# ----------------------
start_time = 60.0      # analysis window start (s)
end_time   = 80.0      # analysis window end (s)
theta_tol  = 0.01      # threshold for success criterion

# ----------------------
# ODE Definition
# ----------------------
def adaptive_oscillator(y, t, epsilon):
    x, v, theta = y
    dxdt = v
    dvdt = -theta * x - delta * v
    dthetadt = epsilon * (x**2 - a) - gamma * (theta - theta0)
    return [dxdt, dvdt, dthetadt]  

# ----------------------
# Utility Functions
# ----------------------
def instantaneous_frequency(x, t, epsilon):
    analytic_signal = hilbert(x)
    phase = np.unwrap(np.angle(analytic_signal))
    freq = np.gradient(phase, t) / (2 * np.pi)
    if epsilon > 0.1:
        print(f"Warning: Large epsilon ({epsilon}) may violate Hilbert transform narrowband assumption.")
    return freq

def get_convergence_window_indices(t, start_time, end_time):
    """Return the start and end indices for the convergence window based on time."""
    start_idx = np.searchsorted(t, start_time, side='left')
    end_idx = np.searchsorted(t, end_time, side='right')
    return start_idx, end_idx

def prepare_plotting_data(t, x_s, x_f, start_time, end_time, epsilon_success, epsilon_fail):
    """Prepare data for plotting: time window and instantaneous frequencies."""
    start_idx, end_idx = get_convergence_window_indices(t, start_time, end_time)
    t_window = t[start_idx:end_idx]
    freq_s = instantaneous_frequency(x_s[start_idx:end_idx], t_window, epsilon_success)
    freq_f = instantaneous_frequency(x_f[start_idx:end_idx], t_window, epsilon_fail)
    return t_window, freq_s, freq_f

# ----------------------
# Simulation Function
# ----------------------
def simulate_case(epsilon, theta_tol=theta_tol):
    if epsilon > 0.1:
        print(f"Using solve_ivp for epsilon={epsilon} to ensure numerical stability")
        sol = solve_ivp(
            lambda t, y: adaptive_oscillator(y, t, epsilon),
            t_span=(t[0], t[-1]),
            y0=x0,
            t_eval=t,
            method='BDF',
            rtol=1e-6,
            atol=1e-8
        )
        x, v, theta = sol.y
    else:
        sol = odeint(adaptive_oscillator, x0, t, args=(epsilon,), rtol=1e-6, atol=1e-8)
        x, v, theta = sol.T
    energy = 0.5 * v**2 + 0.5 * theta * x**2

    start_idx, end_idx = get_convergence_window_indices(t, start_time, end_time)
    steady_window = slice(start_idx, end_idx)
    mean_theta = np.mean(theta[steady_window])
    std_theta = np.std(theta[steady_window])

    # Calculate damped theoretical frequency
    damped_freq = np.sqrt(max(mean_theta - (delta**2 / 4), 0)) / (2 * np.pi)  # Ensure non-negative under sqrt

    status = "Success" if std_theta < theta_tol else "Fail"
    freq = instantaneous_frequency(x[steady_window], t[steady_window], epsilon)
    mean_freq = np.mean(freq)

    return x, v, theta, energy, status, mean_theta, std_theta, mean_freq, damped_freq, theta_tol

# ----------------------
# Plotting Functions                            
# ----------------------
def plot_stacked(t, theta_s, theta_f, x_s, x_f, energy_s, energy_f,
                 eps_s, eps_f, status_s, status_f):
    plt.figure(figsize=(10, 10))
    start_idx, end_idx = get_convergence_window_indices(t, start_time, end_time)
    convergence_window = [t[start_idx], t[end_idx]]

    color_s = 'green' if status_s == "Success" else 'red'
    color_f = 'blue' if status_f == "Success" else 'orange'

    # Theta
    plt.subplot(3, 1, 1)
    plt.axvspan(*convergence_window, color='gray', alpha=0.2)
    plt.plot(t, theta_s, color=color_s, label=f'Theta s (ε={eps_s:.4f}, {status_s})')
    plt.plot(t, theta_f, color=color_f, linestyle='--', label=f'Theta f (ε={eps_f:.4f}, {status_f})')
    plt.xlabel('Time (s)'); plt.ylabel('Theta'); plt.title('Theta Evolution')
    plt.legend(); plt.grid(True)

    # Position
    plt.subplot(3, 1, 2)
    plt.axvspan(*convergence_window, color='gray', alpha=0.2)
    plt.plot(t, x_s, color=color_s, label=f'x s (ε={eps_s:.4f}, {status_s})')
    plt.plot(t, x_f, color=color_f, linestyle='--', label=f'x f (ε={eps_f:.4f}, {status_f})')
    plt.xlabel('Time (s)'); plt.ylabel('x'); plt.title('Position')
    plt.legend(); plt.grid(True)

    # Energy
    plt.subplot(3, 1, 3)                                         
    plt.axvspan(*convergence_window, color='gray', alpha=0.2)                        
    plt.plot(t, energy_s, color=color_s, label=f'Energy s (ε={eps_s:.4f}, {status_s})')
    plt.plot(t, energy_f, color=color_f, linestyle='--', label=f'Energy f (ε={eps_f:.4f}, {status_f})')
    plt.xlabel('Time (s)'); plt.ylabel('Energy'); plt.title('Energy')
    plt.legend(); plt.grid(True)

    plt.tight_layout()
    plt.show()

def plot_phase_space(x_s, v_s, x_f, v_f, eps_s, eps_f, status_s, status_f):
    plt.figure(figsize=(6, 6))
    color_s = 'green' if status_s == "Success" else 'red'
    color_f = 'blue' if status_f == "Success" else 'orange'

    plt.plot(x_s, v_s, color=color_s, label=f'Phase s (ε={eps_s:.4f}, {status_s})')
    plt.plot(x_f, v_f, color=color_f, linestyle='--', label=f'Phase f (ε={eps_f:.4f}, {status_f})')
    plt.xlabel('x'); plt.ylabel('v'); plt.title('Phase Space')
    plt.legend(); plt.grid(True); plt.axis('equal')                                              
    plt.show()

def plot_instantaneous_frequency(t, freq_s, freq_f, eps_s, eps_f, status_s, status_f):
    plt.figure(figsize=(10, 6))
    color_s = 'green' if status_s == "Success" else 'red'
    color_f = 'blue' if status_f == "Success" else 'orange'

    plt.plot(t, freq_s, color=color_s, label=f'IF s (ε={eps_s:.4f}, {status_s})')
    plt.plot(t, freq_f, color=color_f, linestyle='--', label=f'IF f (ε={eps_f:.4f}, {status_f})')
    plt.xlabel('Time (s)'); plt.ylabel('Frequency (Hz)'); plt.title('Instantaneous Frequency')
    plt.legend(); plt.grid(True)
    plt.show()

# ----------------------
# Main Execution
# ----------------------                                
if __name__ == "__main__":
    # Run simulations
    x_s, v_s, theta_s, energy_s, status_s, mean_theta_s, std_theta_s, mean_freq_s, damped_freq_s, tol_s = simulate_case(epsilon_success)
    x_f, v_f, theta_f, energy_f, status_f, mean_theta_f, std_theta_f, mean_freq_f, damped_freq_f, tol_f = simulate_case(epsilon_fail)

    # Console output
    print(f"--- Slow Adaptation (ε = {epsilon_success}) ---")
    print(f"Status: {status_s} (Status criteria std_theta < {tol_s})")
    print(f"Mean θ: {mean_theta_s:.4f}, Std θ: {std_theta_s:.4f}")
    print(f"Observed frequency: {mean_freq_s:.4f} Hz")
    print(f"Damped theoretical frequency: {damped_freq_s:.4f} Hz\n")

    print(f"--- Fast Adaptation (ε = {epsilon_fail}) ---")
    print(f"Status: {status_f} (Status criteria std_theta < {tol_f})")
    print(f"Mean θ: {mean_theta_f:.4f}, Std θ: {std_theta_f:.4f}")
    print(f"Observed frequency: {mean_freq_f:.4f} Hz")
    print(f"Damped theoretical frequency: {damped_freq_f:.4f} Hz\n")

    # Prepare data for plotting
    t_window, freq_s, freq_f = prepare_plotting_data(t, x_s, x_f, start_time, end_time, epsilon_success, epsilon_fail)

    # Generate plots
    plot_stacked(t, theta_s, theta_f, x_s, x_f, energy_s, energy_f,
                 epsilon_success, epsilon_fail, status_s, status_f)

    plot_phase_space(x_s, v_s, x_f, v_f, epsilon_success, epsilon_fail,
                     status_s, status_f)

    plot_instantaneous_frequency(t_window, freq_s, freq_f,
                                 epsilon_success, epsilon_fail,
                                 status_s, status_f)

