import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import hilbert
from dataclasses import dataclass
from typing import Tuple, List, Dict

# ----------------------
# Data Structures
# ----------------------
@dataclass
class SystemParams:
    """System parameters for the adaptive oscillator."""
    delta: float  # Damping coefficient
    a: float      # Target squared amplitude
    gamma: float  # Relaxation rate for theta
    theta0: float # Baseline natural frequency squared

    def validate(self):
        """Validate system parameters."""
        if self.delta < 0:
            raise ValueError("Damping coefficient (delta) must be non-negative")
        if self.a <= 0:
            raise ValueError("Target squared amplitude (a) must be positive")
        if self.gamma < 0:
            raise ValueError("Relaxation rate (gamma) must be non-negative")
        if self.theta0 <= 0:
            raise ValueError("Baseline natural frequency squared (theta0) must be positive")

@dataclass
class AnalysisParams:
    """Analysis parameters for simulation."""
    start_time: float  # Steady-state window start (s)
    end_time: float    # Steady-state window end (s)
    theta_tol: float   # Threshold for success criterion

    def validate(self, t: np.ndarray):
        """Validate analysis parameters against time array."""
        if self.start_time >= self.end_time:
            raise ValueError("start_time must be less than end_time")
        if self.start_time < t[0] or self.end_time > t[-1]:
            raise ValueError("Steady-state window must be within time array bounds")
        if self.theta_tol <= 0:
            raise ValueError("Theta tolerance (theta_tol) must be positive")

@dataclass
class SimulationResult:
    """Results of a single simulation."""
    position: np.ndarray
    velocity: np.ndarray
    theta: np.ndarray
    energy: np.ndarray
    frequency: np.ndarray
    status: str
    mean_theta: float
    std_theta: float
    mean_freq: float
    damped_freq: float
    epsilon: float

# ----------------------
# Utility Functions
# ----------------------
def adaptive_oscillator(t: float, y: np.ndarray, params: SystemParams, epsilon: float) -> np.ndarray:
    """Define the adaptive oscillator ODE system.

    Args:
        t: Time point (float).
        y: State vector [position, velocity, theta].
        params: System parameters.
        epsilon: Adaptation rate.

    Returns:
        Array of derivatives [dx/dt, dv/dt, dtheta/dt].
    """
    x, v, theta = y
    dxdt = v
    dvdt = -theta * x - params.delta * v
    dthetadt = epsilon * (x**2 - params.a) - params.gamma * (theta - params.theta0)
    return np.array([dxdt, dvdt, dthetadt])

def instantaneous_frequency(x: np.ndarray, t: np.ndarray, epsilon: float) -> np.ndarray:
    """Compute instantaneous frequency using Hilbert transform.

    Args:
        x: Position array.
        t: Time array.
        epsilon: Adaptation rate for warning.

    Returns:
        Instantaneous frequency array (Hz).
    """
    if epsilon > 0.1:
        print(f"Warning: Large epsilon ({epsilon}) may violate Hilbert transform narrowband assumption.")
    analytic_signal = hilbert(x)
    phase = np.unwrap(np.angle(analytic_signal))
    return np.gradient(phase, t) / (2 * np.pi)

def get_convergence_window_indices(t: np.ndarray, analysis: AnalysisParams) -> Tuple[int, int]:
    """Return start and end indices for the steady-state window.

    Args:
        t: Time array.
        analysis: Analysis parameters.

    Returns:
        Tuple of (start_idx, end_idx).
    """
    start_idx = np.searchsorted(t, analysis.start_time, side='left')
    end_idx = np.searchsorted(t, analysis.end_time, side='right')
    return start_idx, end_idx

# ----------------------
# Simulation Function
# ----------------------
def simulate_case(t: np.ndarray, x0: List[float], params: SystemParams, analysis: AnalysisParams,
                 epsilon: float) -> SimulationResult:
    """Run simulation for given epsilon and return results.

    Args:
        t: Time array.
        x0: Initial conditions [position, velocity, theta].
        params: System parameters.
        analysis: Analysis parameters.
        epsilon: Adaptation rate.

    Returns:
        SimulationResult object containing simulation outputs and metrics.
    """
    if epsilon <= 0:
        raise ValueError("Adaptation rate (epsilon) must be positive")
    
    sol = solve_ivp(
        fun=lambda t, y: adaptive_oscillator(t, y, params, epsilon),
        t_span=(t[0], t[-1]),
        y0=x0,
        t_eval=t,
        method='BDF',
        rtol=1e-6,
        atol=1e-8
    )
    position, velocity, theta = sol.y
    energy = 0.5 * velocity**2 + 0.5 * theta * position**2
    
    start_idx, end_idx = get_convergence_window_indices(t, analysis)
    steady_window = slice(start_idx, end_idx)
    frequency = instantaneous_frequency(position[steady_window], t[steady_window], epsilon)
    
    mean_theta = np.mean(theta[steady_window])
    std_theta = np.std(theta[steady_window])
    mean_freq = np.mean(frequency)
    damped_freq = np.sqrt(max(mean_theta - (params.delta**2 / 4), 0)) / (2 * np.pi)
    status = "Success" if std_theta < analysis.theta_tol else "Fail"
    
    return SimulationResult(position, velocity, theta, energy, frequency, status,
                           mean_theta, std_theta, mean_freq, damped_freq, epsilon)

# ----------------------
# Plotting Functions
# ----------------------
def plot_helper(ax, t: np.ndarray, data_s: np.ndarray, data_f: np.ndarray, label: str,
                result_s: SimulationResult, result_f: SimulationResult, analysis: AnalysisParams,
                ylabel: str, title: str) -> None:
    """Helper function for plotting time series data.

    Args:
        ax: Matplotlib axis object.
        t: Time array.
        data_s: Data array for success case.
        data_f: Data array for fail case.
        label: Plot label prefix.
        result_s: Simulation result for success case.
        result_f: Simulation result for fail case.
        analysis: Analysis parameters.
        ylabel: Y-axis label.
        title: Plot title.
    """
    color_s = 'green' if result_s.status == "Success" else 'red'
    color_f = 'blue' if result_f.status == "Success" else 'orange'
    start_idx, end_idx = get_convergence_window_indices(t, analysis)
    ax.axvspan(t[start_idx], t[end_idx], color='gray', alpha=0.2)
    ax.plot(t, data_s, color=color_s, label=f'{label} (ε={result_s.epsilon:.4f}, {result_s.status})')
    ax.plot(t, data_f, color=color_f, linestyle='--',
            label=f'{label} (ε={result_f.epsilon:.4f}, {result_f.status})')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

def plot_stacked(t: np.ndarray, result_s: SimulationResult, result_f: SimulationResult,
                 analysis: AnalysisParams) -> None:
    """Plot theta, position, and energy time series.

    Args:
        t: Time array.
        result_s: Simulation result for success case.
        result_f: Simulation result for fail case.
        analysis: Analysis parameters.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))
    plot_helper(ax1, t, result_s.theta, result_f.theta, 'Theta', result_s, result_f, analysis,
                'Theta', 'Theta Evolution')
    plot_helper(ax2, t, result_s.position, result_f.position, 'Position', result_s, result_f, analysis,
                'Position (x)', 'Position Evolution')
    plot_helper(ax3, t, result_s.energy, result_f.energy, 'Energy', result_s, result_f, analysis,
                'Energy', 'Energy Evolution')
    plt.tight_layout()
    plt.show()

def plot_phase_space(result_s: SimulationResult, result_f: SimulationResult) -> None:
    """Plot phase space trajectories.

    Args:
        result_s: Simulation result for success case.
        result_f: Simulation result for fail case.
    """
    plt.figure(figsize=(6, 6))
    color_s = 'green' if result_s.status == "Success" else 'red'
    color_f = 'blue' if result_f.status == "Success" else 'orange'
    plt.plot(result_s.position, result_s.velocity, color=color_s,
             label=f'Phase (ε={result_s.epsilon:.4f}, {result_s.status})')
    plt.plot(result_f.position, result_f.velocity, color=color_f, linestyle='--',
             label=f'Phase (ε={result_f.epsilon:.4f}, {result_f.status})')
    plt.xlabel('Position (x)')
    plt.ylabel('Velocity (v)')
    plt.title('Phase Space')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def plot_instantaneous_frequency(t: np.ndarray, result_s: SimulationResult, result_f: SimulationResult,
                                 analysis: AnalysisParams) -> None:
    """Plot instantaneous frequency in steady-state window.

    Args:
        t: Time array.
        result_s: Simulation result for success case.
        result_f: Simulation result for fail case.
        analysis: Analysis parameters.
    """
    start_idx, end_idx = get_convergence_window_indices(t, analysis)
    t_window = t[start_idx:end_idx]
    plt.figure(figsize=(10, 6))
    color_s = 'green' if result_s.status == "Success" else 'red'
    color_f = 'blue' if result_f.status == "Success" else 'orange'
    plt.plot(t_window, result_s.frequency, color=color_s,
             label=f'Frequency (ε={result_s.epsilon:.4f}, {result_s.status})')
    plt.plot(t_window, result_f.frequency, color=color_f, linestyle='--',
             label=f'Frequency (ε={result_f.epsilon:.4f}, {result_f.status})')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Instantaneous Frequency')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_metrics_bar(result_s: SimulationResult, result_f: SimulationResult) -> None:
    """Plot an enhanced grouped bar chart with annotations and a summary table comparing simulation metrics.

    Args:
        result_s: Simulation result for success case.
        result_f: Simulation result for fail case.
    """
    metrics = ['Mean Theta', 'Std Theta', 'Mean Freq', 'Damped Freq']
    values_s = [result_s.mean_theta, result_s.std_theta, result_s.mean_freq, result_s.damped_freq]
    values_f = [result_f.mean_theta, result_f.std_theta, result_f.mean_freq, result_f.damped_freq]
    
    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot bars
    bars_s = ax.bar(x - width/2, values_s, width, label=f'Success (ε={result_s.epsilon:.4f})',
                    color='forestgreen' if result_s.status == "Success" else 'lightcoral')
    bars_f = ax.bar(x + width/2, values_f, width, label=f'Fail (ε={result_f.epsilon:.4f})',
                    color='royalblue' if result_f.status == "Success" else 'darkorange')
    
    # Add annotations on top of bars
    for bar in bars_s:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height, f'{height:.4f}',
                ha='center', va='bottom', fontsize=9)
    for bar in bars_f:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height, f'{height:.4f}',
                ha='center', va='bottom', fontsize=9)
    
    # Set labels and title
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Values')
    ax.set_title('Comparison of Simulation Metrics (Success vs Fail)')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=0)
    ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Adjust y-axis limit to accommodate annotations
    max_height = max(max(values_s), max(values_f))
    ax.set_ylim(0, max_height * 1.2)
    
    # Add a summary table below the plot
    table_data = [
        ['Metric', 'Success (ε=0.0100)', 'Fail (ε=0.5000)'],
        ['Mean Theta', f'{result_s.mean_theta:.4f}', f'{result_f.mean_theta:.4f}'],
        ['Std Theta', f'{result_s.std_theta:.4f}', f'{result_f.std_theta:.4f}'],
        ['Mean Freq', f'{result_s.mean_freq:.4f}', f'{result_f.mean_freq:.4f}'],
        ['Damped Freq', f'{result_s.damped_freq:.4f}', f'{result_f.damped_freq:.4f}']
    ]
    table = ax.table(cellText=table_data, loc='bottom', cellLoc='center', bbox=[0, -0.9, 1, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    
    # Adjust layout to prevent overlap
    plt.subplots_adjust(bottom=0.4)
    plt.tight_layout()
    plt.show()

# ----------------------
# Main Execution
# ----------------------
if __name__ == "__main__":
    # Parameter Definitions
    # System Parameters
    delta = 0.05        # Damping coefficient
    a = 0.5             # Target squared amplitude
    gamma = 0.05        # Relaxation rate for theta
    theta0 = 1.0        # Baseline natural frequency squared
    
    # Analysis Parameters (defines steady-state window)
    start_time = 60.0   # Steady-state window start (s)
    end_time = 80.0     # Steady-state window end (s)
    theta_tol = 0.01    # Threshold for success criterion
    
    # Simulation Parameters
    t = np.linspace(0, 100, 500)  # Time array
    x0 = [1.0, 0.0, 0.5]          # Initial conditions [position, velocity, theta]
    epsilons = {                  # Adaptation rates
        'success': 0.01,          # Slow adaptation
        'fail': 0.5               # Fast adaptation
    }
    
    # Initialize parameter objects
    params = SystemParams(delta=delta, a=a, gamma=gamma, theta0=theta0)
    params.validate()
    analysis = AnalysisParams(start_time=start_time, end_time=end_time, theta_tol=theta_tol)
    analysis.validate(t)
    
    # Run simulations
    results: Dict[str, SimulationResult] = {}
    for case, epsilon in epsilons.items():
        results[case] = simulate_case(t, x0, params, analysis, epsilon)
    
    # Console output
    for case, result in results.items():
        print(f"--- {case.capitalize()} Adaptation (ε = {result.epsilon}) ---")
        print(f"Status: {result.status} (Status criteria std_theta < {analysis.theta_tol})")
        print(f"Mean θ: {result.mean_theta:.4f}, Std θ: {result.std_theta:.4f}")
        print(f"Observed frequency: {result.mean_freq:.4f} Hz")
        print(f"Damped theoretical frequency: {result.damped_freq:.4f} Hz\n")
    
    # Generate plots
    result_s, result_f = results['success'], results['fail']
    plot_stacked(t, result_s, result_f, analysis)
    plot_phase_space(result_s, result_f)
    plot_instantaneous_frequency(t, result_s, result_f, analysis)
    plot_metrics_bar(result_s, result_f)
