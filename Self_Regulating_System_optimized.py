import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import hilbert

# Configuration dictionary for system and analysis parameters
CONFIG = {
    'system': {
        'delta': 0.05,
        'a': 0.5,
        'gamma': 0.05,
        'theta0': 2.0,
        'epsilon_success': 0.01,
        'epsilon_fail': 0.5,
        't_span': (0, 100),
        'num_points': 500,
        'x0': [1.0, 0.0, 0.5]
    },
    'analysis': {
        'start_time': 60.0,
        'end_time': 80.0,
        'theta_tol': 0.01
    }
}


class AdaptiveOscillator:
    """
    Class for simulating and analyzing an adaptive oscillator system.
    """

    def __init__(self, config=CONFIG):
        """
        Initialize the AdaptiveOscillator with configuration parameters.

        Parameters:
        -----------
        config : dict, optional
            Configuration dictionary containing 'system' and 'analysis' parameters.
            Defaults to the global CONFIG.
        """
        self.system = config['system']
        self.analysis = config['analysis']
        self.delta = self.system['delta']
        self.a = self.system['a']
        self.gamma = self.system['gamma']
        self.theta0 = self.system['theta0']
        self.t_span = self.system['t_span']
        self.num_points = self.system['num_points']
        self.t = np.linspace(self.t_span[0], self.t_span[1], self.num_points)
        self.x0 = self.system['x0']
        self._window_cache = {}
        self.validate_params()

    def validate_params(self):
        """
        Validate the system and analysis parameters to ensure they are valid.
        """
        if self.delta < 0:
            raise ValueError("Damping coefficient (delta) must be non-negative")
        if self.a <= 0:
            raise ValueError("Target amplitude squared (a) must be positive")
        if self.gamma < 0:
            raise ValueError("Relaxation rate (gamma) must be non-negative")
        if self.theta0 <= 0:
            raise ValueError("Baseline natural frequency squared (theta0) must be positive")
        if self.t_span[1] <= self.t_span[0]:
            raise ValueError("End time must be greater than start time")
        if self.num_points < 2:
            raise ValueError("Number of time points must be at least 2")
        if self.analysis['end_time'] <= self.analysis['start_time']:
            raise ValueError("Analysis end time must be greater than start time")
        if self.analysis['theta_tol'] <= 0:
            raise ValueError("Theta tolerance (theta_tol) must be positive")

    def adaptive_oscillator(self, t, y, epsilon):
        """
        Define the ODE system for the adaptive oscillator.

        Parameters:
        -----------
        t : float
            Time point (required by solve_ivp but not used explicitly).
        y : array-like
            State variables [x, v, theta].
        epsilon : float
            Adaptation rate.

        Returns:
        --------
        list
            Derivatives [dxdt, dvdt, dthetadt].
        """
        x, v, theta = y
        dxdt = v
        dvdt = -theta * x - self.delta * v
        dthetadt = epsilon * (x**2 - self.a) - self.gamma * (theta - self.theta0)
        return [dxdt, dvdt, dthetadt]

    def get_convergence_window_indices(self, start_time, end_time):
        """
        Return the start and end indices for the convergence window based on time.
        Uses caching to avoid redundant computations.

        Parameters:
        -----------
        start_time : float
            Start time of the window.
        end_time : float
            End time of the window.

        Returns:
        --------
        tuple
            (start_idx, end_idx)
        """
        key = (start_time, end_time)
        if key not in self._window_cache:
            start_idx = np.searchsorted(self.t, start_time, side='left')
            end_idx = np.searchsorted(self.t, end_time, side='right')
            self._window_cache[key] = (start_idx, end_idx)
        return self._window_cache[key]

    def instantaneous_frequency(self, x, t, epsilon):
        """
        Compute the instantaneous frequency using Hilbert transform.

        Parameters:
        -----------
        x : array-like
            Position signal.
        t : array-like
            Time array.
        epsilon : float
            Adaptation rate (used for warning only).

        Returns:
        --------
        array-like
            Instantaneous frequency.
        """
        if len(x) < 2:
            raise ValueError("Input signal too short for frequency calculation")
        if epsilon > 0.1:
            print(f"Warning: Large epsilon ({epsilon}) may violate Hilbert transform narrowband assumption.")
        analytic_signal = hilbert(x)
        phase = np.unwrap(np.angle(analytic_signal))
        freq = np.gradient(phase, t) / (2 * np.pi)
        if np.any(np.isnan(freq)) or np.any(np.isinf(freq)):
            print(f"Warning: Invalid frequency values detected for epsilon={epsilon}")
        return freq

    def simulate_case(self, epsilon):
        """
        Simulate the adaptive oscillator for a given adaptation rate.

        Parameters:
        -----------
        epsilon : float
            Adaptation rate for theta dynamics.

        Returns:
        --------
        dict
            Simulation results including 'x', 'v', 'theta', 'energy', 'status',
            'mean_theta', 'std_theta', 'mean_freq', 'damped_freq', 'theta_tol'.
        """
        if epsilon <= 0:
            raise ValueError("Adaptation rate (epsilon) must be positive")

        sol = solve_ivp(
            self.adaptive_oscillator,
            t_span=(self.t[0], self.t[-1]),
            y0=self.x0,
            t_eval=self.t,
            method='BDF',
            rtol=1e-6,
            atol=1e-8,
            args=(epsilon,)
        )
        if not sol.success:
            raise RuntimeError(f"Solver failed for epsilon={epsilon}: {sol.message}")
        x, v, theta = sol.y

        energy = 0.5 * v**2 + 0.5 * theta * x**2

        start_idx, end_idx = self.get_convergence_window_indices(
            self.analysis['start_time'], self.analysis['end_time']
        )
        steady_window = slice(start_idx, end_idx)
        mean_theta = np.mean(theta[steady_window])
        std_theta = np.std(theta[steady_window])

        # Calculate damped theoretical frequency
        damped_freq = np.sqrt(max(mean_theta - (self.delta**2 / 4), 0)) / (2 * np.pi)

        status = "Success" if std_theta < self.analysis['theta_tol'] else "Fail"
        freq = self.instantaneous_frequency(x[steady_window], self.t[steady_window], epsilon)
        mean_freq = np.mean(freq)

        return {
            'x': x,
            'v': v,
            'theta': theta,
            'energy': energy,
            'status': status,
            'mean_theta': mean_theta,
            'std_theta': std_theta,
            'mean_freq': mean_freq,
            'damped_freq': damped_freq,
            'theta_tol': self.analysis['theta_tol']
        }

    def prepare_plotting_data(self, x_s, x_f, epsilon_success, epsilon_fail):
        """
        Prepare data for plotting: time window and instantaneous frequencies.

        Parameters:
        -----------
        x_s : array-like
            Position for success case.
        x_f : array-like
            Position for fail case.
        epsilon_success : float
            Epsilon for success case.
        epsilon_fail : float
            Epsilon for fail case.

        Returns:
        --------
        tuple
            (t_window, freq_s, freq_f)
        """
        start_idx, end_idx = self.get_convergence_window_indices(
            self.analysis['start_time'], self.analysis['end_time']
        )
        t_window = self.t[start_idx:end_idx]
        freq_s = self.instantaneous_frequency(x_s[start_idx:end_idx], t_window, epsilon_success)
        freq_f = self.instantaneous_frequency(x_f[start_idx:end_idx], t_window, epsilon_fail)
        return t_window, freq_s, freq_f


# Plotting functions (extracted for modularity)

def plot_stacked(t, theta_s, theta_f, x_s, x_f, energy_s, energy_f,
                 epsilon_success, epsilon_fail, status_s, status_f,
                 start_time, end_time):
    """
    Plot stacked graphs for theta, position, and energy.

    Parameters:
    -----------
    t : array-like
        Time array.
    theta_s, theta_f : array-like
        Theta for success and fail cases.
    x_s, x_f : array-like
        Position for success and fail cases.
    energy_s, energy_f : array-like
        Energy for success and fail cases.
    epsilon_success, epsilon_fail : float
        Adaptation rates.
    status_s, status_f : str
        Status ("Success" or "Fail").
    start_time, end_time : float
        Convergence window times.
    """
    convergence_window = [start_time, end_time]
    color_s = 'green' if status_s == "Success" else 'red'
    color_f = 'blue' if status_f == "Success" else 'orange'

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    plots = [
        ('Theta', theta_s, theta_f, 'Theta Evolution'),
        ('Position', x_s, x_f, 'Position'),
        ('Energy', energy_s, energy_f, 'Energy')
    ]
    for ax, (label, y_s, y_f, title) in zip(axes, plots):
        ax.axvspan(*convergence_window, color='gray', alpha=0.2)
        ax.plot(t, y_s, color=color_s, label=f'{label} (ε={epsilon_success:.4f}, {status_s})')
        ax.plot(t, y_f, color=color_f, linestyle='--', label=f'{label} (ε={epsilon_fail:.4f}, {status_f})')
        ax.set_ylabel(label)
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.show()


def plot_phase_space(x_s, v_s, x_f, v_f, epsilon_success, epsilon_fail, status_s, status_f):
    """
    Plot phase space for success and fail cases.

    Parameters:
    -----------
    x_s, v_s : array-like
        Position and velocity for success case.
    x_f, v_f : array-like
        Position and velocity for fail case.
    epsilon_success, epsilon_fail : float
        Adaptation rates.
    status_s, status_f : str
        Status ("Success" or "Fail").
    """
    color_s = 'green' if status_s == "Success" else 'red'
    color_f = 'blue' if status_f == "Success" else 'orange'

    plt.figure(figsize=(6, 6))
    plt.plot(x_s, v_s, color=color_s, label=f'Phase (ε={epsilon_success:.4f}, {status_s})')
    plt.plot(x_f, v_f, color=color_f, linestyle='--', label=f'Phase (ε={epsilon_fail:.4f}, {status_f})')
    plt.xlabel('x')
    plt.ylabel('v')
    plt.title('Phase Space')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()


def plot_instantaneous_frequency(t_window, freq_s, freq_f,
                                 epsilon_success, epsilon_fail,
                                 status_s, status_f):
    """
    Plot instantaneous frequency for the convergence window.

    Parameters:
    -----------
    t_window : array-like
        Time window.
    freq_s, freq_f : array-like
        Frequencies for success and fail cases.
    epsilon_success, epsilon_fail : float
        Adaptation rates.
    status_s, status_f : str
        Status ("Success" or "Fail").
    """
    color_s = 'green' if status_s == "Success" else 'red'
    color_f = 'blue' if status_f == "Success" else 'orange'

    plt.figure(figsize=(10, 6))
    plt.plot(t_window, freq_s, color=color_s, label=f'IF (ε={epsilon_success:.4f}, {status_s})')
    plt.plot(t_window, freq_f, color=color_f, linestyle='--', label=f'IF (ε={epsilon_fail:.4f}, {status_f})')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Instantaneous Frequency')
    plt.legend()
    plt.grid(True)
    plt.show()


# Main Execution
if __name__ == "__main__":
    oscillator = AdaptiveOscillator()

    epsilon_success = oscillator.system['epsilon_success']
    epsilon_fail = oscillator.system['epsilon_fail']

    # Run simulations
    results_s = oscillator.simulate_case(epsilon_success)
    results_f = oscillator.simulate_case(epsilon_fail)

    # Console output
    print(f"--- Slow Adaptation (ε = {epsilon_success}) ---")
    print(f"Status: {results_s['status']} (Status criteria std_theta < {results_s['theta_tol']})")
    print(f"Mean θ: {results_s['mean_theta']:.4f}, Std θ: {results_s['std_theta']:.4f}")
    print(f"Observed frequency: {results_s['mean_freq']:.4f} Hz")
    print(f"Damped theoretical frequency: {results_s['damped_freq']:.4f} Hz\n")

    print(f"--- Fast Adaptation (ε = {epsilon_fail}) ---")
    print(f"Status: {results_f['status']} (Status criteria std_theta < {results_f['theta_tol']})")
    print(f"Mean θ: {results_f['mean_theta']:.4f}, Std θ: {results_f['std_theta']:.4f}")
    print(f"Observed frequency: {results_f['mean_freq']:.4f} Hz")
    print(f"Damped theoretical frequency: {results_f['damped_freq']:.4f} Hz\n")

    # Prepare data for plotting
    t_window, freq_s, freq_f = oscillator.prepare_plotting_data(
        results_s['x'], results_f['x'], epsilon_success, epsilon_fail
    )

    # Generate plots
    plot_stacked(
        oscillator.t,
        results_s['theta'], results_f['theta'],
        results_s['x'], results_f['x'],
        results_s['energy'], results_f['energy'],
        epsilon_success, epsilon_fail,
        results_s['status'], results_f['status'],
        oscillator.analysis['start_time'], oscillator.analysis['end_time']
    )

    plot_phase_space(
        results_s['x'], results_s['v'],
        results_f['x'], results_f['v'],
        epsilon_success, epsilon_fail,
        results_s['status'], results_f['status']
    )

    plot_instantaneous_frequency(
        t_window, freq_s, freq_f,
        epsilon_success, epsilon_fail,
        results_s['status'], results_f['status']
    )
