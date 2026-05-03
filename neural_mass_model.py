import numpy as np
import matplotlib.pyplot as plt

class JansenRitModel:
    """
    A single-node Neural Mass Model based on the Jansen-Rit equations (1995).
    Simulates the electrical activity of a cortical column.
    """
    def __init__(self, dt=0.001, A=3.25, B=22.0, p_mean=220.0, p_std=22.0, seed=None):
        # Time step for Euler integration
        self.dt = dt
        self.rng = np.random.default_rng(seed)
        
        # Standard parameters
        self.A = A        # Excitatory synaptic gain (mV)
        self.B = B        # Inhibitory synaptic gain (mV) - increase this for FND/Maladaptive state
        self.a = 100.0    # Excitatory time constant (1/s)
        self.b = 50.0     # Inhibitory time constant (1/s)
        
        self.C = 135.0
        self.C1 = self.C
        self.C2 = 0.8 * self.C
        self.C3 = 0.25 * self.C
        self.C4 = 0.25 * self.C
        
        self.v0 = 6.0     # Firing threshold (mV)
        self.e0 = 2.5     # Maximum firing rate / 2 (1/s)
        self.r = 0.56     # Sigmoid steepness (1/mV)
        
        # Input noise (p)
        self.p_mean = p_mean
        self.p_std = p_std
        
        # State variables: [y0, y1, y2, y3, y4, y5]
        self.state = np.zeros(6)

    def sigmoid(self, v):
        """Non-linear function converting average membrane potential to firing rate."""
        return 2.0 * self.e0 / (1.0 + np.exp(self.r * (self.v0 - v)))

    def step(self, intervention=0.0):
        """
        Advances the simulation by one time step using the Euler method.
        'intervention' is the external signal applied by the RL agent (e.g., TMS).
        """
        y0, y1, y2, y3, y4, y5 = self.state
        
        # Input noise + any external therapeutic intervention
        p = self.rng.normal(self.p_mean, self.p_std) + intervention
        
        # Jansen-Rit ODEs
        dy0 = y3
        dy3 = self.A * self.a * self.sigmoid(y1 - y2) - 2 * self.a * y3 - (self.a ** 2) * y0
        
        dy1 = y4
        dy4 = self.A * self.a * (p + self.C2 * self.sigmoid(self.C1 * y0)) - 2 * self.a * y4 - (self.a ** 2) * y1
        
        dy2 = y5
        dy5 = self.B * self.b * self.C4 * self.sigmoid(self.C3 * y0) - 2 * self.b * y5 - (self.b ** 2) * y2
        
        # Update state
        self.state += self.dt * np.array([dy0, dy1, dy2, dy3, dy4, dy5])
        
        # The output of the model (simulated EEG) is the potential difference on pyramidal cells
        return y1 - y2

    def simulate(self, seconds=1.0):
        """Runs the simulation for a set number of seconds and returns the EEG signal."""
        steps = int(seconds / self.dt)
        signal = np.zeros(steps)
        for i in range(steps):
            signal[i] = self.step()
        return signal

if __name__ == "__main__":
    # Test the model
    print("Running healthy NMM simulation...")
    healthy_model = JansenRitModel(A=3.25, B=22.0)
    healthy_signal = healthy_model.simulate(seconds=2.0)
    
    print("Running maladaptive (FND) NMM simulation...")
    # Simulating a maladaptive state by increasing the excitatory gain and decreasing inhibition
    maladaptive_model = JansenRitModel(A=4.5, B=10.0) 
    maladaptive_signal = maladaptive_model.simulate(seconds=2.0)
    
    # Plot the results
    t = np.arange(0, 2.0, 0.001)
    
    plt.figure(figsize=(12, 6))
    plt.plot(t, healthy_signal, label="Healthy State (Alpha/Beta rhythms)", color="green")
    plt.plot(t, maladaptive_signal, label="Maladaptive State (Spiking/Seizure)", color="red", alpha=0.7)
    plt.title("Neural Mass Model: Simulated EEG Output")
    plt.xlabel("Time (s)")
    plt.ylabel("Potential difference (mV)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("nmm_output.png")
    print("Saved plot to nmm_output.png")
