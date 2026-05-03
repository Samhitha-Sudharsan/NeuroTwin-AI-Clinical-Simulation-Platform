# NeuroTwin AI 🧠💻

NeuroTwin AI is an experimental framework for modeling **Clinical Digital Twins** in neurological rehabilitation. It bridges the gap between computational neuroscience and applied behavioral analysis.

By using Neural Mass Models (NMM) and Reinforcement Learning (RL), this framework simulates how behavioral interventions or neuromodulation (like TMS) alter the physical brain state, aiming to optimize therapies for conditions like Functional Neurological Disorder (FND), Stroke recovery, and Chemobrain.

## 📁 Repository Structure

### 1. Documentation
*   `neuro_ai_startup_thesis.md`: Original academic thesis merging behavior, neuroscience, and AI.
*   `neurotwin_pitch_deck.md`: Business and market plan for hospitals/clinics.
*   `neurotwin_theoretical_architecture.md`: Deep-dive into the math (Jansen-Rit models, Dynamic Causal Modeling, RL).

### 2. Python Backend Engine
*   `neural_mass_model.py`: The core physics engine simulating a cortical column using Jansen-Rit ODEs.
*   `rl_environment.py`: A custom `Gymnasium` environment wrapping the brain simulation, ready for Deep Reinforcement Learning agents to test interventions.
*   `requirements.txt`: Python dependencies (`numpy`, `scipy`, `gymnasium`, `torch`).

### 3. Interactive Web UI
*   `frontend/`: Contains a fast, client-side Javascript port of the Digital Twin.
*   Open `frontend/index.html` in your browser to interact with a live, 60fps simulation of the brain. You can adjust the excitatory/inhibitory gains to induce maladaptive states (seizures/FND) and apply simulated TMS pulses to observe plasticity.

## 🚀 Getting Started

### Running the Python Environment
1. Create a virtual environment: `python -m venv venv`
2. Activate it: `.\venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Mac/Linux)
3. Install dependencies: `pip install -r requirements.txt`
4. Run the model: `python neural_mass_model.py`
5. Test the RL environment: `python rl_environment.py`

### Viewing the Interactive Dashboard
Simply open `frontend/index.html` in any modern web browser. No server required!
