// Jansen-Rit Neural Mass Model (JS Implementation)
class JansenRitModel {
    constructor() {
        this.dt = 0.001; // 1ms step
        
        // Params (Linked to UI)
        this.A = 3.25;
        this.B = 22.0;
        this.p_mean = 220.0;
        this.p_std = 22.0;
        
        // Constants
        this.a = 100.0;
        this.b = 50.0;
        this.C = 135.0;
        this.C1 = this.C;
        this.C2 = 0.8 * this.C;
        this.C3 = 0.25 * this.C;
        this.C4 = 0.25 * this.C;
        
        this.v0 = 6.0;
        this.e0 = 2.5;
        this.r = 0.56;
        
        this.reset();
    }

    reset() {
        this.state = [0, 0, 0, 0, 0, 0]; // y0 to y5
    }

    sigmoid(v) {
        return (2.0 * this.e0) / (1.0 + Math.exp(this.r * (this.v0 - v)));
    }

    // Gaussian random approximation
    randomNormal() {
        let u = 0, v = 0;
        while(u === 0) u = Math.random();
        while(v === 0) v = Math.random();
        let num = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
        return num;
    }

    step(intervention = 0) {
        let [y0, y1, y2, y3, y4, y5] = this.state;
        
        let p = (this.randomNormal() * this.p_std) + this.p_mean + intervention;
        
        let dy0 = y3;
        let dy3 = this.A * this.a * this.sigmoid(y1 - y2) - 2 * this.a * y3 - (this.a * this.a) * y0;
        
        let dy1 = y4;
        let dy4 = this.A * this.a * (p + this.C2 * this.sigmoid(this.C1 * y0)) - 2 * this.a * y4 - (this.a * this.a) * y1;
        
        let dy2 = y5;
        let dy5 = this.B * this.b * this.C4 * this.sigmoid(this.C3 * y0) - 2 * this.b * y5 - (this.b * this.b) * y2;
        
        this.state[0] += dy0 * this.dt;
        this.state[1] += dy1 * this.dt;
        this.state[2] += dy2 * this.dt;
        this.state[3] += dy3 * this.dt;
        this.state[4] += dy4 * this.dt;
        this.state[5] += dy5 * this.dt;
        
        return this.state[1] - this.state[2]; // EEG Output
    }
}

// UI & Chart Logic
const model = new JansenRitModel();
let tmsPulse = 0;
const MAX_DATA_POINTS = 300;

// Initialize Chart.js
const ctx = document.getElementById('eegChart').getContext('2d');
const eegChart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: Array(MAX_DATA_POINTS).fill(''),
        datasets: [{
            label: 'EEG Output (mV)',
            data: Array(MAX_DATA_POINTS).fill(0),
            borderColor: '#00f0ff',
            borderWidth: 2,
            tension: 0.1,
            pointRadius: 0
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        scales: {
            y: {
                min: -10,
                max: 10,
                grid: { color: 'rgba(255, 255, 255, 0.1)' },
                ticks: { color: '#a0a5b5' }
            },
            x: {
                grid: { display: false },
                ticks: { display: false }
            }
        },
        plugins: {
            legend: { display: false }
        }
    }
});

// UI Elements
const excGainInput = document.getElementById('excGain');
const excVal = document.getElementById('excVal');
const inhGainInput = document.getElementById('inhGain');
const inhVal = document.getElementById('inhVal');
const noiseMeanInput = document.getElementById('noiseMean');
const noiseVal = document.getElementById('noiseVal');
const btnTms = document.getElementById('btn-tms');
const btnReset = document.getElementById('btn-reset');
const statusIndicator = document.getElementById('system-status');
const statusText = statusIndicator.querySelector('.status-text');

// Event Listeners
excGainInput.addEventListener('input', (e) => {
    model.A = parseFloat(e.target.value);
    excVal.textContent = model.A.toFixed(2) + ' mV';
    checkSystemStatus();
});

inhGainInput.addEventListener('input', (e) => {
    model.B = parseFloat(e.target.value);
    inhVal.textContent = model.B.toFixed(1) + ' mV';
    checkSystemStatus();
});

noiseMeanInput.addEventListener('input', (e) => {
    model.p_mean = parseFloat(e.target.value);
    noiseVal.textContent = model.p_mean.toFixed(1);
    checkSystemStatus();
});

btnReset.addEventListener('click', () => {
    model.reset();
    excGainInput.value = 3.25; model.A = 3.25; excVal.textContent = '3.25 mV';
    inhGainInput.value = 22.0; model.B = 22.0; inhVal.textContent = '22.0 mV';
    noiseMeanInput.value = 220.0; model.p_mean = 220.0; noiseVal.textContent = '220.0';
    checkSystemStatus();
    eegChart.data.datasets[0].data.fill(0);
});

btnTms.addEventListener('mousedown', () => tmsPulse = 1000);
btnTms.addEventListener('mouseup', () => tmsPulse = 0);
btnTms.addEventListener('mouseleave', () => tmsPulse = 0);

function checkSystemStatus() {
    // Basic heuristic to warn if we are in FND/Seizure territory
    if (model.A > 4.0 || model.B < 15.0) {
        statusIndicator.classList.add('danger');
        statusText.textContent = "Maladaptive State (FND Risk)";
        eegChart.data.datasets[0].borderColor = '#ff2a5f'; // Red chart
    } else {
        statusIndicator.classList.remove('danger');
        statusText.textContent = "Healthy State";
        eegChart.data.datasets[0].borderColor = '#00f0ff'; // Cyan chart
    }
}

// Main Simulation Loop
function updateSimulation() {
    // Run 15 simulation steps per render frame (~15ms simulated time per frame)
    let eegValue = 0;
    for(let i=0; i<15; i++) {
        eegValue = model.step(tmsPulse);
    }
    
    // Auto-scale chart if spikes are massive
    if (Math.abs(eegValue) > 10) {
        eegChart.options.scales.y.min = -Math.abs(eegValue) - 5;
        eegChart.options.scales.y.max = Math.abs(eegValue) + 5;
    } else {
        eegChart.options.scales.y.min = -10;
        eegChart.options.scales.y.max = 10;
    }

    // Update chart array
    const dataArray = eegChart.data.datasets[0].data;
    dataArray.shift();
    dataArray.push(eegValue);
    eegChart.update();

    requestAnimationFrame(updateSimulation);
}

// Start
checkSystemStatus();
updateSimulation();
