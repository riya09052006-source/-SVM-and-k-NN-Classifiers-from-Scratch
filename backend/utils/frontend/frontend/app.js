const BACKEND_URL = 'http://127.0.0.1:5000';

let currentX = null;
let currentY = null;

// DOM Elements
const datasetTypeSelect = document.getElementById('dataset-type');
const sampleSizeInput = document.getElementById('sample-size');
const sampleSizeVal = document.getElementById('sample-size-val');
const noiseLevelInput = document.getElementById('noise-level');
const noiseLevelVal = document.getElementById('noise-level-val');
const btnGenerate = document.getElementById('btn-generate');

const knnMetricSelect = document.getElementById('knn-metric');
const knnAutoKCheck = document.getElementById('knn-auto-k');
const knnKGroup = document.getElementById('knn-k-group');
const knnKInput = document.getElementById('knn-k');
const knnKVal = document.getElementById('knn-k-val');
const btnTrainKnn = document.getElementById('btn-train-knn');

const svmKernelSelect = document.getElementById('svm-kernel');
const svmCInput = document.getElementById('svm-c');
const svmCVal = document.getElementById('svm-c-val');
const svmDegreeGroup = document.getElementById('svm-degree-group');
const svmDegreeInput = document.getElementById('svm-degree');
const svmDegreeVal = document.getElementById('svm-degree-val');
const svmGammaGroup = document.getElementById('svm-gamma-group');
const svmGammaInput = document.getElementById('svm-gamma');
const svmGammaVal = document.getElementById('svm-gamma-val');
const btnTrainSvm = document.getElementById('btn-train-svm');

const statusDot = document.querySelector('.status-dot');
const statusText = document.getElementById('status-text');

const knnPlotContainer = document.getElementById('knn-plot-container');
const svmPlotContainer = document.getElementById('svm-plot-container');
const elbowPlotContainer = document.getElementById('elbow-plot-container');

const knnBadge = document.getElementById('knn-badge');
const svmBadge = document.getElementById('svm-badge');

const knnAcc = document.getElementById('knn-acc');
const knnPrec = document.getElementById('knn-prec');
const knnRec = document.getElementById('knn-rec');
const knnF1 = document.getElementById('knn-f1');

const svmAcc = document.getElementById('svm-acc');
const svmPrec = document.getElementById('svm-prec');
const svmRec = document.getElementById('svm-rec');
const svmF1 = document.getElementById('svm-f1');

// UI synchronization
sampleSizeInput.addEventListener('input', (e) => { sampleSizeVal.textContent = e.target.value; });
noiseLevelInput.addEventListener('input', (e) => { noiseLevelVal.textContent = parseFloat(e.target.value).toFixed(2); });
knnKInput.addEventListener('input', (e) => { knnKVal.textContent = e.target.value; });
svmCInput.addEventListener('input', (e) => { svmCVal.textContent = parseFloat(e.target.value).toFixed(1); });
svmDegreeInput.addEventListener('input', (e) => { svmDegreeVal.textContent = e.target.value; });
svmGammaInput.addEventListener('input', (e) => { svmGammaVal.textContent = parseFloat(e.target.value).toFixed(1); });

knnAutoKCheck.addEventListener('change', (e) => {
    knnKGroup.style.display = e.target.checked ? 'none' : 'flex';
});

svmKernelSelect.addEventListener('change', (e) => 
    {
    const kernel = e.target.value;
    svmDegreeGroup.style.display = kernel === 'polynomial' ? 'flex' : 'none';
    svmGammaGroup.style.display = kernel === 'rbf' ? 'flex' : 'none';
});

function showLoader(container) 
{
    container.innerHTML = '<div class="spinner"></div>';
}

async function checkConnection() 
{
    try {
        const response = await fetch(`${BACKEND_URL}/api/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ dataset_type: 'moons', samples: 10, noise: 0.1 })
        });
        if (response.ok) {
            statusDot.className = 'status-dot green';
            statusText.textContent = 'Backend Connected';
            return true;
        }
    } catch (err) {
        statusDot.className = 'status-dot red';
        statusText.textContent = 'Backend Offline';
    }
    return false;
}

async function generateDataset() 
{
    showLoader(knnPlotContainer);
    showLoader(svmPlotContainer);
    elbowPlotContainer.innerHTML = '<div class="placeholder-text">Waiting for model execution...</div>';
    
    const requestData = {
        dataset_type: datasetTypeSelect.value,
        samples: parseInt(sampleSizeInput.value),
        noise: parseFloat(noiseLevelInput.value)
    };
    
    try {
        const response = await fetch(`${BACKEND_URL}/api/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestData)
        });
        const data = await response.json();
        currentX = data.X;
        currentY = data.y;
        
        knnPlotContainer.innerHTML = '<div class="placeholder-text">Dataset loaded.<br>Click "Train k-NN" to visualize.</div>';
        svmPlotContainer.innerHTML = '<div class="placeholder-text">Dataset loaded.<br>Click "Train SVM" to visualize.</div>';
        checkConnection();
    } catch (err) {
        knnPlotContainer.innerHTML = `<div class="placeholder-text" style="color:var(--accent-magenta);">Error connecting to Flask server.</div>`;
        svmPlotContainer.innerHTML = `<div class="placeholder-text">Error connecting to Flask server.</div>`;
    }
}

async function trainKNN() 
{
    if (!currentX || !currentY) return alert('Please generate a dataset first.');
    showLoader(knnPlotContainer);
    if (knnAutoKCheck.checked) showLoader(elbowPlotContainer);
    
    const requestData = {
        X: currentX, y: currentY,
        k: parseInt(knnKInput.value),
        metric: knnMetricSelect.value,
        auto_k: knnAutoKCheck.checked
    };
    
    try {
        const response = await fetch(`${BACKEND_URL}/api/train_knn`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestData)
        });
        const data = await response.json();
        
        knnPlotContainer.innerHTML = `<img src="data:image/png;base64,${data.decision_boundary_img}" alt="k-NN Boundary">`;
        knnBadge.textContent = `k=${data.k}`;
        
        if (data.elbow_curve_img) {
            elbowPlotContainer.innerHTML = `<img src="data:image/png;base64,${data.elbow_curve_img}" alt="Elbow Curve">`;
        } else {
            elbowPlotContainer.innerHTML = '<div class="placeholder-text">Elbow Curve disabled.</div>';
        }
        
        knnAcc.textContent = (data.metrics.accuracy * 100).toFixed(1) + '%';
        knnPrec.textContent = data.metrics.precision.toFixed(3);
        knnRec.textContent = data.metrics.recall.toFixed(3);
        knnF1.textContent = data.metrics.f1.toFixed(3);
        checkConnection();
    } catch (err) {
        knnPlotContainer.innerHTML = '<div class="placeholder-text" style="color:var(--accent-magenta);">Error training k-NN.</div>';
    }
}

async function trainSVM() 
{
    if (!currentX || !currentY) return alert('Please generate a dataset first.');
    showLoader(svmPlotContainer);
    
    const requestData = {
        X: currentX, y: currentY,
        C: parseFloat(svmCInput.value),
        kernel: svmKernelSelect.value,
        degree: parseInt(svmDegreeInput.value),
        gamma: svmKernelSelect.value === 'rbf' ? parseFloat(svmGammaInput.value) : null
    };
    
    try {
        const response = await fetch(`${BACKEND_URL}/api/train_svm`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestData)
        });
        const data = await response.json();
        
        if (data.success) {
            svmPlotContainer.innerHTML = `<img src="data:image/png;base64,${data.decision_boundary_img}" alt="SVM Boundary">`;
            svmBadge.textContent = svmKernelSelect.value.toUpperCase();
            
            svmAcc.textContent = (data.metrics.accuracy * 100).toFixed(1) + '%';
            svmPrec.textContent = data.metrics.precision.toFixed(3);
            svmRec.textContent = data.metrics.recall.toFixed(3);
            svmF1.textContent = data.metrics.f1.toFixed(3);
        }
        checkConnection();
    } catch (err) {
        svmPlotContainer.innerHTML = `<div class="placeholder-text" style="color:var(--accent-magenta);">Error training SVM.</div>`;
    }
}

btnGenerate.addEventListener('click', generateDataset);
btnTrainKnn.addEventListener('click', trainKNN);
btnTrainSvm.addEventListener('click', trainSVM);

window.addEventListener('load', async () => {
    const isConnected = await checkConnection();
    if (isConnected) generateDataset();
});