// API Configuration
const API_BASE_URL = 'http://localhost:8000';

// Form Elements
const form = document.getElementById('predictionForm');
const resultsSection = document.getElementById('resultsSection');
const resultsContent = document.getElementById('resultsContent');
const loadingSpinner = document.getElementById('loadingSpinner');

// Sample Data for Testing
const sampleData = {
    age: 58,
    sex: 1,
    chest_pain_type: 1,
    resting_blood_pressure: 134,
    cholesterol: 246,
    fasting_blood_sugar: 0,
    resting_ecg: 0,
    max_heart_rate: 155,
    exercise_induced_angina: 0,
    st_depression: 0.4,
    st_slope: 1,
    num_major_vessels: 1,
    thalassemia: 2
};

// Event Listeners
form.addEventListener('submit', handleFormSubmit);

// Form Submit Handler
async function handleFormSubmit(e) {
    e.preventDefault();
    
    // Get form data
    const formData = new FormData(form);
    const patientData = {};
    
    for (let [key, value] of formData.entries()) {
        patientData[key] = key === 'st_depression' ? parseFloat(value) : parseInt(value);
    }
    
    // Show loading spinner
    showLoading();
    hideResults();
    
    try {
        // Make API call
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(patientData)
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        
        // Display results
        displayResults(result);
        
    } catch (error) {
        console.error('Error:', error);
        displayError(error.message);
    } finally {
        hideLoading();
    }
}

// Display Results
function displayResults(result) {
    const isPositive = result.prediction === 1;
    const probability = (result.probability * 100).toFixed(1);
    
    const riskClass = result.risk_level.toLowerCase().replace(' ', '-');
    
    const resultsHTML = `
        <div class="result-card">
            <div class="result-header">
                <div class="result-title">
                    <div class="result-icon ${isPositive ? 'positive' : 'negative'}">
                        <i class="fas ${isPositive ? 'fa-exclamation-triangle' : 'fa-check-circle'}"></i>
                    </div>
                    <div class="result-text">
                        <h3>${result.prediction_label}</h3>
                        <p>${isPositive ? 'Elevated risk detected' : 'Low risk detected'}</p>
                    </div>
                </div>
                <div class="risk-badge ${riskClass}">
                    ${result.risk_level}
                </div>
            </div>
            
            <div class="result-body">
                <div class="metric-row">
                    <span class="metric-label">Prediction Confidence</span>
                    <span class="metric-value">${probability}%</span>
                </div>
                
                <div class="probability-container">
                    <div class="metric-label">Risk Probability</div>
                    <div class="probability-bar">
                        <div class="probability-fill" style="width: ${probability}%">
                            ${probability}%
                        </div>
                    </div>
                </div>
                
                <div class="metric-row">
                    <span class="metric-label">Prediction</span>
                    <span class="metric-value">${isPositive ? 'Positive' : 'Negative'}</span>
                </div>
                
                <div class="result-message">
                    <p><strong><i class="fas fa-info-circle"></i> Clinical Assessment:</strong></p>
                    <p>${result.message}</p>
                </div>
                
                ${isPositive ? `
                    <div class="result-message" style="border-left-color: var(--danger-color);">
                        <p><strong><i class="fas fa-heartbeat"></i> Recommended Actions:</strong></p>
                        <ul style="margin-left: 20px; margin-top: 10px; line-height: 1.8;">
                            <li>Consult with a cardiologist immediately</li>
                            <li>Consider additional diagnostic tests (ECG, echocardiogram)</li>
                            <li>Discuss lifestyle modifications and treatment options</li>
                            <li>Monitor blood pressure and cholesterol levels regularly</li>
                        </ul>
                    </div>
                ` : `
                    <div class="result-message" style="border-left-color: var(--success-color);">
                        <p><strong><i class="fas fa-heart"></i> Health Maintenance:</strong></p>
                        <ul style="margin-left: 20px; margin-top: 10px; line-height: 1.8;">
                            <li>Continue regular health check-ups</li>
                            <li>Maintain a healthy diet and exercise routine</li>
                            <li>Monitor cardiovascular health indicators</li>
                            <li>Stay informed about heart disease risk factors</li>
                        </ul>
                    </div>
                `}
            </div>
        </div>
    `;
    
    resultsContent.innerHTML = resultsHTML;
    showResults();
    
    // Smooth scroll to results
    setTimeout(() => {
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }, 100);
}

// Display Error
function displayError(errorMessage) {
    const errorHTML = `
        <div class="result-card" style="border-left: 4px solid var(--danger-color);">
            <div class="result-header">
                <div class="result-title">
                    <div class="result-icon" style="background: rgba(239, 68, 68, 0.1); color: var(--danger-color);">
                        <i class="fas fa-times-circle"></i>
                    </div>
                    <div class="result-text">
                        <h3>Prediction Error</h3>
                        <p>Unable to process the request</p>
                    </div>
                </div>
            </div>
            <div class="result-body">
                <div class="result-message" style="border-left-color: var(--danger-color);">
                    <p><strong><i class="fas fa-exclamation-circle"></i> Error Details:</strong></p>
                    <p>${errorMessage}</p>
                    <p style="margin-top: 15px;">Please ensure:</p>
                    <ul style="margin-left: 20px; margin-top: 10px;">
                        <li>The API server is running (uvicorn api.main:app --reload)</li>
                        <li>All form fields are filled correctly</li>
                        <li>The API is accessible at ${API_BASE_URL}</li>
                    </ul>
                </div>
            </div>
        </div>
    `;
    
    resultsContent.innerHTML = errorHTML;
    showResults();
}

// Loading Functions
function showLoading() {
    loadingSpinner.classList.remove('hidden');
}

function hideLoading() {
    loadingSpinner.classList.add('hidden');
}

function showResults() {
    resultsSection.classList.remove('hidden');
}

function hideResults() {
    resultsSection.classList.add('hidden');
}

// Reset Form
function resetForm() {
    form.reset();
    hideResults();
    
    // Smooth scroll to top
    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });
}

// Fill Sample Data
function fillSampleData() {
    Object.keys(sampleData).forEach(key => {
        const input = document.getElementById(key);
        if (input) {
            input.value = sampleData[key];
        }
    });
    
    // Show notification
    showNotification('Sample data loaded successfully!', 'success');
}

// Show Notification
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: ${type === 'success' ? 'var(--success-color)' : 'var(--info-color)'};
        color: white;
        padding: 15px 25px;
        border-radius: 10px;
        box-shadow: var(--shadow-lg);
        z-index: 1000;
        animation: slideIn 0.3s ease;
    `;
    notification.innerHTML = `
        <i class="fas fa-${type === 'success' ? 'check' : 'info'}-circle"></i>
        <span style="margin-left: 10px;">${message}</span>
    `;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

// Add animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from { transform: translateX(400px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    @keyframes slideOut {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(400px); opacity: 0; }
    }
`;
document.head.appendChild(style);

// Check API Health on Page Load
async function checkAPIHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();
        
        if (data.status === 'healthy') {
            console.log('✓ API is healthy and ready');
        } else {
            console.warn('⚠ API health check failed');
        }
    } catch (error) {
        console.error('✗ API is not accessible:', error);
        showNotification('Warning: API server may not be running. Please start it with: uvicorn api.main:app --reload', 'warning');
    }
}

// Warning notification style
const warningStyle = document.createElement('style');
warningStyle.textContent = `
    .notification.warning {
        background: var(--warning-color) !important;
        max-width: 400px;
    }
`;
document.head.appendChild(warningStyle);

// Initialize
window.addEventListener('DOMContentLoaded', () => {
    checkAPIHealth();
    console.log('Heart Disease Prediction System initialized');
});

// Form Validation
form.addEventListener('input', (e) => {
    const input = e.target;
    
    if (input.validity.valid) {
        input.style.borderColor = 'var(--success-color)';
    } else {
        input.style.borderColor = 'var(--danger-color)';
    }
});

// Keyboard Shortcuts
document.addEventListener('keydown', (e) => {
    // Ctrl/Cmd + Enter to submit form
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        form.dispatchEvent(new Event('submit', { cancelable: true }));
    }
    
    // Ctrl/Cmd + R to reset form
    if ((e.ctrlKey || e.metaKey) && e.key === 'r') {
        e.preventDefault();
        resetForm();
    }
    
    // Ctrl/Cmd + L to load sample data
    if ((e.ctrlKey || e.metaKey) && e.key === 'l') {
        e.preventDefault();
        fillSampleData();
    }
});
