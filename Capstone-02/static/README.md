# Heart Disease Prediction - Web Frontend

## 🎨 Overview

This is a modern, responsive web interface for the Heart Disease Prediction API. The frontend provides an intuitive way for users to input patient data and receive instant predictions with detailed risk assessments.

## ✨ Features

- **Modern UI/UX Design** - Clean, professional interface with gradient backgrounds
- **Responsive Layout** - Works seamlessly on desktop, tablet, and mobile devices
- **Real-time Validation** - Instant feedback on form inputs
- **Interactive Results** - Visual risk indicators and probability bars
- **Sample Data Loading** - Quick testing with pre-filled sample data
- **Animated Elements** - Smooth transitions and heartbeat animations
- **Keyboard Shortcuts** - Fast navigation and actions
- **Error Handling** - Clear error messages and API status checks

## 🚀 Quick Start

### Prerequisites
- FastAPI server running on port 8000
- Modern web browser (Chrome, Firefox, Safari, Edge)

### Running the Frontend

#### Option 1: Integrated with FastAPI (Recommended)
```powershell
# Start the API server (from project root)
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Access the frontend
# Open browser to: http://localhost:8000
```

#### Option 2: Standalone HTML
```powershell
# Open the HTML file directly
start static/index.html

# Note: You'll need to update API_BASE_URL in script.js if API is on different port
```

## 📁 File Structure

```
static/
├── index.html          # Main HTML structure
├── style.css           # Styles and animations
└── script.js           # JavaScript logic and API calls
```

## 🎯 Usage Guide

### 1. Open the Interface
Navigate to `http://localhost:8000` in your web browser.

### 2. Fill Patient Information
Enter all required patient data:
- **Demographics:** Age, Sex
- **Vital Signs:** Blood Pressure, Cholesterol, Blood Sugar
- **Cardiac Tests:** ECG Results, Heart Rate, Chest Pain Type
- **Advanced Tests:** ST Depression, Vessel Count, Thalassemia

### 3. Submit for Prediction
Click the **"Predict Risk"** button to analyze the data.

### 4. View Results
The system will display:
- **Prediction:** Heart Disease Detected or No Heart Disease
- **Risk Level:** Low, Moderate, High, or Very High Risk
- **Probability:** Confidence percentage with visual bar
- **Recommendations:** Actionable next steps based on results

## ⌨️ Keyboard Shortcuts

- `Ctrl/Cmd + Enter` - Submit form for prediction
- `Ctrl/Cmd + R` - Reset form (clears all fields)
- `Ctrl/Cmd + L` - Load sample data for testing

## 🎨 UI Components

### Header
- Animated heartbeat icon
- Project title and subtitle
- Gradient background

### Info Card
- Brief description of the system
- User guidance

### Form Section
- 13 input fields with icons
- Validation indicators
- Help text for each field
- Three action buttons

### Results Section
- Prediction outcome with icon
- Risk level badge
- Probability visualization
- Detailed metrics
- Clinical recommendations
- Suggested actions

### Loading State
- Animated spinner
- Status message

## 🔧 Customization

### Changing Colors
Edit `style.css` and modify the CSS variables:
```css
:root {
    --primary-color: #2563eb;
    --secondary-color: #7c3aed;
    --success-color: #10b981;
    --danger-color: #ef4444;
    /* ... more colors */
}
```

### Updating API URL
Edit `script.js`:
```javascript
const API_BASE_URL = 'http://localhost:8000';
```

### Modifying Sample Data
Edit `script.js`:
```javascript
const sampleData = {
    age: 58,
    sex: 1,
    // ... modify values
};
```

## 📊 Form Fields Reference

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| Age | Number | 0-120 | Patient age in years |
| Sex | Select | 0-1 | 0=Female, 1=Male |
| Chest Pain Type | Select | 0-3 | Type of chest pain |
| Resting BP | Number | 80-200 | Blood pressure in mm Hg |
| Cholesterol | Number | 100-600 | Serum cholesterol in mg/dl |
| Fasting Blood Sugar | Select | 0-1 | >120 mg/dl (0=No, 1=Yes) |
| Resting ECG | Select | 0-2 | ECG results |
| Max Heart Rate | Number | 60-220 | Maximum heart rate |
| Exercise Angina | Select | 0-1 | Exercise induced angina |
| ST Depression | Number | 0-10 | ST depression value |
| ST Slope | Select | 0-2 | ST segment slope |
| Major Vessels | Select | 0-3 | Number of vessels |
| Thalassemia | Select | 0-3 | Blood disorder test |

## 🎯 Risk Level Interpretation

- **Low Risk (< 30%)** - Continue regular monitoring
- **Moderate Risk (30-60%)** - Consider preventive measures
- **High Risk (60-80%)** - Seek medical consultation
- **Very High Risk (≥ 80%)** - Immediate medical attention recommended

## 🐛 Troubleshooting

### Issue: "API is not accessible"
**Solution:** Ensure the API server is running:
```powershell
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### Issue: "CORS Error"
**Solution:** The API includes CORS middleware. If issues persist, check browser console.

### Issue: "Form validation errors"
**Solution:** Ensure all fields are filled with valid values within specified ranges.

### Issue: "Results not displaying"
**Solution:** 
1. Check browser console for errors
2. Verify API response in Network tab
3. Ensure model files are in `models/` directory

## 🌐 Browser Compatibility

- ✅ Chrome 90+
- ✅ Firefox 88+
- ✅ Safari 14+
- ✅ Edge 90+

## 📱 Responsive Breakpoints

- **Desktop:** > 768px (full grid layout)
- **Tablet:** 768px (responsive grid)
- **Mobile:** < 768px (single column)

## 🎨 Design Credits

- **Font:** Poppins (Google Fonts)
- **Icons:** Font Awesome 6.0
- **Color Scheme:** Modern gradient with purple/blue tones
- **UI Pattern:** Material Design inspired

## 📝 API Integration

The frontend communicates with these API endpoints:

### GET `/health`
Check API server status

### POST `/predict`
Submit patient data for prediction
```json
{
  "age": 58,
  "sex": 1,
  "chest_pain_type": 1,
  // ... other fields
}
```

### GET `/model-info`
Get model metadata and parameters

## ⚠️ Important Notes

1. **Medical Disclaimer:** This tool is for educational purposes only
2. **Data Privacy:** No data is stored; all processing is session-based
3. **Accuracy:** Results depend on model training and input data quality
4. **Professional Advice:** Always consult healthcare professionals

## 🚀 Future Enhancements

- [ ] Export results as PDF
- [ ] Multiple patient batch processing
- [ ] Result history tracking
- [ ] Data visualization charts
- [ ] Mobile app version
- [ ] Multilingual support
- [ ] Dark mode toggle
- [ ] Voice input capability

## 📞 Support

For issues or questions:
1. Check the main README.md
2. Review PROJECT_SUMMARY.md
3. Check API documentation at `/docs`
4. Review browser console for errors

## 📄 License

Part of the Heart Disease Detection Capstone Project - Educational Use

---

**Built with ❤️ using HTML, CSS, JavaScript, and FastAPI**
