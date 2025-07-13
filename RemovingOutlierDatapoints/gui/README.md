# CSV Time Series Analysis Tool

A modern desktop application for CSV time series analysis with region-based statistical analysis. Features an Electron frontend with drag-and-drop functionality and a Python Flask backend for advanced data processing.

## 🚀 Features

### **Frontend (Electron App)**
- **Beautiful Modern UI**: Gradient backgrounds, smooth animations, and intuitive design
- **Drag & Drop Upload**: Simply drag CSV files directly into the application window
- **Interactive Charts**: Click on charts to select region boundaries
- **Smart Column Detection**: Automatically identifies time columns in your data
- **Real-time Validation**: Visual feedback for all user inputs
- **Cross-platform**: Works on Windows, macOS, and Linux

### **Backend (Python Flask)**
- **Region Analysis**: Divide time series data into custom regions
- **Statistical Analysis**: Calculate mean, median, std dev, quartiles, and more for each region
- **Flexible Region Sizing**: Custom first region with equally divided remaining regions
- **REST API**: Clean JSON API for frontend communication
- **Error Handling**: Comprehensive validation and error reporting

## 📋 Requirements

### **Frontend**
- Node.js 16+ 
- npm or yarn

### **Backend**
- Python 3.8+
- Required packages (see `backend_requirements.txt`)

## 🛠️ Installation

### **1. Install Node.js Dependencies**
```bash
# In the project root directory
npm install
```

### **2. Install Python Dependencies**
```bash
# Install Python backend requirements
pip install -r backend_requirements.txt
```

## 🚀 Usage

### **Step 1: Start the Python Backend**
```bash
# Option 1: Use the startup script (recommended)
python start_backend.py

# Option 2: Start directly
python backend.py
```

The backend will start on `http://127.0.0.1:5000`

### **Step 2: Start the Electron Frontend**
```bash
# In a new terminal window
npm start
```

### **Step 3: Analyze Your Data**

1. **Upload CSV File**: 
   - Drag & drop a CSV file into the application window
   - Or click "Browse Files" to select manually

2. **Generate Plot**: 
   - Select your time column (X-axis)
   - Select your data column (Y-axis)  
   - Click "Generate Plot"

3. **Set Up Regions**:
   - Click on the chart to select the end of the first region
   - Enter the total number of regions you want (2-20)
   - Click "Next: Analyze Regions"

4. **View Results**:
   - Get statistical analysis for each region
   - View mean, median, standard deviation, and more

## 📊 API Endpoints

### **Health Check**
```http
GET /health
```
Returns backend status and timestamp.

### **Analyze CSV Data**
```http
POST /analyze
```
**Body:**
```json
{
  "csv_content": "string",
  "time_column": "string", 
  "response": "string",
  "first_region_end_index": 100,
  "num_regions": 5
}
```

### **Get Regions Summary**
```http
POST /regions-summary
```
**Body:**
```json
{
  "total_data_points": 1000,
  "first_region_end_index": 100,
  "num_regions": 5
}
```

## 📁 Project Structure

```
├── package.json              # Electron app configuration
├── main.js                   # Electron main process
├── index.html                # Frontend HTML
├── styles.css                # Frontend styles
├── renderer.js               # Frontend JavaScript
├── backend.py                # Python Flask server
├── start_backend.py          # Backend startup script
├── backend_requirements.txt  # Python dependencies
├── requirements.txt          # Legacy requirements (for reference)
└── README.md                 # This file
```

## 🔧 Development

### **Frontend Development**
```bash
# Start with developer tools
npm run dev
```

### **Backend Development**
```bash
# Start with debug mode
python backend.py
```

### **Building for Distribution**
```bash
# Build Electron app
npm run build
```

## 📈 Example Workflow

1. **Load Data**: Upload a CSV with time series data (e.g., sensor readings over time)
2. **Visualize**: Generate an interactive plot to understand your data
3. **Define Regions**: Click to mark where your first analysis region should end
4. **Set Parameters**: Specify how many total regions you want to analyze
5. **Analyze**: Get detailed statistics for each region
6. **Compare**: Use the statistics to compare different time periods in your data

## 🛡️ Error Handling

The application includes comprehensive error handling:

- **CSV Validation**: Checks for proper headers and data formats
- **Column Validation**: Ensures selected columns exist and contain valid data
- **Region Validation**: Validates region boundaries and sizes
- **Backend Communication**: Handles network errors and API failures
- **User Feedback**: Clear error messages and loading indicators

## 🔍 Troubleshooting

### **Backend Issues**
- **"Module not found"**: Run `pip install -r backend_requirements.txt`
- **"Port already in use"**: Kill any existing processes on port 5000
- **"Backend not accessible"**: Ensure the backend is running before starting the frontend

### **Frontend Issues**
- **"npm command not found"**: Install Node.js from [nodejs.org](https://nodejs.org)
- **"Electron fails to start"**: Try `npm install` to reinstall dependencies
- **"Drag & drop not working"**: This requires the tkinterdnd2 equivalent for Electron (built-in)

### **Data Issues**
- **"No data found"**: Ensure your CSV has headers and data rows
- **"Column not found"**: Check that column names match exactly (case-sensitive)
- **"Invalid region"**: Ensure the first region end point is within your data range

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test both frontend and backend
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details. 