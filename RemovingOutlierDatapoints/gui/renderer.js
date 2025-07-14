const { ipcRenderer } = require('electron');

// Application State
let currentData = null;
let currentChart = null;
let selectedRegionStart = null;
let selectedRegionEnd = null;
let selectedFirstRegionEnd = null;
let currentSelectionMode = 'region-start';
let chartClickEnabled = false;
let csvContent = null;

// Removed zoom state - now using larger, simpler preview

let PORT = 5001;


// Backend configuration
const BACKEND_URL = `http://127.0.0.1:${PORT}`;
let backendRunning = false;

// DOM Elements
const landingPage = document.getElementById('landing-page');
const plottingPage = document.getElementById('plotting-page');
const dropZone = document.getElementById('drop-zone');
const browseBtn = document.getElementById('browse-btn');
const backBtn = document.getElementById('back-btn');
const timeColumnSelect = document.getElementById('time-column');
const dataColumnSelect = document.getElementById('data-column');
const plotBtn = document.getElementById('plot-btn');
const chartCanvas = document.getElementById('chart');
const chartPlaceholder = document.getElementById('chart-placeholder');
const loadingOverlay = document.getElementById('loading-overlay');

// Region Analysis DOM Elements
const regionAnalysis = document.getElementById('region-analysis');
const regionStartDisplay = document.getElementById('region-start');
const regionEndDisplay = document.getElementById('region-end');
const firstRegionEnd = document.getElementById('first-region-end');
const clearRegionStartBtn = document.getElementById('clear-region-start');
const clearRegionEndBtn = document.getElementById('clear-region-end');
const clearFirstRegionEndBtn = document.getElementById('clear-first-region-end');
const clearAllSelectionsBtn = document.getElementById('clear-all-selections');
const nextBtn = document.getElementById('next-btn');
const regionInfo = document.getElementById('region-info');
const selectedRegionStartInfo = document.getElementById('selected-region-start-info');
const selectedRegionEndInfo = document.getElementById('selected-region-end-info');
const selectedFirstRegionEndInfo = document.getElementById('selected-first-region-end-info');
const totalRegionsInfo = document.getElementById('total-regions-info');
const pointsPerRegionInfo = document.getElementById('points-per-region');

// Mode selection buttons
const modeRegionStartBtn = document.getElementById('mode-region-start');
const modeRegionEndBtn = document.getElementById('mode-region-end');
const modeFirstRegionEndBtn = document.getElementById('mode-first-region-end');

// Modal elements
const errorModal = document.getElementById('error-modal');
const successModal = document.getElementById('success-modal');
const previewModal = document.getElementById('preview-modal');
const errorMessage = document.getElementById('error-message');
const errorClose = document.getElementById('error-close');
const successClose = document.getElementById('success-close');
const previewClose = document.getElementById('preview-close');

// Preview modal elements
const previewImage = document.getElementById('preview-image');
const previewImageContainer = document.getElementById('preview-image-container');
const previewRegionsCount = document.getElementById('preview-regions-count');
const previewShiftsCount = document.getElementById('preview-shifts-count');
const pValueSlider = document.getElementById('p-value-slider');
const pValueDisplay = document.getElementById('p-value-display');
const livePreviewStatus = document.getElementById('live-preview-status');
const previewContinueBtn = document.getElementById('preview-continue-btn');
const regressionIndicesInput = document.getElementById('regression-indices');

// File info elements
const fileName = document.getElementById('file-name');
const fileStats = document.getElementById('file-stats');
const successFilename = document.getElementById('success-filename');
const successRows = document.getElementById('success-rows');
const successColumns = document.getElementById('success-columns');

// Removed zoom and pan functions - now using larger, simpler preview image display

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    setupDragAndDrop();
    checkBackendHealth();
});

// Event Listeners
function setupEventListeners() {
    // Browse button
    browseBtn.addEventListener('click', handleBrowseClick);
    
    // Back button
    backBtn.addEventListener('click', () => switchToPage('landing'));
    
    // Column selects
    timeColumnSelect.addEventListener('change', checkPlotButtonState);
    dataColumnSelect.addEventListener('change', checkPlotButtonState);
    
    // Plot button
    plotBtn.addEventListener('click', generatePlot);
    
    // Region analysis controls
    clearRegionStartBtn.addEventListener('click', () => clearRegionSelection('region-start'));
    clearRegionEndBtn.addEventListener('click', () => clearRegionSelection('region-end'));
    clearFirstRegionEndBtn.addEventListener('click', () => clearRegionSelection('first-region-end'));
    clearAllSelectionsBtn.addEventListener('click', () => clearRegionSelection('all'));
    nextBtn.addEventListener('click', handleNextClick);
    
    // Mode selection buttons
    modeRegionStartBtn.addEventListener('click', () => setSelectionMode('region-start'));
    modeRegionEndBtn.addEventListener('click', () => setSelectionMode('region-end'));
    modeFirstRegionEndBtn.addEventListener('click', () => setSelectionMode('first-region-end'));
    
    // Modal close buttons
    errorClose.addEventListener('click', () => hideModal('error'));
    successClose.addEventListener('click', () => hideModal('success'));
    previewClose.addEventListener('click', () => hideModal('preview'));
    
    // Preview modal controls
    pValueSlider.addEventListener('input', handlePValueSliderChange);
    previewContinueBtn.addEventListener('click', handlePreviewContinue);
    
    // Click outside modal to close
    errorModal.addEventListener('click', (e) => {
        if (e.target === errorModal) hideModal('error');
    });
    successModal.addEventListener('click', (e) => {
        if (e.target === successModal) hideModal('success');
    });
    previewModal.addEventListener('click', (e) => {
        if (e.target === previewModal) hideModal('preview');
    });
    
    // Drop zone click
    dropZone.addEventListener('click', handleBrowseClick);
}

// Drag and Drop Setup
function setupDragAndDrop() {
    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });

    // Highlight drop zone when item is dragged over it
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, unhighlight, false);
    });

    // Handle dropped files
    dropZone.addEventListener('drop', handleDrop, false);
}

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

function highlight() {
    dropZone.classList.add('drag-over');
}

function unhighlight() {
    dropZone.classList.remove('drag-over');
}

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;

    if (files.length > 0) {
        const file = files[0];
        if (file.name.toLowerCase().endsWith('.csv')) {
            handleFileLoad(file.path);
        } else {
            showError('Please drop a CSV file (.csv extension required)');
        }
    }
}

// File Handling
async function handleBrowseClick() {
    try {
        showLoading();
        const result = await ipcRenderer.invoke('select-csv-file');
        hideLoading();
        
        if (result.success) {
            await processCSVData(result.content, result.filename);
        } else if (!result.cancelled) {
            showError(result.error || 'Failed to read file');
        }
    } catch (error) {
        hideLoading();
        showError('Error selecting file: ' + error.message);
    }
}

async function handleFileLoad(filePath) {
    try {
        showLoading();
        const result = await ipcRenderer.invoke('read-file', filePath);
        hideLoading();
        
        if (result.success) {
            await processCSVData(result.content, result.filename);
        } else {
            showError(result.error || 'Failed to read file');
        }
    } catch (error) {
        hideLoading();
        showError('Error reading file: ' + error.message);
    }
}

async function processCSVData(csvContentParam, filename) {
    try {
        showLoading();
        
        // Store CSV content for backend communication
        csvContent = csvContentParam;
        
        // Parse CSV using Papa Parse
        const parseResult = Papa.parse(csvContentParam, {
            header: true,
            skipEmptyLines: true,
            dynamicTyping: true,
            transformHeader: (header) => header.trim()
        });

        if (parseResult.errors.length > 0) {
            console.warn('CSV parsing warnings:', parseResult.errors);
        }

        if (!parseResult.data || parseResult.data.length === 0) {
            throw new Error('No data found in CSV file');
        }

        currentData = {
            data: parseResult.data,
            columns: Object.keys(parseResult.data[0]),
            filename: filename,
            rowCount: parseResult.data.length
        };

        hideLoading();
        
        // Update UI
        updateFileInfo();
        populateColumnSelects();
        autoSelectColumns();
        switchToPage('plotting');
        
        // Show success modal
        showSuccess();
        
    } catch (error) {
        hideLoading();
        showError('Error processing CSV: ' + error.message);
    }
}

// UI Updates
function updateFileInfo() {
    if (!currentData) return;
    
    fileName.textContent = currentData.filename;
    fileStats.textContent = `${currentData.rowCount.toLocaleString()} rows × ${currentData.columns.length} columns`;
}

function populateColumnSelects() {
    if (!currentData) return;
    
    // Clear existing options
    timeColumnSelect.innerHTML = '<option value="">Select time column...</option>';
    dataColumnSelect.innerHTML = '<option value="">Select data column...</option>';
    
    // Add column options
    currentData.columns.forEach(column => {
        const timeOption = document.createElement('option');
        timeOption.value = column;
        timeOption.textContent = column;
        timeColumnSelect.appendChild(timeOption);
        
        const dataOption = document.createElement('option');
        dataOption.value = column;
        dataOption.textContent = column;
        dataColumnSelect.appendChild(dataOption);
    });
}

function autoSelectColumns() {
    if (!currentData) return;
    
    const columns = currentData.columns;
    
    // Try to auto-select time column
    const timeKeywords = ['time', 'timestamp', 'date', 'datetime', 'Time', 'Timestamp', 'Date', 'DateTime'];
    let timeColumnSelected = false;
    
    for (const keyword of timeKeywords) {
        for (const column of columns) {
            if (column.toLowerCase().includes(keyword.toLowerCase())) {
                timeColumnSelect.value = column;
                timeColumnSelected = true;
                break;
            }
        }
        if (timeColumnSelected) break;
    }
    
    // If no time column found, select first column
    if (!timeColumnSelected && columns.length > 0) {
        timeColumnSelect.value = columns[0];
    }
    
    // Select data column (different from time column)
    const selectedTimeColumn = timeColumnSelect.value;
    for (const column of columns) {
        if (column !== selectedTimeColumn) {
            dataColumnSelect.value = column;
            break;
        }
    }
    
    checkPlotButtonState();
}

function checkPlotButtonState() {
    const timeColumn = timeColumnSelect.value;
    const dataColumn = dataColumnSelect.value;
    
    plotBtn.disabled = !timeColumn || !dataColumn || timeColumn === dataColumn;
}

// Chart Generation
async function generatePlot() {
    if (!currentData) return;
    
    const timeColumn = timeColumnSelect.value;
    const dataColumn = dataColumnSelect.value;
    
    if (!timeColumn || !dataColumn) {
        showError('Please select both time and data columns');
        return;
    }
    
    try {
        showLoading();
        
        // Prepare chart data
        const chartData = prepareChartData(timeColumn, dataColumn);
        
        // Determine if time column contains timestamps
        const isTimeBased = chartData.datasets[0].data.length > 0 && chartData.datasets[0].data[0].x > 1000000000;
        
        // Destroy existing chart if it exists
        if (currentChart) {
            currentChart.destroy();
        }
        
        // Create new chart
        const ctx = chartCanvas.getContext('2d');
        currentChart = new Chart(ctx, {
            type: 'line',
            data: chartData,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'nearest',
                    axis: 'x',
                    intersect: false
                },
                plugins: {
                    title: {
                        display: true,
                        text: `${dataColumn} vs ${timeColumn}`,
                        font: {
                            size: 16,
                            weight: 'bold'
                        }
                    },
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            title: function(context) {
                                const point = context[0];
                                if (isTimeBased) {
                                    return `${timeColumn}: ${new Date(point.parsed.x).toLocaleString()}`;
                                }
                                return `${timeColumn}: ${point.parsed.x.toLocaleString()}`;
                            },
                            label: function(context) {
                                return `${dataColumn}: ${context.parsed.y.toLocaleString()}`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        display: true,
                        type: isTimeBased ? 'time' : 'linear',
                        title: {
                            display: true,
                            text: timeColumn,
                            font: {
                                size: 14,
                                weight: 'bold'
                            }
                        },
                        ticks: {
                            callback: function(value, index, values) {
                                if (isTimeBased) {
                                    return new Date(value).toLocaleString();
                                }
                                return value;
                            },
                            maxTicksLimit: 10
                        }
                    },
                    y: {
                        display: true,
                        title: {
                            display: true,
                            text: dataColumn,
                            font: {
                                size: 14,
                                weight: 'bold'
                            }
                        },
                        ticks: {
                            callback: function(value) {
                                return value.toLocaleString();
                            }
                        }
                    }
                },
                elements: {
                    point: {
                        radius: 1,
                        hoverRadius: 4
                    },
                    line: {
                        borderWidth: 2
                    }
                }
            }
        });
        
        // Hide placeholder, show chart
        chartPlaceholder.classList.add('hidden');
        chartCanvas.style.display = 'block';
        
        // Show region analysis section
        regionAnalysis.classList.remove('hidden');
        
        // Set default region selections
        await setDefaultRegionSelections();
        enableChartSelection();
        
        hideLoading();
        
    } catch (error) {
        hideLoading();
        console.error('Error generating plot:', error);
        showError('Failed to generate plot. Please check your data.');
    }
}

function prepareChartData(timeColumn, dataColumn) {
    const data = currentData.data;
    const chartPoints = [];
    
    for (let i = 0; i < data.length; i++) {
        const row = data[i];
        const timeValue = row[timeColumn];
        const dataValue = row[dataColumn];
        
        // Skip rows with missing data
        if (timeValue == null || dataValue == null) continue;
        
        let parsedTime;
        
        // Enhanced time parsing
        if (typeof timeValue === 'number') {
            parsedTime = timeValue;
        } else if (typeof timeValue === 'string') {
            // Remove any extra whitespace
            const cleanTimeValue = timeValue.toString().trim();
            
            // Try to parse as number first (for seconds, minutes, etc.)
            const numValue = parseFloat(cleanTimeValue);
            if (!isNaN(numValue)) {
                parsedTime = numValue;
            } else {
                // Try various date formats
                let dateValue = new Date(cleanTimeValue);
                
                // If that fails, try some common formats
                if (isNaN(dateValue.getTime())) {
                    // Try YYYY-MM-DD HH:mm:ss format
                    if (cleanTimeValue.match(/^\d{4}-\d{2}-\d{2}/)) {
                        dateValue = new Date(cleanTimeValue);
                    }
                    // Try MM/DD/YYYY format
                    else if (cleanTimeValue.match(/^\d{1,2}\/\d{1,2}\/\d{4}/)) {
                        dateValue = new Date(cleanTimeValue);
                    }
                    // Try DD-MM-YYYY format
                    else if (cleanTimeValue.match(/^\d{1,2}-\d{1,2}-\d{4}/)) {
                        const parts = cleanTimeValue.split('-');
                        dateValue = new Date(`${parts[2]}-${parts[1]}-${parts[0]}`);
                    }
                }
                
                if (!isNaN(dateValue.getTime())) {
                    parsedTime = dateValue.getTime();
                } else {
                    // Use row index as fallback to ensure proper spacing
                    parsedTime = i;
                }
            }
        } else {
            parsedTime = i; // Use index as fallback
        }
        
        // Ensure data value is numeric
        let numericDataValue;
        if (typeof dataValue === 'number') {
            numericDataValue = dataValue;
        } else {
            // Handle string numbers and remove any non-numeric characters except decimal point and minus
            const cleanDataValue = dataValue.toString().replace(/[^\d.-]/g, '');
            numericDataValue = parseFloat(cleanDataValue);
        }
        
        if (isNaN(numericDataValue)) continue;
        
        chartPoints.push({
            x: parsedTime,
            y: numericDataValue
        });
    }
    
    // Sort by x value
    chartPoints.sort((a, b) => a.x - b.x);
    
    // If we have very few unique X values but many points, use index-based X values
    const uniqueXValues = new Set(chartPoints.map(p => p.x));
    if (uniqueXValues.size < 10 && chartPoints.length > 100) {
        console.log('Using index-based X values for better visualization');
        chartPoints.forEach((point, index) => {
            point.x = index;
        });
    }
    
    return {
        datasets: [{
            label: dataColumn,
            data: chartPoints,
            borderColor: '#667eea',
            backgroundColor: 'rgba(102, 126, 234, 0.1)',
            fill: false,
            tension: 0.1
        }]
    };
}

// Page Navigation
function switchToPage(page) {
    if (page === 'landing') {
        landingPage.classList.add('active');
        plottingPage.classList.remove('active');
        
        // Reset chart
        if (currentChart) {
            currentChart.destroy();
            currentChart = null;
        }
        chartPlaceholder.classList.remove('hidden');
        chartCanvas.style.display = 'none';
        
        // Reset region analysis
        regionAnalysis.classList.add('hidden');
        disableChartSelection();
        clearRegionSelection();
        nextBtn.disabled = true;
        
        // Reset data
        currentData = null;
    } else if (page === 'plotting') {
        landingPage.classList.remove('active');
        plottingPage.classList.add('active');
    }
}

// Modal Functions
function showError(message) {
    errorMessage.textContent = message;
    errorModal.classList.add('show');
}

function showSuccess() {
    if (!currentData) return;
    
    successFilename.textContent = currentData.filename;
    successRows.textContent = currentData.rowCount.toLocaleString();
    successColumns.textContent = currentData.columns.length.toString();
    successModal.classList.add('show');
}

function hideModal(type) {
    if (type === 'error') {
        errorModal.classList.remove('show');
    } else if (type === 'success') {
        successModal.classList.remove('show');
    } else if (type === 'preview') {
        previewModal.classList.remove('show');
    }
}

// Loading Functions
function showLoading() {
    loadingOverlay.classList.add('show');
}

function hideLoading() {
    loadingOverlay.classList.remove('show');
}

// Region Analysis Functions
function enableChartSelection() {
    chartClickEnabled = true;
    const chartContainer = document.querySelector('.chart-container');
    chartContainer.classList.add('selection-mode');
    
    // Add click handler to chart
    chartCanvas.addEventListener('click', handleChartClick);
}

function disableChartSelection() {
    chartClickEnabled = false;
    const chartContainer = document.querySelector('.chart-container');
    chartContainer.classList.remove('selection-mode');
    
    // Remove click handler
    chartCanvas.removeEventListener('click', handleChartClick);
}

function setSelectionMode(mode) {
    currentSelectionMode = mode;
    
    // Update active mode button
    document.querySelectorAll('.mode-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    const activeBtn = document.querySelector(`.mode-btn[data-mode="${mode}"]`);
    if (activeBtn) {
        activeBtn.classList.add('active');
    }
    
    // Update chart cursor or instruction based on mode
    if (currentChart && chartClickEnabled) {
        currentChart.canvas.style.cursor = 'crosshair';
    }
}

function handleChartClick(event) {
    if (!chartClickEnabled || !currentChart) return;
    
    try {
        const rect = chartCanvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        
        // Get chart position
        const canvasPosition = Chart.helpers.getRelativePosition(event, currentChart);
        
        // Get data value
        const dataX = currentChart.scales.x.getValueForPixel(canvasPosition.x);
        const dataY = currentChart.scales.y.getValueForPixel(canvasPosition.y);
        
        if (dataX != null && dataY != null) {
            // Find the closest data point
            const dataset = currentChart.data.datasets[0];
            let closestPoint = null;
            let minDistance = Infinity;
            
            dataset.data.forEach((point, index) => {
                const distance = Math.abs(point.x - dataX);
                if (distance < minDistance) {
                    minDistance = distance;
                    closestPoint = { ...point, chartIndex: index };
                }
            });
            
            if (closestPoint) {
                // Find the original data index by matching the time value
                const timeColumn = timeColumnSelect.value;
                let originalIndex = -1;
                for (let i = 0; i < currentData.data.length; i++) {
                    const row = currentData.data[i];
                    let timeValue = row[timeColumn];
                    
                    // Parse time value the same way as in prepareChartData
                    if (typeof timeValue === 'number') {
                        // Already numeric
                    } else if (typeof timeValue === 'string') {
                        const numValue = parseFloat(timeValue.toString().trim());
                        if (!isNaN(numValue)) {
                            timeValue = numValue;
                        } else {
                            // Try date parsing
                            const dateValue = new Date(timeValue.toString().trim());
                            if (!isNaN(dateValue.getTime())) {
                                timeValue = dateValue.getTime();
                            } else {
                                timeValue = i; // fallback to index
                            }
                        }
                    } else {
                        timeValue = i; // fallback to index
                    }
                    
                    // Check if this matches our selected point (with small tolerance for floating point)
                    if (Math.abs(timeValue - closestPoint.x) < 0.001) {
                        originalIndex = i;
                        break;
                    }
                }
                
                // Add the original index to the selected point
                const selectedPoint = { ...closestPoint, originalIndex };
                
                // Update the appropriate selection based on current mode
                switch (currentSelectionMode) {
                    case 'region-start':
                        selectedRegionStart = selectedPoint;
                        break;
                    case 'region-end':
                        selectedRegionEnd = selectedPoint;
                        break;
                    case 'first-region-end':
                        selectedFirstRegionEnd = selectedPoint;
                        break;
                }
                
                updateRegionSelection();
                addRegionMarker(closestPoint.x);
                checkNextButtonState();
            }
        }
    } catch (error) {
        console.error('Error handling chart click:', error);
    }
}

function updateRegionSelection() {
        const timeColumn = timeColumnSelect.value;
        
    // Helper function to format display value
    function formatDisplayValue(selectedPoint) {
        if (!selectedPoint) return 'Click on chart to select';
        
        if (selectedPoint.x > 1000000000) {
            // Timestamp
            return new Date(selectedPoint.x).toLocaleString();
        } else {
            return `${timeColumn}: ${selectedPoint.x.toLocaleString()}`;
        }
    }
    
    // Update Region Start display
    const regionStartValue = formatDisplayValue(selectedRegionStart);
    regionStartDisplay.textContent = regionStartValue;
    if (selectedRegionStart) {
        regionStartDisplay.classList.add('selected');
        clearRegionStartBtn.style.display = 'inline-block';
    } else {
        regionStartDisplay.classList.remove('selected');
        clearRegionStartBtn.style.display = 'none';
    }
    
    // Update Region End display
    const regionEndValue = formatDisplayValue(selectedRegionEnd);
    regionEndDisplay.textContent = regionEndValue;
    if (selectedRegionEnd) {
        regionEndDisplay.classList.add('selected');
        clearRegionEndBtn.style.display = 'inline-block';
    } else {
        regionEndDisplay.classList.remove('selected');
        clearRegionEndBtn.style.display = 'none';
        }
        
    // Update First Region End display
    const firstRegionEndValue = formatDisplayValue(selectedFirstRegionEnd);
    firstRegionEnd.textContent = firstRegionEndValue;
    if (selectedFirstRegionEnd) {
        firstRegionEnd.classList.add('selected');
        clearFirstRegionEndBtn.style.display = 'inline-block';
    } else {
        firstRegionEnd.classList.remove('selected');
        clearFirstRegionEndBtn.style.display = 'none';
    }
        
        updateRegionInfo();
    }

function clearRegionSelection(type = 'all') {
    let shouldUpdateMarkers = false;
    
    if (type === 'region-start' || type === 'all') {
        selectedRegionStart = null;
        shouldUpdateMarkers = true;
    }
    
    if (type === 'region-end' || type === 'all') {
    selectedRegionEnd = null;
        shouldUpdateMarkers = true;
    }
    
    if (type === 'first-region-end' || type === 'all') {
        selectedFirstRegionEnd = null;
        shouldUpdateMarkers = true;
    }
    
    if (shouldUpdateMarkers) {
        updateRegionSelection();
        updateRegionMarkers();
    checkNextButtonState();
    }
}



function checkNextButtonState() {
    // At least one selection is required to proceed
    const hasSelection = selectedRegionStart !== null || selectedRegionEnd !== null || selectedFirstRegionEnd !== null;
    
    nextBtn.disabled = !hasSelection;
}

function updateRegionInfo() {
    const hasAnySelection = selectedRegionStart || selectedRegionEnd || selectedFirstRegionEnd;
    
    if (hasAnySelection) {
        // Helper function to format values for display
        function formatInfoValue(selectedPoint) {
            if (!selectedPoint) return '-';
            
            return selectedPoint.x > 1000000000 
                ? new Date(selectedPoint.x).toLocaleString()
                : selectedPoint.x.toLocaleString();
        }
        
        // Update info display for all selections
        selectedRegionStartInfo.textContent = formatInfoValue(selectedRegionStart);
        selectedRegionEndInfo.textContent = formatInfoValue(selectedRegionEnd);
        selectedFirstRegionEndInfo.textContent = formatInfoValue(selectedFirstRegionEnd);
        
        // Update region count and points info
        totalRegionsInfo.textContent = 'Auto-detected based on selections';
        
        // Calculate points info based on selections
        const totalPoints = currentData.rowCount;
        let pointsInfo = '';
        
        if (selectedRegionStart && selectedRegionEnd) {
            const regionPoints = Math.abs(selectedRegionEnd.originalIndex - selectedRegionStart.originalIndex) + 1;
            pointsInfo = `Selected region: ${regionPoints} points`;
        } else if (selectedFirstRegionEnd) {
            const firstRegionPoints = selectedFirstRegionEnd.originalIndex + 1;
            pointsInfo = `First region: ${firstRegionPoints} points, others auto-detected`;
        } else {
            pointsInfo = 'Region boundaries set for analysis';
        }
        
        pointsPerRegionInfo.textContent = pointsInfo;
        
        regionInfo.classList.remove('hidden');
    } else {
        regionInfo.classList.add('hidden');
    }
}

function addRegionMarker(xValue) {
    // This function now just calls updateRegionMarkers to redraw all markers
    updateRegionMarkers();
}

function updateRegionMarkers() {
    if (!currentChart) return;
    
    // Remove existing vertical lines plugin if it exists
    if (currentChart.options.plugins && currentChart.options.plugins.verticalLines) {
        delete currentChart.options.plugins.verticalLines;
    }
    
    // Create markers for all active selections with distinct colors
    const markers = [];
    
    if (selectedRegionStart) {
        markers.push({
            x: selectedRegionStart.x,
            color: '#28a745',  // Green for start
            label: 'Region Start',
            emoji: '🟢',
            lineStyle: [5, 5]  // Dashed line
        });
    }
    
    if (selectedRegionEnd) {
        markers.push({
            x: selectedRegionEnd.x,
            color: '#dc3545',  // Red for end
            label: 'Region End',
            emoji: '🔴',
            lineStyle: [10, 5]  // Different dash pattern
        });
    }
    
    if (selectedFirstRegionEnd) {
        markers.push({
            x: selectedFirstRegionEnd.x,
            color: '#ffc107',  // Yellow/Gold for first region end
            label: 'First Region End',
            emoji: '🟡',
            lineStyle: [15, 5]  // Another dash pattern
        });
    }
    
    if (markers.length === 0) {
        // If no markers, just update the chart to remove any existing lines
        currentChart.update('none');
        return;
    }
    
    // Create and register the vertical lines plugin
    const verticalLinesPlugin = {
        id: 'verticalLines',
        beforeDraw: function(chart, args, options) {
            const ctx = chart.ctx;
            const xAxis = chart.scales.x;
            const yAxis = chart.scales.y;
            
            markers.forEach((marker, index) => {
                const x = xAxis.getPixelForValue(marker.x);
                
                // Draw vertical line
                ctx.save();
                ctx.beginPath();
                ctx.moveTo(x, yAxis.top);
                ctx.lineTo(x, yAxis.bottom);
                ctx.lineWidth = 4;
                ctx.strokeStyle = marker.color;
                ctx.setLineDash(marker.lineStyle);
                ctx.stroke();
                ctx.restore();
                
                // Add label with emoji at the top
                ctx.save();
                ctx.fillStyle = marker.color;
                ctx.font = 'bold 14px Arial';
                ctx.textAlign = 'center';
                ctx.shadowColor = 'rgba(0, 0, 0, 0.5)';
                ctx.shadowBlur = 3;
                ctx.shadowOffsetX = 1;
                ctx.shadowOffsetY = 1;
                
                const labelY = yAxis.top - 10 - (index * 25); // Stagger labels vertically
                ctx.fillText(`${marker.emoji} ${marker.label}`, x, labelY);
                ctx.restore();
                
                // Add a small circle at the intersection point
                const dataPoint = chart.data.datasets[0].data.find(point => 
                    Math.abs(point.x - marker.x) < 0.001
                );
                if (dataPoint) {
                    const y = yAxis.getPixelForValue(dataPoint.y);
                    ctx.save();
                    ctx.beginPath();
                    ctx.arc(x, y, 6, 0, 2 * Math.PI);
                    ctx.fillStyle = marker.color;
                    ctx.fill();
                    ctx.strokeStyle = 'white';
                    ctx.lineWidth = 2;
                    ctx.stroke();
                    ctx.restore();
                }
            });
        }
    };
    
    // Unregister any existing plugin and register the new one
    Chart.unregister(verticalLinesPlugin);
    Chart.register(verticalLinesPlugin);
    
    // Update chart with animation disabled for better performance
    currentChart.update('none');
}

async function handleNextClick() {
    // Check that at least one region boundary is selected
    if (!selectedRegionStart && !selectedRegionEnd && !selectedFirstRegionEnd) {
        showError('Please select at least one region boundary');
        return;
    }
    
    if (!backendRunning) {
        showError('Python backend is not running. Please start the backend server first.');
        return;
    }
    
    try {
        showLoading();
        
        // Prepare data for preview with all region selections
        const previewData = {
            csv_content: csvContent,
            time_column: timeColumnSelect.value,
            response: dataColumnSelect.value,
            region_start_index: selectedRegionStart ? selectedRegionStart.x : null,
            region_end_index: selectedRegionEnd ? selectedRegionEnd.x : null,
            first_region_end_index: selectedFirstRegionEnd ? selectedFirstRegionEnd.x : null
        };
        
        console.log('Sending data to backend for preview:', {
            time_column: previewData.time_column,
            response: previewData.response,
            region_start_index: previewData.region_start_index,
            region_end_index: previewData.region_end_index,
            first_region_end_index: previewData.first_region_end_index
        });
        
        // Send to Python backend for preview
        const response = await fetch(`${BACKEND_URL}/preview`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(previewData)
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Preview generation failed');
        }
        
        const results = await response.json();
        hideLoading();
        
        // Display preview
        showPreview(results);
        
    } catch (error) {
        hideLoading();
        console.error('Preview error:', error);
        showError(`Preview failed: ${error.message}`);
    }
}

// Preview Functions
function showPreview(previewData) {
    // Set preview image
    previewImage.src = `data:image/png;base64,${previewData.preview_image}`;
    
    // Set preview stats
    previewRegionsCount.textContent = previewData.detected_regions_count;
    previewShiftsCount.textContent = previewData.shifts_count;
    
    // Initialize regression indices with default values based on region count
    const regionCount = previewData.detected_regions_count;
    const defaultIndices = Array.from({length: regionCount}, (_, i) => i + 1).join(',');
    regressionIndicesInput.value = defaultIndices;
    
    // Initialize slider and display
    pValueSlider.value = 70;
    pValueDisplay.textContent = 70;
    livePreviewStatus.textContent = 'Move slider to see live preview';
    livePreviewStatus.className = 'live-status';
    
    // Show preview modal
    previewModal.classList.add('show');
}

// Live preview functionality
let previewUpdateTimeout;
let lastPreviewData = null;

async function handlePValueSliderChange() {
    const newPValue = parseFloat(pValueSlider.value);
    pValueDisplay.textContent = newPValue;
    
    // Update status
    livePreviewStatus.textContent = 'Updating preview...';
    livePreviewStatus.className = 'live-status updating';
    
    // Clear previous timeout
    if (previewUpdateTimeout) {
        clearTimeout(previewUpdateTimeout);
    }
    
    // Debounce the preview update
    previewUpdateTimeout = setTimeout(() => {
        updateLivePreview(newPValue);
    }, 500); // Wait 500ms after user stops moving slider
}

async function updateLivePreview(pValue) {
    try {
        // Check if any region selection exists
        if (!selectedRegionStart && !selectedRegionEnd && !selectedFirstRegionEnd) return;
        
        // Show loading state
        previewImageContainer.classList.add('loading');
        
        // Get and validate regression indices
        let regressionIndices = null;
        const indicesInput = regressionIndicesInput.value.trim();
        if (indicesInput) {
            try {
                regressionIndices = indicesInput.split(',').map(x => parseFloat(x.trim())).filter(x => !isNaN(x));
            } catch (e) {
                console.warn('Invalid regression indices format:', indicesInput);
            }
        }
        
        // Prepare data for backend with all region selections
        const previewData = {
            csv_content: csvContent,
            time_column: timeColumnSelect.value,
            response: dataColumnSelect.value,
            region_start_index: selectedRegionStart ? selectedRegionStart.x : null,
            region_end_index: selectedRegionEnd ? selectedRegionEnd.x : null,
            first_region_end_index: selectedFirstRegionEnd ? selectedFirstRegionEnd.x : null,
            p_value: pValue,
            regression_indices: regressionIndices
        };
        
        // Send to Python backend for live preview
        const response = await fetch(`${BACKEND_URL}/live-preview`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(previewData)
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Live preview update failed');
    }
    
        const results = await response.json();
        
        // Update preview with new data
        previewImage.src = `data:image/png;base64,${results.preview_image}`;
        previewRegionsCount.textContent = results.detected_regions_count;
        previewShiftsCount.textContent = results.shifts_count;
        
        // Update regression indices based on new region count (only if user hasn't entered custom ones)
        const regionCount = results.detected_regions_count;
        if (!indicesInput) {
            const defaultIndices = Array.from({length: regionCount}, (_, i) => i + 1).join(',');
            regressionIndicesInput.value = defaultIndices;
        }
        
        // Store the updated data
        lastPreviewData = results;
        
        // Update status
        livePreviewStatus.textContent = `Updated with p-value ${pValue}`;
        livePreviewStatus.className = 'live-status updated';
        
    } catch (error) {
        console.error('Live preview error:', error);
        livePreviewStatus.textContent = 'Preview update failed';
        livePreviewStatus.className = 'live-status';
    } finally {
        // Remove loading state
        previewImageContainer.classList.remove('loading');
    }
}

async function handlePreviewContinue() {
    const currentPValue = parseFloat(pValueSlider.value);
    
    // Hide preview modal and run analysis with current p-value
    hideModal('preview');
    await runAnalysis(currentPValue);
}

async function runAnalysis(pValue = 70) {
    try {
        showLoading();
        
        // Get and validate regression indices
        let regressionIndices = null;
        const indicesInput = regressionIndicesInput.value.trim();
        if (indicesInput) {
            try {
                regressionIndices = indicesInput.split(',').map(x => parseFloat(x.trim())).filter(x => !isNaN(x));
            } catch (e) {
                console.warn('Invalid regression indices format:', indicesInput);
            }
        }
        
        // Prepare data for backend with all region selections
        const analysisData = {
            csv_content: csvContent,
            time_column: timeColumnSelect.value,
            response: dataColumnSelect.value,
            region_start_index: selectedRegionStart ? selectedRegionStart.x : null,
            region_end_index: selectedRegionEnd ? selectedRegionEnd.x : null,
            first_region_end_index: selectedFirstRegionEnd ? selectedFirstRegionEnd.x : null,
            p_value: pValue,
            regression_indices: regressionIndices
        };
        
        console.log('Sending data to backend for analysis:', {
            time_column: analysisData.time_column,
            response: analysisData.response,
            region_start_index: analysisData.region_start_index,
            region_end_index: analysisData.region_end_index,
            first_region_end_index: analysisData.first_region_end_index,
            p_value: analysisData.p_value,
            regression_indices: analysisData.regression_indices
        });
        
        // Send to Python backend
        const response = await fetch(`${BACKEND_URL}/analyze`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(analysisData)
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Backend analysis failed');
        }
        
        const results = await response.json();
        hideLoading();
        
        // Display results
        displayAnalysisResults(results);
        
    } catch (error) {
        hideLoading();
        console.error('Analysis error:', error);
        showError(`Analysis failed: ${error.message}`);
    }
}

// Backend Communication Functions
async function checkBackendHealth() {
    try {
        const response = await fetch(`${BACKEND_URL}/health`, {
            method: 'GET',
            timeout: 5000
        });
        
        if (response.ok) {
            const data = await response.json();
            backendRunning = true;
            console.log('Backend is running:', data);
        } else {
            backendRunning = false;
            console.warn('Backend health check failed');
        }
    } catch (error) {
        backendRunning = false;
        console.warn('Backend not accessible:', error.message);
    }
}

function displayAnalysisResults(results) {
    console.log('Analysis results:', results);
    
    // Create a formatted results message
    let message = `🎉 Analysis Complete!\n\n`;
    message += `📊 Processed ${results.metadata.total_data_points} data points\n`;
    message += `🔍 Detected ${results.analysis.detected_regions_count || 'Auto'} regions\n`;
    message += `🔗 Merged to ${results.analysis.merged_regions_count || results.analysis.regions.length} final regions\n`;
    message += `📈 P-value used: ${results.metadata.p_value}\n\n`;
    
    // Add region summaries
    if (results.analysis && results.analysis.regions) {
        message += `Detected ${results.analysis.regions.length} merged regions:\n\n`;
        results.analysis.regions.forEach((region, index) => {
            message += `Region ${index + 1}: ${region[0]} - ${region[1]}\n`;
        });
        
        if (results.analysis.regional_means) {
            message += `\nRegional means:\n`;
            results.analysis.regional_means.forEach((mean, index) => {
                message += `  • Region ${index + 1}: ${mean.toFixed(2)}\n`;
            });
        }
    }
    
    message += `\n📁 Analysis files have been saved and zipped.\n`;
    message += `Click the download button below to get your results.`;
    
    // Build image previews HTML
    let imagePreviews = '';
    if (results.analysis && results.analysis.analysis_images) {
        const images = results.analysis.analysis_images;
        const imageData = [
            { key: 'regions_plot', title: '📊 Regions Plot', description: 'Regional analysis with outlier removal' },
            { key: 'final_plot', title: '📈 Final Plot', description: 'Data and left derivatives visualization' },
            { key: 'regression_plot', title: '📉 Regression Plot', description: 'Statistical regression analysis' }
        ];
        
        imagePreviews = `
            <div class="results-images">
                <h3>🖼️ Generated Images</h3>
                <div class="images-grid">
                    ${imageData.map(img => {
                        if (images[img.key]) {
                            return `
                                <div class="image-preview">
                                    <div class="image-header">
                                        <h4>${img.title}</h4>
                                        <p>${img.description}</p>
                                    </div>
                                    <div class="image-container">
                                        <img src="data:image/png;base64,${images[img.key]}" alt="${img.title}" class="analysis-image">
                                    </div>
                                    <div class="image-actions">
                                        <button class="save-image-btn" onclick="saveImage('${img.key}', '${img.title}')">
                                            💾 Save Image
                                        </button>
                                    </div>
                                </div>
                            `;
                        }
                        return '';
                    }).join('')}
                </div>
            </div>
        `;
    }
    
    // Create a more sophisticated results display
    const resultsModal = document.createElement('div');
    resultsModal.className = 'results-modal';
    resultsModal.innerHTML = `
        <div class="results-content">
            <div class="results-header">
                <h2>🎉 Analysis Complete!</h2>
                <button class="close-results" onclick="this.parentElement.parentElement.parentElement.remove()">×</button>
            </div>
            <div class="results-body">
                <div class="results-summary">
                    <h3>📊 Summary</h3>
                    <p><strong>Data Points:</strong> ${results.metadata.total_data_points}</p>
                    <p><strong>Regions Detected:</strong> ${results.analysis.detected_regions_count || 'Auto'}</p>
                    <p><strong>Final Regions:</strong> ${results.analysis.merged_regions_count || results.analysis.regions.length}</p>
                    <p><strong>P-value Used:</strong> ${results.metadata.p_value}</p>
                    <p><strong>Processing Time:</strong> ${new Date(results.metadata.processing_timestamp).toLocaleString()}</p>
                </div>
                
                ${results.analysis && results.analysis.regions ? `
                <div class="results-regions">
                    <h3>🔢 Detected Regions</h3>
                    <div class="regions-list">
                        ${results.analysis.regions.map((region, index) => `
                            <div class="region-item">
                                <span class="region-number">Region ${index + 1}:</span>
                                <span class="region-range">${region[0].toFixed(2)} - ${region[1].toFixed(2)}</span>
                                ${results.analysis.regional_means && results.analysis.regional_means[index] ? 
                                    `<span class="region-mean">Mean: ${results.analysis.regional_means[index].toFixed(2)}</span>` : ''}
                            </div>
                        `).join('')}
                    </div>
                </div>
                ` : ''}
                
                ${imagePreviews}
                
                <div class="results-download">
                    <h3>📁 Download Results</h3>
                    <p>Your analysis results include all images and CSV data:</p>
                    <ul>
                        <li>📈 Final Plot (with left derivatives)</li>
                        <li>📊 Regional Analysis Plot</li>
                        <li>📉 Regression Plot</li>
                        <li>📄 CSV data with outliers removed</li>
                    </ul>
                    <button class="download-btn" onclick="downloadResults('${results.analysis.zip_path}')">
                        📥 Download Complete ZIP File
                    </button>
                </div>
            </div>
        </div>
    `;
    
    // Add CSS for the results modal
    const resultsStyle = document.createElement('style');
    resultsStyle.textContent = `
        .results-modal {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.8);
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 10000;
        }
        
        .results-content {
            background: white;
            border-radius: 12px;
            max-width: 900px;
            max-height: 90vh;
            overflow-y: auto;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        }
        
        .results-header {
            padding: 20px;
            border-bottom: 1px solid #eee;
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: #f8f9fa;
            border-radius: 12px 12px 0 0;
        }
        
        .results-header h2 {
            margin: 0;
            color: #333;
        }
        
        .close-results {
            background: none;
            border: none;
            font-size: 24px;
            cursor: pointer;
            color: #666;
            padding: 5px;
            border-radius: 50%;
            width: 35px;
            height: 35px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .close-results:hover {
            background: #f0f0f0;
        }
        
        .results-body {
            padding: 20px;
        }
        
        .results-summary, .results-regions, .results-images, .results-download {
            margin-bottom: 25px;
        }
        
        .results-summary h3, .results-regions h3, .results-images h3, .results-download h3 {
            margin: 0 0 15px 0;
            color: #333;
            font-size: 18px;
        }
        
        .results-summary p {
            margin: 8px 0;
            color: #666;
        }
        
        .regions-list {
            max-height: 200px;
            overflow-y: auto;
        }
        
        .region-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 12px;
            background: #f8f9fa;
            border-radius: 6px;
            margin-bottom: 8px;
        }
        
        .region-number {
            font-weight: bold;
            color: #333;
        }
        
        .region-range {
            color: #666;
            font-family: monospace;
        }
        
        .region-mean {
            color: #007bff;
            font-weight: 500;
        }
        
        .images-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 15px;
        }
        
        .image-preview {
            border: 1px solid #ddd;
            border-radius: 8px;
            overflow: hidden;
            background: #f9f9f9;
        }
        
        .image-header {
            padding: 12px;
            background: #f8f9fa;
            border-bottom: 1px solid #eee;
        }
        
        .image-header h4 {
            margin: 0 0 4px 0;
            color: #333;
            font-size: 14px;
        }
        
        .image-header p {
            margin: 0;
            color: #666;
            font-size: 12px;
        }
        
        .image-container {
            padding: 10px;
            text-align: center;
        }
        
        .analysis-image {
            max-width: 100%;
            height: auto;
            border-radius: 4px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            cursor: pointer;
            transition: transform 0.2s;
        }
        
        .analysis-image:hover {
            transform: scale(1.02);
        }
        
        .image-actions {
            padding: 10px;
            border-top: 1px solid #eee;
        }
        
        .save-image-btn {
            background: #28a745;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            font-size: 12px;
            cursor: pointer;
            width: 100%;
            transition: background 0.3s;
        }
        
        .save-image-btn:hover {
            background: #218838;
        }
        
        .save-image-btn:active {
            transform: translateY(1px);
        }
        
        .results-download ul {
            margin: 10px 0;
            padding-left: 20px;
        }
        
        .results-download li {
            margin: 5px 0;
            color: #666;
        }
        
        .download-btn {
            background: #007bff;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            font-size: 16px;
            cursor: pointer;
            transition: background 0.3s;
            width: 100%;
            margin-top: 10px;
        }
        
        .download-btn:hover {
            background: #0056b3;
        }
        
        .download-btn:active {
            transform: translateY(1px);
        }
        
        @media (max-width: 768px) {
            .results-content {
                max-width: 95%;
                margin: 10px;
            }
            
            .images-grid {
                grid-template-columns: 1fr;
            }
        }
    `;
    
    document.head.appendChild(resultsStyle);
    document.body.appendChild(resultsModal);
    
    // Close modal when clicking outside
    resultsModal.addEventListener('click', (e) => {
        if (e.target === resultsModal) {
            resultsModal.remove();
        }
    });
    
    // Add click event to enlarge images
    resultsModal.addEventListener('click', (e) => {
        if (e.target.classList.contains('analysis-image')) {
            enlargeImage(e.target);
        }
    });
}

// Function to handle download
function downloadResults(zipPath) {
    try {
        // Extract filename from path
        const filename = zipPath.split('/').pop() || zipPath.split('\\').pop();
        
        // Create download URL
        const downloadUrl = `${BACKEND_URL}/download/${filename}`;
        
        // Create a temporary link to trigger download
        const link = document.createElement('a');
        link.href = downloadUrl;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        // Show success message
        const downloadBtn = document.querySelector('.download-btn');
        if (downloadBtn) {
            const originalText = downloadBtn.textContent;
            downloadBtn.textContent = '✅ Download Started!';
            downloadBtn.style.background = '#28a745';
            
            setTimeout(() => {
                downloadBtn.textContent = originalText;
                downloadBtn.style.background = '#007bff';
            }, 2000);
        }
        
    } catch (error) {
        console.error('Download error:', error);
        alert('Failed to download results. Please try again.');
    }
}

async function getRegionsSummary(totalPoints, firstRegionEnd, numRegions) {
    try {
        const summaryData = {
            csv_content: csvContent,
            time_column: timeColumnSelect.value,
            response: dataColumnSelect.value,
            first_region_end_index: selectedRegionEnd.x  // Send x-coordinate (time value) instead of index
        };
        
        const response = await fetch(`${BACKEND_URL}/preview`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(summaryData)
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Preview failed');
        }
        
        const results = await response.json();
        return results;
        
    } catch (error) {
        console.error('Preview error:', error);
        throw error;
    }
}

// Function to get and set default region selections
async function setDefaultRegionSelections() {
    try {
        if (!csvContent || !backendRunning) return;
        
        const defaultData = {
            csv_content: csvContent,
            time_column: timeColumnSelect.value,
            response: dataColumnSelect.value
        };
        
        const response = await fetch(`${BACKEND_URL}/defaults`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(defaultData)
        });
        
        if (!response.ok) {
            console.warn('Failed to get default selections');
            return;
        }
        
        const results = await response.json();
        const defaults = results.defaults;
        
        // Set default selections if none are currently set
        if (!selectedRegionStart && !selectedRegionEnd && !selectedFirstRegionEnd) {
            // Create selection objects with proper structure
            const timeColumn = timeColumnSelect.value;
            const dataColumn = dataColumnSelect.value;
            
            // Find the original indices for the default values
            const data = currentData.data;
            let startIndex = -1;
            let endIndex = -1;
            
            for (let i = 0; i < data.length; i++) {
                const row = data[i];
                let timeValue = row[timeColumn];
                
                // Parse time value the same way as in prepareChartData
                if (typeof timeValue === 'number') {
                    // Already numeric
                } else if (typeof timeValue === 'string') {
                    const numValue = parseFloat(timeValue.toString().trim());
                    if (!isNaN(numValue)) {
                        timeValue = numValue;
                    } else {
                        // Try date parsing
                        const dateValue = new Date(timeValue.toString().trim());
                        if (!isNaN(dateValue.getTime())) {
                            timeValue = dateValue.getTime();
                        } else {
                            timeValue = i; // fallback to index
                        }
                    }
                } else {
                    timeValue = i; // fallback to index
                }
                
                // Check for start time match
                if (Math.abs(timeValue - defaults.region_start_index) < 0.001) {
                    startIndex = i;
                }
                
                // Check for end time match
                if (Math.abs(timeValue - defaults.region_end_index) < 0.001) {
                    endIndex = i;
                }
            }
            
            // Set default selections
            selectedRegionStart = {
                x: defaults.region_start_index,
                y: startIndex >= 0 ? data[startIndex][dataColumn] : 0,
                originalIndex: startIndex
            };
            
            selectedRegionEnd = {
                x: defaults.region_end_index,
                y: endIndex >= 0 ? data[endIndex][dataColumn] : 0,
                originalIndex: endIndex
            };
            
            // Don't set selectedFirstRegionEnd by default - let user choose
            selectedFirstRegionEnd = null;
            
            // Update the UI
            updateRegionSelection();
            updateRegionMarkers();
            checkNextButtonState();
            
            console.log('Default region selections set:', {
                start: defaults.region_start_index,
                end: defaults.region_end_index,
                first_end: 'not set (user choice)'
            });
        }
        
    } catch (error) {
        console.warn('Error setting default selections:', error);
    }
}

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    // Escape key to close modals
    if (e.key === 'Escape') {
        hideModal('error');
        hideModal('success');
    }
    
    // Ctrl/Cmd + O to open file
    if ((e.ctrlKey || e.metaKey) && e.key === 'o') {
        e.preventDefault();
        if (landingPage.classList.contains('active')) {
            handleBrowseClick();
        }
    }
}); 

// Function to save individual images
function saveImage(imageType, imageTitle) {
    try {
        const imageTypeMap = {
            'regions_plot': 'regions',
            'final_plot': 'final',
            'regression_plot': 'regression'
        };
        
        const mappedType = imageTypeMap[imageType];
        if (!mappedType) {
            console.error('Invalid image type:', imageType);
            return;
        }
        
        // Create download URL
        const downloadUrl = `${BACKEND_URL}/download/image/${mappedType}`;
        
        // Create a temporary link to trigger download
        const link = document.createElement('a');
        link.href = downloadUrl;
        link.download = `${mappedType}_plot.png`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        // Show success feedback
        const saveBtn = event.target;
        const originalText = saveBtn.textContent;
        saveBtn.textContent = '✅ Saved!';
        saveBtn.style.background = '#28a745';
        
        setTimeout(() => {
            saveBtn.textContent = originalText;
            saveBtn.style.background = '#28a745';
        }, 2000);
        
    } catch (error) {
        console.error('Error saving image:', error);
        alert('Failed to save image. Please try again.');
    }
}

// Function to enlarge images for better viewing
function enlargeImage(imageElement) {
    // Create overlay for enlarged image
    const overlay = document.createElement('div');
    overlay.className = 'image-overlay';
    overlay.innerHTML = `
        <div class="enlarged-image-container">
            <img src="${imageElement.src}" alt="${imageElement.alt}" class="enlarged-image">
            <button class="close-enlarged" onclick="this.parentElement.parentElement.remove()">×</button>
        </div>
    `;
    
    // Add styles for enlarged image
    const enlargedStyle = document.createElement('style');
    enlargedStyle.textContent = `
        .image-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.9);
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 20000;
            cursor: zoom-out;
        }
        
        .enlarged-image-container {
            position: relative;
            max-width: 95%;
            max-height: 95%;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        
        .enlarged-image {
            max-width: 100%;
            max-height: 100%;
            border-radius: 8px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        }
        
        .close-enlarged {
            position: absolute;
            top: -10px;
            right: -10px;
            background: white;
            border: none;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            font-size: 20px;
            cursor: pointer;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
            z-index: 20001;
        }
        
        .close-enlarged:hover {
            background: #f0f0f0;
        }
    `;
    
    document.head.appendChild(enlargedStyle);
    document.body.appendChild(overlay);
    
    // Close when clicking outside the image
    overlay.addEventListener('click', (e) => {
        if (e.target === overlay) {
            overlay.remove();
            enlargedStyle.remove();
        }
    });
    
    // Close with Escape key
    const handleKeydown = (e) => {
        if (e.key === 'Escape') {
            overlay.remove();
            enlargedStyle.remove();
            document.removeEventListener('keydown', handleKeydown);
        }
    };
    document.addEventListener('keydown', handleKeydown);
} 