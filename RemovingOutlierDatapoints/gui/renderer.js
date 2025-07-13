const { ipcRenderer } = require('electron');

// Application State
let currentData = null;
let currentChart = null;
let selectedRegionEnd = null;
let regionCount = null;
let chartClickEnabled = false;
let csvContent = null;

// Backend configuration
const BACKEND_URL = 'http://127.0.0.1:5000';
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
const firstRegionEnd = document.getElementById('first-region-end');
const clearSelectionBtn = document.getElementById('clear-selection');
const regionCountInput = document.getElementById('region-count');
const nextBtn = document.getElementById('next-btn');
const regionInfo = document.getElementById('region-info');
const selectedPointInfo = document.getElementById('selected-point-info');
const totalRegionsInfo = document.getElementById('total-regions-info');
const pointsPerRegionInfo = document.getElementById('points-per-region');

// Modal elements
const errorModal = document.getElementById('error-modal');
const successModal = document.getElementById('success-modal');
const errorMessage = document.getElementById('error-message');
const errorClose = document.getElementById('error-close');
const successClose = document.getElementById('success-close');

// File info elements
const fileName = document.getElementById('file-name');
const fileStats = document.getElementById('file-stats');
const successFilename = document.getElementById('success-filename');
const successRows = document.getElementById('success-rows');
const successColumns = document.getElementById('success-columns');

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
    clearSelectionBtn.addEventListener('click', clearRegionSelection);
    regionCountInput.addEventListener('input', handleRegionCountChange);
    nextBtn.addEventListener('click', handleNextClick);
    
    // Modal close buttons
    errorClose.addEventListener('click', () => hideModal('error'));
    successClose.addEventListener('click', () => hideModal('success'));
    
    // Click outside modal to close
    errorModal.addEventListener('click', (e) => {
        if (e.target === errorModal) hideModal('error');
    });
    successModal.addEventListener('click', (e) => {
        if (e.target === successModal) hideModal('success');
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
function generatePlot() {
    if (!currentData) return;
    
    const timeColumn = timeColumnSelect.value;
    const dataColumn = dataColumnSelect.value;
    
    if (!timeColumn || !dataColumn) {
        showError('Please select both time and data columns');
        return;
    }
    
    try {
        showLoading();
        
        // Prepare data for Chart.js
        const chartData = prepareChartData(timeColumn, dataColumn);
        
        // Destroy existing chart if it exists
        if (currentChart) {
            currentChart.destroy();
        }
        
        // Determine if we're using time-based data
        const isTimeBased = chartData.datasets[0].data.some(point => 
            point.x > 1000000000 // Likely a timestamp
        );
        
        // Create new chart
        const ctx = chartCanvas.getContext('2d');
        currentChart = new Chart(ctx, {
            type: 'line',
            data: chartData,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    intersect: false,
                    mode: 'index'
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
                                    return new Date(point.parsed.x).toLocaleString();
                                }
                                return `${timeColumn}: ${point.parsed.x}`;
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
        enableChartSelection();
        
        hideLoading();
        
    } catch (error) {
        hideLoading();
        showError('Error generating plot: ' + error.message);
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
        regionCountInput.value = '';
        regionCount = null;
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
                    closestPoint = { ...point, index };
                }
            });
            
            if (closestPoint) {
                selectedRegionEnd = closestPoint;
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
    if (selectedRegionEnd) {
        const timeColumn = timeColumnSelect.value;
        const dataColumn = dataColumnSelect.value;
        
        // Format the display value
        let displayValue;
        if (selectedRegionEnd.x > 1000000000) {
            // Timestamp
            displayValue = new Date(selectedRegionEnd.x).toLocaleString();
        } else {
            displayValue = `${timeColumn}: ${selectedRegionEnd.x.toLocaleString()}`;
        }
        
        firstRegionEnd.textContent = displayValue;
        firstRegionEnd.classList.add('selected');
        clearSelectionBtn.style.display = 'inline-block';
        
        updateRegionInfo();
    }
}

function clearRegionSelection() {
    selectedRegionEnd = null;
    firstRegionEnd.textContent = 'Click on chart to select';
    firstRegionEnd.classList.remove('selected');
    clearSelectionBtn.style.display = 'none';
    
    // Remove marker from chart
    removeRegionMarker();
    checkNextButtonState();
    updateRegionInfo();
}

function handleRegionCountChange() {
    const value = parseInt(regionCountInput.value);
    regionCount = (value >= 2 && value <= 20) ? value : null;
    
    checkNextButtonState();
    updateRegionInfo();
}

function checkNextButtonState() {
    const hasSelection = selectedRegionEnd !== null;
    const hasValidCount = regionCount !== null;
    
    nextBtn.disabled = !(hasSelection && hasValidCount);
}

function updateRegionInfo() {
    if (selectedRegionEnd && regionCount) {
        // Calculate points per region
        const totalPoints = currentData.rowCount;
        const firstRegionPoints = selectedRegionEnd.index + 1;
        const remainingPoints = totalPoints - firstRegionPoints;
        const pointsPerRegion = Math.floor(remainingPoints / (regionCount - 1));
        
        // Update info display
        selectedPointInfo.textContent = selectedRegionEnd.x > 1000000000 
            ? new Date(selectedRegionEnd.x).toLocaleString()
            : selectedRegionEnd.x.toLocaleString();
        totalRegionsInfo.textContent = regionCount.toString();
        pointsPerRegionInfo.textContent = `First: ${firstRegionPoints}, Others: ~${pointsPerRegion}`;
        
        regionInfo.classList.remove('hidden');
    } else {
        regionInfo.classList.add('hidden');
    }
}

function addRegionMarker(xValue) {
    // Remove existing marker
    removeRegionMarker();
    
    if (!currentChart) return;
    
    // Add vertical line plugin
    const verticalLinePlugin = {
        id: 'verticalLine',
        beforeDraw: function(chart, args, options) {
            const ctx = chart.ctx;
            const xAxis = chart.scales.x;
            const yAxis = chart.scales.y;
            
            const x = xAxis.getPixelForValue(xValue);
            
            ctx.save();
            ctx.beginPath();
            ctx.moveTo(x, yAxis.top);
            ctx.lineTo(x, yAxis.bottom);
            ctx.lineWidth = 3;
            ctx.strokeStyle = '#ff6b6b';
            ctx.setLineDash([5, 5]);
            ctx.stroke();
            ctx.restore();
            
            // Add label
            ctx.save();
            ctx.fillStyle = '#ff6b6b';
            ctx.font = 'bold 12px Arial';
            ctx.textAlign = 'center';
            ctx.fillText('End of Region 1', x, yAxis.top - 5);
            ctx.restore();
        }
    };
    
    // Register and update chart
    Chart.register(verticalLinePlugin);
    currentChart.update();
}

function removeRegionMarker() {
    if (currentChart) {
        // Unregister the plugin and update
        Chart.unregister('verticalLine');
        currentChart.update();
    }
}

async function handleNextClick() {
    if (!selectedRegionEnd || !regionCount) {
        showError('Please select the end of the first region and enter the number of regions');
        return;
    }
    
    if (!backendRunning) {
        showError('Python backend is not running. Please start the backend server first.');
        return;
    }
    
    try {
        showLoading();
        
        // Prepare data for backend
        const analysisData = {
            csv_content: csvContent,
            time_column: timeColumnSelect.value,
            response: dataColumnSelect.value,
            first_region_end_index: selectedRegionEnd.index,
            num_regions: regionCount
        };
        
        console.log('Sending data to backend:', {
            time_column: analysisData.time_column,
            response: analysisData.response,
            first_region_end_index: analysisData.first_region_end_index,
            num_regions: analysisData.num_regions
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
    message += `🔢 Created ${results.metadata.num_regions} regions\n\n`;
    
    // Add region summaries
    results.analysis.forEach(region => {
        message += `Region ${region.region_id}:\n`;
        message += `  • Size: ${region.size} points\n`;
        message += `  • Mean: ${region.statistics.mean.toFixed(2)}\n`;
        message += `  • Std Dev: ${region.statistics.std.toFixed(2)}\n`;
        message += `  • Range: ${region.statistics.min.toFixed(2)} - ${region.statistics.max.toFixed(2)}\n\n`;
    });
    
    // Show results in a modal (for now)
    alert(message);
    
    // Here you could create a more sophisticated results display
    // with charts, tables, export options, etc.
}

async function getRegionsSummary(totalPoints, firstRegionEnd, numRegions) {
    try {
        const response = await fetch(`${BACKEND_URL}/regions-summary`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                total_data_points: totalPoints,
                first_region_end_index: firstRegionEnd,
                num_regions: numRegions
            })
        });
        
        if (response.ok) {
            return await response.json();
        }
    } catch (error) {
        console.warn('Failed to get regions summary:', error);
    }
    return null;
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