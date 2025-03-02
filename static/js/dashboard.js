// Global variables for charts
let sentimentTimelineChart = null;
let sentimentDistributionChart = null;
let pricePredictionChart = null;

// Dashboard state
let currentPeriod = 30;
let charts = {};

// Initialize dashboard
document.addEventListener('DOMContentLoaded', function() {
    refreshData();
});

function refreshData() {
    const days = document.getElementById('timeRange').value;
    fetch(`/api/data?days=${days}`)
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                showError(data.error);
                return;
            }
            updateDashboard(data);
        })
        .catch(error => {
            console.error('Error refreshing data:', error);
            showError('Failed to fetch dashboard data');
        });
}

function updateDashboard(data) {
    if (!data) {
        showError('No data available');
        return;
    }
    
    updateSummaryStats(data);
    updateSentimentTimeline(data);
    updateSentimentDistribution(data);
    updateRecentArticles(data);
    
    // Add LSTM predictions
    if (!data.lstm_data) {
        fetch('/api/predict')
            .then(response => response.json())
            .then(predictions => {
                if (predictions.success) {
                    updatePricePredictions(predictions);
                }
            })
            .catch(error => {
                console.error('Error fetching predictions:', error);
            });
    } else {
        updatePricePredictions(data);
    }
}

function updateSummaryStats(data) {
    if (!data.total_articles) return;
    
    const summaryHtml = `
        <p><strong>Total Articles:</strong> ${data.total_articles}</p>
        <p><strong>Sentiment Distribution:</strong></p>
        <ul class="list-unstyled">
            ${Object.entries(data.sentiment_distribution || {}).map(([key, value]) => 
                `<li>${key.charAt(0).toUpperCase() + key.slice(1)}: ${value}</li>`
            ).join('')}
        </ul>
        <p><strong>Date Range:</strong></p>
        <p>${new Date(data.date_range.start).toLocaleDateString()} - 
           ${new Date(data.date_range.end).toLocaleDateString()}</p>
    `;
    document.getElementById('summaryStats').innerHTML = summaryHtml;
}

function updateSentimentTimeline(data) {
    if (!data.timeline_data) return;
    
    const trace1 = {
        x: data.timeline_data.dates,
        y: data.timeline_data.positive,
        name: 'Positive',
        type: 'scatter',
        mode: 'lines+markers',
        stackgroup: 'one',
        line: {color: '#28a745'}
    };

    const trace2 = {
        x: data.timeline_data.dates,
        y: data.timeline_data.negative,
        name: 'Negative',
        type: 'scatter',
        mode: 'lines+markers',
        stackgroup: 'one',
        line: {color: '#dc3545'}
    };

    const trace3 = {
        x: data.timeline_data.dates,
        y: data.timeline_data.neutral,
        name: 'Neutral',
        type: 'scatter',
        mode: 'lines+markers',
        stackgroup: 'one',
        line: {color: '#ffc107'}
    };

    const layout = {
        title: 'Sentiment Timeline',
        yaxis: {title: 'Number of Articles'},
        hovermode: 'closest',
        showlegend: true,
        legend: {
            orientation: 'h',
            y: -0.2
        }
    };

    Plotly.newPlot('sentimentTimelineChart', [trace1, trace2, trace3], layout);
}

function updateSentimentDistribution(data) {
    if (!data.sentiment_distribution) return;
    
    const values = Object.values(data.sentiment_distribution);
    const labels = Object.keys(data.sentiment_distribution).map(
        label => label.charAt(0).toUpperCase() + label.slice(1)
    );

    const trace = {
        values: values,
        labels: labels,
        type: 'pie',
        marker: {
            colors: ['#28a745', '#dc3545', '#ffc107']
        }
    };

    const layout = {
        title: 'Sentiment Distribution',
        showlegend: true,
        legend: {
            orientation: 'h',
            y: -0.2
        }
    };

    Plotly.newPlot('sentimentDistributionChart', [trace], layout);
}

function updatePricePredictions(data) {
    if (!data.lstm_data) return;
    
    const trace1 = {
        x: data.lstm_data.dates,
        y: data.lstm_data.actual_prices,
        name: 'Actual Prices',
        type: 'scatter',
        mode: 'lines',
        line: {color: '#17a2b8'}
    };

    const trace2 = {
        x: data.lstm_data.dates,
        y: data.lstm_data.predicted_prices,
        name: 'Predicted Prices',
        type: 'scatter',
        mode: 'lines',
        line: {
            color: '#007bff',
            dash: 'dot'
        }
    };

    const layout = {
        title: 'Real Estate Price Predictions',
        yaxis: {
            title: 'Price Index',
            tickformat: ',.0f'
        },
        xaxis: {title: 'Date'},
        hovermode: 'closest',
        showlegend: true,
        legend: {
            orientation: 'h',
            y: -0.2
        }
    };

    Plotly.newPlot('pricePredictionChart', [trace1, trace2], layout);
}

function updateRecentArticles(data) {
    const tbody = document.getElementById('articlesTableBody');
    if (!tbody || !data.articles) return;
    
    tbody.innerHTML = ''; // Clear existing content
    
    data.articles?.slice(0, 10).forEach(article => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${new Date(article.date_of_article).toLocaleDateString()}</td>
            <td>${article.title}</td>
            <td>${article.sentiment}</td>
            <td>${article.sentiment_score.toFixed(2)}</td>
        `;
        tbody.appendChild(row);
    });
}

function showError(message) {
    console.error('Dashboard error:', message);
    const alertHtml = `
        <div class="alert alert-danger alert-dismissible fade show" role="alert">
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        </div>
    `;
    document.querySelector('.container-fluid').insertAdjacentHTML('afterbegin', alertHtml);
} 