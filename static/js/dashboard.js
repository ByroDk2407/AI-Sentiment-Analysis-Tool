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
    
    // Fetch predictions separately
    fetch('/api/predict')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                updatePredictions(data);  // Make sure we're using the right function name
            } else {
                console.error('Prediction error:', data.error);
            }
        })
        .catch(error => {
            console.error('Error fetching predictions:', error);
        });
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

function updatePredictions(data) {
    console.log('Prediction data received:', data);  // Debug log
    
    if (!data.predictions) {
        console.error('No prediction data available');
        return;
    }
    
    const trace1 = {
        x: data.predictions.dates,
        y: data.predictions.actual_sentiment_scores,
        name: 'Actual Sentiment',
        type: 'scatter',
        mode: 'lines',
        line: {color: '#17a2b8'}
    };

    const trace2 = {
        x: data.predictions.dates,
        y: data.predictions.confidence_scores,
        name: 'Prediction Confidence',
        type: 'scatter',
        mode: 'lines',
        line: {
            color: '#007bff',
            dash: 'dot'
        }
    };

    // Add markers for predicted sentiment
    const colors = {
        'positive': '#28a745',
        'neutral': '#ffc107',
        'negative': '#dc3545'
    };

    const trace3 = {
        x: data.predictions.dates,
        y: data.predictions.confidence_scores,
        text: data.predictions.sentiments,
        mode: 'markers',
        marker: {
            size: 10,
            color: data.predictions.sentiments.map(s => colors[s])
        },
        name: 'Predicted Sentiment',
        hovertemplate: '%{text}<br>Confidence: %{y:.2f}<extra></extra>'
    };

    const layout = {
        title: 'Market Sentiment Predictions',
        yaxis: {
            title: 'Sentiment Score / Confidence',
            range: [-1, 1]
        },
        xaxis: {title: 'Date'},
        showlegend: true,
        height: 400,
        margin: {t: 40}
    };

    Plotly.newPlot('predictionChart', [trace1, trace2, trace3], layout);

    // Update market reports
    if (data.market_report) {
        updateMarketReport('currentMarketReport', data.market_report.current);
        updateMarketReport('futureMarketReport', data.market_report.future);
    }
}

function updateMarketReport(elementId, report) {
    if (!report) return;
    
    const container = document.getElementById(elementId);
    const indicator = container.querySelector('.sentiment-indicator');
    const reportText = container.querySelector('.report-text');
    
    // Update sentiment indicator
    indicator.className = 'sentiment-indicator';
    indicator.classList.add(`sentiment-${report.sentiment}`);
    
    // Update report text
    reportText.innerHTML = report.analysis;
    
    // Add confidence indicator
    const confidenceBar = document.createElement('div');
    confidenceBar.className = 'progress mt-2';
    confidenceBar.innerHTML = `
        <div class="progress-bar" role="progressbar" 
             style="width: ${report.confidence * 100}%"
             aria-valuenow="${report.confidence * 100}" 
             aria-valuemin="0" aria-valuemax="100">
            Confidence: ${(report.confidence * 100).toFixed(1)}%
        </div>
    `;
    
    // Replace old confidence bar if exists
    const oldBar = container.querySelector('.progress');
    if (oldBar) oldBar.remove();
    container.appendChild(confidenceBar);
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

function sendMessage() {
    const input = document.getElementById('chatInput');
    const message = input.value.trim();
    
    if (!message) return;
    
    // Add user message to chat
    addChatMessage(message, 'user');
    input.value = '';
    
    // Create message container for bot response
    const botMessageId = `msg-${Date.now()}`;
    const botMessage = document.createElement('div');
    botMessage.id = botMessageId;
    botMessage.className = 'chat-message bot-message';
    botMessage.textContent = 'Thinking...';
    
    const chatMessages = document.getElementById('chatMessages');
    chatMessages.appendChild(botMessage);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    // Send the message
    fetch('/api/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ message: message })
    })
    .then(response => {
        console.log('Response status:', response.status);
        return response.json();
    })
    .then(data => {
        console.log('Parsed response data:', data);
        if (data.success) {
            botMessage.textContent = data.response;
        } else {
            botMessage.textContent = `Error: ${data.error}${data.details ? '\n' + data.details : ''}`;
        }
        chatMessages.scrollTop = chatMessages.scrollHeight;
    })
    .catch(error => {
        console.error('Chat error:', error);
        botMessage.textContent = "Sorry, I encountered an error. Please try again.";
    });
}

function addChatMessage(message, type) {
    const chatMessages = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    const messageId = `msg-${Date.now()}`;
    messageDiv.id = messageId;
    messageDiv.className = `chat-message ${type}-message`;
    messageDiv.textContent = message;
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return messageId;
}

// Add event listener for Enter key
document.getElementById('chatInput').addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        sendMessage();
    }
}); 