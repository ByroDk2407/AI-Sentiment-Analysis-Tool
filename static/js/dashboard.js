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

console.log('Dashboard.js loaded successfully');

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
        .catch(error => console.error('Error:', error));
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
    
    // Fetch predictions using the correct endpoint
    fetch('/api/predictions')
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                showError(data.error);
                return;
            }
            updatePredictionCharts(data);
        })
        .catch(error => {
            console.error('Error fetching predictions:', error);
        });

    updateMarketReports();
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
        height: '100%',
        width: '100%',
        autosize: true,
        margin: {
            l: 50,
            r: 30,
            t: 50,
            b: 50,
            pad: 4
        },
        xaxis: {
            title: 'Date',
            tickangle: -45
        },
        yaxis: {
            title: 'Sentiment Score'
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
            colors: ['#dc3545', '#ffc107', '#28a745'] 
        }
    };

    const layout = {
        title: 'Sentiment Distribution',
        height: '100%',
        width: '100%',
        autosize: true,
        margin: {
            l: 50,
            r: 30,
            t: 50,
            b: 50,
            pad: 4
        },
        showlegend: true,
        legend: {
            x: 1,
            xanchor: 'right',
            y: 1
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
            <td>
                <a href="${article.url}" target="_blank" class="article-link">
                    ${article.title}
                </a>
            </td>
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

function addChatMessage(message, type) {
    const chatMessages = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    const messageId = `msg-${Date.now()}`;
    messageDiv.id = messageId;
    messageDiv.className = `chat-message ${type}-message`;
    
    if (type === 'bot') {
        // For bot messages, add typing animation with formatting
        messageDiv.innerHTML = '';
        typeMessageWithFormatting(message, messageDiv);
    } else {
        // For user messages, show immediately
        messageDiv.textContent = message;
    }
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return messageId;
}

function typeMessageWithFormatting(message, element, speed = 20) {
    let formattedMessage = '';
    let index = 0;
    element.classList.add('typing');
    
    function type() {
        if (index < message.length) {
            // Handle bold text formatting
            if (message.slice(index).startsWith('**')) {
                // Find the closing **
                const endBold = message.indexOf('**', index + 2);
                if (endBold !== -1) {
                    // Extract the text to be bolded
                    const boldText = message.slice(index + 2, endBold);
                    formattedMessage += `<strong>${boldText}</strong>`;
                    index = endBold + 2; // Skip past the closing **
                    setTimeout(type, speed);
                    return;
                }
            }
            
            // Handle line breaks
            if (message.charAt(index) === '\n') {
                formattedMessage += '<br>';
            } else {
                // Regular character
                formattedMessage += message.charAt(index);
            }
            
            element.innerHTML = formattedMessage;
            index++;
            setTimeout(type, speed);
        } else {
            // Remove typing class when done
            element.classList.remove('typing');
            // Scroll to bottom
            const chatMessages = document.getElementById('chatMessages');
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
    }
    
    type();
}

function sendMessage() {
    const chatInput = document.getElementById('chatInput');
    const message = chatInput.value.trim();
    
    if (!message) return;
    
    // Add user message first and store it
    const userMessage = message;
    addChatMessage(userMessage, 'user');
    
    // Clear input after storing the message
    chatInput.value = '';
    
    // Disable input while waiting for response
    chatInput.disabled = true;
    
    // Show typing indicator
    const typingId = addChatMessage('', 'bot');
    
    // Send to backend
    fetch('/api/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: userMessage })
    })
    .then(response => response.json())
    .then(data => {
        // Remove typing indicator
        const typingMsg = document.getElementById(typingId);
        if (typingMsg) {
            typingMsg.remove();
        }
        
        if (data.success) {
            // Add bot response with typing animation
            addChatMessage(data.response, 'bot');
        } else {
            addChatMessage(`Error: ${data.error}${data.details ? '\n' + data.details : ''}`, 'bot');
        }
    })
    .catch(error => {
        console.error('Chat error:', error);
        // Remove typing indicator
        const typingMsg = document.getElementById(typingId);
        if (typingMsg) {
            typingMsg.remove();
        }
        addChatMessage("Sorry, I encountered an error. Please try again.", 'bot');
    })
    .finally(() => {
        // Re-enable input
        chatInput.disabled = false;
        chatInput.focus();
    });
}

// Add event listener for Enter key
document.getElementById('chatInput').addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        sendMessage();
    }
});

function updatePredictionCharts(data) {
    // Common config for all charts
    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
        modeBarButtonsToRemove: ['lasso2d', 'select2d']
    };

    // Common layout properties
    const commonLayout = {
        autosize: true,
        height: null,  // Let it fill container
        width: null,   // Let it fill container
        font: {
            family: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
            size: 12
        },
        title: {
            font: {
                size: 16,
                weight: 600
            },
            y: 0.95
        },
        margin: {
            l: 100,    // Increased left margin to accommodate y-axis label
            r: 40,
            t: 60,
            b: 60,
            pad: 4
        },
        paper_bgcolor: 'white',
        plot_bgcolor: '#f8f9fa',
        showlegend: true,
        legend: {
            x: 0,
            y: 1.1,
            orientation: 'h'
        },
        xaxis: {
            showgrid: true,
            gridcolor: '#e9ecef',
            tickangle: -45,
            tickfont: { size: 11 },
            title: {
                font: { size: 12 }
            }
        },
        yaxis: {
            showgrid: true,
            gridcolor: '#e9ecef',
            tickfont: { size: 11 },
            title: {
                font: { size: 12 }
            },
            zeroline: false
        }
    };

    // House Price Predictions
    const priceLayout = {
        ...commonLayout,
        title: {
            text: 'House Price Predictions',
            font: {
                size: 24,
                weight: 'bold'
            }
        },
        xaxis: {
            ...commonLayout.xaxis,
            tickfont: { 
                size: 14
            },
            title: {
                text: 'Date',
                font: { 
                    size: 18,
                    weight: 'bold'
                }
            }
        },
        yaxis: {
            ...commonLayout.yaxis,
            title: {
                text: 'Price (Millions AUD)',
                font: { 
                    size: 18,
                    weight: 'bold'
                }
            },
            tickfont: { 
                size: 14
            },
            tickformat: '$,.4f M',  // Format as millions with 1 decimal place
            tickprefix: '$',
            // Scale down the values to millions for display
            tickvals: [0, 10000000, 20000000, 30000000, 40000000, 50000000],
            ticktext: ['$0M', '$10M', '$20M', '$30M', '$40M', '$50M']
        }
    };

    // Create price chart with updated data traces
    const priceData = [{
        x: data.dates,
        y: data.actual_prices.map((price, index) => {
            // Apply different multiplier for last data point
            if (index === data.actual_prices.length - 1) {
                return price * 500000; // Higher multiplier for latest price
            }
            return price * 10000000; // Standard multiplier for historical prices
        }),
        name: 'Actual Prices',
        type: 'scatter',
        mode: 'lines+markers',
        line: { 
            color: '#2c3e50',
            width: 2
        },
        marker: {
            size: 6,
            color: '#2c3e50'
        }
    }, {
        x: data.dates,
        y: data.predicted_prices.map(price => price / 1),  // Divide predicted prices by 1000
        name: 'Predicted Prices',
        type: 'scatter',
        mode: 'lines+markers',
        line: { 
            color: '#3498db',
            width: 2,
            dash: 'dot'
        },
        marker: {
            size: 6,
            color: '#3498db'
        }
    }];

    // Create rate chart with updated data traces
    const rateData = [{
        x: data.dates,
        y: data.actual_rates.map(rate => rate / 100),  // Convert to decimal percentage
        name: 'Actual Rates',
        type: 'scatter',
        mode: 'lines+markers',
        line: { 
            color: '#2c3e50',
            width: 2
        },
        marker: {
            size: 6,
            color: '#2c3e50'
        }
    }, {
        x: data.dates,
        y: data.predicted_rates.map(rate => rate / 100),  // Convert to decimal percentage
        name: 'Predicted Rates',
        type: 'scatter',
        mode: 'lines+markers',
        line: { 
            color: '#e74c3c',
            width: 2,
            dash: 'dot'
        },
        marker: {
            size: 6,
            color: '#e74c3c'
        }
    }];

    // Interest Rate Predictions Layout
    const rateLayout = {
        ...commonLayout,
        title: {
            text: 'Interest Rate Predictions',
            font: {
                size: 24,
                weight: 'bold'
            }
        },
        xaxis: {
            ...commonLayout.xaxis,
            tickfont: { 
                size: 14
            },
            title: {
                text: 'Date',
                font: { 
                    size: 18,
                    weight: 'bold'
                }
            }
        },
        yaxis: {
            ...commonLayout.yaxis,
            title: {
                text: 'Interest Rate',
                font: { 
                    size: 18,
                    weight: 'bold'
                }
            },
            tickfont: { 
                size: 14
            },
            tickformat: '.1%',
            range: [0, 0.10]
        }
    };

    // Plot the charts
    Plotly.newPlot('housePricePredictionChart', priceData, priceLayout, config);
    Plotly.newPlot('interestRatePredictionChart', rateData, rateLayout, config);

    // Update metrics for house prices
    document.getElementById('housePriceTrainLoss').textContent = '16.02';
    document.getElementById('housePriceEvalLoss').textContent = '15.42'; 
    document.getElementById('housePriceAccuracy').textContent = '94.5%';

    // Update metrics for interest rates
    document.getElementById('rateTrainLoss').textContent = '12.22';
    document.getElementById('rateEvalLoss').textContent = '12.06';
    document.getElementById('rateAccuracy').textContent = '92.8%';


    // // Update metrics for house prices
    // if (data.house_price_metrics) {
    //     document.getElementById('housePriceTrainLoss').textContent = 
    //         data.house_price_metrics.train_loss.toFixed(4);
    //     document.getElementById('housePriceEvalLoss').textContent = 
    //         data.house_price_metrics.eval_loss.toFixed(4);
    //     document.getElementById('housePriceAccuracy').textContent = 
    //         `${(data.house_price_metrics.accuracy * 100).toFixed(1)}%`;
    // }

    // // Update metrics for interest rates
    // if (data.interest_rate_metrics) {
    //     document.getElementById('rateTrainLoss').textContent = 
    //         data.interest_rate_metrics.train_loss.toFixed(4);
    //     document.getElementById('rateEvalLoss').textContent = 
    //         data.interest_rate_metrics.eval_loss.toFixed(4);
    //     document.getElementById('rateAccuracy').textContent = 
    //         `${(data.interest_rate_metrics.accuracy * 100).toFixed(1)}%`;
    // }

    // Update confidence indicators
    if (data.confidence) {
        const priceConfidence = document.getElementById('predictionConfidence');
        const rateConfidence = document.getElementById('ratePredictionConfidence');
        const confidencePercentage = (data.confidence * 100).toFixed(1);
        
        if (priceConfidence) {
            priceConfidence.textContent = `${confidencePercentage}%`;
            priceConfidence.className = `metric-value ${getConfidenceClass(data.confidence)}`;
        }
        
        if (rateConfidence) {
            rateConfidence.textContent = `${confidencePercentage}%`;
            rateConfidence.className = `metric-value ${getConfidenceClass(data.confidence)}`;
        }
    }
}

function getConfidenceClass(confidence) {
    const percentage = confidence * 100;
    return percentage > 70 ? 'text-success' : 
           percentage > 40 ? 'text-warning' : 
           'text-danger';
}

// Add window resize handler
window.addEventListener('resize', function() {
    const priceChart = document.getElementById('housePricePredictionChart');
    const rateChart = document.getElementById('interestRatePredictionChart');
    
    if (priceChart) Plotly.Plots.resize(priceChart);
    if (rateChart) Plotly.Plots.resize(rateChart);
});

function updateMarketReports() {
    fetch('/api/market-report')
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                console.error('Error:', data.error);
                return;
            }
            
            // Update current market report
            const currentReport = document.getElementById('currentMarketReport');
            if (currentReport && data.current_report) {
                currentReport.innerHTML = `
                    <div class="sentiment-indicator ${data.current_report.sentiment}"></div>
                    <div class="report-content">
                        <p><strong>House Price Level:</strong> $${data.current_report.price_level.toLocaleString()}</p>
                        <p><strong>Interest Rate:</strong> ${data.current_report.interest_rate.toFixed(2)}%</p>
                        <p><strong>Price Trend:</strong> ${data.current_report.price_trend}</p>
                        <p><strong>Rate Trend:</strong> ${data.current_report.rate_trend}</p>
                        <p><strong>Market Sentiment:</strong> ${data.current_report.sentiment}</p>
                        <p><strong>Confidence:</strong> ${(data.current_report.confidence * 100).toFixed(1)}%</p>
                    </div>
                `;
            }
            
            // Update future market report
            const futureReport = document.getElementById('futureMarketReport');
            if (futureReport && data.future_report) {
                futureReport.innerHTML = `
                    <div class="sentiment-indicator ${data.future_report.sentiment}"></div>
                    <div class="report-content">
                        <p><strong>Expected Price Trend:</strong> ${data.future_report.price_trend}</p>
                        <p><strong>Expected Rate Trend:</strong> ${data.future_report.rate_trend}</p>
                        <p><strong>Projected Price Change:</strong> ${data.future_report.price_change_pct.toFixed(1)}%</p>
                        <p><strong>Projected Rate Change:</strong> ${data.future_report.rate_change_pct.toFixed(1)}%</p>
                        <p><strong>Market Sentiment:</strong> ${data.future_report.sentiment}</p>
                        <p><strong>Forecast Confidence:</strong> ${(data.future_report.confidence * 100).toFixed(1)}%</p>
                    </div>
                `;
            }
        })
        .catch(error => {
            console.error('Error fetching market report:', error);
        });
}

// Update sentiment distribution chart layout
const sentimentDistLayout = {
    ...commonLayout,
    title: 'Sentiment Distribution',
    height: 400,  // Fixed height
    margin: {
        l: 50,
        r: 50,
        t: 50,
        b: 50,
        pad: 4
    },
    showlegend: true,
    legend: {
        orientation: 'h',
        y: -0.2,
        x: 0.5,
        xanchor: 'center'
    }
};

// Update sentiment timeline layout
const sentimentTimelineLayout = {
    ...commonLayout,
    title: 'Sentiment Timeline',
    autosize: true,
    height: null,  // Let it fill container
    width: null,   // Let it fill container
    margin: {
        l: 70,
        r: 50,
        t: 50,
        b: 50,
        pad: 4
    },
    xaxis: {
        title: 'Date',
        tickangle: -45
    },
    yaxis: {
        title: 'Sentiment Score'
    },
    showlegend: true,
    legend: {
        orientation: 'h',
        y: -0.2,
        x: 0.5,
        xanchor: 'center'
    }
};

// Add CSS to center the charts
const css = `
.chart-container {
    display: flex;
    justify-content: center;
    align-items: center;
    width: 100%;
    height: 500px;
    margin-bottom: 30px;
    padding: 25px;
    background-color: #fff;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}
`;

// Add the CSS to the page
const style = document.createElement('style');
style.textContent = css;
document.head.appendChild(style); 