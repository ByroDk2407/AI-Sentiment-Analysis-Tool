// Dashboard state
let currentPeriod = 30;
let charts = {};

// Initialize dashboard
$(document).ready(function() {
    // Sidebar toggle
    $('#sidebarCollapse').on('click', function() {
        $('#sidebar').toggleClass('active');
    });

    // Time period buttons
    $('.btn-group .btn').on('click', function() {
        // Update active state
        $('.btn-group .btn').removeClass('active');
        $(this).addClass('active');
        
        // Update period and refresh data
        currentPeriod = parseInt($(this).data('period'));
        updateDashboard();
    });

    // Initial load
    updateDashboard();
    
    // Update every 5 minutes
    setInterval(updateDashboard, 300000);
});

// Update all dashboard components
function updateDashboard() {
    $.get(`/api/data?days=${currentPeriod}`)
        .done(function(data) {
            if (data.error) {
                showError(data.error);
                return;
            }
            updateCharts(data);
            updateStats(data);
            updateDateRange(data.date_range);
        })
        .fail(function(jqXHR, textStatus, errorThrown) {
            showError(`Failed to fetch data: ${errorThrown}`);
        });
}

// Update date range display
function updateDateRange(dateRange) {
    if (dateRange.start && dateRange.end) {
        const start = new Date(dateRange.start).toLocaleDateString();
        const end = new Date(dateRange.end).toLocaleDateString();
        $('#dateRange').html(`
            <i class="fas fa-calendar"></i>
            <strong>Date Range:</strong> ${start} - ${end}
        `);
    }
}

// Update statistics cards
function updateStats(data) {
    $('#totalArticles').text(data.total_articles || 0);
    
    const sentiment = data.overall_sentiment || 'Unknown';
    $('#overallSentiment').text(sentiment.charAt(0).toUpperCase() + sentiment.slice(1));
    
    // Update sentiment icon and color
    const sentimentIcon = $('#overallSentiment').closest('.stat-card').find('i');
    sentimentIcon.removeClass().addClass('fas');
    
    switch(sentiment.toLowerCase()) {
        case 'positive':
            sentimentIcon.addClass('fa-smile text-success');
            break;
        case 'negative':
            sentimentIcon.addClass('fa-frown text-danger');
            break;
        default:
            sentimentIcon.addClass('fa-meh text-warning');
    }
}

// Update charts with time period information
function updateCharts(data) {
    // Update sentiment pie chart
    updateSentimentPie(data.sentiment_distribution);
    
    // Update source distribution
    updateSourceBar(data.source_distribution);
    
    // Update timeline with proper date range
    updateSentimentTimeline(data.timeline_data);
    
    // Update summary stats
    $('#totalArticles').text(data.total_articles);
    $('#timeRange').text(currentPeriod + ' days');
    
    // Update date range display
    if (data.date_range.start && data.date_range.end) {
        const start = new Date(data.date_range.start).toLocaleDateString();
        const end = new Date(data.date_range.end).toLocaleDateString();
        $('#dateRange').text(`${start} - ${end}`);
    }
}

// Sentiment Distribution Pie Chart
function updateSentimentPie(distribution) {
    const colors = {
        positive: '#2ecc71',
        negative: '#e74c3c',
        neutral: '#95a5a6'
    };

    const config = {
        data: [{
            values: Object.values(distribution),
            labels: Object.keys(distribution),
            type: 'pie',
            hole: 0.4,
            marker: {
                colors: Object.keys(distribution).map(key => colors[key.toLowerCase()])
            }
        }],
        layout: {
            showlegend: true,
            legend: { orientation: 'h' },
            margin: { t: 10, l: 0, r: 0, b: 0 },
            height: 300
        }
    };

    Plotly.newPlot('sentimentPie', config.data, config.layout);
}

// Sources Distribution Bar Chart
function updateSourceBar(distribution) {
    const config = {
        data: [{
            x: Object.keys(distribution),
            y: Object.values(distribution),
            type: 'bar',
            marker: {
                color: '#3498db'
            }
        }],
        layout: {
            margin: { t: 10, l: 40, r: 10, b: 60 },
            height: 300,
            xaxis: {
                tickangle: -45
            }
        }
    };

    Plotly.newPlot('sourceBar', config.data, config.layout);
}

// Sentiment Timeline
function updateSentimentTimeline(timelineData) {
    const colors = {
        positive: '#2ecc71',
        negative: '#e74c3c',
        neutral: '#95a5a6'
    };

    const traces = Object.keys(colors).map(sentiment => ({
        x: timelineData.dates,
        y: timelineData[sentiment],
        name: sentiment.charAt(0).toUpperCase() + sentiment.slice(1),
        type: 'scatter',
        mode: 'lines+markers',
        line: { color: colors[sentiment] }
    }));

    const layout = {
        margin: { t: 10, l: 40, r: 10, b: 40 },
        height: 300,
        legend: { orientation: 'h', y: -0.2 },
        yaxis: { title: 'Number of Articles' },
        xaxis: { title: 'Date' }
    };

    Plotly.newPlot('sentimentTimeline', traces, layout);
}

// Show error message
function showError(message) {
    console.error('Dashboard error:', message);
    // Add error alert to the page
    const alertHtml = `
        <div class="alert alert-danger alert-dismissible fade show" role="alert">
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        </div>
    `;
    $('#content').prepend(alertHtml);
} 