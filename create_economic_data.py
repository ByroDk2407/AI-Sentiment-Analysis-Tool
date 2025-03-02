import pandas as pd
import numpy as np
from datetime import datetime

# Set random seed for reproducibility
np.random.seed(42)

# Create date range
start_date = '2025-03-02'
end_date = '2025-05-03'
date_range = pd.date_range(start=start_date, end=end_date, freq='D')

# Generate random data with realistic ranges
# Inflation: typically between 0-10%
inflation = np.random.normal(loc=2.5, scale=0.5, size=len(date_range))
# GDP: typically between -2% to 6%
gdp = np.random.normal(loc=2.0, scale=0.8, size=len(date_range))
# Interest rates: typically between 2-8%
interest_rates = np.random.normal(loc=4.5, scale=0.3, size=len(date_range))

# Create DataFrame
economic_data = pd.DataFrame({
    'Date': date_range,
    'Inflation': np.round(inflation, 2),
    'GDP': np.round(gdp, 2),
    'Interest_Rate': np.round(interest_rates, 2)
})

# Set Date as index
economic_data.set_index('Date', inplace=True)

# Save to CSV
economic_data.to_csv('datasets/economic_data.csv')

# Display first few rows
print(economic_data.head())
