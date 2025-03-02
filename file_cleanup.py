import pandas as pd

# Load the CSV file
df = pd.read_csv('datasets/aus-property-sales-sep2018-april2020.csv')

# Check if date_sold column exists
if 'date_sold' in df.columns:
    # Convert date_sold to datetime and then to YYYY-MM-DD format
    df['date_sold'] = pd.to_datetime(df['date_sold']).dt.strftime('%Y-%m-%d')
    # Replace "2021" with "2025" in the date_sold column
    df['date_sold'] = df['date_sold'].astype(str).str.replace('2021', '2025')
    # Save the modified DataFrame to a new CSV file
    output_path = 'datasets/aus-property.csv'
    df.to_csv(output_path, index=False)
    # Show first few rows as confirmation
    print("File has been updated. Here are the first few rows:")
    df = df.dropna()
    print("\nNULL entries have been removed. Updated row count:", len(df))
    print(df.head())
else:
    print("The column 'date_sold' was not found in the CSV file.")
    # Remove any NULL entries from the DataFrame
    df = df.dropna()
    print("\nNULL entries have been removed. Updated row count:", len(df))
