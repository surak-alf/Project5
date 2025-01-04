
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno

def load_data(file_path, **kwargs):
  try:
      df = pd.read_csv(file_path, **kwargs) 
      return df
  except FileNotFoundError:
      print(f"Error: File not found at {file_path}")
      return None
  except Exception as e:
      print(f"Error loading data: {e}")
      return None


def column_summary(df):
  summary_data = []

  for col_name in df.columns:
      col_dtype = df[col_name].dtype
      num_of_nulls = df[col_name].isnull().sum()
      num_of_non_nulls = df[col_name].notnull().sum()
      num_of_distinct_values = df[col_name].nunique()

      summary_data.append({
          'col_name': col_name,
          'col_dtype': col_dtype,
          'num_of_nulls': num_of_nulls,
          'num_of_non_nulls': num_of_non_nulls,
          'num_of_distinct_values': num_of_distinct_values,
      })

  summary_df = pd.DataFrame(summary_data)
  return summary_df

def duplicates_info(df):
  
  pd.set_option('display.max_rows', 500)
  pd.set_option('display.max_columns', 500)
  pd.set_option('display.width', 1000)

  print(df.head())
  print(df.describe())
  print(f"Number of duplicate rows: {df.duplicated().sum()}")
def plot_missingno_matrix_bar(df):
  # Matrix plot
  msno.matrix(df)
  plt.title("Missingno Matrix Plot")
  plt.show()

  # Bar plot
  msno.bar(df)
  plt.title("Missingno Bar Plot")
  plt.show()   
def filling_missing_values(df):

  # Fill missing values with zero
  df['competition_opened_month'] = df['competition_opened_month'].fillna(0)
  df['competition_opened_year'] = df['competition_opened_year'].fillna(0)
  df['promo2_start_week'] = df['promo2_start_week'].fillna(0)
  df['promo2_start_year'] = df['promo2_start_year'].fillna(0)
  df['promo_interval'] = df['promo_interval'].fillna('0')   

def plot_countplot_assortment_level(df):
  sns.countplot(x='assortment_level', data=df)

  # Customize the plot
  plt.title('Distribution of Assortment Levels')
  plt.xlabel('Assortment Level')
  plt.ylabel('Count')

  # Show the plot
  plt.show()
def plot_histogram_competition_distance(df, bins=30):
  sns.histplot(data=df, x='competition_distance_m', bins=bins)

  # Customize the plot
  plt.title('Distribution of Competition Distance')
  plt.xlabel('Competition Distance (meters)')
  plt.ylabel('Frequency')

  # Show the plot
  plt.show()
def plot_countplot_competition_opened_month(df):
  sns.countplot(x='competition_opened_month', data=df)

  # Customize the plot
  plt.title('Distribution of Competition Opened Month')
  plt.xlabel('Month')
  plt.ylabel('Count')
  plt.xticks(rotation=45)

  # Show the plot
  plt.show()
def plot_countplot_competition_opened_year(df):
  sns.countplot(x='competition_opened_year', data=df)

  # Customize the plot
  plt.title('Distribution of Competition Opened Year')
  plt.xlabel('Year')
  plt.ylabel('Count')
  plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

  # Adjust figure size and tight layout
  plt.figure(figsize=(15, 6))  # Increase width to 15 inches
  plt.tight_layout()  # Improve layout for better spacing

  # Show the plot
  plt.show()
def plot_countplot_promo_active(df):
  # Generate sample data for 'promo_active'
  df['promo_active'] = np.random.choice([0, 1], size=len(df))

  # Create the countplot for 'promo_active'
  sns.countplot(x='promo_active', data=df)

  # Customize the plot
  plt.title('Distribution of Promo Active')
  plt.xlabel('Promo Active')
  plt.ylabel('Count')

  # Show the plot
  plt.show()    
def plot_countplot_promo2_start_week(df):

  sns.countplot(x='promo2_start_week', data=df)

  # Customize the plot
  plt.title('Distribution of Promo2 Start Week')
  plt.xlabel('Week of the Year')
  plt.ylabel('Count')
  plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

  # Show the plot
  plt.show()
def plot_countplot_promo2_start_year(df):
  sns.countplot(x='promo2_start_year', data=df)

  # Customize the plot
  plt.title('Distribution of Promo2 Start Year')
  plt.xlabel('Year')
  plt.ylabel('Count')
  plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

  # Show the plot
  plt.show()
def plot_countplot_promo_interval(df):
  # Join month lists into comma-separated strings (handling missing values)
  df['promo_interval'] = df['promo_interval'].apply(lambda x: ','.join(x) if pd.notna(x) else np.nan)

  # Create the countplot for 'promo_interval'
  sns.countplot(x='promo_interval', data=df)

  # Customize the plot
  plt.title('Distribution of Promo Intervals')
  plt.xlabel('Promo Interval')
  plt.ylabel('Count')

  # Rotate x-axis labels for better readability and adjust horizontal alignment
  plt.xticks(rotation=45, ha='right')

  # Show the plot
  plt.show()
def plot_countplot_day_of_week(df):

  plt.figure(figsize=(8, 6))  # Set figure size
  sns.countplot(x='day_of_week', data=df)

  # Customize the plot
  plt.title('Distribution of Day of Week')
  plt.xlabel('Day of Week')
  plt.ylabel('Frequency')
  plt.xticks(rotation=45)

  # Show the plot
  plt.show()
def plot_countplot_store_open(df):
  plt.figure(figsize=(6, 4))  
  sns.countplot(x='store_open', data=df)

  # Customize the plot
  plt.title('Distribution of Store Open')
  plt.xlabel('Store Open')
  plt.ylabel('Frequency')
  plt.xticks(rotation=45)

  # Show the plot
  plt.show()
def plot_countplots(df, figsize=(6, 4)):
  # Create and customize countplot for 'promotion'
  plt.figure(figsize=figsize)
  sns.countplot(x='promotion', data=df)
  plt.title('Distribution of Promotions')
  plt.xlabel('Promotion (0: No Promo, 1: Promo)')
  plt.ylabel('Count')
  plt.xticks([0, 1], ['No Promo', 'Promo'])  # Set explicit labels for 0 and 1
  plt.show()

  # Create and customize countplot for 'state_holiday'
  plt.figure(figsize=figsize)
  sns.countplot(x='state_holiday', data=df)
  plt.title('Distribution of State Holiday')
  plt.xlabel('State Holiday')
  plt.ylabel('Frequency')
  plt.xticks(rotation=45)  
  plt.show()

  # Create and customize countplot for 'school_holiday'
  plt.figure(figsize=figsize)
  sns.countplot(x='school_holiday', data=df)
  plt.title('Distribution of School Holiday')
  plt.xlabel('School Holiday')
  plt.ylabel('Frequency')
  plt.xticks(rotation=45) 
  plt.show()
def plot_distributions_and_countplots(df):
  # Sales distribution
  plt.figure(figsize=(10, 6))
  sns.histplot(df['sales'], bins=30, kde=True)
  plt.title('Distribution of Sales')
  plt.xlabel('Sales')
  plt.ylabel('Frequency')
  plt.show()

  # Customers distribution
  plt.figure(figsize=(10, 6))
  sns.histplot(df['customers'], bins=30, kde=True)
  plt.title('Distribution of Customers')
  plt.xlabel('Customers')
  plt.ylabel('Frequency')
  plt.show()

  # Day of week countplot
  plt.figure(figsize=(8, 6))
  sns.countplot(x='day_of_week', data=df)
  plt.title('Distribution of Day of Week')
  plt.xlabel('Day of Week')
  plt.ylabel('Frequency')
  plt.xticks(rotation=45)
  plt.show()

  # Store open countplot
  plt.figure(figsize=(6, 4))
  sns.countplot(x='store_open', data=df)
  plt.title('Distribution of Store Open')
  plt.xlabel('Store Open')
  plt.ylabel('Frequency')
  plt.show()

  # Promotion countplot (with explicit labels)
  plt.figure(figsize=(6, 4))
  sns.countplot(x='promotion', data=df)
  plt.title('Distribution of Promotions')
  plt.xlabel('Promotion (0: No Promo, 1: Promo)')
  plt.ylabel('Count')
  plt.xticks([0, 1], ['No Promo', 'Promo'])  # Set explicit labels for 0 and 1
  plt.show()

  # State holiday countplot
  plt.figure(figsize=(6, 4))
  sns.countplot(x='state_holiday', data=df)
  plt.title('Distribution of State Holiday')
  plt.xlabel('State Holiday')
  plt.ylabel('Frequency')
  plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
  plt.show()

  # School holiday countplot
  plt.figure(figsize=(6, 4))
  sns.countplot(x='school_holiday', data=df)
  plt.title('Distribution of School Holiday')
  plt.xlabel('School Holiday')
  plt.ylabel('Frequency')
  plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
  plt.show()