
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
       




def analyze_numericals(df):
  """
  Performs univariate analysis on numerical columns in a DataFrame.

  Args:
      df (pd.DataFrame): The DataFrame containing the data.

  Returns:
      None
  """
  numerical_columns = df.select_dtypes(include=[np.number]).columns

  for column in numerical_columns:
    # Continuous variables (more than 10 unique values)
    if len(df[column].unique()) > 10:
      plt.figure(figsize=(8, 6))
      sns.histplot(df[column], kde=True)
      plt.title(f'Histogram of {column}')
      plt.xlabel(column)
      plt.ylabel('Frequency')
      plt.show()
    # Discrete or ordinal variables (less than or equal to 10 unique values)
    else:
      plt.figure(figsize=(8, 6))
      ax = sns.countplot(x=column, data=df)
      plt.title(f'Count of {column}')
      plt.xlabel(column)
      plt.ylabel('Count')

      # Annotate each bar with its count
      for p in ax.patches:
          ax.annotate(format(p.get_height(), '.0f'), 
                      (p.get_x() + p.get_width() / 2., p.get_height()), 
                      ha = 'center', va = 'center', 
                      xytext = (0, 5), 
                      textcoords = 'offset points')
      plt.show()

