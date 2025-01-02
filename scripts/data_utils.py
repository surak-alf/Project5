
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(file_path, **kwargs):
  """
  Loads data from a file into a pandas DataFrame. 

  Args:
      file_path (str): Path to the data file.
      **kwargs: Additional arguments to pass to pandas.read_csv().

  Returns:
      pd.DataFrame: The loaded DataFrame, or None if errors occur.
  """
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
  """
  Generates a summary of the columns in a DataFrame.

  Args:
      df (pd.DataFrame): The DataFrame to summarize.

  Returns:
      pd.DataFrame: A DataFrame containing summary information for each column.
  """
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

# Assuming you have loaded your data into a DataFrame named 'df'
# analyze_numericals(df)
