
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
def preprocess_and_visualize_store_data(store_df, df):
  # Rename columns for convenience
  store_df.rename(columns={
      'Store': 'store_id',
      'StoreType': 'store_type',
      'Assortment': 'assortment_level',
      'CompetitionDistance': 'competition_distance_m',
      'CompetitionOpenSinceMonth': 'competition_opened_month',
      'CompetitionOpenSinceYear': 'competition_opened_year',
      'Promo2': 'promo_active',
      'Promo2SinceWeek': 'promo2_start_week',
      'Promo2SinceYear': 'promo2_start_year',
      'PromoInterval': 'promo_interval'
  }, inplace=True)

  # Merge store data with training data
  store_train_df = pd.merge(store_df, df, on='store_id', how='left')

  # Fill missing values
  store_train_df['competition_opened_month'] = store_train_df['competition_opened_month'].fillna(0)
  store_train_df['competition_opened_year'] = store_train_df['competition_opened_year'].fillna(0)
  store_train_df['promo2_start_week'] = store_train_df['promo2_start_week'].fillna(0)
  store_train_df['promo2_start_year'] = store_train_df['promo2_start_year'].fillna(0)
  store_train_df['promo_interval'] = store_train_df['promo_interval'].fillna('0')

  # Impute missing competition distance with mean
  mean_distance = store_train_df['competition_distance_m'].mean()
  store_train_df['competition_distance_m'].fillna(mean_distance, inplace=True)

  # Create a scatter plot (customers vs. sales)
  sns.scatterplot(x='customers', y='sales', data=store_train_df, palette='viridis')
  plt.title('Sales vs. Customers')
  plt.xlabel('Number of Customers')
  plt.ylabel('Sales')
  plt.show()

  # Create a histogram of sales
  sns.histplot(store_train_df['sales'], bins=30, kde=True, color='skyblue')
  plt.title('Distribution of Sales')
  plt.xlabel('Sales')
  plt.ylabel('Frequency')
  plt.show()

def analyze_promotions(store_train_df):
  # Calculate average sales during promotions and non-promotions
  avg_sales_promo = store_train_df[store_train_df['promotion'] == 1]['sales'].mean()
  avg_sales_no_promo = store_train_df[store_train_df['promotion'] == 0]['sales'].mean()

  print(f"Average Sales during Promotion: {avg_sales_promo:.2f}")
  print(f"Average Sales without Promotion: {avg_sales_no_promo:.2f}")

  # Sales distribution by promotion status (box plot)
  plt.figure(figsize=(8, 6))
  sns.boxplot(x='promotion', y='sales', data=store_train_df, palette='pastel')
  plt.title('Sales Distribution by Promotion Status')
  plt.xlabel('Promotion Active (1: Yes, 0: No)')
  plt.ylabel('Sales')
  plt.show()

  # Average sales comparison (bar plot)
  plt.figure(figsize=(6, 4))
  sns.barplot(x=['Promotion', 'No Promotion'], y=[avg_sales_promo, avg_sales_no_promo], palette='viridis')
  plt.title('Average Sales Comparison')
  plt.xlabel('Promotion Status')
  plt.ylabel('Average Sales')
  plt.show()

  # Promotion count by store
  store_promotions = store_train_df.groupby('store_id')['promotion'].sum()
  promo_counts_df = pd.DataFrame({'store_id': store_promotions.index, 'Promotion_Count': store_promotions.values})

  # Promotion count visualizations (bar, scatter, line, histogram)
  plt.figure(figsize=(10, 6))
  sns.barplot(x='store_id', y='Promotion_Count', data=promo_counts_df, palette='viridis')
  plt.xticks(rotation=45)
  plt.xlabel('Store ID')
  plt.ylabel('Number of Promotions')
  plt.title('Promotion Count by Store')
  plt.show()

  plt.figure(figsize=(8, 6))
  sns.scatterplot(x='store_id', y='Promotion_Count', data=promo_counts_df, color='blue')
  plt.xlabel('Store ID')
  plt.ylabel('Number of Promotions')
  plt.title('Promotion Count by Store')
  plt.show()

  plt.figure(figsize=(12, 6))
  sns.lineplot(x='store_id', y='Promotion_Count', data=promo_counts_df, marker='o')
  plt.xlabel('Store ID')
  plt.ylabel('Number of Promotions')
  plt.title('Promotion Count by Store')
  plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
  plt.show()

  plt.figure(figsize=(8, 6))
  sns.histplot(promo_counts_df['Promotion_Count'], bins=20, kde=True, color='skyblue')
  plt.xlabel('Number of Promotions')
  plt.ylabel('Frequency')
  plt.title('Distribution of Promotion Counts Across Stores')
  plt.show()

def analyze_sales_by_daytype(store_train_df):

  # Create a new column to identify weekdays/weekends
  store_train_df['date'] = pd.to_datetime(store_train_df['date'])
  store_train_df['day_type'] = store_train_df['date'].dt.dayofweek.apply(lambda x: 'Weekday' if x < 5 else 'Weekend')

  # Calculate average sales for weekdays and weekends
  avg_sales_weekday = store_train_df[store_train_df['day_type'] == 'Weekday']['sales'].mean()
  avg_sales_weekend = store_train_df[store_train_df['day_type'] == 'Weekend']['sales'].mean()

  print(f"Average Sales on Weekdays: {avg_sales_weekday:.2f}")
  print(f"Average Sales on Weekends: {avg_sales_weekend:.2f}")

  # Create a bar plot to compare average sales
  plt.figure(figsize=(6, 4))
  sns.barplot(x=['Weekday', 'Weekend'], y=[avg_sales_weekday, avg_sales_weekend], palette='viridis')
  plt.title('Average Sales by Day Type')
  plt.xlabel('Day Type')
  plt.ylabel('Average Sales')
  plt.show()

def analyze_sales_by_assortment_and_competition(store_train_df):
  # Average sales by assortment level
  avg_sales_by_assortment = store_train_df.groupby('assortment_level')['sales'].mean()

  # Bar plot for average sales by assortment level
  plt.figure(figsize=(8, 6))
  sns.barplot(x=avg_sales_by_assortment.index, y=avg_sales_by_assortment.values, palette='viridis')
  plt.xlabel('Assortment Level')
  plt.ylabel('Average Sales')
  plt.title('Average Sales by Assortment Level')
  plt.show()

  # Box plot for sales distribution by assortment level
  plt.figure(figsize=(8, 6))
  sns.boxplot(x='assortment_level', y='sales', data=store_train_df, palette='pastel')
  plt.xlabel('Assortment Level')
  plt.ylabel('Sales')
  plt.title('Sales Distribution by Assortment Level')
  plt.show()

  # Sales vs. competition distance (scatter plot)
  plt.figure(figsize=(10, 6))
  sns.scatterplot(x='competition_distance_m', y='sales', data=store_train_df)
  plt.xlabel('Competition Distance (meters)')
  plt.ylabel('Sales')
  plt.title('Sales vs. Competition Distance')
  plt.show()

  # Sales vs. competition distance (line plot, sorted by distance)
  store_train_df_sorted = store_train_df.sort_values(by='competition_distance_m')
  plt.figure(figsize=(10, 6))
  sns.lineplot(x='competition_distance_m', y='sales', data=store_train_df_sorted)
  plt.xlabel('Competition Distance (meters)')
  plt.ylabel('Sales')
  plt.title('Sales vs. Competition Distance')
  plt.show()

def analyze_sales_by_holidays(df):

  # Create a new column 'holiday' combining StateHoliday and SchoolHoliday
  df['holiday'] = df['state_holiday'].astype(str) + df['school_holiday'].astype(str)
  df['holiday'] = df['holiday'].apply(lambda x: 'Holiday' if x != '00' else 'No Holiday')

  # Calculate average sales during holidays and non-holidays
  avg_sales_holiday = df[df['holiday'] == 'Holiday']['sales'].mean()
  avg_sales_non_holiday = df[df['holiday'] == 'No Holiday']['sales'].mean()

  # Print the average sales
  print(f"Average Sales during Holidays: {avg_sales_holiday:.2f}")
  print(f"Average Sales on Non-Holiday Days: {avg_sales_non_holiday:.2f}")

  # Create a DataFrame for plotting
  data = {'Holiday Period': ['Holiday', 'Non-Holiday'], 'Average Sales': [avg_sales_holiday, avg_sales_non_holiday]}
  plot_df = pd.DataFrame(data)

  # Create a bar plot to compare average sales
  plt.figure(figsize=(8, 6))
  ax = sns.barplot(x='Holiday Period', y='Average Sales', data=plot_df)

  # Annotate the bars with the average sales values
  for p in ax.patches:
    ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=11, color='black', xytext=(0, 5),
                textcoords='offset points')

  plt.title('Average Sales by Holiday Period')
  plt.xlabel('Holiday Period')
  plt.ylabel('Average Sales')
  plt.show()      