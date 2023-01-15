#!/usr/bin/env python
# coding: utf-8

# # TEST

# In[6]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def import_process():
    """
    Import dataframes from Apple Health.
    Compute total daily insulin.
    Compute average daily insulin blood glucose.
    Create master dataframe 'df' by merging on 'Date'.
    Compute additional features:
    - 'Total energy burned(kcal)': basal + active calories burned (daily basis)
    - 'Insulin sensitivity': ratio of total carbs to total insulin (daily basis)
    """
    energy = pd.read_csv('./source_files/energy.csv')
    insulin = pd.read_csv('./source_files/insulin.csv')
    glucose = pd.read_csv('./source_files/glucose.csv')
    
    energy['Date'] = pd.to_datetime(energy['Date'])
    insulin['Date'] = pd.to_datetime(insulin['Date'])
    glucose['Date'] = pd.to_datetime(glucose['Date'])
    
    # Compute total daily bolus insulin
    bolus = insulin.query("Purpose=='Bolus'")
    bolus = bolus[['Date', 'Insulin delivery(IU)']]
    bolus.dropna(inplace=True)
    bolus = bolus.groupby(pd.Grouper(key='Date', axis=0, freq='D')).sum().reset_index()

    # Compute average daily blood glucose
    glucose.dropna(inplace=True)
    glucose = glucose.groupby(pd.Grouper(key='Date', axis=0, freq='D')).mean().reset_index()
    
    # Align 'Date' column in energy dataframe
    energy = energy.groupby(pd.Grouper(key='Date', axis=0, freq='D')).mean().reset_index()
    
    # Merge to master df
    df_diabetes = glucose.merge(bolus, on='Date')
    df = df_diabetes.merge(energy, on='Date')
    
    def replace_outliers(df, columns):
        """
        This function takes the 'df' and a list of columns as arguments.
        The syntax is: replace_outliers(df, ['col1', 'col2', ...])
        It replaces a column's outliers (detected outside the 20th and 80th percentiles) 
        with the median value of that column.
        It returns the 'df' with the corrected values.
        
        """
        
        for column in columns:
            # Calculate the median value of the column
            median = df[column].median()
            mean = df[column].mean()
            std = df[column].std()
            
            # Calculate the lower and upper bounds for outliers
            q1, q3 = df[column].quantile([0.2, 0.8])
            iqr = q3 - q1

            # Calculate the lower and upper bounds
            lower_bound = q1 - (1.5 * iqr)
            upper_bound = q3 + (1.5 * iqr)

           # Calculate the lower and upper bounds for outliers
            #lower_bound = mean - (3*std)
            #upper_bound = mean + (3*std)

            # Replace values outside the bounds with the median value
            df[column] = df[column].apply(lambda x: median if x < lower_bound or x > upper_bound else x)
            # Replace 0s with the median value
            df[column] = df[column].apply(lambda x: median if x == 0 else x)
            # Replace NaNs with the median value
            df[column] = df[column].fillna(df[column].median())
        return df
    
    columns= df[df.columns[~df.columns.isin(['Date'])]]
    df = replace_outliers(df, columns)
    
    # Replace spaces with underscores and lowercase labels 
    df.rename(columns = lambda x: x.strip().lower().replace(" ", "_"), inplace=True)
    
    # Compute additional features
    df['total_energy_burned(kcal)'] = df['active_energy_burned(kcal)'] + df['basal_energy_burned(kcal)']
    df['total_energy_balance'] = df['energy_consumed(kcal)'] - df['total_energy_burned(kcal)']
    df['insulin_sensitivity'] = df['carbohydrates(g)']/df['insulin_delivery(iu)']
    
    sensitivity=[]
    for i in df['insulin_delivery(iu)']:
        sensitivity.append((1500 / i))
    df['insulin_sensitivity'] = sensitivity
    
    df.to_csv('diabetes_master.csv')
    
    return df

def fix_dates(df):
    df['week'] = df['date'].dt.week
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year

    import calendar
    df['month'] = df['month'].apply(lambda x: calendar.month_abbr[x])
    
    # create weekly_df
    weekly_df = df.copy()
    weekly_df = weekly_df.groupby(['year','month','week']).mean().reset_index()
    
    #convert week number to week time
    import datetime
    from dateutil.relativedelta import relativedelta
    # update week column with new value
    formatted_week = []
    for week, month, year in zip(weekly_df.week, weekly_df.month, weekly_df.year):
        formatted_week.append(datetime.date(year, 1, 1) + relativedelta(weeks=+week))
    weekly_df['week'] = formatted_week
    weekly_df = weekly_df[:-1]
    
    return df, weekly_df

def display_metrics(weekly_df):
    """
    Display on a weekly average: 
    Insulin sensitivity, Blood Glucose, Kcal burned, Kcal consumed.

    """
    plt.figure(figsize=[20,15])
    color = sns.color_palette()[0]
    plt.subplots_adjust(hspace=0.3)

    # Metric 1
    plt.subplot(2, 2, 1)
    sns.barplot(data=weekly_df,
               x='week',
               y='insulin_sensitivity',
               color=color)
    plt.ylim(0, weekly_df['insulin_sensitivity'].max()*1.5)
    #plt.xticks(rotation=45)
    plt.title('Insulin Sensitivity')
    #plt.savefig('/output_figures/weekly_report_insulin_sensitivity.png');

    # Metric 2
    plt.subplot(2, 2, 2)
    sns.barplot(data=weekly_df,
               x='week',
               y='blood_glucose(mg/dl)',
               color=color)
    plt.ylim(0, weekly_df['blood_glucose(mg/dl)'].max()*1.5)
    plt.ylim(0, 250)
   # plt.xticks(rotation=45)
    plt.title('Average Weekly Blood Glucose Level(mg/dl)')
    #plt.savefig('/output_figures/weekly_report_blood_glucose.png');


    # Metric 3
    plt.subplot(2, 2, 3)
    sns.barplot(data=weekly_df,
               x='week',
               y='total_energy_burned(kcal)',
               color=color)
    plt.ylim(0, weekly_df['total_energy_burned(kcal)'].max()*1.5)
    #plt.xticks(rotation=45)
    plt.title('Kcal burned')
    #plt.savefig('/output_figures/weekly_report_energy_burned.png');

    # Metric 4
    plt.subplot(2, 2, 4)
    sns.barplot(data=weekly_df,
               x='week',
               y='energy_consumed(kcal)',
                #secondary_y='energy_consumed(kcal)',
               color=color)
    plt.ylim(0, weekly_df['energy_consumed(kcal)'].max()*1.5)
    #plt.xticks(rotation=45)
    plt.title('Kcal eaten')
    plt.savefig('./output_figures/weekly_report.png');
    
    
    ### Without seaborn (choose which one to keep)
    metrics = ['insulin_sensitivity', 'blood_glucose(mg/dl)', 'total_energy_burned(kcal)', 'energy_consumed(kcal)']
    #plt.figure(figsize=[20,15])
    color = sns.color_palette()[0]
    plt.subplots_adjust(hspace=0.3)

    
def display_macro_split(weekly_df):
    """
    Plot stacked bar and column charts showing 
    the proportion of the type of macro in each week.
    """
    # compute frequency by row
    weekly_df[['carbohydrates(g)','protein(g)','fat_total(g)']] = weekly_df[['carbohydrates(g)','protein(g)','fat_total(g)']].apply(lambda x: x/x.sum(), axis=1)

    # prepare dataframe
    macros = weekly_df[['week', 'carbohydrates(g)','protein(g)','fat_total(g)']].copy()
    macros = macros.set_index('week')
    results = macros.div(macros.sum(axis=1), axis=0)
    color=sns.dark_palette("#69d", reverse=True, as_cmap=True)
    
    # display graph
    results.plot(kind='bar', 
            stacked=True, 
            colormap=color, 
            figsize=(10, 6))

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
    plt.xlabel("Week")
    plt.ylabel("Proportion")
    plt.xticks(rotation=0)
    plt.title('Macros split - Weekly basis')
    plt.savefig('./output_figures/macros_split.png');
    plt.show();
    
def display_pct_change_GI(weekly_df):
    """
    Compute the % change in average weekly blood glucose level and display as a waterfall graph using plotly.
    Reference: https://plotly.com/python/waterfall-charts/
    """
    pct_change = ((weekly_df['blood_glucose(mg/dl)'].pct_change(periods=1))*100).round(1)
    weekly_df['∆% Glycemic Index'] = pct_change
    weekly_df['∆% Glycemic Index'] = weekly_df['∆% Glycemic Index'].fillna(0)

    import plotly.graph_objects as go

    fig = go.Figure(go.Waterfall(
        name = "20", orientation = "v",
        measure = ["relative"],
        x = weekly_df['week'],
        textposition = "outside",
        text = [i for i in weekly_df['∆% Glycemic Index']],
        y = weekly_df['∆% Glycemic Index'],
        decreasing = {"marker":{"color":"green"}},
        increasing = {"marker":{"color":"red"}},
        #connector = {"line":{"color":"rgb(63, 63, 63)"}},
        base = 0
    ))

    fig.update_layout(
            title={
                'text': "Weekly % Change in Average Blood Glucose",
                'y':0.9,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'})

    fig.update_layout(yaxis_range=[-100,100])
    plt.savefig('./output_figures/waterfall.png');

    fig.show()


# In[ ]:





# In[ ]:




