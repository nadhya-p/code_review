#import necessary modules
import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import datetime
import matplotlib.dates as mdates
import shutil
from dateutil.relativedelta import relativedelta
import scipy.stats as stats
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM
from patsy import dmatrices
from sklearn.preprocessing import StandardScaler



def convert_file_name(key,file_name):
    ''' Converts file names into recording dates for use in regression graphs '''
    
    #get data from childrens.csv to get child dob
    cwd = os.getcwd()
    child_file_path = os.path.join(cwd, "metadata", "children.csv")
    df = pd.read_csv(child_file_path, usecols=["child_id", "child_dob"])
    dob = df.loc[df['child_id'] == key, 'child_dob'].iloc[0]
    dob_datetime = datetime.datetime.strptime(dob, '%Y-%m-%d')
    
    # split the file name into parts
    name, age = file_name.split('_')
    
    year = int(age[:2])
    month = int(age[2:4])
    day = int(age[4:6])
    
    recording_age_months = (year*12) + month + (day/30)
    recording_age_months = round(recording_age_months,2)
    # add the year, month, and day variables to dob_datetime to get the recording date
   # recording_date = dob_datetime + relativedelta(years=year, months=month, days=day)
    
    # get only the date part of recording_date
    #recording_date = recording_date.date()
    #recording_date = recording_date.strftime('%d-%m-%Y')


    return recording_age_months

def import_acoustic_data():
    '''Import the acoustic data for each child so we can make the graphs'''
    
    # Get the current working directory (which will be the scripts folder)
    current_directory = os.getcwd()

    # Move up one directory
    os.chdir('..')
    
    #Get the path for conversations
    root_path = os.getcwd()
    
    #Specify path to accoustic data
    acoustic_path = os.path.join(root_path,'annotations','acoustic','raw')
    
    #print('Path to acoustic data: {}'.format(acoustic_path))
    #print('Path to conversations data: {}'.format(conversations_path))
    
    acoustic_dict = {}
    
    #Get the acoustic files from all of the folders in the acousitc path
    for subdir in os.listdir(acoustic_path):
        subdir_path = os.path.join(acoustic_path, subdir)
        if os.path.isdir(subdir_path):
            files = [f for f in os.listdir(subdir_path) if f.endswith('.csv') and not f.startswith('.')]
            if files:
                basename = os.path.basename(subdir_path)
                key = basename[len('ACOUSTIC_'):]
                acoustic_dict[key] = [os.path.join(subdir_path, f) for f in files]

    return acoustic_dict

    
def import_conversations_data():
    '''Import the conversations data for each child so we can make the graphs'''
    
    #Get the path for conversations
    root_path = os.getcwd()
    
    #Specify path to acoustic data
    conversations_path = os.path.join(root_path,'annotations','conversations','raw')
    
    conversations_dict = {}
    
    #Get the acoustic files from all of the folders in the acoustic path
    for subdir in os.listdir(conversations_path):
        subdir_path = os.path.join(conversations_path, subdir)
        if os.path.isdir(subdir_path):
            files = [f for f in os.listdir(subdir_path) if f.endswith('.csv') and not f.startswith('.')]
            if files:
                basename = os.path.basename(subdir_path)
                key = basename[len('CONVERSATIONS_'):]
                conversations_dict[key] = [os.path.join(subdir_path, f) for f in files]
    
    return conversations_dict

def create_conversations_dataframes(conversations_dict):
    ''' create dataframes from conversations data and get the variables of interest '''
    
    data_dict = {}

    for key in conversations_dict:
        files = conversations_dict[key]
        sub_dict = {
            'segment_duration' : [],
            'age_months': []
        }
        for file in files:
            df = pd.read_csv(file)
            df = df[df['speaker_type'] == 'CHI']
            sub_dict['segment_duration'].append(df['segment_duration'].mean())
            file_name = file.split('/')[-1]  # Get the last element of
            first_name = file_name.split('_')[0]
            recording_date = convert_file_name(key, os.path.splitext(os.path.basename(file))[0])
            sub_dict['age_months'].append(recording_date)
        data_dict[key] = sub_dict
        
    return data_dict

def create_acoustic_dataframes(acoustic_dict):
    ''' create dataframes from conversations data and get the variables of interest '''
    
    data_dict = {}
    
    for key in acoustic_dict:
        files = acoustic_dict[key]
        
        sub_dict = {
            'mean_pitch': [],
            'median_pitch': [],
            'pitch_range': [],
            'p5_semitone': [],
            'p95_semitone': [],
            'age_months': []
        }
            
        for file in files:
            df = pd.read_csv(file)
            df = df[df['speaker_type'] == 'CHI']
            sub_dict['mean_pitch'].append(df['mean_pitch_semitone'].mean())
            sub_dict['median_pitch'].append(df['median_pitch_semitone'].mean())
            sub_dict['pitch_range'].append(df['pitch_range_semitone'].mean())
            sub_dict['p95_semitone'].append(df['p95_pitch_semitone'].mean())
            sub_dict['p5_semitone'].append(df['p5_pitch_semitone'].mean())
            recording_date = convert_file_name(key, os.path.splitext(os.path.basename(file))[0])
            sub_dict['age_months'].append(recording_date)
        data_dict[key] = sub_dict
        
    return data_dict
    

    
    
def main():
    ''' creates the mixed regression model + plots the graphs of each variable and saves them to the cwd '''
    
    #import the data files
    acoustic_data = import_acoustic_data()
    conversations_data = import_conversations_data()
    
    conversations_dict = create_conversations_dataframes(conversations_data)
    acoustic_dict = create_acoustic_dataframes(acoustic_data)
    
        
    ################################ build the mixed regression model ##################################
    df_conv = pd.DataFrame(columns =['Child','utterance_duration','age'])
    
    for child, values in conversations_dict.items():
        for i in range(len(values['age_months'])):
            df_conv = df_conv.append({'Child': child,
                                      'utterance_duration': values['segment_duration'][i],
                                      'age': values['age_months'][i]}, ignore_index=True)
        
    
    df_ac = pd.DataFrame(columns =['Child','mean_pitch','median_pitch', 'pitch_range', 'p5', 'p95', 'age'])
    
    for child, values in acoustic_dict.items():
        for i in range(len(values['age_months'])):
            df_ac = df_ac.append({'Child': child,
                                      'mean_pitch': values['mean_pitch'][i],
                                      'median_pitch': values['median_pitch'][i],
                                      'pitch_range': values['pitch_range'][i],
                                      'p5': values['p5_semitone'][i],
                                      'p95': values['p95_semitone'][i],
                                      'age': values['age_months'][i]}, ignore_index=True)
        

    merged_df = pd.merge(df_ac, df_conv, on=['Child', 'age'])
    df = merged_df.sort_values(by=['Child', 'age'])

    # make catergorical for model
    df['Child'] = pd.Categorical(df['Child'])
    
    # select only the numeric columns
    df_numeric = df.select_dtypes(include=['float64', 'int64'])

    # create a StandardScaler object
    scaler = StandardScaler()

    # fit the scaler to the DataFrame
    scaler.fit(df_numeric)

    # transform the DataFrame
    df_scaled = pd.DataFrame(scaler.transform(df_numeric), columns=df_numeric.columns)
    df_scaled = pd.concat([df_scaled, df[['Child']]], axis=1)
    #df_scaled.to_csv('scaled_df_for_regression.csv', index=False)


    print(df_scaled)
    
    # Create the formula for the mixed-effects model
    model = sm.MixedLM.from_formula('age ~ median_pitch + pitch_range + utterance_duration', data=df_scaled, groups='Child')

    # Fit mixed-effects model
    result = model.fit()
    
    # get the AIC and BIC values
    aic = result.aic
    bic = result.bic

    print(f"AIC: {aic}, BIC: {bic}")

    
    print(result.summary())
    


    ################################# create utterance graph for each child ###########################
    num_rows = len(conversations_dict)

    # Create subplots with one row and as many columns as there are subdictionaries
    fig, axs = plt.subplots(nrows=num_rows, ncols=1, figsize=(10*num_rows, 10*num_rows), sharey=True)

    # Loop through subdictionaries and plot scatter plot on each subplot
    for i, (key, subdict) in enumerate(conversations_dict.items()):
        ax = axs[i]
        
        
        x = subdict['age_months']
        
        # Plot scatter plot with regression line
        sns.regplot(x=x, y=subdict['segment_duration'], ax=ax, ci=None)
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, subdict['segment_duration'])
        #ax.text(0.2, 0.9, "slope = {:.2f}".format(slope), transform=plt.gca().transAxes)
        
        #uncomment if you want to print slopes
        #print(f'The slope of {key}\'s Uterrance Duration Graph is: {slope}')
        
        #uncomment if you want to print r values
        #print(f'The r value of {key}\'s Uterrance Duration Graph is: {r_value}')
        
        #uncomment if you want to print the R-squared values
        #print(f'The R-squared value of {key}\'s Utterance Duration Graph is: {r_value**2:.3f}')
    

        # Set x-axis ticks and labels
        ax.set_xticks(x)
        ax.set_xticklabels(subdict['age_months'], rotation='vertical')
        ax.set_xlabel('Age at Recording (months)', fontsize=14)
        ax.set_ylabel('Utterance Duration', fontsize=14)
        ax.set_title(key, fontsize=17)
        
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.2)
    fig.savefig('longitudinal_utterances.png', dpi=300)

    
    ###################################### create acoustic graphs for each child ########################################
    #mean pitch graphs
    
    # Create the first set of subplots
    num_rows = len(acoustic_dict)
    fig1, axs1 = plt.subplots(num_rows, 1, figsize=(5*num_rows, 10*num_rows), sharey = True)
    
    # Loop through subdictionaries and plot scatter plot on each subplot
    for i, (key, subdict) in enumerate(acoustic_dict.items()):
        ax = axs1[i]
        
        x = subdict['age_months']
        
        # Plot scatter plot with regression line
        sns.regplot(x=x, y=subdict['mean_pitch'], ax=ax, ci=None)
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, subdict['mean_pitch'])
        #print(f'The slope of {key}\'s Mean Pitch Graph is: {slope}')
        #print(f'The r value of {key}\'s Mean Pitch Graph is: {r_value}')
        print(f'The R-squared value of {key}\'s Mean Pitch Graph is: {r_value**2:.3f}')

        
        # Set x-axis ticks and labels
        ax.set_xticks(x)
        ax.set_xticklabels(subdict['age_months'], rotation='vertical')
        ax.set_xlabel('Age at Recording (months)', fontsize=14)
        ax.set_ylabel('Mean Pitch (semitones)', fontsize=14)
        ax.set_title(key, fontsize=17)
    
    fig1.suptitle("Mean Pitch by Child", fontsize=20)
    plt.tight_layout()
    fig1.subplots_adjust(hspace=0.2)
    fig1.savefig('longitudinal_mean_pitch_graphs.png', dpi=300)

    #median_pitch graphs
    fig2, axs2 = plt.subplots(num_rows, 1, figsize=(5*num_rows, 10*num_rows), sharey = True)
    
    for i, (key, subdict) in enumerate(acoustic_dict.items()):
        ax = axs2[i]
        
        # Convert date strings to datetime objects
        #dates = [datetime.datetime.strptime(date_str, '%d-%m-%Y') for date_str in subdict['child_names']]
        # Convert datetime objects to matplotlib dates
        #x = mdates.date2num(dates)
        x = subdict['age_months']
        
        # Plot scatter plot with regression line
        sns.regplot(x=x, y=subdict['median_pitch'], ax=ax, ci=None)
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, subdict['median_pitch'])
        
        #uncomment if you want to print slopes
        #print(f'The slope of {key}\'s Median Pitch Graph is: {slope}')
        
        #uncomment if you want to print r values
        #print(f'The r value of {key}\'s Median Pitch Graph is: {r_value}')

        # Set x-axis ticks and labels
        ax.set_xticks(x)
        ax.set_xticklabels(subdict['age_months'], rotation='vertical')
        ax.set_xlabel('Age at Recording (months)', fontsize=14)
        ax.set_ylabel('Median Pitch (semitones)', fontsize=14)
        ax.set_title(key, fontsize=17)
        
    fig2.suptitle("Median Pitch by Child", fontsize=20)
    plt.tight_layout()
    fig2.subplots_adjust(hspace=0.2)
    fig2.savefig('longitudinal_median_pitch_graphs.png', dpi=300)
    
    #pitch_range_pitch graphs
    fig3, axs3 = plt.subplots(num_rows, 1, figsize=(5*num_rows, 10*num_rows), sharey = True)
    
    for i, (key, subdict) in enumerate(acoustic_dict.items()):
        ax = axs3[i]
        
        
        x = subdict['age_months']
        
        # Plot scatter plot with regression line
        sns.regplot(x=x, y=subdict['pitch_range'], ax=ax, ci=None)
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, subdict['pitch_range'])
        
        #uncomment if you want to print slopes
        #print(f'The slope of {key}\'s Pitch Range Graph is: {slope}')
        
        #uncomment if you want to print r values
        #print(f'The r value of {key}\'s Pitch Range Graph is: {r_value}')

        
        # Set x-axis ticks and labels
        ax.set_xticks(x)
        ax.set_xticklabels(subdict['age_months'], rotation='vertical')
        ax.set_xlabel('Age at Recording (months)', fontsize=14)
        ax.set_ylabel('Pitch Range (semitones)', fontsize=14)
        ax.set_title(key, fontsize=17)
        
    fig3.suptitle("Pitch Range by Child", fontsize=20)
    plt.tight_layout()
    fig3.subplots_adjust(hspace=0.2)
    fig3.savefig('longitudinal_pitch_range_graphs.png', dpi=300)
    
    #p5 graphs
    fig4, axs4 = plt.subplots(num_rows, 1, figsize=(5*num_rows, 10*num_rows), sharey = True)
    
    for i, (key, subdict) in enumerate(acoustic_dict.items()):
        ax = axs4[i]
        
        
        x = subdict['age_months']
        
        # Plot scatter plot with regression line
        sns.regplot(x=x, y=subdict['p5_semitone'], ax=ax, ci=None)
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, subdict['p5_semitone'])
        
        #uncomment if you want to print slopes
        #print(f'The slope of {key}\'s p5 Pitch Graph is: {slope}')
        
        #uncomment if you want to print r values
        #print(f'The r value of {key}\'s p5 Pitch Graph is: {r_value}')
        
        # Set x-axis ticks and labels
        ax.set_xticks(x)
        ax.set_xticklabels(subdict['age_months'], rotation='vertical')
        ax.set_xlabel('Age at Recording (months)', fontsize=14)
        ax.set_ylabel('p5 Pitch (semitones)', fontsize=14)
        ax.set_title(key, fontsize=17)
    
    fig4.suptitle("p5 Pitch by Child", fontsize=20)
    plt.tight_layout()
    fig4.subplots_adjust(hspace=0.2)
    fig4.savefig('longitudinal_p5_graphs.png', dpi=300)
    
    #p95 graphs
    fig5, axs5 = plt.subplots(num_rows, 1, figsize=(5*num_rows, 10*num_rows), sharey = True)
    
    for i, (key, subdict) in enumerate(acoustic_dict.items()):
        ax = axs5[i]
        
        x = subdict['age_months']
        
        # Plot scatter plot with regression line
        sns.regplot(x=x, y=subdict['p95_semitone'], ax=ax, ci=None)
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, subdict['p95_semitone'])
        
        #uncomment if you want to print slopes
       # print(f'The slope of {key}\'s p95 Pitch Graph is: {slope}')
       
        #uncomment if you want to print r values
        #print(f'The r value of {key}\'s p95 Pitch Graph is: {r_value}')

        # Set x-axis ticks and labels
        ax.set_xticks(x)
        ax.set_xticklabels(subdict['age_months'], rotation='vertical')
        ax.set_xlabel('Age at Recording (months)', fontsize=14)
        ax.set_ylabel('p95 Pitch (semitones)', fontsize=14)
        ax.set_title(key, fontsize=17)
        
    fig5.suptitle("p95 Pitch by Child", fontsize=20)
    plt.tight_layout()
    fig5.subplots_adjust(hspace=0.2)
    fig5.savefig('longitudinal_p95_graphs.png', dpi=300)
    
    ######################## save graphs in graphs folder ######################################
    # Define the directory to search for .png files
    dir_path = os.getcwd()

    # Create the 'graphs' subfolder if it doesn't exist
    if not os.path.exists(os.path.join(dir_path, 'graphs')):
        os.makedirs(os.path.join(dir_path, 'graphs'))

    # Create the 'longitudinal' subfolder within 'graphs' if it doesn't exist
    graphs_dir = os.path.join(dir_path, 'graphs')
    if not os.path.exists(os.path.join(graphs_dir, 'longitudinal')):
        os.makedirs(os.path.join(graphs_dir, 'longitudinal'))

    # Get a list of all .png files in the current directory
    png_files = [f for f in os.listdir(dir_path) if f.endswith('.png')]

    # Move all .png files to the appropriate subfolder
    for png_file in png_files:
        src_path = os.path.join(dir_path, png_file)
        if 'longitudinal' in png_file:
            dest_path = os.path.join(graphs_dir, 'longitudinal', png_file)
        else:
            dest_path = os.path.join(graphs_dir, png_file)
        shutil.move(src_path, dest_path)
        
    # Print the message
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    path_to_graphs = os.path.join(os.getcwd(),"graphs")
    print(f"Graphs created in {path_to_graphs}. Updated at {now}.")
    
    
if __name__ == '__main__':
    main()
