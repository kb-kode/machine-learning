#!/usr/bin/env python
"""
AirBnB Rio de Janeiro Data Challenge
Python 3.7
"""
# Import
import os
import logging
import operator

import pandas as pd
import numpy as np
from datetime import datetime
#plotting packages
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf

#stats packages
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder


# Functions
def main():
    """
    Function:
        - reads data
        - cleans data
        - calculates and appends supporting features
        - run analysis and output graphs as collection of PDFs
        - run model prediction

    """

    # Develop dataset for analysis and model prediction. Save file.
    
    file_path = './data/'
    
    print ('Cleaning and enhancing features...')
    data = generate_features(file_path)
    data.to_csv(file_path + 'df_cleaned.csv')
    print ('Cleaning and enhancing COMPLETED, file created.')
    
    #Generate EDA Plots
    
    eda_plots(data)
    
    #Create dataframe for model
    model_df = prep_model_df(data)
    #Run RF model
    auc, feat_imp = run_model(model_df, 0.20)
    print ('AUC: ' + str(auc))
    print (feat_imp)
    
    #Create pdf for partial dependence plots and distribution plots
    logging.info('Creating partial dependence plots')
    
    #set cutoff for xlims for graphs MANUAL
    cutoff = {}
    cutoff['m_interactions'] = 50
    cutoff['m_first_message_length_in_characters'] = 500
    cutoff['total_reviews'] = 50
    cutoff['inq_to_checkin'] = 200
    cutoff['num_nights'] = 50
    cutoff['host_words_in_user_profile'] = 175
    
    pdf = matplotlib.backends.backend_pdf.PdfPages('./data/dist_univariate.pdf')
    
    for col,_ in feat_imp[:6]:
        dist_uni_plot(model_df,col,'booked',pdf,xlim = cutoff[col])
    
    pdf.close()
    
def generate_features(file_path):
    """
    *Return: DataFrame of Cleaned+Enhanced dataset.
    file_path: directory where csv files are stored (string) 
    """
    
    def convert_to_datetime(df, col):
        df[col] = np.where(df[col].notnull(),df[col],"")
        df[col] = [x.split('.',1)[0] for x in df[col]]
        df[col] = df[col].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S') if x != '' else '')
    
    logging.info('Importing data from: %s' % (file_path))

    listings = pd.read_csv(os.path.join(file_path, 'listings.csv'))
    isinstance(listings, pd.DataFrame)
    users = pd.read_csv(os.path.join(file_path, 'users.csv'))
    isinstance(users, pd.DataFrame)
    contacts = pd.read_csv(os.path.join(file_path, 'contacts.csv'))
    isinstance(contacts, pd.DataFrame)
    
    n_samples, n_features = listings.shape
    logging.info('Listing    # of Observations: %s' % (n_samples))
    logging.info('Listing    # of Features: %s' % (n_features))
    n_samples, n_features = users.shape
    logging.info('Users    # of Observations: %s' % (n_samples))
    logging.info('Users    # of Features: %s' % (n_features))
    n_samples, n_features = contacts.shape
    logging.info('Contacts    # of Observations: %s' % (n_samples))
    logging.info('Contacts    # of Features: %s' % (n_features))

    #Convert timestamps to datetime in Contacts
    logging.info('Convert timestamps to datetime in Contacts table')
    for col_ts in ['ts_interaction_first','ts_reply_at_first','ts_accepted_at_first','ts_booking_at']:
        convert_to_datetime(contacts, col_ts)
    contacts['ds_checkin_first'] = contacts['ds_checkin_first'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    contacts['ds_checkout_first'] = contacts['ds_checkout_first'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))  
    
    #Length of time a host takes to reply to an inquiry (minutes)
    logging.info('Adding Feature: reply_inq_mins: Length of time(minutes) host replies to inquiry')
    contacts['reply_inq_mins'] = (contacts['ts_reply_at_first']-contacts['ts_interaction_first']).apply(lambda x: x.total_seconds())/60

    #Length of time for host to accept an inquiry (minutes)
    logging.info('Adding Feature: accept_inq_mins: Length of time(minutes) host takes to accept an inquiry')
    contacts['accept_inq_mins'] = (contacts['ts_accepted_at_first']-contacts['ts_interaction_first']).apply(lambda x: x.total_seconds())/60

    #Length of time for host to accept booking from inquiry(minutes)
    logging.info('Adding Feature: book_inq_mins: Length of time(minutes) to confirm booking from inquiry')
    contacts['book_inq_mins'] = (contacts['ts_booking_at']-contacts['ts_interaction_first']).apply(lambda x: x.total_seconds())/60

    #identifier that the contact lead to a booking (0=failed, 1=succeeded)
    logging.info('Adding Feature: booked: User successfully booked an Airbnb')
    bookings = []
    for book in contacts['ts_booking_at']:
        if book is pd.Timestamp('NaT'):
            bookings.append(0)
        else:
            bookings.append(1)
    contacts['booked'] = bookings

    #Length of stay
    logging.info('Adding Feature: num_nights: Number of nights requested')
    contacts['num_nights'] = (contacts['ds_checkout_first']-contacts['ds_checkin_first']).apply(lambda x: x.days-1)

    #length of time between check-in date and inquiry date
    logging.info('Adding Feature: inq_to_checkin: Length of time(days) between inquiry and check-in date')
    contacts['inq_to_checkin'] = (contacts['ds_checkin_first']-contacts['ts_interaction_first']).apply(lambda x: x.days)

    #get day of week for check in date (Mon=0, Tues=1 ... Sun=6)
    logging.info('Adding Feature: checkin_day_of_week: Check-in Day of Week (Mon=0, Tues=1...Sun=6)')
    contacts['checkin_day_of_week'] = contacts['ds_checkin_first'].apply(lambda x: x.weekday())
    
    
    #Merge datasets
    logging.info('Merge Users and Contacts Table')
    df = contacts.merge(
        users,
        right_on='id_user_anon',
        left_on='id_host_anon',
        how='left').merge(
                users,
                right_on='id_user_anon',
                left_on='id_guest_anon',
                how='left')
    
    logging.info('Merge Listings and Merged Table')
    df = df.merge(listings,on='id_listing_anon',how='left')
    
    #Clean column names
    logging.info('Clean columns')
    del df['id_user_anon_x']
    del df['id_user_anon_y']
    df.columns = ['id_guest_anon', 'id_host_anon', 'id_listing_anon',
           'ts_interaction_first', 'ts_reply_at_first', 'ts_accepted_at_first',
           'ts_booking_at', 'ds_checkin_first', 'ds_checkout_first', 'm_guests',
           'm_interactions', 'm_first_message_length_in_characters',
           'contact_channel_first', 'guest_user_stage_first', 'reply_inq_mins',
           'accept_inq_mins', 'book_inq_mins', 'booked', 'num_nights',
           'inq_to_checkin', 'checkin_day_of_week',  'host_country',
           'host_words_in_user_profile', 'guest_country',
           'guest_words_in_user_profile', 'room_type', 'listing_neighborhood',
           'total_reviews']
    
    n_samples, n_features = df.shape
    logging.info('Complete Data    # of Observations: %s' % (n_samples))
    logging.info('Complete Data    # of Features: %s' % (n_features))
    
    #Check and drop duplicates
    logging.info('Number of duplicate rows: %s' % (df.duplicated().sum()))
    df.drop_duplicates(inplace=True)
    
    #Check number of null or NaT values in each column
    logging.info('Generating feature null dictionary')
    col_null = {}
    for col in df.columns:
        if df[col].dtype == np.dtype('<M8[ns]'):
            num_null = len([x for x in df[col] if x is pd.Timestamp('NaT')])
            if num_null > 0:
                logging.warn('%s null values found in column: %s' % (num_null, col))
                col_null[col] = num_null
        else:
            num_null = df[col].isnull().sum()
            if num_null > 0:
                logging.warn('%s null values found in column: %s' % (num_null, col))
                col_null[col] = num_null

    #Drop nulls if safe
    logging.info('Dropping Nulls if safe to (< 15)')
    for col in col_null.keys():
        if col_null[col] < 15:
            df = df[pd.notnull(df[col])]
            df = df.reset_index(drop=True)
    
    # Convert # of guests to int
    df['m_guests'] = df['m_guests'].astype(int)

    # Convert # of first message length to int
    df['m_first_message_length_in_characters'] = df['m_first_message_length_in_characters'].astype(int)

    # Convert # of total reviews to int
    df['total_reviews'] = df['total_reviews'].astype(int)
    
    #Remove rows where total_reviews < 0: Check with product team to see why this is occuring
    logging.info('Removing rows where total_reviews < 0')
    df = df[df['total_reviews'] >= 0]
    df = df.reset_index(drop=True)
    
    #create column where host replied to an inquiry
    logging.info('Creating column where host replied to an inquiry (0,1)')
    df['host_replied'] = np.where(df['reply_inq_mins'].isnull(),0,1)
    
    return df
    
    
    
    
def eda_plots(df):
    """
    *Return: Plots of key metrics used to monitor over time
            Univariate and Distribution plots of important features
    data: DataFrame of cleaned+enhanced dataset
    """
    #graph the percent of all airbnbs booked on any given day
    def graph_booking(data,unique_date,dates_booked):
        bookings_per_day = pd.concat([pd.DataFrame(dates_booked.keys()),pd.DataFrame(dates_booked.values())],axis=1)
        bookings_per_day.columns = ['date','day_booked']
        bookings_per_day.sort_values('date',inplace=True)
        bookings_per_day = unique_date.merge(bookings_per_day,how='left').fillna(0)

        logging.info('Number of unique listings: %s' % len(set(data['id_listing_anon'])))
        total_listings = len(set(data['id_listing_anon']))

        percent_day_booked = pd.DataFrame(bookings_per_day['day_booked']/total_listings*100)
        percent_day_booked['date'] = unique_date
        percent_day_booked = percent_day_booked.set_index('date')

        #Plot             
        plt.figure(figsize=(20,6))
        plt.xticks(rotation=45)
        plt.plot(percent_day_booked)
        plt.xlabel('date',fontsize=15)
        plt.ylabel('% Listings Booked',fontsize=15)
        plt.title('% of Total Listings Booked by Day')
        plt.savefig('./data/percent_of_total_listings_booked.png', bbox_inches='tight')
        plt.close()
        plt.show()
    
    def graph_inquiry_bookings(data, col, unique_date, metric = None, cohort = None):
    
        def plot(user_inq,metric,cohort=None):   
            plt.figure(figsize=(20,6))
            plt.xticks(rotation=45)
            plt.plot(user_inq)
            plt.xlabel('date',fontsize=15)
            plt.ylabel('# of %s' % metric,fontsize=15)
            if cohort is not None:
                plt.title('# %s per Day %s' % (metric,str(cohort)))
                plt.savefig('./data/%s_%s.png' % (metric,str(cohort)), bbox_inches='tight')
                plt.close()
            else:
                plt.title('# %s per Day' % metric)
                print ('./data/%s.png' % metric)
                plt.savefig('./data/%s.png' % metric, bbox_inches='tight')
                plt.close()
            plt.show()

        def process(user_inq,col,unique_date):
            user_inq.reset_index(drop=True,inplace=True)
            user_inq = user_inq.apply(lambda x: pd.Timestamp(x.date()))
            user_inq = pd.DataFrame(user_inq.value_counts()).reset_index()
            user_inq.columns = ['date',col]
            user_inq = unique_date.merge(user_inq,how='left')
            user_inq.fillna(0,inplace=True)
            user_inq = user_inq.set_index('date')
            return user_inq

        if cohort is not None:
            for c in np.unique(data[cohort]):
                user_inq = data[data[cohort] == c][col]
                user_inq = process(user_inq,col,unique_date)
                plot(user_inq,metric,cohort = c)
        else:
            user_inq = data[col]
            user_inq = process(user_inq,col,unique_date)
            plot(user_inq,metric)
    
    def graph_cumsum_users(data, col, unique_date, metric = None, cohort = None):
    
        def plot(users,metric,cohort=None):   
            plt.figure(figsize=(20,6))
            plt.xticks(rotation=45)
            plt.plot(users)
            plt.xlabel('date',fontsize=15)
            plt.ylabel('# of %s' % metric,fontsize=15)
            if cohort is not None:
                plt.title('Cumulative Sum of %s per Day %s' % (metric,str(cohort)))
                plt.savefig('./data/cumsum__%s_%s.png' % (metric,str(cohort)), bbox_inches='tight')
                plt.close()
            else:
                plt.title('Cumulative Sum of %s per Day' % metric)
                plt.savefig('./data/cumsum__%s.png' % metric, bbox_inches='tight')
                plt.close()
            plt.show()

        def process(users_df,col,unique_date):
            users_df.reset_index(drop=True,inplace=True)
            users_df = users_df.groupby('id_guest_anon')[col].min()
            users_df = users_df.apply(lambda x: pd.Timestamp(x.date()))
            users_df = pd.DataFrame(users_df.value_counts()).reset_index()
            users_df.columns = ['date',col]
            users_df = unique_date.merge(users_df,how='left')
            users_df.fillna(0,inplace=True)
            users_df = users_df.set_index('date')
            users_df[col] = users_df[col].cumsum()
            return users_df

        if cohort is not None:
            for c in np.unique(data[cohort]):
                users_df = data[data[cohort] == c][['id_guest_anon',col]]
                users_df = process(users_df,col,unique_date)
                plot(users_df,metric,cohort = c)
        else:
            users_df = data[['id_guest_anon',col]]
            users_df = process(users_df,col,unique_date)
            plot(users_df,metric)
    
    logging.info('Starting graph generations...')
    
    #Create unique date list
    logging.info('Creating base unique dates')
    min_checkin = df['ds_checkin_first'].min()
    max_checkin = df['ds_checkin_first'].max()
    unique_date = pd.date_range(start=min_checkin,end=max_checkin, freq='D')
    unique_date = pd.DataFrame(unique_date,columns=['date'])
    
    #Get count of airbnbs booked on any given day
    logging.info('Generating graph for key metric: Ratio of airbnbs booked to total potential listings on any given day.')
    dates_booked = {}
    #Only getting count of booked airbnbs
    booked = df[df['booked']==1]
    for c_in,c_out in zip(booked['ds_checkin_first'],booked['ds_checkout_first']):
        booked_date_range = pd.date_range(start=c_in,end=c_out, freq='D')
        #A user's checkout date availability is not restricted from another user to book starting on that date
        for dt in booked_date_range[:-1]:
            if dt not in dates_booked:
                dates_booked[dt] = 1
            else:
                dates_booked[dt] += 1
    
    #Generate % total listings booked by day
    graph_booking(df,unique_date,dates_booked)
    #Generate # Inquiries per day
    graph_inquiry_bookings(df,'ts_interaction_first',unique_date, metric = 'Inquiries', cohort = 'guest_user_stage_first')
    #Generate # Bookings per day
    graph_inquiry_bookings(df,'ts_booking_at',unique_date,metric = 'Bookings',  cohort = 'guest_user_stage_first')
    #Generate cumsum users first inquiry
    graph_cumsum_users(df,'ts_interaction_first',unique_date,metric = 'Inquiries',  cohort = 'guest_user_stage_first')
    #Generate cumsum users first booking
    graph_cumsum_users(df,'ts_booking_at',unique_date,metric = 'Bookings',  cohort = 'guest_user_stage_first')
    
    
def dist_uni_plot(data,col,target, pdf, cohort=None,xlim = None):
    
    def run():
        if cohort is not None:
            cohorts = set(data[cohort])
            for c in cohorts:
                data_cohort = data[data[cohort] == c]
                plot(data_cohort,col,target,cohort = c)
        else:
            plot(data,col,target,cohort=None)

    def plot(data,col,target,cohort=None):
        data = data[data[col].notnull()]
        grouped = data[[col,target]].groupby(col).mean().reset_index()
        hist_kws = {'histtype':'bar', 'edgecolor':'black', 'alpha':0.2}
        fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(18,6))
        sns.distplot(data[data[target]==0][col],label='Booked 0',
                    ax=ax[0], hist_kws = hist_kws)
        sns.distplot(data[data[target]==1][col],label='Booked 1',
                    ax=ax[0], hist_kws = hist_kws)

        if cohort is not None:
            ax[0].set_title('Count Plot of {}'.format(col) + '  {}'.format(cohort), fontsize=16)
        else:
            ax[0].set_title('Count Plot of {}'.format(col), fontsize=16)
        ax[0].legend()
        ax[1].plot(grouped[col],grouped[target],'.-')
        if cohort is not None:
            ax[1].set_title('Mean Conversion Rate vs. {}'.format(col) + '  {}'.format(cohort), fontsize=16)
        else:
            ax[1].set_title('Mean Conversion Rate vs. {}'.format(col), fontsize=16)
        ax[1].set_xlabel('{}'.format(col))
        plt.xlim([0,xlim])
        ax[1].set_ylabel('Mean Convertion Rate')
        ax[1].grid(True)
        plt.tight_layout()
        pdf.savefig( fig )
        
        
    run()
    


def prep_model_df(df):
    """
    Create prepared model for running machine learning algorithm
    *Return: Altered DataFrame
    """
    #Begin prepping dataframe
    logging.info('Beginning preparation to feature engineer for model purposes')
    #Excluding contact_channel_first since it correlates too high with bookings target
    feature_cols = [
            'm_guests',
            'm_interactions', 
            'm_first_message_length_in_characters',
            #'contact_channel_first', 
            'guest_user_stage_first',
            'booked', 
            'num_nights',
            'inq_to_checkin',
            'checkin_day_of_week',
            'host_country',
            'host_words_in_user_profile',
            'guest_country',
            'guest_words_in_user_profile',
            'room_type',
            'listing_neighborhood',
            'total_reviews',
            'host_replied']
    
    prep_df = df[feature_cols]
    
    cat_feature = [
    #'contact_channel_first',
    'guest_user_stage_first',
    'host_country',
    'checkin_day_of_week',
    'guest_country',
    'room_type',
    'listing_neighborhood',
    'host_replied']
    
    df_train = prep_df[cat_feature]
    tar_feature = 'booked'
    
    #Encoder for categorical variables
    def encoder(data, categorical_feature):
        enc = OneHotEncoder()
        array_sparse = enc.fit_transform(np.array(data[categorical_feature]).reshape(-1,1))
        array = array_sparse.toarray()
        column_names = [x.replace("x0_",'{}_'.format(categorical_feature)).replace(".0","") for x in enc.get_feature_names()]
        return array_sparse, array, column_names
    
    logging.info('One Hot Encoding Cat variables...')
    for cat in cat_feature:
        array_sparse, array, column_names = encoder(df_train, cat)
        for i,col in enumerate(column_names[:-1]):
            df_train[col] = array[:,i]
      
    #Drop categorical features after one hot encoding
    prep_df.drop(cat_feature, axis=1, inplace=True)
    df_train.drop(cat_feature, axis=1, inplace=True)
    
    model_df = pd.concat([prep_df,df_train],axis=1)
    
    return model_df
    
    
def run_model(model_df, test_size):
    """
    Run Random Forest to predict a user's likelihood to book an airbnb.

    *Return: AUC of model
             Feature Importance sorted
    model_df: DataFrame of fix data for model
    test_size: float between 0.0 - 1.0
    """
    X_train, X_test, y_train, y_test = train_test_split(
    model_df.drop('booked',axis=1), model_df['booked'], test_size=test_size, random_state=42)

    # Create, output AUC
    clf = RandomForestClassifier(n_jobs=2, random_state=0)
    clf.fit(X_train, y_train)
    
    auc = roc_auc_score(y_true=y_test, y_score=clf.predict(X_test))
    print (auc)
    
    feat_import = {}
    for i,x in zip(model_df.drop('booked', axis=1).columns,clf.feature_importances_):
        feat_import[i] = x
    
    sorted_x = sorted(feat_import.items(), key=operator.itemgetter(1),reverse=True)
    
    return auc, sorted_x


# Main section
if __name__ == '__main__':
    main()
