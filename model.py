import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load and preprocess the data
data = pd.read_csv('c:/Users/ansha/Desktop/predictive_lead_data.csv')

def preprocess_data(data):
    data = data.rename(columns={'Lead Origin':'Origin','Lead Source':'Source','Total Time Spent on Website':'Time Spend on Site',
                            'Page Views Per Visit':'Page Viewed','Last Activity':'LA','How did you hear about X Education':'Hear about',
                            'What is your current occupation':'Job','What matters most to you in choosing a course':'Purpose',
                            'Digital Advertisement':'Digital Ad','Through Recommendations':'Recommendations',
                            'Receive More Updates About Our Courses':'Subscribe',
                            'Update me on Supply Chain Content':'Supply Chain','Get updates on DM Content':'DM',
                            'Asymmetrique Activity Index':'Activity Index',
                            'Asymmetrique Profile Index':'Profile Index','Asymmetrique Activity Score':'Activity Score',
                            'Asymmetrique Profile Score':'Profile Score','I agree to pay the amount through cheque':'Pay by Cheque',
                            'A free copy of Mastering The Interview':'Free copy','X Education Forums': 'Forum_ad'})
   
    data.replace('Select', np.nan, inplace=True)
    data.drop(['Prospect ID', 'Lead Number'], inplace=True, axis=1)
    col_to_drop = (data.isnull().mean() * 100).round(2)[(data.isnull().mean() * 100).round(2) > 30].index
    data.drop(columns=col_to_drop, inplace=True)

    for col in ['TotalVisits', 'Page Viewed']:
        data[col] = data[col].fillna(data[col].median())
    for col in ['LA','Source']:
        data[col] = data[col].fillna(data[col].mode()[0])
    data['Job'].fillna('Unemployed', inplace=True)
    data['Purpose'] = data['Purpose'].fillna(data['Purpose'].mode()[0])
    data['Country'] = data['Country'].fillna(data['Country'].mode()[0])

    mapping_dict = {
        'google': 'Search Source',
        'Google': 'Search Source',
        'bing': 'Search Source',
        'Organic Search': 'Search Source',
        'Olark Chat': 'Online Chat',
        'Reference': 'Referral Sites',
        'Facebook': 'Social Media',
        'youtubechannel': 'Social Media',
        'welearnblog_Home': 'Welingak Website',
        'WeLearn': 'Welingak Website',
        'blog': 'Others',
        'Pay per Click Ads': 'Others',
        'Click2call': 'Others',
        'Press_Release': 'Others',
        'NC_EDM': 'Others',
        'testone': 'Others',
        'Live Chat': 'Online Chat'
    }
    data['Source'] = data['Source'].map(mapping_dict).fillna(data['Source'])

    activity_counts = data.groupby('LA')['LA'].transform('count')
    data['LA'] = data['LA'].where(activity_counts >= 30, 'Others')

    activity_counts = data.groupby('Last Notable Activity')['Last Notable Activity'].transform('count')
    data['Last Notable Activity'] = data['Last Notable Activity'].where(activity_counts >= 30, 'Others')

    data.loc[data['Country'].isin(['India', 'Singapore', 'Hong Kong', 'Philippines', 'Asia/Pacific Region', 'Bangladesh',
                                   'China', 'Sri Lanka', 'Malaysia', 'Vietnam', 'Indonesia']), 'Country'] = 'Asia'
    data.loc[data['Country'].isin(['United States', 'Canada']), 'Country'] = 'America'
    data.loc[data['Country'].isin(['United Arab Emirates', 'Saudi Arabia', 'Qatar', 'Bahrain', 'Oman', 'Kuwait']), 'Country'] = 'Middle_East'
    data.loc[data['Country'].isin(['United Kingdom', 'France', 'Germany', 'Sweden', 'Italy', 'Netherlands', 'Belgium',
                                    'Switzerland', 'Denmark', 'Russia']), 'Country'] = 'Europe'
    data.loc[data['Country'] == 'Australia', 'Country'] = 'Australia'
    data.loc[data['Country'].isin(['South Africa', 'Nigeria', 'Uganda', 'Ghana', 'Kenya', 'Tanzania', 'Liberia']), 'Country'] = 'Africa'

    data.rename(columns={'Country': 'Region'}, inplace=True)

    data.drop(['Magazine', 'Search', 'Newspaper Article', 'Forum_ad', 'Newspaper', 'Digital Ad', 'Purpose',
               'Recommendations', 'Subscribe', 'Supply Chain', 'DM', 'Pay by Cheque', 'Do Not Call'], axis=1, inplace=True)

    q1 = data['TotalVisits'].quantile(0.25)
    q3 = data['TotalVisits'].quantile(0.75)
    iqr = q3 - q1
    LL = q1 - (1.5 * iqr)
    UL = q3 + (1.5 * iqr)
    outlier = data[(data['TotalVisits'] < LL) | (data['TotalVisits'] > UL)].index
    data.drop(outlier, inplace=True)

    q1 = data['Page Viewed'].quantile(0.25)
    q3 = data['Page Viewed'].quantile(0.75)
    iqr = q3 - q1
    LL = q1 - (1.5 * iqr)
    UL = q3 + (1.5 * iqr)
    outl = data[(data['Page Viewed'] < LL) | (data['Page Viewed'] > UL)].index
    data.drop(outl, inplace=True)

    data.drop('Last Notable Activity', inplace=True, axis=1)

    le = LabelEncoder()
    data['Do Not Email'] = le.fit_transform(data['Do Not Email'])
    data['Free copy'] = le.fit_transform(data['Free copy'])

    data = pd.get_dummies(data)  
    data = data.astype(int)  
    data.drop(columns=['Source_Others','LA_Others','Region_unknown','Job_Other'],inplace=True)     

    return data

# Define features and target
X = data.drop('Converted', axis=1)
y = data['Converted']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Save model and scaler
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# def predict(model, scaler, data):
#     data_scaled = scaler.transform([data])
#     return model.predict(data_scaled)


