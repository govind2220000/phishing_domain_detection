#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
import seaborn as sns


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as ss
from collections import Counter
import math
from scipy import stats


# In[3]:


data = pd.read_csv('dataset_full(1).csv')


# In[4]:


data


# In[7]:


X=data.iloc[:,:-1]
X


# In[8]:


y=data['phishing']
y


# In[10]:


#importing the ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt


# In[11]:


model = ExtraTreesClassifier()


# In[12]:


model.fit(X,y)

print(model.feature_importances_)


# In[14]:


feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()


# # Filter Feature Selection - Pearson Correlation

# In[15]:


feature_name = list(X.columns)
# no of maximum features we need to select
num_feats=25


# In[17]:


cor_list = []
for i in feature_name:
         cor= np.corrcoef(X[i],y)[0,1]
         cor_list.append(cor)
cor_list


# In[18]:


cor_list =[0 if np.isnan(i) else i for i in cor_list]
cor_list


# In[19]:


feature_set= X.iloc[:,np.argsort(np.abs(cor_list))].columns.tolist()
feature_set


# In[21]:


cor_feature = X.iloc[:,np.argsort(np.abs(cor_list))[-10:]].columns.tolist()
cor_feature


# In[22]:


def cor_selector(X, y,num_feats):
    # Your code goes here (Multiple lines
    cor_list = []
    for i in feature_name:
        cor= np.corrcoef(X[i],y)[0,1]
        cor_list.append(cor)
        
    cor_list =[0 if np.isnan(i) else i for i in cor_list]
    cor_feature = X.iloc[:,np.argsort(np.abs(cor_list))[-num_feats:]].columns.tolist()
    cor_support =[True if i in feature_set else False for i in feature_name] 
    
    # Your code ends here
    return cor_support, cor_feature


# In[23]:


cor_support, cor_feature = cor_selector(X, y,num_feats)
print(str(len(cor_feature)), 'selected features')


# In[24]:


cor_feature


# # Embedded Selection - Lasso: SelectFromModel

# In[28]:


from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler


# In[29]:


lr= LogisticRegression(penalty='l1', solver='liblinear', max_iter=50000)
embedded_lr_selector = SelectFromModel(lr, max_features=30)
embedded_lr_selector = embedded_lr_selector.fit(X,y)
embedded_lr_support = embedded_lr_selector.get_support()
embedded_lr_support
embedded_lr_features = X.loc[:, embedded_lr_support].columns.tolist()
embedded_lr_features


# In[30]:


def embedded_log_reg_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    embedded_lr_support = embedded_lr_selector.get_support()
    embedded_lr_features = X.loc[:, embedded_lr_support].columns.tolist()

    # Your code ends here
    return embedded_lr_support, embedded_lr_features


# In[31]:


embedded_lr_support, embedded_lr_features = embedded_log_reg_selector(X, y, num_feats)
print(str(len(embedded_lr_features)), 'selected features')


# In[32]:


embedded_lr_features


# # Tree based(Random Forest): SelectFromModel

# In[33]:


from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier


# In[34]:


rf = RandomForestClassifier(n_estimators=100)


# In[35]:


embedded_rf_selector = SelectFromModel(rf, max_features=24)
embedded_rf_selector


# In[36]:


embedded_rf_selector1 = embedded_rf_selector.fit(X,y)
embedded_rf_selector1


# In[37]:


embedded_rf_support = embedded_rf_selector1.get_support()
embedded_rf_support


# In[38]:


embedded_rf_features = X.loc[:, embedded_rf_support].columns.tolist()
embedded_rf_features


# In[39]:


def embedded_rf_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    
    embedded_rf_support = embedded_rf_selector1.get_support()

    embedded_rf_feature = X.loc[:, embedded_rf_support].columns.tolist()
    # Your code ends here
    return embedded_rf_support, embedded_rf_feature


# In[40]:


embedded_rf_support, embedded_rf_feature = embedded_rf_selector(X, y, num_feats)
print(str(len(embedded_rf_feature)), 'selected features')


# In[41]:


embedded_rf_features


# # Putting all of it together: AutoFeatureSelector

# In[48]:


pd.set_option('display.max_rows', None)
# put all selection together
feature_selection_df = pd.DataFrame({'Feature':feature_name, 'Pearson':cor_support,'Logistics':embedded_lr_support,
                                    'Random Forest':embedded_rf_support})
# count the selected times for each feature
feature_selection_df['Total'] = np.sum(feature_selection_df, axis=1)
# display the top 100
feature_selection_df = feature_selection_df.sort_values(['Total','Feature'] , ascending=False)
feature_selection_df.index = range(1, len(feature_selection_df)+1)
df_new=feature_selection_df.head(num_feats)


# In[49]:


df_new


# In[50]:



# Converting a specific Dataframe 
# column to list using Series.tolist()
Name_list = df_new["Feature"].tolist()
  
print("Converting name to list:")
  
# displaying list
Name_list


# In[55]:


data_new =data.filter(['qty_underline_file',
 'qty_tilde_file',
 'qty_space_file',
 'qty_hyphen_directory',
 'qty_exclamation_file',
 'qty_dot_directory',
 'url_shortened',
 'url_google_index',
 'ttl_hostname',
 'tls_ssl_certificate',
 'tld_present_params',
 'time_response',
 'time_domain_expiration',
 'time_domain_activation',
 'qty_underline_domain',
 'qty_tld_url',
 'qty_slash_url',
 'qty_questionmark_url',
 'qty_questionmark_directory',
 'qty_plus_url',
 'qty_plus_directory',
 'qty_percent_file',
 'qty_params',
 'qty_hyphen_domain',
 'qty_hashtag_file','phishing'])


# In[56]:


data_new.to_csv('phising_data_new.csv', index=False)


# In[62]:


from sklearn.model_selection import train_test_split

y=data_new["phishing"].copy()
X=data_new.drop('phishing',axis=1).copy()
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.8,shuffle=True,random_state=123)


# In[57]:


data_new['phishing'].value_counts()


# In[58]:


sns.countplot(data_new['phishing'])


# Sampling technique is used to balance the data

# In[63]:


from imblearn.over_sampling import RandomOverSampler
os=RandomOverSampler(sampling_strategy='all')


# In[64]:


X_train_res,y_train_res=os.fit_resample(X_train,y_train)


# In[ ]:





# In[65]:


from collections import Counter
print('Resampled dataset shape %s' % Counter(y_train_res))


# In[82]:


ax=  y_train_res.value_counts().plot.pie(autopct='%.2f')


# In[66]:


import joblib as jb


# In[69]:


RSEED=50
rf = RandomForestClassifier(random_state= RSEED)
from pprint import pprint
# Look at parameters used by our current forest

#print('Parameters currently in use:\n')
pprint(rf.get_params())


# In[70]:


from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 200, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']+list(np.arange(0.5, 1, 0.1))
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(3, 20, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
pprint(random_grid)


# In[71]:


rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 10, cv = 3, verbose=2, random_state=RSEED,n_jobs = -1)


# In[72]:


rf_random.fit(X_train_res, y_train_res)


# In[73]:


print(rf_random.best_params_)


# In[74]:


random_cv=rf_random.best_estimator_
random_cv


# In[75]:


y_pred1 = random_cv.predict(X_test)
y_pred1


# In[76]:


print(random_cv.score(X_test,y_test))
print(random_cv.score(X_train,y_train))


# In[78]:


from sklearn.metrics import confusion_matrix


# In[79]:


confusion_matrix(y_test,y_pred1)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




