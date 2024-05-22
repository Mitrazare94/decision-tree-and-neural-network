#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree

data = {
    'Patient ID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Sore Throat': ['Yes', 'No', 'Yes', 'Yes', 'No', 'No', 'No', 'Yes', 'No', 'Yes'],
    'Fever': ['Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'No', 'No', 'No', 'Yes'],
    'Swollen Glands': ['Yes', 'No', 'No', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No'],
    'Congestion': ['Yes', 'Yes', 'No', 'No', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'Yes'],
    'Headache': ['Yes', 'Yes', 'No', 'No', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes'],
    'Diagnosis': ['Strep throat', 'Allergy', 'Cold', 'Strep throat', 'Cold', 'Allergy', 'Strep throat', 'Allergy', 'Cold', 'Cold']
}

df = pd.DataFrame(data)

X = df[['Sore Throat', 'Fever', 'Swollen Glands', 'Congestion', 'Headache']]
y = df['Diagnosis']

le_X = LabelEncoder()
le_y = LabelEncoder()

X_encoded = X.apply(le_X.fit_transform)
y_encoded = le_y.fit_transform(y)

clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(X_encoded, y_encoded)

plt.figure(figsize=(20,10))
plot_tree(clf, feature_names=X.columns, class_names=le_y.classes_, filled=True)
plt.show()

new_samples = {
    'Patient ID': [11, 12, 13],
    'Sore Throat': ['No', 'Yes', 'No'],
    'Fever': ['No', 'Yes', 'No'],
    'Swollen Glands': ['Yes', 'No', 'No'],
    'Congestion': ['Yes', 'No', 'No'],
    'Headache': ['Yes', 'Yes', 'Yes']
}

new_df = pd.DataFrame(new_samples)

new_X_encoded = new_df[['Sore Throat', 'Fever', 'Swollen Glands', 'Congestion', 'Headache']].apply(le_X.transform)
predictions_encoded = clf.predict(new_X_encoded)
predictions = le_y.inverse_transform(predictions_encoded)

new_df['Diagnosis'] = predictions
new_df


# In[ ]:




