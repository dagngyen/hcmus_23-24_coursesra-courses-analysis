import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import SGDRegressor

data = pd.read_csv('Data/processed_data.csv')

# handle outlier for duration
Q1 = data['duration'].quantile(0.25)
Q3 = data['duration'].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5*IQR
upper_bound = Q3 + 1.5*IQR

data.loc[data['duration'] > upper_bound, 'duration'] = int(upper_bound)
data.loc[data['duration'] < lower_bound, 'duration'] = max(int(lower_bound), 0)


# # handle for enrollment
Q1 = data['enrollment'].quantile(0.25)
Q3 = data['enrollment'].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5*IQR
upper_bound = Q3 + 1.5*IQR

data.loc[data['enrollment'] > upper_bound, 'enrollment'] = int(upper_bound)
data.loc[data['enrollment'] < lower_bound, 'enrollment'] = max(int(lower_bound), 0)

# define numerical and categorical features
numerical_features = ['enrollment', 'duration', 'instructor_rate']
categorical_features = ['general', 'specify', 'language', 'level', 'offered by']

# select specific features
selected_features = ['enrollment', 'duration', 'instructor_rate', 'general', 'specify', 'language', 'level', 'offered by']
# handle missing values
data.dropna(inplace=True)

# select features and target variable
X = data[selected_features]
y = data['rating']

# split the data into training, testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# create transformers for numerical and categorical features
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(sparse=False, drop='first', handle_unknown='ignore'))
])

# combine transformers into a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# defime best param we got above
param = {'alpha': 0.001,
 'learning_rate': 'adaptive',
 'max_iter': 500}

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', SGDRegressor(**param))
])

model.fit(X_train, y_train)

# saving model
pickle.dump(model, open('Phase 3: Deploying model/model.pkl','wb'))