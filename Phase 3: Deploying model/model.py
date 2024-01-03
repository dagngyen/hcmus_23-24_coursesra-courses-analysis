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