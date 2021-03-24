"""
split train and val
"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=0)


"""
pipeline example
"""
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC())])
 # The pipeline can be used as any other estimator
 # and avoids leaking the test set into the train set
pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)


"""
missing values
"""
#for dataframe data
#drop
data.dropna()
# replace all NA's with 0
data.fillna(0)
# replace all NA's the value that comes directly after it in the same column, 
# then replace all the remaining na's with 0
data.fillna(method='bfill', axis=0).fillna(0)
#with sklearn
from sklearn.impute import SimpleImputer
preprocessor = SimpleImputer(missing_values=np.nan, strategy='mean')
preprocessor.fit(X)
X_prep = preprocessor.transform(X)
#learning based on other columns
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
preprocessor = IterativeImputer(max_iter=10, random_state=0)
#nearest neighbour
from sklearn.impute import KNNImputer
preprocessor = KNNImputer(n_neighbors=5, weights="distance")
preprocessor.fit(X)
X_prep = preprocessor.transform(X)