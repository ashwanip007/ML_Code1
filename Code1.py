import os
import tarfile
from six.moves import urllib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, LabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
import hashlib

DOWNLOAD_ROOT = "https://github.com/ageron/handson-ml/tree/master/"
HOUSING_PATH = "datasets/housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):           # Pandas
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


# fetch_housing_data()                                    Not called (Error in fetching details)
housing = load_housing_data()
print(housing.head())                                     # Print is necessary
print(housing.info())
print(housing["ocean_proximity"].value_counts())
# print(housing.describe()) : Return statistical information of the attribute (Count, mean, std, min, max, 25%)


housing.hist(bins=50, figsize=(20, 15))                      # Matplotlib
plt.show()

"""
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]
    
    >>> train_set, test_set = split_train_test(housing, 0.2)
    >>> print(len(train_set), "train +", len(test_set), "test"
    16512 train + 4128 test
    #This works but not perfect, it will generate different test set over time MLA will get to see whole dataset
    #Set RNG seed (eg np.random.seed(42)) before calling np.random.permutation(), (Generate same shuffled indices)
    #Again it will break if you fetch an updated dataset

# import hashlib
def test_set_check(identifier, test_ratio, hash):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio

def split_train_test_by_id(data, test_ratio, id_column, hash = hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    return data.loc[~in_test_set], data.loc[in_test_set]
# Unfortunately, it doesn't have identifier column; so use row index as id

housing_with_id = housing.reset_index()                                         # adds an 'index' column
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")
# Make sure new data appended at end. Else...

housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")
# Stable for a few million year. Else Scikit-Learn provide split_train_test
"""
# from sklearn.model_selection train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
# Same as split_train_test defined earlier with direct random_state parameter and you can pass multiple dataset with
# identical number of rows, and it will split them on the same indices

housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)
# ceil to have discrete category (also merging all categories greater than 5 into category 5)

# This one is pretty straight forward and different from text
strat_train_set, strat_test_set = train_test_split(housing, test_size=0.2, random_state=42, stratify=housing["income_cat"])

print(housing["income_cat"].value_counts() / len(housing))

"""
print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))
print(strat_test_set.head())
Test set generated using stratified sampling has income category proportion identical to full dataset
Removing the income_cat to make it original
"""

for sets in (strat_train_set, strat_test_set):
    sets.drop(["income_cat"], axis=1, inplace=True)
# Making copy without harming thr training set

housing = strat_train_set.copy()

"""
U can remove alpha but it is much easier to visualize to see high density,
option s (radius of the circle), option c (color represent the price)
option cmap (Used predefined color map jet, which ranges from blue to red (high))
"""
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, s=housing["population"]/100, label="population",
             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)
# plt.legend()   Check why this isn't showing
plt.show()
# Computing the standard correlation coefficient b/w every pair of attribute
corr_matrix = housing.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))

# Another way to check correlation b/w attributes use pandas scatter_matrix function 11^2 too high
attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
# scatter_matrix(housing[attributes], figsize=(12,8)) this will show all 16, main diagonal histogram
housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
plt.show()

# Trying different meaningful combinations of the attribute to see useful result
housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["population_per_household"] = housing["population"] / housing["households"]

corr_matrix = housing.corr()
# The bedrooms_per_room is much more correlated than the total number of rooms or bedrooms.
print(corr_matrix["median_house_value"].sort_values(ascending=False))

# Prepare the data for ML Algorithm (Instead of doing manually always write function)
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()
# Separated predictors and labels as transformation is not common(Drop does not effect the set)--(next)-->Data Cleaning

""""
Remember, total_bedrooms attribute has some missing value, Option: Remove corresponding district/whole attribute/Set the
value to some value(0/mean/median)
housing.dropna(subset=["total_bedrooms"])    #1
housing.drop("total_bedrooms", axis=1)       #2
median=housing["total_bedrooms"].median()
housing["total_bedrooms"].fillna(median)     #3
//Don't forget to save median value
Scikit-Learn provides a handy class SimpleImputer, Different from text it has been changed.
"""
imputer = SimpleImputer(strategy="median")
# Removing non numerical attribute
housing_num = housing.drop("ocean_proximity", axis=1)
# Fit imputer instance to training data using fit
imputer.fit(housing_num)
"""Computed the median value of each attribute and stored in statistics_ instance variable
print(imputer.statistics_)
print(housing_num.median().values)
Check both values are same
We can use this trained imputer to transform the training set by the learned medians
"""
X = imputer.transform(housing_num)
# This result is plain Numpy 2D array putting back in Pandas DataFrame
housing_tr = pd.DataFrame(X, columns=housing_num.columns)

# MLA prefers to work with numbers, we have left ocean_proximity beacause it was a text.

encoder = LabelEncoder()
housing_cat = housing["ocean_proximity"]
housing_cat_encoded = encoder.fit_transform(housing_cat)
print(housing_cat_encoded)
print(encoder.classes_)
"""
Issue MLA will assume two nearby values are similar than two distant value.
So ScikitLearn provide OneHotEncoder
"""
encoder = OneHotEncoder()
# Note fit_transform expects a 2D array, but housing_cat_encoded is 1D array
housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1, 1))
# print(housing_cat_1hot) ; Result will not be as text but it will be a SciPy sparse matrix, instead of NumPy array, So
print(housing_cat_1hot.toarray())

"""
We can apply both the transformation in one shot (category to integer category then to one-hot-vector)
from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
housing_cat_1hot = encoder.fit_transform(housing_cat)
print(housing_cat_1hot)
# Notice this return dense NumPy array by default
"""
# Custom Transformers
# from sklearn.base import BaseEstimator, TransformerMixin
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6
# To be used for slicing


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):                 # No *args or kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self                                                 # Nothing else to do

    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)

"""
np.c_ is used for concatenating the array along the 1st axis. add_bedrooms_per_room set to true by default (sensible
default). Generally add hyper parameter for which you are not 100% sure. Automate it to find great combination.

### Feature Scaling (Two Common Way's)###
Min-Max: Subtracting min value dividing by max minus min (ScikitLearn's MinMaxScaler with feature_range).
Standardization: First subtracts the mean value (Standardized values always have 0 mean) then divide by hte variance
so that the resulting variance have the unit variance. (Less affected by outliers) (StanderedScaler)

### Transformation Pipelines ###
All but the last estimator must be transformers (i.e., they must have a fit_transform() method).
When you call the pipeline’s fit() method, it calls fit_transform() sequentially on all transformers.
Passing output of each as parameter to next call, until it reaches final estimator, for it just calls fit() method.

num_pipeline = Pipeline([
            ('imputer', Imputer(strategy="median")),
            ('attribs_adder', CombinedAttributesAdder()),
            ('std_scaler', StandardScaler()),
        ])
housing_num_tr = num_pipeline.fit_transform(housing_num)

In this example, the last estimator is a StandardScaler, which is a transformer, so the pipeline has a transform()
method that applies all the transforms to the data in sequence. This pipeline is for numerical values, for categorical
value ? SKL provide FeatureUnion class for this. You give a list of transformers (Which can be entire transformers
pipeline), and when transform() method is called it runs each transformer's transform() method in parallel, waits for
their output, and then concatenates them and return the result.
"""


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values

# There is nothing in SKL to handle Pandas DataFrame so we need to write a simple custome transformer

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attribs)),
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler())
        ])

cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs)),
    ('label_binarizer', LabelBinarizer())
])

full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline)
])

# We can run the whole pipeline simply:
housing_prepared = full_pipeline.fit_transform(housing_num)
print(housing_prepared)

"""
class LabelBinarizerPipelineFriendly(LabelBinarizer):
    def fit(self, X, y=None):
        # this would allow us to fit the model based on the X input.
        super(LabelBinarizerPipelineFriendly, self).fit(X)
    def transform(self, X, y=None):
        return super(LabelBinarizerPipelineFriendly, self).transform(X)

    def fit_transform(self, X, y=None):
        return super(LabelBinarizerPipelineFriendly, self).fit(X).transform(X)
"""