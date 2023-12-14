# %% [markdown]
# ### Naive Bayes

# %% [markdown]
# Of the Naive Bayes classifiers we are going to use Multinomial Naive Bayes, because we are dealing with discrete values (ie text sentiment to determine a rating).

# %%
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

# try some various hyperparameters
naive_bayes = MultinomialNB(alpha=0.1)

naive_bayes.fit(X_train, y_train)
print('Multinomial Naive Bayes Trained!')

test_model(naive_bayes)

# %% [markdown]
# ### Logistic Regression

# %%

# try some various hyperparameters
logistic_regression = LogisticRegression(
    max_iter=1000, n_jobs=-1, solver='saga')

logistic_regression.fit(X_train, y_train)
print('Logistic Regression Trained!')

test_model(logistic_regression)

# %% [markdown]
# ### Support Vector Machine

# %%

# try some various hyperparameters
svc = LinearSVC(max_iter=1000)

svc.fit(X_train, y_train)
print('SVC Trained!')

test_model(svc)
