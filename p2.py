import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import  accuracy_score
from sklearn.svm import SVC
import numpy as np

nba_stats = pd.read_csv(r'D:\Personal\Academics\UTA\3rd semester\Data Mining\P2\P2\nba2021.csv')#reading the csv file that is needed
nba_stats = nba_stats[nba_stats['MP']>=8]#filter data to only include rows with at least 8 min of play
nba_stats['BLK/TRB'] = nba_stats['BLK'] / nba_stats['TRB'] #feature engineering by adding a new column using existing coloumn
nba_stats= nba_stats.drop(columns=['Player','Tm','GS','G','eFG%','FGA','FG%','3P%','FT%']) #dropped rows as per a better accuracy using feature selection and manual selection
features = nba_stats.drop(columns=['Pos']) #created a dataframe for features
target = nba_stats['Pos'] #created a dataframe for target
target_to_numeric_dict = {"PG": 1,"SG": 2,"SF": 3,"PF": 4,"C": 5} #mapped targets to numerical as model was performing better
target = target.map(target_to_numeric_dict) #stored the numerical target
X_train, X_test, y_train, y_test = train_test_split(features, target, random_state=0, stratify=target, test_size=0.25) #used test train split with stratify as true
scaler = StandardScaler() #used standard scalar to scale the data
X_train = scaler.fit_transform(X_train) #fitting the scaled the data and coverting x_train to scaled version
X_test = scaler.transform(X_test) #coverting x_test to scale it
model = SVC(C=3,kernel='linear', max_iter=-1) #select svm model with kernal as linear and C value 3 while max iteration is 100000
model.fit(X_train, y_train) #fitting the data to model to train it
y_pred = model.predict(X_test) #predicting the test set
y_train_pred = model.predict(X_train) #predicting the train set
numeric_to_target_dict = {v: k for k, v in target_to_numeric_dict.items()} #a dictionary to reverse back to orginal classification
y_pred = [numeric_to_target_dict[value] for value in y_pred] #converting y_pred to original classification
y_test = [numeric_to_target_dict[value] for value in y_test] #converting y_test to original classification
y_train_pred = [numeric_to_target_dict[value] for value in y_train_pred] #converting y_train_pred to original classification
y_train = [numeric_to_target_dict[value] for value in y_train] #converting y_train to original classification
test_accuracy = accuracy_score(y_test, y_pred) #checking the accuracy of test set
train_accuracy = accuracy_score(y_train, y_train_pred) #checking the accuracy of train set 
Confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True) #code to make the confusion matrix 
features = scaler.fit_transform(features) #scaling the whole feature
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0) #using k fold version to check accuracy at various folds
scores = cross_val_score(model, features, target, cv=skf) #calculating accuracy with kfold method
average_accuracy = np.mean(scores) # finding average k fold cross val accuracy
print(f'Classification Method Used: {model}') #printing what method we are using
print(f'Train Accuracy of SVM Model: {train_accuracy:.2%}') #printing train accuracy
print(f'Test Accuracy of SVM Model: {test_accuracy:.2%}') #printing Test Accuracy
print("Confusion matrix:") #printing confusion matrix
print(Confusion_matrix) #printing confusion matrix
for fold, accuracy in enumerate(scores, 1): #running a loop to go through different folds
    print(f"Fold {fold} Accuracy: {accuracy:.2%}") #print each folds accuracy
print(f"Average accuracy across all folds: {average_accuracy:.2%}") #printing the average kfold cross val score
