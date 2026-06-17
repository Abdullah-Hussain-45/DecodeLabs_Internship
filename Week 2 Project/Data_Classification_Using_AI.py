from sklearn.datasets import load_iris
import pandas as pd
#Step 1: Load the dataset and adding the column names to the dataset and displaying the first 5 rows of the dataset
#Load the iris dataset
iris = load_iris()

df = pd.DataFrame(iris.data,columns=iris.feature_names)
df['target'] = iris.target
print(df.head())

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# define X as input feature and y as target variable
X = iris.data
y = iris.target

#split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state= 42)

print("Total Samples: ", len(X))
print("Training Samples: " , len(X_train))
print("Testing Samples: ", len(X_test))

#Step 2:Scaling data to get mean = 0 and standard deviation = 1

#Scaling the features using StandardScaler
scaler = StandardScaler()

#Fit the scaler on the training data and transform both training and testing data
X_train_scaler = scaler.fit_transform(X_train)
X_test_scaler = scaler.transform(X_test)

#printing the scaled data
print("Scaled Training Data: \n", X_train_scaler[:5])
print("Scaled Testing Data: \n", X_test_scaler[:5])


#importing the KNN classifier
from sklearn.neighbors import KNeighborsClassifier

#Creating the KNN model with k=5
model = KNeighborsClassifier(n_neighbors=5)

model = model.fit(X_train_scaler, y_train)

#Predict the labels for the test set
y_pred = model.predict(X_test_scaler)

#Evaluate the Model
print("Actual Labels: ",y_test)
print("Predicted Labels: ", y_pred)

#Step 4: Validate the Output of Model
#importing the confusion matrix and classification metrics
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score

#Calculate the Accuracy of the model
y_accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {y_accuracy * 100:.2f}%")

#Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix: \n", conf_matrix)

#Confusion Matrix Table
for i,class_name in enumerate(iris.target_names):
    #true positives for each class
    tp = conf_matrix[i, i]
    #false positives for each class
    fp = conf_matrix[:, i].sum() - tp
    #false negatives for each class
    fn = conf_matrix[i, :].sum() - tp
    #true negatives for each class
    tn = conf_matrix.sum() - (tp + fp + fn)
    
    data = {
        'Predicted Positive' : [tp, fp],
        'Predicted Negative' : [fn, tn]
    }
    df_matrix = pd.DataFrame(data, index=['Actual Positive (Y)', 'Actual Negative (N)'])
    
    print(f"Category of Flower: {class_name.upper()}")
    print(df_matrix)
    print("\n")
          
    
    

#Classification Report
class_report  = classification_report(y_test, y_pred)
print("Classification Report: \n", class_report)

