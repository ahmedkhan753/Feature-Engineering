import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
class DataProcessor:

    def __init__(self, file_path):
        """Initializes the DataProcessor with a CSV file."""
        try:
            self.df = pd.read_csv(file_path)
            self.df = self.df.iloc[:,2:]
            print(f"Data successfully loaded from: {file_path}")
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
            self.df = None
            
    def show_info(self):
        """Prints the head and info of the internal DataFrame."""
        if self.df is not None:
            print("\n--- DataFrame Head ---")
            print(self.df.head()) 
            print("\n--- DataFrame Info ---")
            self.df.info()

    def train_test_split(self, test_size=0.2, random_state=42):
        """Splits the DataFrame into training and testing sets."""
        if self.df is not None:
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(self.df.drop('Purchased', axis=1), self.df['Purchased'], test_size=0.3, random_state=random_state)
            return X_train, X_test, y_train, y_test
        else:
            print("DataFrame is not loaded.")
            return None, None, None, None
    
    def standarScalar(self, X_train, X_test):
        """Applies standard scaling to the training and testing feature sets."""
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        return X_train_scaled, X_test_scaled 

    def before_after_scaling_plot(self, x_train, X_train_scaled):
        '''Plotes the feautures before and after scaling'''
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

        ax1.scatter(x_train['Age'], x_train['EstimatedSalary'], color='red')
        ax1.set_title('Before Scaling')
        ax2.scatter(X_train_scaled['Age'], X_train_scaled['EstimatedSalary'], color='blue')
        ax2.set_title('After Scaling')
        plt.show()

from sklearn.linear_model import LogisticRegression
processor = DataProcessor('Social_Network_Ads.csv')

if processor.df is not None:
    # 1. Properly capture the returned variables from your methods
    X_train, X_test, y_train, y_test = processor.train_test_split()
    X_train_scaled, X_test_scaled = processor.standarScalar(X_train, X_test)
    
    # 2. Show the plots
    processor.before_after_scaling_plot(X_train, X_train_scaled)

    # 3. Use the captured variables for training (NOT the function names)
    lr = LogisticRegression()
    lr_scaled = LogisticRegression()
    dt = sklearn.tree.DecisionTreeClassifier()
    dt_scaled = sklearn.tree.DecisionTreeClassifier()

    # Training without scaling (Using variables returned by train_test_split)
    lr.fit(X_train, y_train)
    dt.fit(X_train, y_train)

    # Training with scaling (Using variables returned by standarScalar)
    lr_scaled.fit(X_train_scaled, y_train) # y_train is the same for both
    dt_scaled.fit(X_train_scaled, y_train) # y_train is the same for both

    # 4. Predict and check accuracy
    y_pred = lr.predict(X_test)
    y_pred_scaled = lr_scaled.predict(X_test_scaled)
    y_pred_dt = dt.predict(X_test)
    y_pred_dt_scaled = dt_scaled.predict(X_test_scaled)

    from sklearn.metrics import accuracy_score
    print("Accuracy without Scaling:", accuracy_score(y_test, y_pred))
    print("Accuracy with Scaling:", accuracy_score(y_test, y_pred_scaled))
    print("Decision Tree Accuracy without Scaling:", accuracy_score(y_test, y_pred_dt))
    print("Decision Tree Accuracy with Scaling:", accuracy_score(y_test, y_pred_dt_scaled))
