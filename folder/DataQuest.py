import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
file_path = "shuffled_dataset.xlsx"
df = pd.read_excel(file_path)

X = df[['Age', 'Gender', 'Number_of_Lanes', 'Lane_Width', 'Road_Type', 'Alcohol_Consumption', 'Crash_Type', 'Seatbelt_Usage', 'Speed_Limit']]
y = df['Crash_Severity']

# Manually splitting into training (80%) and testing (20%)
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Standardize the features (feature scaling)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Function to train k-NN classifier
def train_knn(k):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    return model

# 1. Accuracy Plot for different k values
k_values = range(1, 21)  # Testing k from 1 to 20
accuracies = []

for k in k_values:
    model = train_knn(k)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

# Plotting the accuracy
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, marker='o')
plt.title('KNN Accuracy for Different k Values')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.xticks(k_values)
plt.grid()
plt.show()

# Set k value to 17 for final model
k_value = 17
model = train_knn(k_value)
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy for k={k_value}: {accuracy}")

# 2. Confusion Matrix and Visualization
conf_matrix = confusion_matrix(y_test, y_pred)

# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Low', 'Medium', 'High'],  # Adjust based on your classes
            yticklabels=['Low', 'Medium', 'High'])  # Adjust based on your classes
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# 3. Classification Report Visualization
report = classification_report(y_test, y_pred, output_dict=True)
labels = list(report.keys())[:-3]  # Exclude 'accuracy', 'macro avg', 'weighted avg'
precision = [report[label]['precision'] for label in labels]
recall = [report[label]['recall'] for label in labels]
f1_score = [report[label]['f1-score'] for label in labels]

x = range(len(labels))  # the label locations
width = 0.25  # the width of the bars

# Plotting the metrics
plt.figure(figsize=(10, 6))
plt.bar([p - width for p in x], precision, width, label='Precision')
plt.bar(x, recall, width, label='Recall')
plt.bar([p + width for p in x], f1_score, width, label='F1 Score')

plt.xlabel('Classes')
plt.ylabel('Scores')
plt.title('Classification Report Metrics')
plt.xticks(x, labels)
plt.legend()
plt.show()

# Safety Tips
safety_tips = {
    "Age": "Younger drivers (especially those under 25) should undergo defensive driving courses to improve their skills and awareness on the road.",
    "Gender": "Regardless of gender, all drivers should be educated on safe driving practices and the importance of avoiding distractions while driving.",
    "Number_of_Lanes": "When driving on multi-lane roads, always check mirrors and blind spots before changing lanes. Use turn signals to indicate lane changes.",
    "Lane_Width": "Be aware of lane width, especially in narrow lanes. Maintain a safe distance from other vehicles and avoid sudden movements.",
    "Road_Type": "Adjust driving behavior according to the type of road. For example, on highways, maintain a safe speed and keep a safe following distance; on residential streets, be vigilant for pedestrians and cyclists.",
    "Alcohol_Consumption": "Never drive under the influence of alcohol or drugs. Always have a designated driver or use public transportation or rideshare services if consuming alcohol .",
    "Crash_Type": "Understand common crash types (e.g., rear-end collisions, side impacts) and take preventive measures, such as maintaining a safe following distance and being cautious at intersections.",
    "Seatbelt_Usage": "Always wear a seatbelt, and ensure that all passengers in the vehicle do the same. Seatbelts significantly reduce the risk of injury in the event of a crash.",
    "Speed_Limit": "Always adhere to posted speed limits. Adjust your speed according to road conditions, weather, and traffic. Remember that speeding increases the severity of crashes."
}

# Print Safety Tips
print("\n### Safety Tips ###\n")
for feature, tip in safety_tips.items():
    print(f"{feature}: {tip}\n")

# Time and Space Complexity
print("Time Complexity: O(n * k) where n is the number of test samples and k is the number of neighbors.")
print("Space Complexity: O(n) for storing the training dataset and O(k) for storing the neighbors during prediction.")