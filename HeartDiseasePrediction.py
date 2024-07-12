import tkinter as tk
from tkinter import ttk, W
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load your dataset
df = pd.read_csv('heart.csv')
x = df[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']]
y = df['target']

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# Create and train the model
logr = LogisticRegression(max_iter=1000)
logr.fit(x_train, y_train)

#to print count of train data and test data
print("Number of train data:",len(x_train))#53	1	0	140	203	1	0	155	1	3.1	0	0	3	0
print("Number of test data:",len(x_test))#58	0	0	100	248	0	0	122	0	1	1	0	2	1

print("Train data accuracy:",logr.score(x_train,y_train))
print("Test data accuracy:",logr.score(x_test,y_test))

def predict():
    input_data = [int(age_entry.get()), int(sex_combobox.get()), int(cp_combobox.get()), int(trestbps_entry.get()),
        int(chol_entry.get()), int(fbs_combobox.get()), int(restecg_combobox.get()), int(thalach_entry.get()),
        int(exang_combobox.get()), float(oldpeak_entry.get()), int(slope_combobox.get()), int(ca_combobox.get()),
        int(thal_combobox.get())]

    input_data_as_numpy = np.asarray([input_data])
    prediction = logr.predict(input_data_as_numpy)
    print(prediction)
    result_label.config(text=f"Prediction: {'Heart Disease' if prediction[0] == 1 else 'No Heart Disease'}")

def draw_histograms(dataframe, features, rows, cols):
        fig = plt.figure(figsize=(10, 10))
        for i, feature in enumerate(features):
            ax = fig.add_subplot(rows, cols, i + 1)
            dataframe[feature].hist(bins=20, ax=ax, facecolor='midnightblue')
            ax.set_title(feature + " Distribution", color='DarkRed')

        fig.tight_layout()
        plt.show()
draw_histograms(df, df.columns, 6, 3)

def reset():
    # Clear all entry fields
    age_entry.delete(0, tk.END)
    sex_combobox.set("")
    cp_combobox.set("")
    trestbps_entry.delete(0, tk.END)
    chol_entry.delete(0, tk.END)
    fbs_combobox.set("")
    restecg_combobox.set("")
    thalach_entry.delete(0, tk.END)
    exang_combobox.set("")
    oldpeak_entry.delete(0, tk.END)
    slope_combobox.set("")
    ca_combobox.set("")
    thal_combobox.set("")
    result_label.config(text="")


# Create GUI
root = tk.Tk()
root.geometry("800x800")
root.config(background="#FFF0F5")
root.title("Heart Disease Prediction")

# Labels
tk.Label(root, text="Enter Patient Information",font=("Times New Roman",24),bg="white").grid(row=0, column=2,pady=10)

tk.Label(root, text="Age:",font=("Aerial Black",14),bg="#FFF0F5").grid(row=2, column=2,sticky=W)
age_entry = tk.Entry(root)
age_entry.grid(row=2, column=3)


tk.Label(root, text="Sex:",font=("Aerial Black",14),bg="#FFF0F5").grid(row=3, column=2,sticky=W)
sex_combobox = ttk.Combobox(root,values=["1","0"],state="readonly")
sex_combobox.set("select Here")
sex_combobox.grid(row=3, column=3)


tk.Label(root, text="Chest Pain:",font=("Aerial Black",14),bg="#FFF0F5").grid(row=4, column=2,sticky=W)
cp_combobox = ttk.Combobox(root,values=["0", "1", "2", "3"], state="readonly")
cp_combobox.set("select Here")
cp_combobox.grid(row=4, column=3)

tk.Label(root, text="Blood Pressure:",font=("Aerial Black",14),bg="#FFF0F5").grid(row=5, column=2,sticky=W)
trestbps_entry = tk.Entry(root)
trestbps_entry.grid(row=5, column=3)


tk.Label(root, text="Cholesterol:",font=("Aerial Black",14),bg="#FFF0F5").grid(row=6, column=2,sticky=W)
chol_entry = tk.Entry(root)
chol_entry.grid(row=6, column=3)


tk.Label(root, text="Blood Sugar:",font=("Aerial Black",14),bg="#FFF0F5").grid(row=7, column=2,sticky=W)
fbs_combobox = ttk.Combobox(root,values=["0", "1"], state="readonly")
fbs_combobox.set("select Here")
fbs_combobox.grid(row=7, column=3)


tk.Label(root, text="ECG Results:",font=("Aerial Black",14),bg="#FFF0F5").grid(row=8, column=2,sticky=W)
restecg_combobox = ttk.Combobox(root,values=["0","1","2"], state="readonly")
restecg_combobox.set("select Here")
restecg_combobox.grid(row=8, column=3)


tk.Label(root, text="Max Heart Rate:",font=("Aerial Black",14),bg="#FFF0F5").grid(row=9, column=2,sticky=W)
thalach_entry = tk.Entry(root)
thalach_entry.grid(row=9, column=3)


tk.Label(root, text="ExerciseInducedAngina:",font=("Aerial Black",14),bg="#FFF0F5").grid(row=10, column=2,sticky=W)
exang_combobox = ttk.Combobox(root, values=["0", "1"], state="readonly")
exang_combobox.set("select Here")
exang_combobox.grid(row=10, column=3)

tk.Label(root, text="ST Depression:",font=("Aerial Black",14),bg="#FFF0F5").grid(row=11, column=2,sticky=W)
oldpeak_entry = tk.Entry(root)
oldpeak_entry.grid(row=11, column=3)


tk.Label(root, text="Slope of ST Segment:",font=("Aerial Black",14),bg="#FFF0F5").grid(row=12, column=2,sticky=W)
slope_combobox = ttk.Combobox(root,values=["0", "1", "2"], state="readonly")
slope_combobox.set("select Here")
slope_combobox.grid(row=12, column=3)


tk.Label(root, text="Fluoroscopy:",font=("Aerial Black",14),bg="#FFF0F5").grid(row=13, column=2,sticky=W)
ca_combobox = ttk.Combobox(root,values=["0", "1", "2", "3","4"], state="readonly")
ca_combobox.set("select Here")
ca_combobox.grid(row=13, column=3)


tk.Label(root, text="Thalassemia:",font=("Aerial Black",14),bg="#FFF0F5").grid(row=14, column=2,sticky=W)
thal_combobox = ttk.Combobox(root,values=[" 0","1","2","3"], state="readonly")
thal_combobox.set("select Here")
thal_combobox.grid(row=14, column=3)

# Button to make a prediction and reset
predict_button = tk.Button(root, text="Predict", command=predict,bg="#8B1A1A",fg="white",font=("Aerial Black",12))
predict_button.grid(row=16, column=2,columnspan=2,pady=10,sticky=W )

reset_button = tk.Button(root, text="Reset", command=reset,bg="#000080",fg="white",font=("Aerial Black",12))
reset_button.grid(row=16, column=3,columnspan=2,pady=10,sticky=W)

# Display the result
result_label = tk.Label(root, text="",font=("Aerial Black",18))
result_label.grid(row=18, column=1,columnspan=2, pady=10,sticky=W)

root.mainloop()

