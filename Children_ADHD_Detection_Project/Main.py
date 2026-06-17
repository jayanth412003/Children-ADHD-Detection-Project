
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

dataset = None
model = None
scaler = StandardScaler()

def upload_dataset():
    global dataset
    file = filedialog.askopenfilename(filetypes=[("CSV Files","*.csv")])
    if file:
        dataset = pd.read_csv(file)
        text.insert(tk.END, f"Loaded: {file}\n")

def preprocess():
    global dataset
    if dataset is None:
        return
    dataset = dataset.fillna(0)
    text.insert(tk.END, "Dataset preprocessed\n")

def train_svm():
    global dataset, model
    if dataset is None:
        return
    X = dataset.iloc[:,:-1]
    y = dataset.iloc[:,-1]
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = SVC(kernel="rbf")
    model.fit(X_train,y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test,pred)
    text.insert(tk.END, f"SVM Accuracy: {acc*100:.2f}%\n")

root = tk.Tk()
root.title("Children ADHD Disease Detection using Pose Estimation Technique")
root.geometry("900x600")

tk.Button(root,text="Upload ADHD Pose Dataset",command=upload_dataset).pack(pady=5)
tk.Button(root,text="Preprocess Dataset",command=preprocess).pack(pady=5)
tk.Button(root,text="Train SVM Algorithm",command=train_svm).pack(pady=5)

text = tk.Text(root,height=20,width=80)
text.pack()

root.mainloop()
