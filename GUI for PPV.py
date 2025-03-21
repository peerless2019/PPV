import tkinter as tk
from tkinter import messagebox
import numpy as np
import joblib
import os

root = tk.Tk()
root.title("Prediction of Peak Particle velocity in open pit blasting")
root.geometry("710x600")
frame = tk.Frame(root, padx=10, pady=10)
frame.pack(fill=tk.BOTH, expand=True)

def create_label_entry(frame, text, row, col):
    label = tk.Label(frame, text=text, font=('Times New Roman', 12), anchor='w', width=40)
    label.grid(row=row, column=col, padx=10, pady=7, sticky='w')
    entry = tk.Entry(frame, font=('Times New Roman', 12), bg='#FFFFFF', highlightthickness=0, bd=1, width=20, justify='left')
    entry.grid(row=row, column=col+1, padx=10, pady=5)
    return entry

label_title = tk.Label(frame, text='GUI Model for PPV Prediction', font=('Times New Roman', 18, 'bold'), fg='#000000')
label_title.grid(row=0, column=0, columnspan=2, pady=10, sticky='n')

label_params = tk.Label(frame, text='Definition of Parameters', font=('Times New Roman', 14, 'bold'), fg='#000000')
label_params.grid(row=2, column=0, columnspan=2, pady=10, sticky='w')

entries = {}
param_labels = [
    'Input1: Spacing (m)',
    'Input2: Stemming (m)',
    'Input3: Burden (m)',
    'Input4: Charge Length (m)',
    'Input5: Powder factor (kg/m^3)',
    'Input6: Maximum Charge Per Delay Time (kg)',
    'Input7: Charge Weight (kg)',
    'Input8: Distance From Measuring Point to Blasting Face (m)'
]
for i, text in enumerate(param_labels):
    entries[text] = create_label_entry(frame, text, i+3, 0)

label_output = tk.Label(frame, text='Output:', font=('Times New Roman', 16, 'bold'), fg='#000000')
label_output.grid(row=11, column=0, pady=10, sticky='w')
label_result = tk.Label(frame, text='Peak Particle Velocity (cm/s)', font=('Times New Roman', 16, 'bold'), fg='#000000')
label_result.grid(row=12, column=0, pady=10, sticky='w')

output_value = tk.Label(frame, text="0.0", font=('Times New Roman', 16, 'bold'), fg='#C00000', relief="sunken", width=15)
output_value.grid(row=12, column=1, pady=10)

def clear_entries():
    for entry in entries.values():
        entry.delete(0, tk.END)
        entry.insert(0, "0.0")
    output_value.config(text="0.0")

def predict():
    try:
        input_values = []
        for label in param_labels:
            value = entries[label].get()
            if not value.strip(): 
                value = "0.0"
            try:
                input_values.append(float(value))
            except ValueError:
                messagebox.showerror("Error", f"Invalid input for {label}. Please enter numeric values.")
                return
        input_data = np.array([input_values])
        prediction = model_xgb.predict(input_data)
        output_value.config(text=f"{prediction[0]:.4f}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

model_path_xgb = "D:/Code/PSO-SVR.joblib"
if not os.path.exists(model_path_xgb):
    messagebox.showerror("Error", "PSO-SVR model file not found.")
    root.destroy()
    exit()

model_xgb = joblib.load(model_path_xgb)

button_frame = tk.Frame(frame)
button_frame.grid(row=13, column=0, columnspan=2, pady=20)

clear_button = tk.Button(button_frame, text="Clear", font=('Times New Roman', 14), command=clear_entries)
clear_button.grid(row=0, column=0, padx=10)
predict_button = tk.Button(button_frame, text="Predict", font=('Times New Roman', 14), command=predict)
predict_button.grid(row=0, column=1, padx=10)

clear_entries()

root.mainloop()