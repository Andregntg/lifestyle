import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv('sample_data/student_lifestyle_dataset.csv')

# Mengonversi target 'Stress_Level' menjadi numerik
data['Stress_Level_Encoded'] = data['Stress_Level'].map({'Low': 0, 'Moderate': 1, 'High': 2})

# Memilih fitur dan target
X = data[['Study_Hours_Per_Day', 'Physical_Activity_Hours_Per_Day']]  # Fitur
y = data['Stress_Level_Encoded']  # Target (Stress Level)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Streamlit UI
st.title('Prediksi Tingkat Stres')
st.write("Masukkan data untuk memprediksi tingkat stres berdasarkan jam belajar dan aktivitas fisik:")

study_hours = st.number_input("Study Hours per Day", min_value=0, max_value=24, value=5)
activity_hours = st.number_input("Physical Activity Hours per Day", min_value=0, max_value=24, value=3)

# Buat DataFrame untuk inputan
new_data = pd.DataFrame({
    'Study_Hours_Per_Day': [study_hours],
    'Physical_Activity_Hours_Per_Day': [activity_hours]
})

# Prediksi menggunakan model
predicted_stress = rf_model.predict(new_data)

# Menampilkan hasil prediksi
stress_labels = {0: 'Low', 1: 'Moderate', 2: 'High'}
predicted_stress_label = stress_labels[predicted_stress[0]]

st.write(f"Predicted Stress Level: {predicted_stress_label}")
