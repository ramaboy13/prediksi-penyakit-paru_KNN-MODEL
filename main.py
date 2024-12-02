from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load model dan scaler
with open('model/knn_model.pkl', 'rb') as file:
    model = pickle.load(file)
with open('model/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

@app.route('/')
def home():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Mengambil data dari form input
        data = [
            float(request.form['Usia']),
            float(request.form['Jenis_Kelamin']),
            float(request.form['Merokok']),
            float(request.form['Aktivitas_Begadang']),
            float(request.form['Aktivitas_Olahraga']),
            float(request.form['Penyakit_Bawaan']),
        ]

        # Transformasi data menggunakan scaler
        scaled_data = scaler.transform([data])

        # Prediksi probabilitas
        probability = model.predict_proba(scaled_data)[0][1]  # Probabilitas kelas 1

        # Mengonversi ke persentase
        prediction_percentage = probability * 100

        return render_template(
            'form.html',
            prediction=f"Peluang terkena penyakit paru-paru: {prediction_percentage:.2f}%"
        )
    except Exception as e:
        return render_template('form.html', prediction=f"Terjadi kesalahan: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
