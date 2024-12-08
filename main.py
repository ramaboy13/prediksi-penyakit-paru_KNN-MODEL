from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__, static_folder='public')

# Load model dan scaler
with open('model/knn_model.pkl', 'rb') as file:
    model = pickle.load(file)
with open('model/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('form.html')  # Form input
    try:
        # Mengambil data dari form input
        usia = float(request.form.get('Usia', 0))
        kategori_usia = "muda" if usia < 40 else "tua"

        data = [
            usia,
            float(request.form.get('Jenis_Kelamin', 0)),
            float(request.form.get('Merokok', 0)),
            float(request.form.get('Aktivitas_Begadang', 0)),
            float(request.form.get('Aktivitas_Olahraga', 0)),
            float(request.form.get('Penyakit_Bawaan', 0)),
        ]

        # Pastikan semua field diisi
        if None in data:
            return render_template('form.html', error="Harap isi semua field dengan benar!")

        # Transformasi data menggunakan scaler
        scaled_data = scaler.transform([data])

        # Prediksi probabilitas
        probability = model.predict_proba(scaled_data)[0][1]  # Probabilitas kelas 1

        # Mengonversi ke persentase
        prediction_percentage = probability * 100

        # Menentukan output berdasarkan probabilitas
        if prediction_percentage < 50:  # Ambang batas 50%
            message = (
                f"Anda termasuk kategori {kategori_usia} dan tidak berisiko terkena penyakit paru-paru. "
                f"Pertahankan gaya hidup sehat dan tetap jaga kesehatan Anda! 😊"
            )
        else:
            message = (
                f"Anda termasuk kategori {kategori_usia} dan berisiko terkena penyakit paru-paru. "
                f"Jangan khawatir, lakukan perubahan positif pada pola hidup Anda. "
                f"Selalu ingat, kesehatan adalah investasi terbaik! 💪"
            )

        return render_template('result.html', prediction=message)
    except Exception as e:
        return render_template('form.html', error=f"Terjadi kesalahan: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
