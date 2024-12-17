from flask import Flask, request, render_template
import pickle
import numpy as np
import random

app = Flask(__name__, static_folder='public')

# Load model dan scaler
with open('model/knn_model.pkl', 'rb') as file:
    model = pickle.load(file)
with open('model/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Recommendation Dictionaries
RECOMMENDATIONS = {
    'low_risk': [
        "Tetap pertahankan gaya hidup sehat yang sudah Anda jalani",
        "Lakukan pemeriksaan kesehatan rutin minimal setahun sekali",
        "Hindari paparan asap rokok dan polusi udara",
        "Tingkatkan aktivitas fisik dengan olahraga teratur",
        "Konsumsi makanan bergizi dan cukup tidur"
    ],
    'high_risk': [
        "Segera konsultasi dengan dokter spesialis paru-paru",
        "Kurangi aktivitas yang berisiko terpapar polusi",
        "Hindari merokok dan lingkungan perokok",
        "Tingkatkan sistem kekebalan tubuh melalui diet sehat",
        "Gunakan masker saat beraktivitas di luar ruangan"
    ]
}

HEALTHY_FOODS = {
    'low_risk': [
        "Bayam kaya antioksidan untuk kesehatan paru-paru",
        "Apel mengandung flavonoid yang melindungi paru-paru",
        "Ikan salmon mengandung omega-3 untuk kesehatan pernapasan",
        "Brokoli kaya vitamin C untuk meningkatkan sistem kekebalan",
        "Kacang-kacangan sebagai sumber protein dan mineral"
    ],
    'high_risk': [
        "Kunyit dengan sifat anti-inflamasi untuk paru-paru",
        "Jahe membantu mengurangi peradangan saluran pernapasan",
        "Wortel kaya beta-karoten untuk kesehatan paru-paru",
        "Bawang putih mengandung senyawa anti-inflamasi",
        "Teh hijau dengan antioksidan tinggi untuk melindungi paru-paru"
    ]
}

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
        
        # Kategori Usia Baru
        if 25 <= usia <= 44:
            kategori_usia = "muda"
        elif 44 < usia <= 60:
            kategori_usia = "paruh baya"
        elif 60 < usia <= 75:
            kategori_usia = "tua"
        elif 75 < usia <= 90:
            kategori_usia = "pikun"
        elif usia > 90:
            kategori_usia = "manula"
        else:
            kategori_usia = "di bawah kategori"
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
                f"Anda termasuk kategori {kategori_usia} dan TIDAK berisiko terkena penyakit paru-paru. "
                f"Pertahankan gaya hidup sehat dan tetap jaga kesehatan Anda! 😊 "
            )
            recommendations = random.sample(RECOMMENDATIONS['low_risk'], 3)
            healthy_foods = random.sample(HEALTHY_FOODS['low_risk'], 3)
        else:
            message = (
                f"Anda termasuk kategori {kategori_usia} dan BERISIKO terkena penyakit paru-paru. "
                f"Jangan khawatir, lakukan perubahan positif pada pola hidup Anda. "
                f"Selalu ingat, kesehatan adalah investasi terbaik! 💪 "
            )
            recommendations = random.sample(RECOMMENDATIONS['high_risk'], 3)
            healthy_foods = random.sample(HEALTHY_FOODS['high_risk'], 3)

        return render_template(
            'result.html', 
            prediction=message, 
            recommendations=recommendations,
            healthy_foods=healthy_foods
        )
    except Exception as e:
        return render_template('form.html', error=f"Terjadi kesalahan: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)