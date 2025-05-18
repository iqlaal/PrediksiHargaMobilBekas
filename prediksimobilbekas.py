import streamlit as st
import pickle
import pandas as pd

# Load model dan encoder
model = pickle.load(open('best_random_forest_model.sav', 'rb'))
encoders = pickle.load(open('best_label_encoders.sav', 'rb'))

# Ambil opsi dari encoder yang sudah dipelajari
brand_options = list(encoders['brand'].classes_)
model_options = list(encoders['model'].classes_)
transmission_options = list(encoders['transmission'].classes_)
fueltype_options = list(encoders['fueltype'].classes_)

# Konfigurasi halaman
st.set_page_config(page_title="Prediksi Harga Mobil Bekas", layout="centered")
st.title("Prediksi Harga Mobil Bekas")
st.write("Masukkan spesifikasi mobil untuk memprediksi harga jualnya.")

# Form input pengguna
brand_input = st.selectbox("Merek Mobil", brand_options)
model_input = st.selectbox("Model Mobil", model_options)
year_input = st.number_input("Tahun Mobil", min_value=2000, max_value=2023, value=2015)
transmission_input = st.selectbox("Jenis Transmisi", transmission_options)
mileage_input = st.number_input("Jarak Tempuh (mileage) dalam mil", min_value=0, value=30000)
fueltype_input = st.selectbox("Jenis Bahan Bakar", fueltype_options)
tax_input = st.number_input("Biaya Pajak (£)", min_value=0, value=150)
mpg_input = st.number_input("Konsumsi BBM (mpg)", min_value=0.0, value=50.0)
enginesize_input = st.number_input("Ukuran Mesin (L)", min_value=0.0, value=1.4)

# Masukkan ke DataFrame
input_data = pd.DataFrame({
    'brand': [brand_input],
    'model': [model_input],
    'year': [year_input],
    'transmission': [transmission_input],
    'mileage': [mileage_input],
    'fueltype': [fueltype_input],
    'tax': [tax_input],
    'mpg': [mpg_input],
    'enginesize': [enginesize_input]
})

# Encode kolom kategorikal
for col in ['brand', 'model', 'transmission', 'fueltype']:
    encoder = encoders.get(col)
    if encoder:
        input_data[col] = encoder.transform(input_data[col])

# Tombol prediksi
if st.button("Prediksi Harga"):
    # Prediksi harga
    predicted_price = model.predict(input_data)[0]
    
    # Konversi ke Rupiah (misal 1 pound = Rp21.000)
    harga_rupiah = int(predicted_price * 21000)
    
    st.success(f"Perkiraan Harga Mobil Bekas: Rp {harga_rupiah:,.0f}")
