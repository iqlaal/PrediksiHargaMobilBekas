import streamlit as st
import pickle
import pandas as pd

# Load model dan encoder dari file .sav
model = pickle.load(open('best_random_forest_model.sav', 'rb'))
encoders = pickle.load(open('best_label_encoders.sav', 'rb'))

# Ambil opsi dari encoder (kelas-kelas untuk dropdown)
brand_options = list(encoders['brand'].classes_)
model_options = list(encoders['model'].classes_)
transmission_options = list(encoders['transmission'].classes_)
fueltype_options = list(encoders['fuelType'].classes_)

# Konfigurasi halaman Streamlit
st.set_page_config(page_title="Prediksi Harga Mobil Bekas", layout="centered")
st.title("Prediksi Harga Mobil Bekas")
st.write("Masukkan spesifikasi mobil untuk memprediksi harga jualnya.")

# Input pengguna lewat form
brand_input = st.selectbox("Merek Mobil", brand_options)
model_input = st.selectbox("Model Mobil", model_options)
year_input = st.number_input("Tahun Mobil", min_value=2011, max_value=2020, value=2015)
transmission_input = st.selectbox("Jenis Transmisi", transmission_options)
mileage_km = st.number_input("Jarak Tempuh (kilometer)", min_value=0, value=48000)
fueltype_input = st.selectbox("Jenis Bahan Bakar", fueltype_options)
tax_rupiah = st.number_input("Biaya Pajak (Rupiah)", min_value=0, value=3150000)
mpg_input = st.number_input("Konsumsi BBM (mpg)", min_value=0.0, value=50.0)
enginesize_input = st.number_input("Ukuran Mesin (L)", min_value=0.0, value=1.4)

# Konversi satuan ke format model
mileage_mil = mileage_km / 1.60934          # km ke mil
tax_pound = tax_rupiah / 21000              # Rupiah ke Pound (ubah sesuai kurs terkini)

# Buat DataFrame input untuk model (perhatikan nama kolom harus sama persis saat training)
input_data = pd.DataFrame({
    'brand': [brand_input],
    'model': [model_input],
    'year': [year_input],
    'transmission': [transmission_input],
    'mileage': [mileage_mil],
    'fuelType': [fueltype_input],
    'tax': [tax_pound],
    'mpg': [mpg_input],
    'engineSize': [enginesize_input]   # pastikan ini sama huruf besar kecilnya dengan saat training
})

# Encode fitur kategorikal dengan validasi input
for col in ['brand', 'model', 'transmission', 'fuelType']:
    encoder = encoders.get(col)
    if encoder:
        val = input_data.at[0, col]
        if val not in encoder.classes_:
            st.error(f"❌ Nilai '{val}' tidak ditemukan pada opsi {col}.")
            st.stop()
        input_data[col] = encoder.transform(input_data[col])

# Pastikan urutan kolom sesuai fitur model saat training
input_data = input_data[list(model.feature_names_in_)]

# Tombol prediksi
if st.button("Prediksi Harga"):
    predicted_price = model.predict(input_data)[0]
    harga_rupiah = int(predicted_price * 21000)  # konversi GBP ke IDR sesuai kurs
    st.success(f"Perkiraan Harga Mobil Bekas: Rp {harga_rupiah:,.0f}")

# Footer / informasi
st.markdown("---")
st.markdown("**Nama :** Muhammad Istiqlal  \n**NPM :** 51421006  \n**Skripsi Jurusan Informatika – Universitas Gunadarma**")
