import streamlit as st
import pickle
import pandas as pd

# Load model dan encoder
model = pickle.load(open('best_random_forest_model.sav', 'rb'))
encoders = pickle.load(open('best_label_encoders.sav', 'rb'))

# Load mapping brand -> model dari pickle
with open('brand_model_mapping.pkl', 'rb') as f:
    brand_model_df = pickle.load(f)

# Ambil daftar brand unik (urutkan)
brand_options = sorted(brand_model_df['brand'].unique())

# Konfigurasi halaman
st.set_page_config(page_title="Prediksi Harga Mobil Bekas", layout="centered")
st.title("Prediksi Harga Mobil Bekas")
st.write("Masukkan spesifikasi mobil untuk memprediksi harga jualnya.")

# Pilih brand
brand_input = st.selectbox("Merek Mobil", brand_options)

# Filter model berdasarkan brand yang dipilih
filtered_models = brand_model_df.loc[brand_model_df['brand'] == brand_input, 'model'].unique()
filtered_models = sorted(filtered_models)

# Pilih model dari filtered list
model_input = st.selectbox("Model Mobil", filtered_models)

# Input lain
transmission_options = list(encoders['transmission'].classes_)
fueltype_options = list(encoders['fuelType'].classes_)

year_input = st.number_input("Tahun Mobil", min_value=2011, max_value=2020, value=2015)
transmission_input = st.selectbox("Jenis Transmisi", transmission_options)
mileage_km = st.number_input("Jarak Tempuh (kilometer)", min_value=0, value=48000)
fueltype_input = st.selectbox("Jenis Bahan Bakar", fueltype_options)
tax_rupiah = st.number_input("Biaya Pajak (Rupiah)", min_value=0, value=3150000)
mpg_input = st.number_input("Konsumsi BBM (mpg)", min_value=0.0, value=50.0)
enginesize_input = st.number_input("Ukuran Mesin (L)", min_value=0.0, value=1.4)

# Konversi
mileage_mil = mileage_km / 1.60934
tax_pound = tax_rupiah / 21000

# Siapkan dataframe input
input_data = pd.DataFrame({
    'brand': [brand_input],
    'model': [model_input],
    'year': [year_input],
    'transmission': [transmission_input],
    'mileage': [mileage_mil],
    'fuelType': [fueltype_input],
    'tax': [tax_pound],
    'mpg': [mpg_input],
    'engineSize': [enginesize_input]
})

# Encode fitur kategorikal dengan validasi
for col in ['brand', 'model', 'transmission', 'fuelType']:
    encoder = encoders.get(col)
    if encoder:
        val = input_data.at[0, col]
        if val not in encoder.classes_:
            st.error(f"❌ Nilai '{val}' tidak ditemukan pada opsi {col}.")
            st.stop()
        input_data[col] = encoder.transform(input_data[col])

# Pastikan urutan kolom sesuai saat training
input_data = input_data[list(model.feature_names_in_)]

# Tombol prediksi
if st.button("Prediksi Harga"):
    predicted_price = model.predict(input_data)[0]
    harga_rupiah = int(predicted_price * 21000)
    st.success(f"Perkiraan Harga Mobil Bekas: Rp {harga_rupiah:,.0f}")

# Footer
st.markdown("---")
st.markdown("**Nama :** Muhammad Istiqlal  \n**NPM :** 51421006  \n**Skripsi Jurusan Informatika – Universitas Gunadarma**")
