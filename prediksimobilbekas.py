import streamlit as st
import pickle
import pandas as pd

# Load model dan encoder
model = pickle.load(open('best_random_forest_model.sav', 'rb'))
encoders = pickle.load(open('best_label_encoders.sav', 'rb'))

# Load dataframe brand-model mapping dari pickle
with open('brand_model_mapping.pkl', 'rb') as f:
    brand_model_df = pickle.load(f)

# Bersihkan spasi di brand dan model
brand_model_df['brand'] = brand_model_df['brand'].str.strip()
brand_model_df['model'] = brand_model_df['model'].str.strip()

# Ambil daftar brand unik, urutkan
brand_options = sorted(brand_model_df['brand'].unique())

# Konfigurasi halaman Streamlit
st.set_page_config(page_title="Prediksi Harga Mobil Bekas", layout="centered")
st.title("Prediksi Harga Mobil Bekas")
st.write("Masukkan spesifikasi mobil untuk memprediksi harga jualnya.")

# Input
brand_input = st.selectbox("Merek Mobil", brand_options)
filtered_models = brand_model_df.loc[brand_model_df['brand'] == brand_input, 'model'].unique()
filtered_models = sorted([m.strip() for m in filtered_models])
model_input = st.selectbox("Model Mobil", filtered_models)

transmission_options = list(encoders['transmission'].classes_)
fueltype_options = list(encoders['fuelType'].classes_)

year_input = st.number_input("Tahun Mobil", min_value=2011, max_value=2020, value=2020)
transmission_input = st.selectbox("Jenis Transmisi", transmission_options)
mileage_km = st.number_input("Jarak Tempuh (kilometer)", min_value=0, value=50000)
fueltype_input = st.selectbox("Jenis Bahan Bakar", fueltype_options)
tax_rupiah = st.number_input("Biaya Pajak (Rupiah)", min_value=0, value=2000000)
mpg_input = st.number_input("Konsumsi BBM (mpg)", min_value=0.0, value=40.0)
enginesize_input = st.number_input("Ukuran Mesin (L)", min_value=0.0, value=1.5)

def cek_input_valid(nilai):
    if nilai <= 0:
        st.warning("⚠️ Harap lengkapi data, data tidak boleh kosong.")
        st.stop()

cek_input_valid(mileage_km)
cek_input_valid(tax_rupiah)
cek_input_valid(mpg_input)
cek_input_valid(enginesize_input)

# Konversi satuan dan siapkan data prediksi
mileage_mil = mileage_km / 1.60934
tax_pound = tax_rupiah / 21000

input_data = pd.DataFrame({
    'brand': [brand_input.strip()],
    'model': [model_input.strip()],
    'year': [year_input],
    'transmission': [transmission_input.strip()],
    'mileage': [mileage_mil],
    'fuelType': [fueltype_input.strip()],
    'tax': [tax_pound],
    'mpg': [mpg_input],
    'engineSize': [enginesize_input]
})

# Encoding label
for col in ['brand', 'model', 'transmission', 'fuelType']:
    encoder = encoders.get(col)
    if encoder:
        val = input_data.at[0, col]
        if val not in encoder.classes_:
            st.error("⚠️ Harap lengkapi data, data tidak boleh kosong.")
            st.stop()
        input_data[col] = encoder.transform([val])

# Pastikan urutan kolom sesuai model
input_data = input_data[list(model.feature_names_in_)]

# Buat dua kolom untuk tombol dan hasil
col_button, col_result = st.columns([3, 13])

with col_button:
    pred_button = st.button("Prediksi Harga")

with col_result:
    if pred_button:
        predicted_price = model.predict(input_data)[0]
        kurs_gbp_to_idr = 21000

        # Penyesuaian berdasarkan merek
        brand_factors = {
            'Hyundai': 0.75,
            'Ford': 0.65
        }
        faktor_penyesuaian = brand_factors.get(brand_input, 0.7)  # 0.7 jika merek tidak ditemukan

        harga_rupiah = int(predicted_price * kurs_gbp_to_idr * faktor_penyesuaian)
        st.success(f"Perkiraan Harga Mobil Bekas: Rp {harga_rupiah:,.0f}")
    else:
        st.write("")

# Footer
st.markdown("---")
st.markdown("**Nama :** Muhammad Istiqlal  \n**NPM :** 51421006  \n**Skripsi Jurusan Informatika – Universitas Gunadarma**")
