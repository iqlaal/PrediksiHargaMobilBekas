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

# Pilih brand
brand_input = st.selectbox("Merek Mobil", brand_options)

# Filter model sesuai brand
filtered_models = brand_model_df.loc[brand_model_df['brand'] == brand_input, 'model'].unique()
filtered_models = sorted([m.strip() for m in filtered_models])

# Pilih model sesuai filter brand
model_input = st.selectbox("Model Mobil", filtered_models)

# Input lainnya
transmission_options = list(encoders['transmission'].classes_)
fueltype_options = list(encoders['fuelType'].classes_)

year_input = st.number_input("Tahun Mobil", min_value=2011, max_value=2020, value=2020)
transmission_input = st.selectbox("Jenis Transmisi", transmission_options)
mileage_km = st.number_input("Jarak Tempuh (kilometer)", min_value=0, value=50000)
fueltype_input = st.selectbox("Jenis Bahan Bakar", fueltype_options)
tax_rupiah = st.number_input("Biaya Pajak (Rupiah)", min_value=0, value=2000000)
mpg_input = st.number_input("Konsumsi BBM (mpg)", min_value=0.0, value=40.0)
enginesize_input = st.number_input("Ukuran Mesin (L)", min_value=0.0, value=1.5)

# Fungsi validasi input numerik
def cek_input_valid(nilai):
    if nilai <= 0:
        st.warning("⚠️ Harap lengkapi data, data tidak boleh kosong.")
        st.stop()

# Validasi input numerik
cek_input_valid(mileage_km)
cek_input_valid(tax_rupiah)
cek_input_valid(mpg_input)
cek_input_valid(enginesize_input)

# Konversi satuan
mileage_mil = mileage_km / 1.60934
tax_pound = tax_rupiah / 21000

# Siapkan DataFrame input
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

# Encode fitur kategorikal
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

# Tombol dan hasil prediksi berdampingan dengan posisi hasil naik sedikit
col1, col2 = st.columns([1, 3])  # kolom hasil dibuat lebih lebar

with col1:
    pred_button = st.button("Prediksi Harga")

with col2:
    if pred_button:
        predicted_price = model.predict(input_data)[0]

        # Konversi kurs dan penyesuaian
        kurs_gbp_to_idr = 21000
        faktor_penyesuaian = 0.4

        harga_rupiah = int(predicted_price * kurs_gbp_to_idr * faktor_penyesuaian)
        # Pakai markdown dengan margin-top untuk posisi hasil sedikit naik
        st.markdown(f"<div style='margin-top:10px;'>"
                    f"<span style='font-size:18px; color:green; font-weight:bold;'>"
                    f"Perkiraan Harga Mobil Bekas: Rp {harga_rupiah:,.0f}"
                    f"</span></div>", unsafe_allow_html=True)
    else:
        st.write("")  # ruang kosong supaya layout tetap rapi

# Footer
st.markdown("---")
st.markdown("**Nama :** Muhammad Istiqlal  \n**NPM :** 51421006  \n**Skripsi Jurusan Informatika – Universitas Gunadarma**")
