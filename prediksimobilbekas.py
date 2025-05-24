import streamlit as st
import pickle
import pandas as pd

# Load model dan encoder
model = pickle.load(open('best_random_forest_model.sav', 'rb'))
encoders = pickle.load(open('best_label_encoders.sav', 'rb'))

# Misal kamu punya DataFrame asli (belum di-encode) atau bisa load CSV:
# Contoh minimal buat dictionary brand ke model (pastikan data asli sudah ada)
# Jika kamu tidak punya dataset asli di sini, kamu harus siapkan file CSV atau DataFrame
# yang berisi semua brand dan model, lalu buat mappingnya

# Contoh data mapping (harus kamu sesuaikan dengan data asli)
brand_model_map = {
    'ford': ['Fiesta', 'Focus', 'Galaxy', 'Mondeo', 'Mustang', 'Kuga', 'Edge', 'Tourneo Connect', 'Tourneo Custom'],
    'hyundai': ['I10', 'I20', 'I30', 'I40', 'I800', 'IX20', 'IX35', 'Ioniq', 'Kona', 'Santa Fe', 'Tucson', 'Veloster']
}

# Ambil opsi brand
brand_options = list(encoders['brand'].classes_)
brand_options_lower = [b.lower() for b in brand_options]  # asumsikan key lowercase sesuai brand_model_map

# Pilih brand dulu
brand_input = st.selectbox("Merek Mobil", brand_options)

# Berdasarkan brand terpilih, ambil list model yg sesuai
brand_key = brand_input.lower()  # samakan format dengan brand_model_map key
model_list_for_brand = brand_model_map.get(brand_key, [])

# Supaya bisa buat pilihan model yang sudah di-encode, kita perlu reverse encode model_list_for_brand
# mapping encoded_model -> original_model
model_encoder = encoders['model']
encoded_models_for_brand = []

for model_name in model_list_for_brand:
    try:
        encoded_val = model_encoder.transform([model_name])[0]
        encoded_models_for_brand.append((encoded_val, model_name))
    except:
        # kalau model_name tidak ditemukan di encoder, abaikan saja
        pass

# Sort list model sesuai nama (optional)
encoded_models_for_brand.sort(key=lambda x: x[1])

# Ambil list model asli yang sesuai brand untuk pilihan model
model_options = [m[1] for m in encoded_models_for_brand]

# Pilih model dari opsi yang sudah difilter sesuai brand
model_input = st.selectbox("Model Mobil", model_options)

# Input lain
year_input = st.number_input("Tahun Mobil", min_value=2000, max_value=2023, value=2015)
transmission_options = list(encoders['transmission'].classes_)
transmission_input = st.selectbox("Jenis Transmisi", transmission_options)

mileage_km = st.number_input("Jarak Tempuh (kilometer)", min_value=0, value=48000)
fueltype_options = list(encoders['fuelType'].classes_)
fueltype_input = st.selectbox("Jenis Bahan Bakar", fueltype_options)

tax_rupiah = st.number_input("Biaya Pajak (Rupiah)", min_value=0, value=3150000)
mpg_input = st.number_input("Konsumsi BBM (mpg)", min_value=0.0, value=50.0)
enginesize_input = st.number_input("Ukuran Mesin (L)", min_value=0.0, value=1.4)

# Konversi untuk model
mileage_mil = mileage_km / 1.60934
tax_pound = tax_rupiah / 21000

input_data = pd.DataFrame({
    'brand': [brand_input],
    'model': [model_input],
    'year': [year_input],
    'transmission': [transmission_input],
    'mileage': [mileage_mil],
    'fuelType': [fueltype_input],
    'tax': [tax_pound],
    'mpg': [mpg_input],
    'enginesize': [enginesize_input]
})

# Encoding fitur kategorikal
for col in ['brand', 'model', 'transmission', 'fuelType']:
    encoder = encoders.get(col)
    if encoder:
        input_data[col] = encoder.transform(input_data[col])

if st.button("Prediksi Harga"):
    predicted_price = model.predict(input_data)[0]
    harga_rupiah = int(predicted_price * 21000)
    st.success(f"Perkiraan Harga Mobil Bekas: Rp {harga_rupiah:,.0f}")

st.markdown("---")
st.markdown("**Nama :** Muhammad Istiqlal  \n**NPM :** 51421006  \n**Skripsi Jurusan Informatika – Universitas Gunadarma**")
