import streamlit as st
import pickle
import pandas as pd

# Load model dan encoder (pastikan ini model terbaru sudah termasuk fitur 'month')
model = pickle.load(open('best_random_forest_model.sav', 'rb'))
encoders = pickle.load(open('best_label_encoders.sav', 'rb'))

# Load dataframe brand-model mapping
with open('brand_model_mapping.pkl', 'rb') as f:
    brand_model_df = pickle.load(f)

brand_model_df['brand'] = brand_model_df['brand'].str.strip()
brand_model_df['model'] = brand_model_df['model'].str.strip()

brand_options = sorted(brand_model_df['brand'].unique())

st.set_page_config(page_title="Prediksi Harga Mobil Bekas", layout="centered")
st.title("Prediksi Harga Mobil Bekas")
st.write("Masukkan spesifikasi mobil dan bulan ke depan untuk prediksi harga.")

# Input dasar
brand_input = st.selectbox("Merek Mobil", brand_options)
filtered_models = brand_model_df.loc[brand_model_df['brand'] == brand_input, 'model'].unique()
filtered_models = sorted([m.strip() for m in filtered_models])
model_input = st.selectbox("Model Mobil", filtered_models)

transmission_options = list(encoders['transmission'].classes_)
fueltype_options = list(encoders['fuelType'].classes_)

year_input = st.number_input("Tahun Mobil", min_value=2011, max_value=2020, value=2020)
transmission_input = st.selectbox("Jenis Transmisi", transmission_options)
mileage_km = st.number_input("Jarak Tempuh (km)", min_value=0, value=50000)
fueltype_input = st.selectbox("Jenis Bahan Bakar", fueltype_options)
tax_rupiah = st.number_input("Biaya Pajak (Rp)", min_value=0, value=2000000)
mpg_input = st.number_input("Konsumsi BBM (mpg)", min_value=0.0, value=40.0)
enginesize_input = st.number_input("Ukuran Mesin (L)", min_value=0.0, value=1.5)

# Input bulan prediksi dinamis
months_to_predict = st.text_input(
    "Masukkan bulan ke depan untuk prediksi, pisahkan dengan koma (contoh: 0,1,2,3,6,12,24)",
    value="0,1,2,3,6,12,24"
)
try:
    months_list = [int(m.strip()) for m in months_to_predict.split(",")]
except Exception:
    st.error("Format input bulan salah, masukkan angka dipisah koma.")
    st.stop()

# Validasi input
def cek_input_valid(nilai):
    if nilai <= 0:
        st.warning("⚠️ Harap lengkapi data, nilai tidak boleh <= 0.")
        st.stop()

cek_input_valid(mileage_km)
cek_input_valid(tax_rupiah)
cek_input_valid(mpg_input)
cek_input_valid(enginesize_input)

# Konversi satuan
mileage_mil = mileage_km / 1.60934
tax_pound = tax_rupiah / 21000

# Siapkan input dasar tanpa 'month'
input_base = {
    'brand': brand_input.strip(),
    'model': model_input.strip(),
    'year': year_input,
    'transmission': transmission_input.strip(),
    'mileage': mileage_mil,
    'fuelType': fueltype_input.strip(),
    'tax': tax_pound,
    'mpg': mpg_input,
    'engineSize': enginesize_input
}

categorical_cols = ['brand', 'model', 'transmission', 'fuelType']
feature_order = list(model.feature_names_in_)

if st.button("Prediksi Harga"):
    kurs_gbp_to_idr = 21000
    brand_factors = {
        'Hyundai': 0.75,
        'Ford': 0.65
    }
    faktor_penyesuaian = brand_factors.get(brand_input, 0.7)

    hasil_prediksi = []

    for bulan in months_list:
        input_data = input_base.copy()
        input_data['month'] = bulan

        input_df = pd.DataFrame([input_data])

        # Encoding kolom kategorikal
        for col in categorical_cols:
            encoder = encoders.get(col)
            val = input_df.at[0, col]
            if val not in encoder.classes_:
                st.error(f"⚠️ Nilai '{val}' tidak dikenali pada kolom '{col}'.")
                st.stop()
            input_df[col] = encoder.transform([val])

        # Pastikan urutan kolom sesuai model
        input_df = input_df[feature_order]

        # Prediksi harga GBP
        pred_gbp = model.predict(input_df)[0]
        pred_rp = int(pred_gbp * kurs_gbp_to_idr * faktor_penyesuaian)

        label = "Harga Saat Ini" if bulan == 0 else f"{bulan} bulan ke depan"
        hasil_prediksi.append((label, pred_rp))

    # Tampilkan hasil prediksi dengan bullet dan bold
    st.success("✅ Hasil Prediksi Harga Mobil:")
    for label, harga in hasil_prediksi:
        st.markdown(f"🔹 **{label}:** Rp {harga:,.0f}")
