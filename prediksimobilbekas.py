import streamlit as st
import pickle
import pandas as pd

# Load model dan encoder
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

year_input = st.number_input("Tahun Mobil", min_value=2011, max_value=2020, value=2015)
transmission_input = st.selectbox("Jenis Transmisi", transmission_options)
mileage_km = st.number_input("Jarak Tempuh (km)", min_value=0, value=50000)
fueltype_input = st.selectbox("Jenis Bahan Bakar", fueltype_options)
tax_rupiah = st.number_input("Biaya Pajak (Rp)", min_value=0, value=2000000)
mpg_input = st.number_input("Konsumsi BBM (mpg)", min_value=0.0, value=32.0)
enginesize_input = st.number_input("Ukuran Mesin (L)", min_value=0.0, value=1.5)

# Input pilih bulan prediksi (1 sampai 12)
prediksi_bulan_ke = st.number_input(
    "Pilih bulan ke depan untuk prediksi harga mobil (1-12):",
    min_value=1,
    max_value=12,
    value=1,
    step=1
)

# Validasi input
def cek_input_valid(nilai, nama):
    if nilai <= 0:
        st.warning(f"⚠️ Harap lengkapi data, nilai '{nama}' tidak boleh <= 0.")
        st.stop()

cek_input_valid(mileage_km, 'Jarak Tempuh (km)')
cek_input_valid(tax_rupiah, 'Biaya Pajak (Rp)')
cek_input_valid(mpg_input, 'Konsumsi BBM (mpg)')
cek_input_valid(enginesize_input, 'Ukuran Mesin (L)')

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

    # Gunakan lowercase untuk pencocokan
    brand_input_lower = brand_input.strip().lower()

    # Definisikan faktor dalam lowercase
    brand_factors = {
        'hyundai': 0.55,
        'ford': 0.65
    }

    faktor_penyesuaian = brand_factors.get(brand_input_lower, 0.7)

    # Prediksi harga saat ini (bulan 0)
    input_df_now = pd.DataFrame([input_base])
    for col in categorical_cols:
        encoder = encoders.get(col)
        val = input_df_now.at[0, col]
        if val not in encoder.classes_:
            st.error(f"⚠️ Nilai '{val}' tidak dikenali pada kolom '{col}'.")
            st.stop()
        input_df_now[col] = encoder.transform([val])
    input_df_now = input_df_now[feature_order]
    pred_gbp_now = model.predict(input_df_now)[0]
    harga_saat_ini = int(pred_gbp_now * kurs_gbp_to_idr * faktor_penyesuaian)

    st.success(f"✅ Harga Saat Ini: Rp {harga_saat_ini:,.0f}")

    # Dummy depresiasi bulanan manual berdasarkan merk
    if brand_input_lower == 'hyundai':
        monthly_depreciation = 0.01  # 1%
    elif brand_input_lower == 'ford':
        monthly_depreciation = 0.015  # 1.5%
    else:
        monthly_depreciation = 0.0125  # default

    # Hitung harga prediksi bulan ke-n
    harga_dummy = int(harga_saat_ini * ((1 - monthly_depreciation) ** prediksi_bulan_ke))

    # Format label bulan
    label_bulan = "bulan depan" if prediksi_bulan_ke == 1 else f"{prediksi_bulan_ke} bulan ke depan"

    st.success(f"✅ Prediksi Harga Mobil {label_bulan} adalah: Rp {harga_dummy:,.0f}")
