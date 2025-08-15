import streamlit as st
import pickle
import pandas as pd

# ===== Fungsi Format =====
def format_ribuan(val):
    try:
        val_int = int(str(val).replace(",", "").replace(".", ""))
        return f"{val_int:,}"
    except:
        return val

def parse_angka(val):
    try:
        return int(str(val).replace(",", "").replace(".", ""))
    except:
        return 0

# ===== Inisialisasi Session State =====
if "mileage_km" not in st.session_state:
    st.session_state.mileage_km = "50,000"
if "user_budget" not in st.session_state:
    st.session_state.user_budget = "100,000,000"

# ===== Event Handler =====
def update_mileage():
    st.session_state.mileage_km = format_ribuan(st.session_state.mileage_km)

def update_budget():
    st.session_state.user_budget = format_ribuan(st.session_state.user_budget)

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

# ===== Input Real-Time Format Ribuan (Hanya Mileage dan Budget) =====
mileage_km_str = st.text_input("Jarak Tempuh (km)", key="mileage_km", on_change=update_mileage)
mileage_km = parse_angka(mileage_km_str)

fueltype_input = st.selectbox("Jenis Bahan Bakar", fueltype_options)

enginesize_input = st.number_input("Ukuran Mesin (L)", min_value=0.0, value=1.5)

prediksi_bulan_ke = st.number_input(
    "Pilih bulan ke depan untuk prediksi harga mobil (1-12):",
    min_value=1,
    max_value=12,
    value=1,
    step=1
)

user_budget_str = st.text_input("Masukkan budget Anda (Rp)", key="user_budget", on_change=update_budget)
user_budget = parse_angka(user_budget_str)

# ===== Validasi Input =====
def cek_input_valid(nilai, nama):
    if nilai <= 0:
        st.warning(f"⚠️ Harap lengkapi data, nilai '{nama}' tidak boleh <= 0.")
        st.stop()

cek_input_valid(mileage_km, 'Jarak Tempuh (km)')
cek_input_valid(enginesize_input, 'Ukuran Mesin (L)')

# ===== Ambil Pajak & MPG Otomatis =====
default_row = brand_model_df[
    (brand_model_df['brand'] == brand_input) &
    (brand_model_df['model'] == model_input)
].iloc[0]

tax_rupiah = default_row['tax_default']
mpg_input = default_row['mpg_default']

# ===== Konversi Satuan =====
mileage_mil = mileage_km / 1.60934
tax_pound = tax_rupiah / 21000

# ===== Siapkan Input Model =====
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

# ===== Prediksi =====
if st.button("Prediksi Harga"):
    kurs_gbp_to_idr = 21000
    brand_input_lower = brand_input.strip().lower()

    brand_factors = {
        'hyundai': 0.55,
        'ford': 0.65
    }

    faktor_penyesuaian = brand_factors.get(brand_input_lower, 0.7)

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

    # Depresiasi harga
    if brand_input_lower == 'hyundai':
        monthly_depreciation = 0.01
    elif brand_input_lower == 'ford':
        monthly_depreciation = 0.015
    else:
        monthly_depreciation = 0.0125

    harga_dummy = int(harga_saat_ini * ((1 - monthly_depreciation) ** prediksi_bulan_ke))

    label_bulan = "bulan depan" if prediksi_bulan_ke == 1 else f"{prediksi_bulan_ke} bulan ke depan"
    st.success(f"✅ Prediksi Harga Mobil {label_bulan} adalah: Rp {harga_dummy:,.0f}")

    # Cek apakah budget cukup
    if user_budget < harga_dummy:
        st.error(
            f"💰 Budget Anda: Rp {user_budget:,.0f} masih di bawah harga prediksi mobil ({label_bulan}): Rp {harga_dummy:,.0f}.\n\n"
            "👉 Saran: Pilih mobil atau tipe lain, atau pertimbangkan tahun produksi yang lebih tua."
        )
    else:
        st.info(
            f"✅ Budget Anda: Rp {user_budget:,.0f} cukup untuk membeli mobil ini di {label_bulan}."
        )

    # ===== Rekomendasi Alternatif =====
    st.markdown("---")
    st.subheader("🔎 Rekomendasi Mobil Alternatif dalam Budget Anda")

    rekomendasi_list = []

    for th in range(2011, year_input):
        models_same_brand = brand_model_df[brand_model_df['brand'] == brand_input]['model'].unique()
        for m in models_same_brand:
            temp_input = input_base.copy()
            temp_input['model'] = m
            temp_input['year'] = th

            temp_df = pd.DataFrame([temp_input])
            skip = False
            for col in categorical_cols:
                val = temp_df.at[0, col]
                encoder = encoders[col]
                if val not in encoder.classes_:
                    skip = True
                    break
                temp_df[col] = encoder.transform([val])
            if skip:
                continue

            temp_df = temp_df[feature_order]
            pred_gbp = model.predict(temp_df)[0]
            harga_rupiah = int(pred_gbp * kurs_gbp_to_idr * faktor_penyesuaian)
            harga_prediksi = int(harga_rupiah * ((1 - monthly_depreciation) ** prediksi_bulan_ke))

            if harga_prediksi <= user_budget:
                rekomendasi_list.append({
                    'Model': m,
                    'Tahun': th,
                    'Harga Prediksi': f"Rp {harga_prediksi:,.0f}"
                })

    if rekomendasi_list:
        df_rekomendasi = pd.DataFrame(rekomendasi_list)
        df_rekomendasi.index = df_rekomendasi.index + 1  
        st.table(df_rekomendasi)
    else:
        st.write("❌ Tidak ada rekomendasi yang cocok dengan budget Anda untuk merek dan tipe yang lebih tua.")
