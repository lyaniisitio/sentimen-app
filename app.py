import streamlit as st
import pickle
import pandas as pd
import re
import os
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# ======================
# LOAD MODEL
# ======================
model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

# ======================
# PREPROCESSING
# ======================
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

# ======================
# STYLE (SOFT CREAM ELEGANT)
# ======================
st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(135deg, #fdf6f0, #f7efe5);
}

/* Container */
.block-container {
    background-color: rgba(255, 255, 255, 0.9);
    padding: 2rem;
    border-radius: 20px;
}

/* Judul */
h1 {
    text-align: center;
    color: #5a4a42;
}

/* Font */
body {
    font-family: 'Segoe UI', sans-serif;
}

/* Button */
.stButton>button {
    background-color: #e6cfc3;
    color: black;
    border-radius: 12px;
    border: none;
    padding: 10px 20px;
    font-weight: bold;
}

.stButton>button:hover {
    background-color: #d8bfb3;
}

/* Input */
textarea {
    border-radius: 10px !important;
}

</style>
""", unsafe_allow_html=True)

# ======================
# HEADER
# ======================
st.markdown("<h1>👜 Analisis Sentimen Ulasan Toko Fashion</h1>", unsafe_allow_html=True)
st.markdown("---")

# ======================
# GAMBAR
# ======================
if os.path.exists("tas.png"):
    st.image("tas.png", use_container_width=True)
else:
    st.warning("⚠️ File tas.png tidak ditemukan")

# ======================
# DESKRIPSI
# ======================
st.write("""
Web ini dirancang untuk menganalisis sentimen ulasan pelanggan terhadap produk tas fashion 
dari tiga toko, yaitu Les Catino, Mossdoom, dan Priorbags, menggunakan metode 
TF-IDF dan Support Vector Machine (SVM).  
Hasil analisis dikategorikan menjadi sentimen positif, netral, dan negatif.
""")

st.markdown("---")

# ======================
# PILIH TOKO
# ======================
toko = st.selectbox(
    "Pilih Toko",
    ["Les Catino", "Mossdoom", "Priorbags"]
)

st.info(f"Toko yang dipilih: {toko}")

# ======================
# INPUT USER
# ======================
text = st.text_area("Masukkan Ulasan:")

# ======================
# PREDIKSI
# ======================
if st.button("Prediksi Sentimen"):
    if text.strip() != "":
        processed = preprocess(text)
        vector = tfidf.transform([processed])
        hasil = model.predict(vector)[0]

        st.markdown("---")
        st.subheader("🔍 Hasil Analisis")

        if hasil == "positif":
            st.success("💚 Sentimen Positif: Produk dinilai baik oleh pelanggan")
        elif hasil == "netral":
            st.warning("🟡 Sentimen Netral: Ulasan bersifat biasa")
        else:
            st.error("❤️ Sentimen Negatif: Produk memiliki kekurangan")
    else:
        st.warning("Masukkan teks dulu!")

# ======================
# ANALISIS DATASET
# ======================
st.markdown("---")
st.subheader("📊 Analisis Dataset")

try:
    df = pd.read_excel("dataset.xlsx")
    st.info("Menggunakan dataset default")
except:
    st.error("Dataset tidak ditemukan")
    df = None

# ======================
# PROSES DATA
# ======================
if df is not None:
    df['clean'] = df['text_review'].apply(preprocess)
    df['prediksi'] = model.predict(tfidf.transform(df['clean']))

    def get_toko(nama):
        nama = str(nama).lower()
        if "les catino" in nama:
            return "Les Catino"
        elif "mossdoom" in nama:
            return "Mossdoom"
        elif "prior" in nama:
            return "Priorbags"
        else:
            return "Lainnya"

    df['toko'] = df['nama_produk'].apply(get_toko)

    st.subheader("📄 Contoh Data")
    st.dataframe(df.head())

    tabel = pd.crosstab(df['toko'], df['prediksi'])

    st.subheader("📊 Distribusi Sentimen per Toko")
    st.dataframe(tabel)

    st.subheader("📈 Grafik Distribusi Sentimen")
    st.bar_chart(tabel)

    # ======================
    # INSIGHT
    # ======================
    st.subheader("🧠 Insight")

    if 'positif' in tabel.columns:
        toko_terbaik = tabel['positif'].idxmax()
        st.success(f"Toko dengan sentimen positif tertinggi: {toko_terbaik}")

    # ======================
    # WORDCLOUD
    # ======================
    st.subheader("☁️ WordCloud per Toko")

    toko_list = ["Les Catino", "Mossdoom", "Priorbags"]

    for t in toko_list:
        st.write(f"WordCloud - {t}")

        teks = " ".join(df[df['toko'] == t]['clean'])

        if teks.strip() != "":
            wc = WordCloud(width=800, height=400, background_color='white').generate(teks)

            fig, ax = plt.subplots()
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')

            st.pyplot(fig)
        else:
            st.warning(f"Tidak ada data untuk {t}")

    st.caption("Dataset digunakan sebagai data uji dalam sistem analisis sentimen")