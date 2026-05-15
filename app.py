import streamlit as st
import pickle
import pandas as pd
import re
import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# ======================
# LOAD MODEL
# ======================
model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(
    page_title="Analisis Sentimen Fashion",
    page_icon="👜",
    layout="wide"
)

# ======================
# PREPROCESSING
# ======================
def preprocess(text):

    text = str(text).lower()

    slang_dict = {
        'ihh': '',
        'ihhh': '',
        'emm': '',
        'hmm': '',
        'bangettt': 'banget',
        'bagusss': 'bagus',
        'jelekkk': 'jelek',
        'mantapp': 'mantap',
        'bgt': 'banget',
        'gk': 'tidak',
        'ga': 'tidak',
        'nggak': 'tidak'
    }

    for key, value in slang_dict.items():
        text = text.replace(key, value)

    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# ======================
# STYLE WEBSITE
# ======================
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #d9f3ff, #eef9ff);
}

.block-container {
    background-color: rgba(255,255,255,0.92);
    padding: 2rem;
    border-radius: 25px;
}

h1 {
    text-align: center;
    color: #2c4e63;
}

h2, h3 {
    color: #355c74;
}

.stButton>button {
    background-color: #8ed1f0;
    color: black;
    border-radius: 12px;
    border: none;
    padding: 10px 20px;
    font-weight: bold;
}

.stButton>button:hover {
    background-color: #70c2e8;
}

textarea {
    border-radius: 12px !important;
}

</style>
""", unsafe_allow_html=True)

# ======================
# HEADER
# ======================
st.markdown(
    "<h1>👜 Analisis Sentimen Ulasan Toko Fashion</h1>",
    unsafe_allow_html=True
)

st.write("""
Web ini dirancang untuk menganalisis sentimen ulasan pelanggan terhadap produk tas fashion 
dari tiga toko, yaitu Les Catino, Mossdoom, dan Priorbags, menggunakan metode 
TF-IDF dan Support Vector Machine (SVM).

Hasil analisis dikategorikan menjadi sentimen positif, netral, dan negatif.
""")

st.markdown("---")

# ======================
# GAMBAR TAS
# ======================
if os.path.exists("tas.png"):
    st.image("tas.png", use_container_width=True)

# ======================
# PILIH TOKO
# ======================
toko = st.selectbox(
    "Pilih Toko",
    ["Les Catino", "Mossdoom", "Priorbags"]
)

# ======================
# INPUT ULASAN
# ======================
st.subheader("✍️ Prediksi Sentimen")

text = st.text_area("Masukkan ulasan pelanggan:")

# ======================
# PREDIKSI SENTIMEN
# ======================
if st.button("Prediksi Sentimen"):

    if text.strip() != "":

        processed = preprocess(text)

        vector = tfidf.transform([processed])

        hasil = model.predict(vector)[0]

        # ======================
        # HASIL PREPROCESSING
        # ======================
        st.write("### Hasil Preprocessing")

        st.info(processed)

        # ======================
        # HASIL SENTIMEN
        # ======================
        st.write("### Hasil Prediksi Sentimen")

        if hasil == "positif":
            st.success("💚 Sentimen Positif")

        elif hasil == "netral":
            st.warning("🟡 Sentimen Netral")

        else:
            st.error("❤️ Sentimen Negatif")

        # ======================
        # WORDCLOUD INPUT USER
        # ======================
        st.subheader("☁️ WordCloud Input Pengguna")

        st.write("""
WordCloud berikut menampilkan kata-kata yang muncul dari ulasan yang dimasukkan pengguna.
Semakin besar ukuran kata, maka semakin dominan kata tersebut pada ulasan.
""")

        wc_live = WordCloud(
            width=700,
            height=300,
            background_color='white'
        ).generate(processed)

        fig_live, ax_live = plt.subplots(figsize=(10,4))

        ax_live.imshow(wc_live, interpolation='bilinear')

        ax_live.axis('off')

        st.pyplot(fig_live)

    else:
        st.warning("Masukkan teks terlebih dahulu")

# ======================
# LOAD DATASET
# ======================
st.markdown("---")

st.subheader("📊 Analisis Dataset")

try:
    df = pd.read_excel("dataset.xlsx")

except:
    st.error("Dataset tidak ditemukan")
    df = None

# ======================
# PROSES DATASET
# ======================
if df is not None:

    df['clean'] = df['text_review'].apply(preprocess)

    df['prediksi'] = model.predict(
        tfidf.transform(df['clean'])
    )

    # IDENTIFIKASI TOKO
    def get_toko(nama):

        nama = str(nama).lower()

        if 'les catino' in nama:
            return 'Les Catino'

        elif 'mossdoom' in nama:
            return 'Mossdoom'

        elif 'prior' in nama:
            return 'Priorbags'

        else:
            return 'Lainnya'

    df['toko'] = df['nama_produk'].apply(get_toko)

    # ======================
    # TAMPILKAN DATA
    # ======================
    st.subheader("📄 Contoh Dataset")

    st.write("""
Tabel berikut menampilkan sebagian data ulasan pelanggan yang telah melalui proses preprocessing 
dan prediksi sentimen menggunakan model Support Vector Machine (SVM).
""")

    st.dataframe(
        df[['text_review', 'prediksi', 'toko']].head()
    )

    # ======================
    # DISTRIBUSI SENTIMEN
    # ======================
    st.markdown("---")

    st.subheader("📈 Distribusi Sentimen Per Toko")

    st.write("""
Visualisasi berikut menunjukkan distribusi hasil analisis sentimen pada masing-masing toko fashion.
Grafik menampilkan jumlah ulasan dengan kategori positif, netral, dan negatif berdasarkan hasil prediksi model SVM.
""")

    daftar_toko = [
        'Les Catino',
        'Mossdoom',
        'Priorbags'
    ]

    for nama_toko in daftar_toko:

        st.write(f"## 🛍️ {nama_toko}")

        data_toko = df[df['toko'] == nama_toko]

        sentimen_count = data_toko['prediksi'].value_counts()

        fig, ax = plt.subplots(figsize=(5,4))

        ax.bar(
            sentimen_count.index,
            sentimen_count.values
        )

        ax.set_xlabel('Sentimen')
        ax.set_ylabel('Jumlah')
        ax.set_title(f'Distribusi Sentimen {nama_toko}')

        st.pyplot(fig)

    # ======================
    # WORDCLOUD PER TOKO
    # ======================
    st.markdown("---")

    st.subheader("☁️ WordCloud Per Toko")

    st.write("""
WordCloud digunakan untuk menampilkan kata-kata yang paling sering muncul pada ulasan pelanggan di masing-masing toko.
Semakin besar ukuran kata pada WordCloud, maka semakin sering kata tersebut muncul pada data ulasan.
""")

    col1, col2, col3 = st.columns(3)

    toko_list = [
        ('Les Catino', col1),
        ('Mossdoom', col2),
        ('Priorbags', col3)
    ]

    for nama_toko, kolom in toko_list:

        data_toko = df[df['toko'] == nama_toko]

        text_toko = ' '.join(
            data_toko['clean'].astype(str)
        )

        if text_toko.strip() != '':

            wc = WordCloud(
                width=500,
                height=300,
                background_color='white'
            ).generate(text_toko)

            fig_wc, ax_wc = plt.subplots(figsize=(5,3))

            ax_wc.imshow(wc, interpolation='bilinear')

            ax_wc.axis('off')

            kolom.write(f"### {nama_toko}")

            kolom.pyplot(fig_wc)

# ======================
# FOOTER
# ======================
st.markdown("---")

st.caption(
    "Aplikasi Analisis Sentimen Ulasan Produk Fashion Menggunakan TF-IDF dan Support Vector Machine (SVM)"
)