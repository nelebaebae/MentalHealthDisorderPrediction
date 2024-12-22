import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Page Configuration
st.set_page_config(
    page_title="Mental Health Survey",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Styling
st.markdown(
    """
    <style>
    body {
        font-family: Arial, sans-serif;
    }
    .main {
        background-color: #f9f9f9;
    }
    h1, h2, h3 {
        color: white;
        text-align: left;
    }
    .stButton button {
        background-color: #3a506b; 
        color: white; 
        border-radius: 10px; 
        font-size: 18px;
        padding: 10px 20px;
    }
    .stButton button:hover {
        background-color: #2c3e50;
    }
    .question-container {
        margin-bottom: 20px;
        padding: 20px;
        border: 2px solid #ccc;
        border-radius: 10px;
        font-size: 18px;
        text-align: left;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load Model dan Label Encoder
try:
    model = joblib.load("naive_bayes_best_model.pkl")  # File model
    label_encoder = joblib.load("label_encoder.pkl")  # Label encoder
except FileNotFoundError:
    st.error("Model atau Label Encoder tidak ditemukan! Pastikan file `naive_bayes_best_model.pkl` dan `label_encoder.pkl` ada di direktori kerja.")
    st.stop()

# Data Penjelasan Penyakit
# Penjelasan Penyakit dan Penanganannya
mental_health_info = {
    "ADHD": {
        "description": "ADHD adalah gangguan perkembangan saraf yang ditandai dengan kesulitan fokus, hiperaktivitas, dan impulsivitas.",
        "treatment": [
            "Gunakan pengingat atau jadwal harian untuk membantu manajemen waktu.",
            "Pertimbangkan terapi perilaku atau konsultasi dengan psikolog.",
            "Obat-obatan seperti stimulan dapat diresepkan oleh dokter."
        ]
    },
    "ASD": {
        "description": "ASD adalah gangguan perkembangan yang memengaruhi komunikasi, interaksi sosial, dan perilaku.",
        "treatment": [
            "Terapi perilaku untuk membantu interaksi sosial dan komunikasi.",
            "Dukungan keluarga dan lingkungan yang inklusif sangat penting.",
            "Pendidikan khusus sesuai kebutuhan individu."
        ]
    },
    "Loneliness": {
        "description": "Kesepian adalah perasaan terisolasi yang dapat memengaruhi kesehatan mental secara signifikan.",
        "treatment": [
            "Bergabunglah dengan komunitas atau kegiatan sosial.",
            "Berbicara dengan teman, keluarga, atau konselor.",
            "Fokus pada pengembangan hubungan positif."
        ]
    },
    "MDD": {
        "description": "MDD (Major Depressive Disorder) adalah gangguan depresi berat yang ditandai dengan perasaan sedih yang mendalam dan kehilangan minat.",
        "treatment": [
            "Terapi kognitif perilaku (CBT).",
            "Obat antidepresan sesuai resep dokter.",
            "Jaga rutinitas harian yang sehat dan olahraga teratur."
        ]
    },
    "OCD": {
        "description": "OCD adalah gangguan yang menyebabkan pikiran obsesif dan perilaku kompulsif berulang.",
        "treatment": [
            "Terapi CBT untuk mengurangi obsesi dan kompulsi.",
            "Obat SSRI dapat membantu mengurangi gejala.",
            "Buat jurnal untuk mengidentifikasi pola pemicu."
        ]
    },
    "PDD": {
        "description": "PDD (Persistent Depressive Disorder) adalah bentuk depresi jangka panjang yang berlangsung lebih dari dua tahun.",
        "treatment": [
            "Konsultasi rutin dengan psikolog.",
            "Perubahan gaya hidup, seperti diet sehat dan olahraga.",
            "Terapi dukungan dari keluarga atau kelompok."
        ]
    },
    "PTSD": {
        "description": "PTSD adalah gangguan stres pasca-trauma yang terjadi setelah pengalaman traumatis.",
        "treatment": [
            "Terapi trauma untuk mengatasi memori yang menyakitkan.",
            "Meditasi atau latihan pernapasan untuk mengurangi stres.",
            "Konsultasi dengan ahli kesehatan mental untuk perawatan."
        ]
    },
    "anexiety": {
        "description": "Kecemasan adalah gangguan mental yang ditandai dengan rasa takut atau khawatir berlebihan.",
        "treatment": [
            "Latihan pernapasan dalam untuk menenangkan diri.",
            "Olahraga teratur untuk mengurangi stres.",
            "Konsultasi dengan terapis jika kecemasan terus berlanjut."
        ]
    },
    "bipolar": {
        "description": "Bipolar adalah gangguan mood yang menyebabkan perubahan suasana hati yang ekstrem antara mania dan depresi.",
        "treatment": [
            "Pengobatan stabilisasi mood, seperti lithium.",
            "Terapi psikologis untuk mengelola episode mood.",
            "Hindari stres berlebihan dan jaga pola tidur teratur."
        ]
    },
    "eating disorder": {
        "description": "Gangguan makan meliputi pola makan yang tidak teratur, seperti anoreksia atau bulimia.",
        "treatment": [
            "Konsultasi dengan ahli gizi untuk perencanaan makan.",
            "Terapi psikologis untuk mengatasi penyebab emosional.",
            "Dukungan keluarga untuk pemulihan yang optimal."
        ]
    },
    "psychotic depression": {
        "description": "Depresi psikotik adalah bentuk depresi berat yang disertai dengan gejala psikotik seperti delusi atau halusinasi.",
        "treatment": [
            "Pengobatan antipsikotik sesuai anjuran dokter.",
            "Terapi intensif untuk mengelola gejala.",
            "Konsultasi rutin dengan psikiater."
        ]
    },
    "sleeping disorder": {
        "description": "Gangguan tidur mencakup insomnia, sleep apnea, dan masalah lainnya yang memengaruhi kualitas tidur.",
        "treatment": [
            "Tetapkan jadwal tidur yang konsisten.",
            "Hindari kafein dan perangkat elektronik sebelum tidur.",
            "Konsultasi dengan spesialis tidur jika masalah berlanjut."
        ]
    }
}

# Fitur Dataset dan Pertanyaan yang Disesuaikan
questions_mapping = {
    "ag+1:629e": "Berapa usia Anda?",
    "feeling.nervous": "Apakah Anda merasa gugup atau tegang?",
    "panic": "Apakah Anda sering merasa panik tiba-tiba tanpa sebab yang jelas?",
    "breathing.rapidly": "Apakah Anda mengalami napas cepat tanpa sebab yang jelas?",
    "sweating": "Apakah Anda sering berkeringat berlebihan?",
    "trouble.in.concentration": "Apakah Anda merasa kesulitan berkonsentrasi?",
    "having.trouble.in.sleeping": "Apakah Anda mengalami kesulitan tidur atau sering terbangun di malam hari?",
    "having.trouble.with.work": "Apakah Anda merasa sulit menyelesaikan pekerjaan atau tugas harian?",
    "hopelessness": "Apakah Anda merasa kehilangan harapan terhadap masa depan?",
    "anger": "Apakah Anda sering merasa marah tanpa alasan jelas?",
    "over.react": "Apakah Anda merasa reaksi Anda terhadap sesuatu sering berlebihan?",
    "change.in.eating": "Apakah Anda mengalami perubahan pola makan (lebih banyak atau lebih sedikit)?",
    "suicidal.thought": "Apakah Anda pernah memiliki pikiran untuk tidak melanjutkan hidup?",
    "feeling.tired": "Apakah Anda merasa lelah meskipun sudah cukup istirahat?",
    "close.friend": "Apakah Anda merasa hubungan dengan teman dekat semakin sulit atau terasingkan?",
    "social.media.addiction": "Apakah Anda merasa kecanduan media sosial?",
    "weight.gain": "Apakah Anda mengalami kenaikan berat badan tanpa sebab yang jelas?",
    "introvert": "Apakah Anda merasa lebih suka menyendiri atau menghindari keramaian?",
    "popping.up.stressful.memory": "Apakah Anda sering teringat kenangan yang membuat stres?",
    "having.nightmares": "Apakah Anda sering mengalami mimpi buruk?",
    "avoids.people.or.activities": "Apakah Anda sering menghindari orang atau aktivitas yang biasa Anda nikmati?",
    "feeling.negative": "Apakah Anda sering merasa negatif tentang diri sendiri?",
    "trouble.concentrating": "Apakah Anda merasa kesulitan untuk fokus?",
    "blamming.yourself": "Apakah Anda sering menyalahkan diri sendiri?",
    "hallucinations": "Apakah Anda pernah melihat atau mendengar sesuatu yang tidak dirasakan orang lain?",
    "repetitive.behaviour": "Apakah Anda merasa melakukan perilaku tertentu secara berulang tanpa bisa menghentikannya?",
    "seasonally": "Apakah mood Anda berubah secara drastis pada waktu tertentu setiap tahun?",
    "increased.energy": "Apakah Anda merasa memiliki energi yang meningkat secara tiba-tiba?",
}

# Inisialisasi st.session_state untuk kontrol menu
# Inisialisasi st.session_state untuk kontrol menu
if "menu" not in st.session_state:
    st.session_state.menu = "Home"

# Navigasi berdasarkan st.session_state.menu
menu = st.sidebar.radio(
    "Navigasi", ["Home", "Survey", "Hasil Prediksi", "Tentang"],
    index=["Home", "Survey", "Hasil Prediksi", "Tentang"].index(st.session_state.menu)
)

if menu == "Home":
    st.title("Selamat Datang di Mental Health Survey")

    st.image(
        "https://www.snapsurveys.com/blog/wp-content/uploads/2011/08/demographic-sample-580x387.png",  # URL atau path gambar
        caption="Peduli Kesehatan Mental, Langkah Menuju Masa Depan Cerah",
        width=750
    )

    st.markdown(
        """
        <div class="info-box">
            <h2>Mengapa Penting Peduli dengan Kesehatan Mental?</h2>
            <p>
                Gangguan kesehatan mental, seperti <strong>depresi</strong>, <strong>kecemasan</strong>, dan <strong>stres</strong>, 
                sering kali tidak terdiagnosis akibat stigma sosial dan keterbatasan akses layanan kesehatan. 
                Dengan teknologi modern, kita bisa mengidentifikasi masalah ini lebih awal dan memberikan langkah awal yang penting.
            </p>
        </div>
        <div class="info-box">
            <h2>Solusi Kami</h2>
            <p>
                Aplikasi ini menggunakan pendekatan berbasis <strong>machine learning</strong> untuk menganalisis data survei kesehatan mental Anda. 
                Kami bertujuan untuk membantu Anda memahami kondisi Anda lebih baik serta memberikan rekomendasi awal untuk langkah selanjutnya.
            </p>
        </div>
        <div class="info-box">
            <h2>Cara Kerja Aplikasi</h2>
            <ol>
                <li>Isi survei kesehatan mental dengan jujur dan penuh kesadaran.</li>
                <li>Model machine learning akan memproses data Anda secara anonim.</li>
                <li>Dapatkan hasil prediksi dan saran yang sesuai dengan kondisi Anda.</li>
            </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("**Mari mulai perjalanan Anda menuju kesehatan mental yang lebih baik!**")


elif menu == "Survey":
    st.title("Survey Kesehatan Mental")
    st.write("Jawab pertanyaan berikut dengan jujur.")
    
    responses = {}
    for feature, question in questions_mapping.items():
        if feature == "ag+1:629e":
            responses[feature] = st.number_input(question, min_value=1, max_value=120, value=25)
        else:
            responses[feature] = st.radio(question, ["Ya", "Tidak"])

    if st.button("Submit Survey"):
        # Konversi Jawaban
        input_data = {feature: 0 for feature in questions_mapping.keys()}
        for feature, response in responses.items():
            if feature == "ag+1:629e":
                input_data[feature] = response
            else:
                input_data[feature] = 1 if response == "Ya" else 0
        input_df = pd.DataFrame([input_data])

        # Simpan Data untuk Hasil Prediksi
        st.session_state["input_data"] = input_df
        st.session_state["responses"] = responses
        st.success("Jawaban berhasil disimpan. Lihat hasil prediksi pada menu 'Hasil Prediksi'.")
elif menu == "Hasil Prediksi":
    if "input_data" in st.session_state:
        input_df = st.session_state["input_data"]
        responses = st.session_state["responses"]
        try:
            prediction = model.predict(input_df)
            predicted_disorder = label_encoder.inverse_transform(prediction)
        except Exception as e:
            st.error(f"Terjadi kesalahan saat prediksi: {e}")
            st.stop()

        st.header("Hasil Prediksi Diagnosis")
        disorder = predicted_disorder[0]
        st.write(f"Berdasarkan model, Anda mungkin memiliki gangguan: **{disorder}**")

        if disorder in mental_health_info:
            st.subheader("Penjelasan Penyakit")
            st.write(mental_health_info[disorder]["description"])

            st.subheader("Cara Penanganan")
            for step in mental_health_info[disorder]["treatment"]:
                st.write(f"- {step}")
        else:
            st.write("Informasi tentang penyakit ini belum tersedia.")

        st.header("Distribusi Gejala")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(
            list(questions_mapping.values())[1:],
            [1 if responses[feature] == "Ya" else 0 for feature in list(questions_mapping.keys())[1:]],
            color='skyblue'
        )
        ax.set_xlabel("Jumlah Gejala Teridentifikasi")
        ax.set_ylabel("Gejala")
        ax.set_title("Distribusi Gejala Berdasarkan Jawaban Survei")
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("Silakan isi survei terlebih dahulu.")
elif menu == "Tentang":
    st.title("Tentang Aplikasi")
    st.write("Aplikasi ini dirancang untuk memenuhi tugas akhir Mata Kuliah Decision Support System.")
    st.markdown(
        """
        <div class="info-box">
            <h2>Dibuat oleh : </h2>
            <ol>
                <li>140810220068 - Naufal Fakhri Ilyas</li>
                <li>140810220075 - Rayhan Nugrah Kristio</li>
                <li>140810220078 - Anel Fuad Abiyyu</li>
            </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )
