import streamlit as st
import pandas as pd

def about_page():
    st.markdown(
        """
        <div style="text-align: center; color: #008080; font-size: 40px; font-weight: bold;"> 
            <h1>About Drowsiness Detection System</h1>
            <p>Real-time video feed to detect drowsiness using CNN.</p>
        </div>
        """, unsafe_allow_html=True
    )
    
    tab1, tab2, tab3 = st.tabs(["Drowsiness", "Dataset", "Author"])
    
    with tab1:
        st.subheader("Pengertian Drowsiness")
        st.write("""
            Drowsiness atau Kantuk adalah kondisi ketika seseorang merasa ingin tidur. 
            Kondisi ini biasa terjadi pada malam hari atau kadang di siang hari. 
            Kantuk merupakan hal yang wajar, tetapi perlu diatasi jika terjadi secara berlebihan, 
            mengganggu aktivitas, atau menurunkan produktivitas.
        """)
    
    with tab2:   
        st.subheader("Data Gambar Kantuk Ketika Berkendara")
        st.write("""
            Dataset Driver Drowsiness (DDD) adalah kumpulan data yang berisi wajah pengemudi 
            yang diekstraksi dan dipotong dari video pada Real-Life Drowsiness Dataset. 
            Frame-frame tersebut diekstraksi dari video menjadi gambar menggunakan perangkat lunak VLC. 
            Setelah itu, algoritma Viola-Jones digunakan untuk mengekstraksi region of interest dari gambar yang diambil. 
            Dataset yang diperoleh (DDD) telah digunakan untuk melatih dan menguji arsitektur CNN untuk deteksi kantuk pada pengemudi 
            dalam makalah berjudul “Detection and Prediction of Driver Drowsiness for the Prevention of Road Accidents Using Deep Neural Networks Techniques”.
        """)
        url = "https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd?resource=download"
        st.caption("Data mentah dapat diperoleh dari KAGGLE [Driver Drowsiness Dataset (DDD)](%s)" % url)
    
    with tab3:   
        st.subheader("Author")
        st.write("""
            Program ini dikembangkan oleh:
            <table style="border-collapse: collapse; width: 100%; text-align: left;">
                <thead>
                    <tr style="background-color: #008080;">
                        <th style="padding: 8px; border: 1px solid #ddd;">Nama</th>
                        <th style="padding: 8px; border: 1px solid #ddd;">NIM</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td style="padding: 8px; border: 1px solid #ddd;">WAHYU SURYANING TYAS</td>
                        <td style="padding: 8px; border: 1px solid #ddd;">A11.2022.14731</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; border: 1px solid #ddd;">ADEL JANANI JANMABHUMI P.</td>
                        <td style="padding: 8px; border: 1px solid #ddd;">A11.2021.13574</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; border: 1px solid #ddd;">MARGARETA VALENCIA SUCI H.</td>
                        <td style="padding: 8px; border: 1px solid #ddd;">A11.2022.14704</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; border: 1px solid #ddd;">GILARDINHO JAVIERE O.P.S.</td>
                        <td style="padding: 8px; border: 1px solid #ddd;">A11.2022.14732</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; border: 1px solid #ddd;">EKA RIZKY ANGGI SARASWATI</td>
                        <td style="padding: 8px; border: 1px solid #ddd;">A11.2022.14789</td>
                    </tr>
                </tbody>
            </table>
        """, unsafe_allow_html=True)
