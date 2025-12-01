Folder project dibuat otomatis dari notebook:
- notebook_script.py : hasil konversi dari IPYNB
- app.py : aplikasi Streamlit siap pakai
- utils.py : helper preprocess dan postprocess
- requirements.txt : daftar paket

Langkah singkat:
1. Masuk folder project:
   cd C://Users//USER//Downloads//streamlit-deploy
2. Buat virtual env:
   python -m venv .venv
3. Aktifkan venv
   Windows: .venv\Scripts\activate
   macOS/Linux: source .venv/bin/activate
4. Install dependensi:
   pip install -r requirements.txt
5. Jalankan streamlit:
   streamlit run app.py
6. Jika notebook menghasilkan model.h5, letakkan model.h5 di folder ini.
   Atau upload model.h5 lewat UI Streamlit.