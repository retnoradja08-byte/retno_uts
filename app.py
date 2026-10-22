# -*- coding: utf-8 -*-
"""
UTS Business Intelligence — Integrasi Data Mining & BI pada Data Transaksi Toko
Jalankan:  python -m streamlit run app.py
"""
import base64, io, os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import streamlit as st
from PIL import Image
from plotly.subplots import make_subplots
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# ════════════════════════════════════════════════════════════════
#  ⚙️  KONFIGURASI — UBAH DATA DIRI ANDA DI SINI
# ════════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PROFILE = {
    "nama": "STEFANIA LAURA RETNO RADJA",
    "npm": "51230165",
    "kelas": "Kelas B",
    "prodi": "Teknologi Informasi",
    "matkul": "Business Intelligence / Data Mining",
    "ujian": "Ujian Tengah Semester (UTS)",
    "foto": "foto_profil.png",   # taruh file foto di folder yang sama dengan app.py
}
LOGIN_USER, LOGIN_PASS = "admin", "uts2026"
SAMPLE_FILE = os.path.join(BASE_DIR, "P8__UTS_Business_Intelligence.xlsx")

NAVY, BLUE, CYAN, ORANGE = "#0b1f4d", "#1e5bd8", "#22b8f0", "#ff8a1f"
PALETTE = [BLUE, ORANGE, CYAN, "#7c4dff", "#2bb673", "#e5484d", "#f5b700", "#64748b"]

st.set_page_config(page_title="UTS Business Intelligence", page_icon="📊", layout="wide")

# ════════════════════════════════════════════════════════════════
#  🎨  STYLE
# ════════════════════════════════════════════════════════════════
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, button, input {{ font-family:'Poppins',sans-serif; }}
.stApp {{ background: linear-gradient(180deg,#eef3fc 0%,#f8faff 100%); }}
header[data-testid="stHeader"] {{ background: transparent; }}
section[data-testid="stSidebar"] {{ background: linear-gradient(180deg,{NAVY} 0%,#0f2f7a 100%); }}
section[data-testid="stSidebar"] * {{ color:#eaf1ff; }}

.hero {{ background: linear-gradient(120deg,{NAVY} 0%,{BLUE} 60%,{CYAN} 130%); color:#fff; border-radius:24px;
        padding:34px 38px; box-shadow:0 14px 40px rgba(30,91,216,.30); position:relative; overflow:hidden; }}
.hero:after {{ content:""; position:absolute; right:-60px; top:-60px; width:260px; height:260px; border-radius:50%;
        background:radial-gradient(circle,{ORANGE}aa 0%,transparent 70%); }}
.hero h1 {{ margin:0; font-size:2.1rem; font-weight:800; color:#fff; line-height:1.2; }}
.hero p {{ margin:.4rem 0 0; opacity:.92; font-size:1rem; }}
.badge {{ display:inline-block; background:rgba(255,255,255,.16); border:1px solid rgba(255,255,255,.35);
        padding:4px 14px; border-radius:99px; font-size:.78rem; font-weight:600; margin-bottom:10px; }}
.card {{ background:#fff; border-radius:20px; padding:22px 24px; box-shadow:0 6px 24px rgba(15,47,122,.08);
        border:1px solid #e6edfb; height:100%; }}
.card h4 {{ margin:0 0 6px; color:{NAVY}; font-weight:700; }}
.pg-title {{ color:{NAVY}; font-weight:800; font-size:1.7rem; margin:0; }}
.pg-sub {{ color:#5b6b8c; margin:0 0 14px; }}
.tag {{ display:inline-block; background:#e8f0ff; color:{BLUE}; padding:3px 12px; border-radius:99px; font-size:.75rem; font-weight:600; }}
.kpi {{ background:#fff; border-radius:18px; padding:18px 20px; border:1px solid #e6edfb; box-shadow:0 6px 20px rgba(15,47,122,.07);
        border-left:6px solid {BLUE}; }}
.kpi .l {{ color:#5b6b8c; font-size:.8rem; font-weight:600; text-transform:uppercase; letter-spacing:.04em; }}
.kpi .v {{ color:{NAVY}; font-size:1.55rem; font-weight:800; }}
.insight {{ background:#fff8ee; border-left:5px solid {ORANGE}; border-radius:12px; padding:12px 16px; color:#5a3a0a; margin:8px 0; }}
.okbox {{ background:#eefaf3; border-left:5px solid #2bb673; border-radius:12px; padding:12px 16px; color:#14532d; margin:8px 0; }}
.avatar {{ width:96px; height:96px; border-radius:50%; object-fit:cover; border:3px solid {ORANGE}; display:block; margin:auto; }}
.avatar-i {{ width:96px; height:96px; border-radius:50%; background:linear-gradient(135deg,{ORANGE},{BLUE}); color:#fff;
        font-weight:800; font-size:2rem; display:flex; align-items:center; justify-content:center; margin:auto; border:3px solid #fff3; }}
.pf {{ text-align:center; padding:6px 0 10px; }}
.pf b {{ font-size:1rem; }} .pf small {{ opacity:.85; }}
.stButton>button, .stDownloadButton>button {{ border-radius:12px; font-weight:600; }}

/* ========== MENU SIDEBAR (tombol pill) ========== */
section[data-testid="stSidebar"] .stRadio [role="radiogroup"] {{ gap:10px; }}
section[data-testid="stSidebar"] .stRadio label {{
    width:100%; padding:13px 18px; border-radius:14px;
    background:rgba(255,255,255,.08); border:2px solid rgba(255,255,255,.20);
    transition:all .2s ease; cursor:pointer; }}
section[data-testid="stSidebar"] .stRadio label > div:first-child {{ display:none; }}
section[data-testid="stSidebar"] .stRadio label p {{ font-size:.98rem; font-weight:600; }}
section[data-testid="stSidebar"] .stRadio label:hover {{ background:rgba(255,255,255,.22); transform:translateX(5px); }}
section[data-testid="stSidebar"] .stRadio label:has(input:checked) {{
    background:linear-gradient(120deg,{ORANGE},#ffb15c); border-color:#fff;
    box-shadow:0 8px 22px rgba(255,138,31,.55); }}
section[data-testid="stSidebar"] .stRadio label:has(input:checked) * {{ color:#fff !important; font-weight:800; }}

/* ========== SUB MENU (TAB) - BESAR & JELAS ========== */
div[data-testid="stTabs"] [role="tablist"] {{
    gap:12px; background:#ffffff; padding:12px; border-radius:20px;
    border:2px solid {BLUE}33; box-shadow:0 10px 28px rgba(15,47,122,.14);
    flex-wrap:wrap; overflow:visible; }}
div[data-testid="stTabs"] button[role="tab"] {{
    height:auto; min-height:52px; padding:12px 24px; border-radius:14px;
    background:#eef3fc; border:2px solid #c9d9fa; transition:all .2s ease; }}
div[data-testid="stTabs"] button[role="tab"] p,
div[data-testid="stTabs"] button[role="tab"] div {{
    font-size:1.02rem !important; font-weight:700 !important; color:{NAVY} !important; }}
div[data-testid="stTabs"] button[role="tab"]:hover {{
    background:#dbe7ff; border-color:{BLUE}; transform:translateY(-3px);
    box-shadow:0 6px 14px rgba(30,91,216,.22); }}
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {{
    background:linear-gradient(120deg,{BLUE},{CYAN}); border-color:{NAVY};
    box-shadow:0 10px 22px rgba(30,91,216,.50); transform:translateY(-3px); }}
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] p,
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] div {{ color:#ffffff !important; font-weight:800 !important; }}
div[data-testid="stTabs"] [data-baseweb="tab-highlight"],
div[data-testid="stTabs"] [data-baseweb="tab-border"] {{ display:none !important; }}
div[data-testid="stTabs"] [data-baseweb="tab-panel"] {{ padding-top:22px; }}

/* ========== JUDUL SUB BAGIAN (##### 2a, 2b, ...) ========== */
h5 {{ color:{NAVY}; font-weight:800; font-size:1.15rem; background:#fff3e3;
      border-left:6px solid {ORANGE}; border-radius:0 12px 12px 0;
      padding:8px 14px; margin:22px 0 10px; }}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
#  🔧  UTILITAS
# ════════════════════════════════════════════════════════════════
def rp(x):
    return "Rp " + f"{x:,.0f}".replace(",", ".")

def num(x, d=0):
    return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")

def page_head(title, sub, tag=None):
    t = f'<span class="tag">{tag}</span>' if tag else ""
    st.markdown(f'{t}<p class="pg-title">{title}</p><p class="pg-sub">{sub}</p>', unsafe_allow_html=True)

def kpi(col, label, value, color=BLUE):
    col.markdown(f'<div class="kpi" style="border-left-color:{color}"><div class="l">{label}</div><div class="v">{value}</div></div>',
                 unsafe_allow_html=True)

def insight(text, ok=False):
    st.markdown(f'<div class="{"okbox" if ok else "insight"}">{text}</div>', unsafe_allow_html=True)

def style_fig(fig, h=380):
    fig.update_layout(template="plotly_white", height=h, margin=dict(l=10, r=10, t=50, b=10),
                      font=dict(family="Poppins"), colorway=PALETTE, title_font=dict(size=15, color=NAVY))
    return fig

def to_png(data: bytes) -> bytes:
    """Ubah gambar apa pun (JPG/JPEG/WEBP/PNG) menjadi PNG persegi 300x300."""
    img = Image.open(io.BytesIO(data)).convert("RGBA")
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)).resize((300, 300))
    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()

def avatar_html():
    data = st.session_state.get("foto_bytes")
    path = os.path.join(BASE_DIR, PROFILE["foto"])
    if data is None and os.path.exists(path):
        try:
            data = to_png(open(path, "rb").read())
        except Exception:
            data = None
    if data:
        return f'<img class="avatar" src="data:image/png;base64,{base64.b64encode(data).decode()}">'
    ini = "".join(w[0] for w in st.session_state.prof["nama"].split()[:2]).upper() or "?"
    return f'<div class="avatar-i">{ini}</div>'


# ════════════════════════════════════════════════════════════════
#  🧹  DATA: LOAD, PREPROCESSING, FITUR
# ════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_file(name, content):
    bio = io.BytesIO(content)
    if name.lower().endswith(".csv"):
        return pd.read_csv(bio), None, None
    xl = pd.ExcelFile(bio)
    sheet = "Data_Raw" if "Data_Raw" in xl.sheet_names else xl.sheet_names[0]
    raw = xl.parse(sheet)
    prod = xl.parse("Data_Produk") if "Data_Produk" in xl.sheet_names else None
    cust = xl.parse("Data_Pelanggan") if "Data_Pelanggan" in xl.sheet_names else None
    return raw, prod, cust

def canon(s):
    """Samakan variasi huruf/spasi ke bentuk yang paling sering muncul."""
    t = s.astype("string").str.strip()
    key = t.str.lower()
    best = t.groupby(key).agg(lambda x: x.value_counts().index[0])
    return key.map(best)

@st.cache_data(show_spinner=False)
def preprocess(raw, prod, cust, qty_cap):
    df = raw.copy()
    n0, log = len(df), []
    def add(step, masalah, n, tindakan, alasan):
        log.append({"Langkah": step, "Masalah": masalah, "Jumlah": int(n), "Penanganan": tindakan, "Alasan": alasan})

    df["Transaction_Date"] = pd.to_datetime(df["Transaction_Date"], errors="coerce")
    for c in ["Quantity", "Unit_Price", "Discount_Rate", "Line_Total"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 1. teks tidak konsisten
    for c in ["Category", "Payment_Method", "Region", "Customer_Type", "Product_Name"]:
        new = canon(df[c]); chg = ((df[c].astype("string") != new) & df[c].notna()).sum()
        df[c] = new
        if chg: add("1. Format teks", f"Variasi huruf/spasi di {c}", chg, "Seragamkan ke bentuk terbanyak", "Mencegah satu kategori terhitung ganda")

    # 2. duplikat
    d = df.duplicated().sum(); df = df.drop_duplicates().reset_index(drop=True)
    add("2. Duplikasi", "Baris identik (satu item dicatat ulang)", d, "Hapus, simpan 1", "Duplikat menggelembungkan penjualan & support Apriori")

    # 3. referensi produk
    if prod is not None:
        ref = prod.set_index("Product_Name")["Reference_Unit_Price_Rp"]
        pid = prod.set_index("Product_Name")["Product_ID"]
        cat = prod.set_index("Product_Name")["Category"]
    else:
        ref = df[df.Unit_Price > 0].groupby("Product_Name")["Unit_Price"].median()
        pid = df.dropna(subset=["Product_ID"]).groupby("Product_Name")["Product_ID"].agg(lambda x: x.mode()[0])
        cat = df.groupby("Product_Name")["Category"].agg(lambda x: x.mode()[0])
    m = df.Product_ID.isna(); df.loc[m, "Product_ID"] = df.loc[m, "Product_Name"].map(pid)
    if m.sum(): add("3. Missing value", "Product_ID kosong", m.sum(), "Isi dari nama produk (Data_Produk)", "Nama produk lengkap → ID bisa dipulihkan tanpa menebak")
    df["Category"] = df["Product_Name"].map(cat).fillna(df["Category"])

    # 4. atribut pelanggan / transaksi: pulihkan dari baris lain pada nota yang sama
    for c in ["Customer_ID", "Region", "Customer_Type"]:
        m = df[c].isna()
        df[c] = df[c].fillna(df.groupby("Transaction_ID")[c].transform("first"))
        if c != "Customer_ID" and cust is not None:
            df[c] = df[c].fillna(df["Customer_ID"].map(cust.set_index("Customer_ID")[c]))
        rec = m.sum() - df[c].isna().sum()
        if m.sum(): add("3. Missing value", f"{c} kosong", m.sum(), f"Dipulihkan {rec} baris dari nota/master pelanggan", "Satu nota = satu pelanggan, sehingga nilai bisa dipulihkan")
    m = df.Customer_ID.isna().sum()
    if m: add("3. Missing value", "Customer_ID tetap kosong", m, "Dipertahankan untuk Apriori & penjualan, dikeluarkan dari RFM/K-Means", "Tanpa ID tidak mungkin dihitung RFM per pelanggan")
    for c in ["Region", "Customer_Type"]: df[c] = df[c].fillna("Tidak Diketahui")
    m = df.Payment_Method.isna().sum(); df["Payment_Method"] = df["Payment_Method"].fillna("Tidak Diketahui")
    if m: add("3. Missing value", "Payment_Method kosong", m, "Isi 'Tidak Diketahui'", "Kategori; tidak boleh menebak metode bayar")

    # 5. harga tidak wajar
    r = df.Product_Name.map(ref)
    bad = df.Unit_Price.isna() | (df.Unit_Price <= 0) | (df.Unit_Price > 3 * r) | (df.Unit_Price < 0.3 * r)
    df.loc[bad, "Unit_Price"] = r[bad]
    add("4. Nilai tidak wajar", "Unit_Price kosong / ≤0 / jauh dari harga referensi", bad.sum(), "Ganti harga referensi produk", "Harga satu produk relatif tetap (99,7% baris = harga referensi)")

    # 6. quantity tidak wajar
    lo = df.Quantity.isna() | (df.Quantity <= 0) | (df.Quantity > qty_cap)
    rec = (df.Line_Total / (df.Unit_Price * (1 - df.Discount_Rate))).round()
    ok = lo & rec.between(1, qty_cap)
    df.loc[ok, "Quantity"] = rec[ok]
    drop = lo & ~ok
    df = df[~drop].reset_index(drop=True)
    add("4. Nilai tidak wajar", f"Quantity kosong / ≤0 / >{qty_cap}", lo.sum(), f"{ok.sum()} dipulihkan dari Line_Total, {drop.sum()} dihapus", "Kuantitas ≤0 mustahil; 999 = salah input (outlier ekstrem konteks ritel)")

    # 7. Line_Total
    lt = (df.Quantity * df.Unit_Price * (1 - df.Discount_Rate)).round(0)
    chg = (df.Line_Total.isna() | ((df.Line_Total - lt).abs() > 1)).sum()
    df["Line_Total"] = lt
    add("5. Transformasi", "Line_Total kosong / tidak konsisten", chg, "Hitung ulang = Qty × Harga × (1 − Diskon)", "Rumus terbukti konsisten dengan baris yang valid")

    # 8. fitur turunan
    df["Quantity"] = df.Quantity.astype(int)
    df["Date"] = df.Transaction_Date.dt.normalize()
    df["Month"] = df.Transaction_Date.dt.to_period("M").dt.to_timestamp()
    df["Hour"] = df.Transaction_Date.dt.hour
    df["basket_size"] = df.groupby("Transaction_ID")["Product_Name"].transform("nunique")
    add("5. Transformasi", "Tipe data & fitur baru", len(df), "Tanggal→datetime, Qty→int, tambah Month, Hour, basket_size", "Dibutuhkan untuk tren bulanan, RFM, dan Apriori")
    return df, pd.DataFrame(log), n0

@st.cache_data(show_spinner=False)
def make_rfm(df):
    d = df.dropna(subset=["Customer_ID"])
    snap = d.Transaction_Date.max() + pd.Timedelta(days=1)
    g = d.groupby("Customer_ID")
    rfm = pd.DataFrame({
        "Recency": (snap - g.Transaction_Date.max()).dt.days,
        "Frequency": g.Transaction_ID.nunique(),
        "Monetary": g.Line_Total.sum(),
        "Avg_Basket": g.basket_size.mean(),
    }).reset_index()
    return rfm

def feats(rfm, use_log):
    X = rfm[["Recency", "Frequency", "Monetary"]].astype(float)
    return StandardScaler().fit_transform(np.log1p(X) if use_log else X)

@st.cache_data(show_spinner=False)
def kmeans_scan(rfm, use_log, kmax=8):
    X = feats(rfm, use_log); out = []
    for k in range(2, kmax + 1):
        km = KMeans(k, n_init=10, random_state=42).fit(X)
        out.append({"k": k, "Inertia (WCSS)": km.inertia_, "Silhouette": silhouette_score(X, km.labels_)})
    return pd.DataFrame(out)

def label_clusters(prof):
    """Beri label berdasarkan skor gabungan (F + M − R) dari centroid asli."""
    z = (prof[["Recency", "Frequency", "Monetary"]] - prof[["Recency", "Frequency", "Monetary"]].mean()) / \
        prof[["Recency", "Frequency", "Monetary"]].std(ddof=0).replace(0, 1)
    score = z.Frequency + z.Monetary - z.Recency
    order = score.sort_values(ascending=False).index.tolist()
    labels = {}
    for i, c in enumerate(order):
        if i == 0: labels[c] = "Loyal Customers"
        elif i == len(order) - 1: labels[c] = "Churn Risk"
        elif z.Recency[c] <= 0: labels[c] = "Potential Customers" if i == 1 else "New / Active Customers"
        else: labels[c] = "Need Attention" if i == 1 else "Hibernating"
    seen = {}
    for c in labels:
        seen[labels[c]] = seen.get(labels[c], 0) + 1
        if list(labels.values()).count(labels[c]) > 1: labels[c] += f" {seen[labels[c]]}"
    return labels

@st.cache_data(show_spinner=False)
def run_kmeans(rfm, k, use_log):
    X = feats(rfm, use_log)
    km = KMeans(k, n_init=10, random_state=42).fit(X)
    r = rfm.copy(); r["Cluster"] = km.labels_
    prof = r.groupby("Cluster")[["Recency", "Frequency", "Monetary", "Avg_Basket"]].mean()
    lab = label_clusters(prof)
    r["Segmen"] = r.Cluster.map(lab)
    from sklearn.decomposition import PCA
    p = PCA(2, random_state=42).fit_transform(X); r["PC1"], r["PC2"] = p[:, 0], p[:, 1]
    return r, silhouette_score(X, km.labels_)

@st.cache_data(show_spinner=False)
def run_apriori(df, min_sup, min_conf, max_len=3):
    tx = df.groupby(["Transaction_ID", "Product_Name"]).size().unstack(fill_value=0).gt(0)
    fi = apriori(tx, min_support=min_sup, use_colnames=True, max_len=max_len)
    if fi.empty: return tx, fi, pd.DataFrame()
    try:
        ru = association_rules(fi, metric="confidence", min_threshold=min_conf, num_itemsets=len(tx))
    except TypeError:
        ru = association_rules(fi, metric="confidence", min_threshold=min_conf)
    if ru.empty: return tx, fi, ru
    ru["Antecedent"] = ru.antecedents.apply(lambda s: ", ".join(sorted(s)))
    ru["Consequent"] = ru.consequents.apply(lambda s: ", ".join(sorted(s)))
    ru["Aturan"] = ru.Antecedent + "  →  " + ru.Consequent
    return tx, fi, ru.sort_values(["lift", "confidence"], ascending=False).reset_index(drop=True)

def top_rules(ru, n=10, min_lift=1.0):
    if ru.empty: return ru
    r = ru[ru.lift > min_lift]
    r = r if len(r) else ru
    return r.head(n)[["Aturan", "support", "confidence", "lift"]].rename(
        columns={"support": "Support", "confidence": "Confidence", "lift": "Lift"})

def require_data():
    if "clean" not in st.session_state:
        st.warning("Belum ada data. Silakan unggah dataset di menu **📁 Upload Data** terlebih dahulu.")
        st.stop()
    return st.session_state.clean


# ════════════════════════════════════════════════════════════════
#  🔐  LOGIN
# ════════════════════════════════════════════════════════════════
def login_page():
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    c1, c2 = st.columns([1.15, 1], gap="large")
    with c1:
        st.markdown(f"""<div class="hero" style="min-height:430px">
        <span class="badge">UJIAN TENGAH SEMESTER</span>
        <h1>Business Intelligence<br>& Data Mining</h1>
        <p>Integrasi Data Mining dan Business Intelligence pada Data Transaksi Toko —
        EDA, Preprocessing, Apriori, K-Means, dan Dashboard Interaktif.</p><br>
        <p style="font-size:.9rem">📊 Market Analysis &nbsp;•&nbsp; 🧹 Data Preprocessing &nbsp;•&nbsp; 🛒 Apriori &nbsp;•&nbsp; 👥 K-Means</p>
        <br><div style="font-weight:700">{st.session_state.prof['nama']}</div>
        <div style="opacity:.85">NPM {st.session_state.prof['npm']} · {st.session_state.prof['kelas']}</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='pf' style='color:{NAVY}'>{avatar_html()}</div>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align:center;color:{NAVY};margin:0'>Selamat Datang 👋</h3>"
                    f"<p style='text-align:center;color:#5b6b8c'>Masuk untuk membuka dashboard analisis</p>", unsafe_allow_html=True)
        with st.form("login"):
            u = st.text_input("Username", placeholder="admin")
            p = st.text_input("Password", type="password", placeholder="••••••••")
            if st.form_submit_button("🔓 Masuk", width="stretch"):
                if u == LOGIN_USER and p == LOGIN_PASS:
                    st.session_state.auth = True; st.rerun()
                else:
                    st.error("Username atau password salah.")
        st.caption(f"Demo login → username: **{LOGIN_USER}** · password: **{LOGIN_PASS}**")


# ════════════════════════════════════════════════════════════════
#  🏠  BERANDA
# ════════════════════════════════════════════════════════════════
def page_home():
    p = st.session_state.prof
    st.markdown(f"""<div class="hero"><span class="badge">{p['ujian']}</span>
    <h1>Integrasi Data Mining & Business Intelligence<br>pada Data Transaksi Toko</h1>
    <p>Dari data transaksi mentah → informasi → pola pembelian → segmentasi pelanggan → rekomendasi keputusan.</p></div>""",
                unsafe_allow_html=True)
    st.write("")
    c1, c2 = st.columns([1, 2], gap="large")
    with c1:
        st.markdown(f"""<div class="card" style="text-align:center">{avatar_html()}
        <h4 style="margin-top:12px">{p['nama']}</h4>
        <div style="color:#5b6b8c;line-height:1.7">NPM: <b>{p['npm']}</b><br>{p['kelas']}<br>{p['prodi']}</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="card"><h4>📋 Identitas Ujian</h4>
        <table style="width:100%;color:#334">
        <tr><td width="35%">Mata Kuliah</td><td><b>{p['matkul']}</b></td></tr>
        <tr><td>Jenis Ujian</td><td><b>{p['ujian']}</b></td></tr>
        <tr><td>Studi Kasus</td><td>Integrasi Data Mining & BI pada Data Transaksi Toko</td></tr>
        <tr><td>Praktikum</td><td>EDA · Data Preprocessing · Apriori · K-Means · BI Dashboard</td></tr>
        <tr><td>Peran</td><td>Data Analyst & Business Intelligence Analyst</td></tr></table></div>""", unsafe_allow_html=True)
    st.write("")
    st.markdown("#### 🧭 Alur Pengerjaan")
    steps = [("📁", "Upload Data", "Unggah dataset"), ("🔍", "EDA", "10% · Eksplorasi"), ("🧹", "Preprocessing", "25% · Bersihkan"),
             ("🛒", "Apriori", "20% · Pola beli"), ("👥", "K-Means", "20% · Segmentasi"), ("📊", "Dashboard", "15% · BI"),
             ("💡", "Rekomendasi", "10% · Keputusan")]
    for col, (i, t, s) in zip(st.columns(len(steps)), steps):
        col.markdown(f"<div class='card' style='text-align:center;padding:14px 6px'><div style='font-size:1.8rem'>{i}</div>"
                     f"<b style='color:{NAVY};font-size:.85rem'>{t}</b><br><small style='color:#5b6b8c'>{s}</small></div>", unsafe_allow_html=True)
    st.write("")
    if "clean" in st.session_state:
        d = st.session_state.clean
        c = st.columns(4)
        kpi(c[0], "Total Penjualan", rp(d.Line_Total.sum())); kpi(c[1], "Transaksi", num(d.Transaction_ID.nunique()), ORANGE)
        kpi(c[2], "Pelanggan", num(d.Customer_ID.nunique()), CYAN); kpi(c[3], "Rata-rata Basket", num(d.drop_duplicates("Transaction_ID").basket_size.mean(), 2), "#7c4dff")
    else:
        insight("👉 Mulai dengan membuka menu <b>📁 Upload Data</b> di sidebar untuk mengunggah dataset transaksi.")


# ════════════════════════════════════════════════════════════════
#  📁  UPLOAD
# ════════════════════════════════════════════════════════════════
def page_upload():
    page_head("Upload Dataset", "Unggah file transaksi toko (.xlsx atau .csv). Sheet Data_Raw, Data_Produk, Data_Pelanggan terbaca otomatis.", "TAHAP 0")
    f = st.file_uploader("Pilih file dataset", type=["xlsx", "csv"], help="Contoh: P8__UTS_Business_Intelligence.xlsx")
    use_sample = False
    if os.path.exists(SAMPLE_FILE):
        use_sample = st.button("📦 Gunakan dataset contoh dari dosen")
    if f is not None or use_sample:
        name, content = (f.name, f.getvalue()) if f is not None else (os.path.basename(SAMPLE_FILE), open(SAMPLE_FILE, "rb").read())
        raw, prod, cust = load_file(name, content)
        need = {"Transaction_ID", "Transaction_Date", "Product_Name", "Quantity", "Unit_Price", "Line_Total"}
        miss = need - set(raw.columns)
        if miss:
            st.error(f"Kolom wajib tidak ditemukan: {', '.join(sorted(miss))}"); return
        st.session_state.update(raw=raw, prod=prod, cust=cust, fname=name)
        st.session_state.pop("clean", None)
    if "raw" not in st.session_state:
        insight("Belum ada file. Unggah dataset untuk memulai analisis."); return
    raw = st.session_state.raw
    st.success(f"✅ **{st.session_state.fname}** berhasil dimuat — {raw.shape[0]:,} baris × {raw.shape[1]} kolom".replace(",", "."))
    st.dataframe(raw.head(100), width="stretch", height=300)
    cap = st.slider("Batas maksimum kuantitas wajar per item (konteks ritel)", 5, 50, 10,
                    help="Kuantitas di atas batas ini dianggap salah input (mis. 999).")
    if st.button("🚀 Jalankan Preprocessing Otomatis", type="primary") or "clean" not in st.session_state:
        with st.spinner("Membersihkan data…"):
            clean, log, n0 = preprocess(raw, st.session_state.prod, st.session_state.cust, cap)
        st.session_state.update(clean=clean, log=log, n0=n0, cap=cap)
        st.session_state.pop("k_final", None)
        st.success(f"Preprocessing selesai: {n0:,} → {len(clean):,} baris. Lanjut ke menu berikutnya.".replace(",", "."))


# ════════════════════════════════════════════════════════════════
#  🔍  BUSINESS UNDERSTANDING & EDA
# ════════════════════════════════════════════════════════════════
def page_eda():
    page_head("Business Understanding & EDA", "Memahami masalah bisnis dan kondisi awal data mentah.", "TAHAP 1 · 10%")
    if "raw" not in st.session_state:
        st.warning("Unggah dataset terlebih dahulu di menu Upload Data."); st.stop()
    raw = st.session_state.raw.copy()
    raw["Transaction_Date"] = pd.to_datetime(raw["Transaction_Date"], errors="coerce")
    t1, t2 = st.tabs(["🎯 Business Understanding", "🔍 Exploratory Data Analysis"])
    with t1:
        c1, c2 = st.columns(2, gap="large")
        c1.markdown("""<div class="card"><h4>1a. Permasalahan Utama</h4>
        Data transaksi hanya dipakai untuk <b>laporan penjualan rutin</b>. Perusahaan belum tahu:
        <ul><li>produk apa yang sering dibeli bersama,</li><li>pelanggan mana yang bernilai tinggi / berisiko pergi,</li>
        <li>strategi pemasaran apa yang tepat per kelompok pelanggan.</li></ul>
        Ditambah kualitas data buruk (missing, duplikat, outlier, format tidak konsisten).</div>""", unsafe_allow_html=True)
        c2.markdown("""<div class="card"><h4>1b. Pertanyaan Bisnis</h4><ol>
        <li>Kombinasi produk apa yang paling sering dibeli bersamaan (cross-selling/bundling)?</li>
        <li>Segmen pelanggan mana yang paling bernilai dan mana yang berisiko churn?</li>
        <li>Kapan penjualan meningkat/menurun dan kategori apa penyumbang terbesar?</li>
        <li>Wilayah & metode pembayaran mana yang dominan?</li></ol></div>""", unsafe_allow_html=True)
        st.write("")
        st.markdown("""<div class="card"><h4>1c. Manfaat Analisis bagi Manajemen</h4>
        <b>Meningkatkan penjualan:</b> Apriori → bundling, cross-selling, penataan rak berdekatan → menaikkan nilai keranjang.<br>
        <b>Mempertahankan pelanggan:</b> K-Means (RFM) → program loyalitas untuk pelanggan terbaik & kampanye reaktivasi untuk yang hampir hilang.<br>
        <b>Keputusan berbasis data:</b> dashboard KPI memantau tren dan hasil tindakan secara berkelanjutan.</div>""", unsafe_allow_html=True)
    with t2:
        st.markdown("##### 2a. Struktur Data")
        c = st.columns(4)
        kpi(c[0], "Baris", num(len(raw))); kpi(c[1], "Kolom", raw.shape[1], ORANGE)
        kpi(c[2], "Awal Periode", raw.Transaction_Date.min().strftime("%d %b %Y"), CYAN); kpi(c[3], "Akhir Periode", raw.Transaction_Date.max().strftime("%d %b %Y"), "#7c4dff")
        a, b = st.columns([1, 1.3])
        a.dataframe(pd.DataFrame({"Kolom": raw.dtypes.index, "Tipe": raw.dtypes.astype(str).values, "Non-null": raw.notna().sum().values}), width="stretch", height=330, hide_index=True)
        b.dataframe(raw.describe().T.round(2), width="stretch", height=330)
        st.markdown("##### 2b. Statistik Deskriptif")
        c = st.columns(5)
        kpi(c[0], "Transaksi Unik", num(raw.Transaction_ID.nunique())); kpi(c[1], "Pelanggan Unik", num(raw.Customer_ID.nunique()), ORANGE)
        kpi(c[2], "Produk Unik", raw.Product_Name.nunique(), CYAN)
        qv = raw.Quantity[(raw.Quantity > 0) & (raw.Quantity <= 10)]
        kpi(c[3], "Rata-rata Qty (wajar)", num(qv.mean(), 2), "#7c4dff"); kpi(c[4], "Total Penjualan", rp(raw.Line_Total.sum()), "#2bb673")

        st.markdown("##### 2c. Missing Values, Duplikat & Outlier")
        mv = raw.isna().sum().rename("Missing").to_frame(); mv["%"] = (mv.Missing / len(raw) * 100).round(2)
        x, y = st.columns([1, 1.4])
        x.dataframe(mv[mv.Missing > 0], width="stretch")
        x.metric("Baris duplikat penuh", int(raw.duplicated().sum()))
        incon = {c_: int((raw[c_].dropna().astype(str) != raw[c_].dropna().astype(str).str.strip().str.lower().map(
            raw[c_].dropna().astype(str).groupby(raw[c_].dropna().astype(str).str.strip().str.lower()).agg(lambda s: s.value_counts().index[0]))).sum())
            for c_ in ["Category", "Payment_Method"]}
        x.caption(f"Teks tidak konsisten: Category={incon['Category']}, Payment_Method={incon['Payment_Method']} baris")
        rows = []
        for col in ["Quantity", "Unit_Price", "Line_Total"]:
            s = raw[col].dropna(); q1, q3 = s.quantile([.25, .75]); i = q3 - q1
            rows.append({"Kolom": col, "Min": s.min(), "Max": s.max(), "Batas Atas IQR": q3 + 1.5 * i, "Outlier IQR": int((s > q3 + 1.5 * i).sum()), "Nilai ≤ 0": int((s <= 0).sum())})
        y.dataframe(pd.DataFrame(rows).round(0), width="stretch", hide_index=True)
        y.plotly_chart(style_fig(px.box(raw, x="Quantity", title="Sebaran Quantity (outlier: 999 & nilai negatif)", color_discrete_sequence=[ORANGE]), 220), width="stretch")

        st.markdown("##### 2d. Visualisasi Awal")
        g1, g2 = st.columns(2)
        mo = raw.dropna(subset=["Line_Total"]).assign(Bulan=raw.Transaction_Date.dt.to_period("M").dt.to_timestamp()).groupby("Bulan").Line_Total.sum().reset_index()
        g1.plotly_chart(style_fig(px.area(mo, x="Bulan", y="Line_Total", markers=True, title="Tren Penjualan Bulanan (data mentah)", color_discrete_sequence=[BLUE])), width="stretch")
        tp = raw.groupby("Product_Name").Line_Total.sum().nlargest(10).sort_values().reset_index()
        g2.plotly_chart(style_fig(px.bar(tp, x="Line_Total", y="Product_Name", orientation="h", title="10 Produk Penjualan Tertinggi", color_discrete_sequence=[ORANGE])), width="stretch")

        st.markdown("##### 2e. Temuan Awal")
        top = raw.groupby("Product_Name").Line_Total.sum().idxmax()
        pk = mo.loc[mo.Line_Total.idxmax()]; lw = mo.loc[mo.Line_Total.idxmin()]
        insight(f"<b>Temuan 1 — Kualitas data:</b> {int(raw.isna().any(axis=1).sum()):,} baris memuat missing value dan {int(raw.duplicated().sum())} baris duplikat; Quantity memuat nilai negatif/0 dan 999 → wajib dibersihkan sebelum Apriori & K-Means.".replace(",", "."))
        insight(f"<b>Temuan 2 — Penjualan:</b> bulan tertinggi {pk.Bulan:%B %Y} ({rp(pk.Line_Total)}), terendah {lw.Bulan:%B %Y} ({rp(lw.Line_Total)}); produk dengan pendapatan terbesar adalah <b>{top}</b>.")
        insight(f"<b>Temuan 3 — Struktur transaksi:</b> {len(raw):,} baris hanya mewakili {raw.Transaction_ID.nunique():,} nota (≈{len(raw)/raw.Transaction_ID.nunique():.1f} item/nota) → data berbentuk keranjang belanja dan cocok untuk Market Basket Analysis.".replace(",", "."))
        insight(f"<b>Temuan 4 — Segmen & pembayaran:</b> kategori terbesar <b>{raw.groupby('Category').Line_Total.sum().idxmax()}</b>; pelanggan Member = {(raw.Customer_Type=='Member').mean()*100:.0f}% baris.")


# ════════════════════════════════════════════════════════════════
#  🧹  PREPROCESSING
# ════════════════════════════════════════════════════════════════
def page_prep():
    page_head("Data Preprocessing", "Pembersihan, transformasi, dan feature engineering.", "TAHAP 2 · 25%")
    clean = require_data(); log, n0, raw = st.session_state.log, st.session_state.n0, st.session_state.raw
    t1, t2, t3, t4 = st.tabs(["🧼 Pembersihan & Alasan", "📊 Sebelum vs Sesudah", "⚙️ Feature Engineering", "🗂️ Dataset Besar & Unduh"])
    with t1:
        st.dataframe(log, width="stretch", hide_index=True)
        insight("Setiap keputusan didasarkan konteks bisnis: nilai yang <b>bisa dipulihkan</b> dari sumber lain (nota yang sama, master produk, rumus Line_Total) dipulihkan; "
                "nilai yang <b>mustahil</b> dan tidak dapat dipulihkan dihapus agar tidak mendistorsi model.")
    with t2:
        c = st.columns(4)
        removed = n0 - len(clean)
        kpi(c[0], "Baris Awal", num(n0)); kpi(c[1], "Baris Akhir (dipertahankan)", num(len(clean)), "#2bb673")
        kpi(c[2], "Baris Dihapus", num(removed), "#e5484d"); kpi(c[3], "Nilai Diubah/Diisi", num(int(log[~log.Penanganan.str.startswith("Hapus")].Jumlah.sum() - 0)), ORANGE)
        rawn = raw.copy()
        cmp = pd.DataFrame({
            "Indikator": ["Jumlah baris", "Total missing cell", "Baris duplikat", "Quantity ≤0 / >cap", "Unit_Price ≤0 / ekstrem", "Variasi kategori Category", "Variasi Payment_Method", "Total penjualan"],
            "Sebelum": [len(rawn), int(rawn.isna().sum().sum()), int(rawn.duplicated().sum()), int(((rawn.Quantity <= 0) | (rawn.Quantity > st.session_state.cap)).sum()),
                        int(((rawn.Unit_Price <= 0) | (rawn.Unit_Price > 100000)).sum()), rawn.Category.nunique(), rawn.Payment_Method.nunique(), rawn.Line_Total.sum()],
            "Sesudah": [len(clean), int(clean[["Product_ID", "Quantity", "Unit_Price", "Line_Total", "Payment_Method", "Region", "Customer_Type"]].isna().sum().sum()), int(clean.duplicated().sum()),
                        int(((clean.Quantity <= 0) | (clean.Quantity > st.session_state.cap)).sum()), int((clean.Unit_Price <= 0).sum()),
                        clean.Category.nunique(), clean.Payment_Method.nunique(), clean.Line_Total.sum()]})
        cmp[["Sebelum", "Sesudah"]] = cmp[["Sebelum", "Sesudah"]].astype(float).round(0)
        st.dataframe(cmp, width="stretch", hide_index=True)
        f = make_subplots(1, 2, subplot_titles=("Quantity sebelum", "Quantity sesudah"))
        f.add_trace(go.Histogram(x=rawn.Quantity.clip(-3, 12), marker_color="#e5484d"), 1, 1)
        f.add_trace(go.Histogram(x=clean.Quantity, marker_color="#2bb673"), 1, 2)
        st.plotly_chart(style_fig(f, 320).update_layout(showlegend=False), width="stretch")
    with t3:
        tx = clean.drop_duplicates("Transaction_ID")
        st.markdown("**basket_size** = jumlah produk berbeda per transaksi")
        a, b = st.columns(2)
        a.plotly_chart(style_fig(px.histogram(tx, x="basket_size", title="Distribusi basket_size", color_discrete_sequence=[BLUE]), 300), width="stretch")
        rfm = make_rfm(clean)
        b.markdown("**RFM per pelanggan** (snapshot = tanggal terakhir + 1 hari)")
        b.dataframe(rfm.head(8).round(1), width="stretch", hide_index=True)
        st.markdown("""<div class="card"><h4>Persiapan Algoritma</h4>
        <b>Apriori:</b> matriks transaksi × produk bernilai True/False (one-hot per nota; kuantitas tidak dipakai).<br>
        <b>K-Means:</b> matriks R, F, M per pelanggan → <code>log1p</code> (meredam skewness Monetary/Frequency) → <code>StandardScaler</code> (z-score).<br>
        <b>Alasan standardisasi:</b> K-Means berbasis jarak Euclidean; Monetary (jutaan Rupiah) akan mendominasi Recency (hari) bila skala tidak disamakan.</div>""", unsafe_allow_html=True)
        st.dataframe(rfm.describe().round(2), width="stretch")
    with t4:
        st.markdown("""<div class="card"><h4>Strategi Mengolah Dataset Berukuran Besar</h4><ul>
        <li><b>Sampling</b> — ambil sampel acak/stratifikasi untuk eksplorasi cepat dan tuning parameter.</li>
        <li><b>Chunking</b> — <code>pd.read_csv(chunksize=…)</code> untuk memproses bertahap tanpa memuat seluruh file.</li>
        <li><b>Agregasi</b> — ringkas ke level nota/pelanggan (RFM) sebelum K-Means; kurangi ukuran 18 ribu baris → 1.500 pelanggan.</li>
        <li><b>Tipe data efisien</b> — kategori & int8/float32; matriks Apriori boolean/sparse; batasi <code>max_len</code> itemset.</li></ul></div>""", unsafe_allow_html=True)
        st.write("")
        st.download_button("⬇️ Unduh dataset hasil preprocessing (CSV)", clean.to_csv(index=False).encode(), "data_bersih.csv", "text/csv")
        st.download_button("⬇️ Unduh dokumentasi perubahan (CSV)", log.to_csv(index=False).encode(), "log_preprocessing.csv", "text/csv")


# ════════════════════════════════════════════════════════════════
#  🛒  APRIORI
# ════════════════════════════════════════════════════════════════
def rule_network(r):
    G = nx.DiGraph()
    for _, x in r.iterrows():
        G.add_edge(x.Antecedent, x.Consequent, lift=x.lift)
    pos = nx.spring_layout(G, seed=7, k=1.1)
    f = go.Figure()
    for u, v, d in G.edges(data=True):
        f.add_trace(go.Scatter(x=[pos[u][0], pos[v][0]], y=[pos[u][1], pos[v][1]], mode="lines",
                               line=dict(width=1 + 4 * (d["lift"] - 0.9), color="rgba(30,91,216,.45)"), hoverinfo="none", showlegend=False))
        f.add_annotation(x=pos[v][0], y=pos[v][1], ax=pos[u][0], ay=pos[u][1], xref="x", yref="y", axref="x", ayref="y",
                         showarrow=True, arrowhead=3, arrowsize=1.3, arrowcolor="rgba(255,138,31,.9)", standoff=18)
    f.add_trace(go.Scatter(x=[pos[n][0] for n in G], y=[pos[n][1] for n in G], mode="markers+text", text=list(G.nodes),
                           textposition="top center", marker=dict(size=[18 + 6 * G.degree(n) for n in G], color=BLUE, line=dict(color="#fff", width=2)),
                           hoverinfo="text", showlegend=False))
    f.update_xaxes(visible=False); f.update_yaxes(visible=False)
    return style_fig(f, 520).update_layout(title="Network Graph Aturan Asosiasi (tebal garis = lift)")

def page_apriori():
    page_head("Market Basket Analysis — Apriori", "Menemukan kombinasi produk yang sering dibeli bersamaan.", "TAHAP 3 · 20%")
    clean = require_data()
    c1, c2, c3 = st.columns(3)
    sup = c1.slider("Minimum Support", 0.005, 0.10, 0.01, 0.005, format="%.3f")
    conf = c2.slider("Minimum Confidence", 0.05, 0.80, 0.15, 0.05)
    mlen = c3.selectbox("Panjang itemset maksimum", [2, 3], index=0)
    st.session_state.apr = (sup, conf, mlen)
    tx, fi, ru = run_apriori(clean, sup, conf, mlen)
    insight(f"<b>Alasan parameter:</b> {len(tx):,} nota, ±{clean.drop_duplicates('Transaction_ID').basket_size.mean():.1f} produk/nota dari 30 produk → support per pasangan produk wajar rendah. "
            f"Support {sup:.1%} ≈ muncul di minimal {int(sup*len(tx))} nota (cukup sering untuk bertindak, tidak terlalu ketat). Confidence {conf:.0%} agar aturan cukup andal; lift > 1 dipakai sebagai penentu kualitas.".replace(",", "."))
    t1, t2, t3, t4 = st.tabs(["🧮 Matriks & Frequent Itemsets", "📜 Aturan Asosiasi", "🕸️ Visualisasi", "🧠 Interpretasi & Rekomendasi"])
    with t1:
        st.caption(f"Matriks transaksi: {tx.shape[0]:,} nota × {tx.shape[1]} produk".replace(",", "."))
        st.dataframe(tx.head(10).astype(int), width="stretch")
        if fi.empty: st.warning("Tidak ada itemset — turunkan minimum support.")
        else:
            f2 = fi.assign(Itemset=fi.itemsets.apply(lambda s: " + ".join(sorted(s))), Ukuran=fi.itemsets.apply(len)).sort_values("support", ascending=False)
            st.dataframe(f2[["Itemset", "support", "Ukuran"]].rename(columns={"support": "Support"}), width="stretch", height=300, hide_index=True)
    with t2:
        if ru.empty: st.warning("Belum ada aturan — turunkan support/confidence.")
        else:
            st.markdown(f"**{len(ru)} aturan ditemukan.** Kriteria pemeringkatan: **lift tertinggi**, lalu confidence (hanya lift > 1 = hubungan lebih kuat dari kebetulan). Ditampilkan maks. 10 teratas.")
            tr = top_rules(ru)
            st.dataframe(tr.style.format({"Support": "{:.3f}", "Confidence": "{:.1%}", "Lift": "{:.2f}"}).background_gradient(subset=["Lift"], cmap="Blues"), width="stretch", hide_index=True)
            st.download_button("⬇️ Unduh semua aturan (CSV)", ru[["Aturan", "support", "confidence", "lift"]].to_csv(index=False).encode(), "aturan_apriori.csv")
    with t3:
        if ru.empty: st.info("Tidak ada aturan untuk divisualisasikan.")
        else:
            st.plotly_chart(rule_network(ru.head(10)), width="stretch")
            s = ru[(ru.antecedents.apply(len) == 1) & (ru.consequents.apply(len) == 1)]
            if len(s):
                hm = s.pivot_table(index="Antecedent", columns="Consequent", values="lift")
                st.plotly_chart(style_fig(px.imshow(hm, color_continuous_scale="Blues", aspect="auto", title="Heatmap Lift (antecedent → consequent)"), 520), width="stretch")
            st.plotly_chart(style_fig(px.scatter(ru, x="support", y="confidence", size="lift", color="lift", color_continuous_scale="Turbo", hover_name="Aturan", title="Support vs Confidence (ukuran/warna = lift)"), 420), width="stretch")
    with t4:
        if ru.empty: st.info("Belum ada aturan."); return
        r = top_rules(ru, 3)
        b = r.iloc[0]; a, c = b.Aturan.split("  →  ")
        st.markdown(f"""<div class="card"><h4>Makna Metrik pada Aturan Teratas</h4><b>{b.Aturan}</b><ul>
        <li><b>Support {b.Support:.2%}</b> → {b.Support:.2%} dari seluruh nota memuat {a} dan {c} sekaligus (≈{int(b.Support*len(tx))} nota).</li>
        <li><b>Confidence {b.Confidence:.1%}</b> → dari nota yang berisi {a}, {b.Confidence:.1%} juga berisi {c}.</li>
        <li><b>Lift {b.Lift:.2f}</b> → peluang {c} dibeli bersama {a} {b.Lift:.2f}× {'lebih besar' if b.Lift>1 else 'lebih kecil'} dibanding peluang dasarnya{'.' if b.Lift>1 else ' (tidak ada asosiasi positif).'}</li></ul></div>""", unsafe_allow_html=True)
        st.write("")
        st.markdown("##### 💡 Rekomendasi Cross-selling / Bundling / Penataan Rak")
        for i, (_, x) in enumerate(top_rules(ru, 5).iterrows(), 1):
            a, c = x.Aturan.split("  →  ")
            tipe = ["Bundling paket hemat", "Cross-selling di kasir/aplikasi", "Penempatan rak berdekatan", "Promo 'beli A gratis diskon B'", "Rekomendasi otomatis di struk"][(i - 1) % 5]
            st.markdown(f"**{i}. {tipe}: {a} + {c}** — Support {x.Support:.2%}, Confidence {x.Confidence:.1%}, Lift {x.Lift:.2f}. "
                        f"Alasan: pelanggan yang membeli {a} punya kemungkinan {x.Lift:.2f}× lebih tinggi membeli {c} dibanding rata-rata.")
        if ru.lift.max() < 1.15:
            insight("⚠️ <b>Catatan jujur:</b> lift tertinggi hanya mendekati 1, artinya asosiasi antar produk lemah (pola mendekati acak). Rekomendasi sebaiknya diuji dulu lewat A/B test kecil sebelum diterapkan luas.")


# ════════════════════════════════════════════════════════════════
#  👥  K-MEANS
# ════════════════════════════════════════════════════════════════
def get_segments(clean):
    rfm = make_rfm(clean)
    k = st.session_state.get("k_final")
    if k is None:
        sc = kmeans_scan(rfm, True); k = int(sc.loc[sc.Silhouette.idxmax(), "k"]); k = k if k >= 3 else 3
        st.session_state.k_final = k
    return run_kmeans(rfm, k, True)

STRAT = {
    "Loyal": "Program loyalitas/poin eksklusif, akses awal produk baru, bundling premium; jaga dengan layanan prioritas.",
    "Potential": "Tawarkan voucher frekuensi (beli 3× dapat diskon), cross-selling dari aturan Apriori untuk menaikkan nilai keranjang.",
    "New": "Sapaan selamat datang, diskon kunjungan ke-2, ajak jadi Member agar terbentuk kebiasaan belanja.",
    "Need": "Pengingat personal + kupon terbatas waktu sebelum mereka pasif; tonjolkan produk favorit mereka.",
    "Hibernating": "Kampanye 'kami rindu Anda' berdiskon, survei singkat penyebab berhenti.",
    "Churn": "Kampanye reaktivasi (voucher kuat, durasi pendek) dengan batas anggaran; hentikan bila tidak merespons.",
}
def strategy(label):
    for k, v in STRAT.items():
        if label.startswith(k): return v
    return STRAT["Potential"]

def page_kmeans():
    page_head("Segmentasi Pelanggan — K-Means", "Mengelompokkan pelanggan berdasarkan Recency, Frequency, dan Monetary.", "TAHAP 4 · 20%")
    clean = require_data(); rfm = make_rfm(clean)
    t1, t2, t3 = st.tabs(["📐 Penentuan Jumlah Klaster", "🧩 Hasil Klaster", "🏷️ Profil & Strategi"])
    uselog = True
    with t1:
        st.markdown(f"**Fitur:** Recency (hari sejak belanja terakhir), Frequency (jumlah nota), Monetary (total belanja) dari **{len(rfm):,} pelanggan** — ".replace(",", ".") +
                    "`log1p` + StandardScaler agar skala setara dan distribusi tidak miring.")
        sc = kmeans_scan(rfm, uselog)
        f = make_subplots(1, 2, subplot_titles=("Elbow Method (Inertia)", "Silhouette Score"))
        f.add_trace(go.Scatter(x=sc.k, y=sc["Inertia (WCSS)"], mode="lines+markers", line=dict(color=BLUE, width=3)), 1, 1)
        f.add_trace(go.Bar(x=sc.k, y=sc.Silhouette, marker_color=ORANGE), 1, 2)
        st.plotly_chart(style_fig(f, 340).update_layout(showlegend=False), width="stretch")
        st.dataframe(sc.round(4), width="stretch", hide_index=True)
        best = int(sc.loc[sc.Silhouette.idxmax(), "k"])
        k = st.slider("Jumlah klaster yang digunakan", 2, 8, st.session_state.get("k_final", max(best, 3)))
        st.session_state.k_final = k
        insight(f"Silhouette tertinggi pada k={best} ({sc.Silhouette.max():.3f}). Pilih k dengan menimbang <b>siku pada grafik Elbow</b>, silhouette, "
                f"dan <b>kegunaan bisnis</b> (≥3 segmen agar strategi dapat dibedakan: bernilai tinggi, menengah, berisiko). Saat ini k = <b>{k}</b>.")
    k = st.session_state.get("k_final", 3)
    res, sil = run_kmeans(rfm, k, uselog)
    with t2:
        c = st.columns(3); kpi(c[0], "Jumlah Klaster", k); kpi(c[1], "Silhouette", f"{sil:.3f}", ORANGE); kpi(c[2], "Pelanggan Dikelompokkan", num(len(res)), CYAN)
        a, b = st.columns(2)
        a.plotly_chart(style_fig(px.scatter(res, x="PC1", y="PC2", color="Segmen", opacity=.75, title="Scatter PCA 2D", hover_data=["Customer_ID", "Recency", "Frequency", "Monetary"]), 430), width="stretch")
        b.plotly_chart(style_fig(px.scatter_3d(res, x="Recency", y="Frequency", z="Monetary", color="Segmen", opacity=.7, title="Ruang RFM 3D"), 430), width="stretch")
        st.dataframe(res[["Customer_ID", "Recency", "Frequency", "Monetary", "Segmen"]].round(1), width="stretch", height=260, hide_index=True)
    with t3:
        prof = res.groupby("Segmen").agg(Pelanggan=("Customer_ID", "count"), Recency=("Recency", "mean"), Frequency=("Frequency", "mean"),
                                         Monetary=("Monetary", "mean"), Total_Belanja=("Monetary", "sum")).reset_index()
        prof["% Pelanggan"] = prof.Pelanggan / prof.Pelanggan.sum() * 100; prof["% Pendapatan"] = prof.Total_Belanja / prof.Total_Belanja.sum() * 100
        prof = prof.sort_values("Monetary", ascending=False)
        st.dataframe(prof.round(1), width="stretch", hide_index=True)
        rd = res.groupby("Segmen")[["Recency", "Frequency", "Monetary"]].mean()
        rd = (rd - rd.min()) / (rd.max() - rd.min()).replace(0, 1)
        rf = go.Figure()
        for s_ in rd.index: rf.add_trace(go.Scatterpolar(r=rd.loc[s_], theta=rd.columns, fill="toself", name=s_))
        st.plotly_chart(style_fig(rf, 380).update_layout(title="Radar profil (dinormalisasi 0–1)"), width="stretch")
        for _, r_ in prof.iterrows():
            st.markdown(f"""<div class="card" style="margin-bottom:10px"><h4>🏷️ {r_.Segmen}</h4>
            {int(r_.Pelanggan)} pelanggan ({r_['% Pelanggan']:.1f}%) · rata-rata belanja terakhir <b>{r_.Recency:.0f} hari</b> lalu · <b>{r_.Frequency:.1f}</b> transaksi · <b>{rp(r_.Monetary)}</b>/pelanggan · kontribusi pendapatan <b>{r_['% Pendapatan']:.1f}%</b>.<br>
            <b>Strategi:</b> {strategy(r_.Segmen)}</div>""", unsafe_allow_html=True)
        top = prof.sort_values("% Pendapatan", ascending=False).iloc[0]
        ch = prof[prof.Segmen.str.startswith(("Churn", "Hibern"))]
        insight(f"<b>Prioritas 1:</b> {top.Segmen} — menyumbang {top['% Pendapatan']:.1f}% pendapatan; kehilangan segmen ini paling merugikan. " +
                (f"<b>Prioritas 2:</b> {ch.iloc[0].Segmen} ({int(ch.iloc[0].Pelanggan)} pelanggan) untuk reaktivasi bernilai rendah-biaya." if len(ch) else ""))


# ════════════════════════════════════════════════════════════════
#  📊  DASHBOARD
# ════════════════════════════════════════════════════════════════
def page_dash():
    page_head("Business Intelligence Dashboard", "Ringkasan interaktif untuk manajemen.", "TAHAP 5 · 15%")
    clean = require_data()
    seg, _ = get_segments(clean)
    df = clean.merge(seg[["Customer_ID", "Segmen"]], on="Customer_ID", how="left"); df["Segmen"] = df.Segmen.fillna("Tanpa ID")
    with st.expander("🎛️ Filter Interaktif", expanded=True):
        a, b, c, d, e = st.columns([1.3, 1, 1, 1, 1])
        dmin, dmax = df.Date.min().date(), df.Date.max().date()
        rng = a.date_input("Periode transaksi", (dmin, dmax), min_value=dmin, max_value=dmax)
        cats = b.multiselect("Kategori", sorted(df.Category.unique()))
        prods = c.multiselect("Produk", sorted(df.Product_Name.unique()))
        segs = d.multiselect("Segmen", sorted(df.Segmen.unique()))
        regs = e.multiselect("Wilayah", sorted(df.Region.unique()))
    if len(rng) == 2: df = df[(df.Date.dt.date >= rng[0]) & (df.Date.dt.date <= rng[1])]
    if cats: df = df[df.Category.isin(cats)]
    if prods: df = df[df.Product_Name.isin(prods)]
    if segs: df = df[df.Segmen.isin(segs)]
    if regs: df = df[df.Region.isin(regs)]
    if df.empty: st.warning("Tidak ada data untuk filter ini."); st.stop()
    tx = df.drop_duplicates("Transaction_ID")
    c = st.columns(4)
    kpi(c[0], "💰 Total Pendapatan", rp(df.Line_Total.sum())); kpi(c[1], "🧾 Jumlah Transaksi", num(tx.shape[0]), ORANGE)
    kpi(c[2], "👤 Pelanggan Unik", num(df.Customer_ID.nunique()), CYAN); kpi(c[3], "🛍️ Rata-rata basket_size", num(df.groupby("Transaction_ID").Product_Name.nunique().mean(), 2), "#7c4dff")
    st.write("")
    t1, t2, t3, t4 = st.tabs(["📈 Penjualan", "🛒 Apriori", "👥 K-Means", "⭐ Bonus: Apriori × Segmen"])
    with t1:
        m = df.groupby("Month").Line_Total.sum().reset_index()
        a, b = st.columns([1.4, 1])
        a.plotly_chart(style_fig(px.area(m, x="Month", y="Line_Total", markers=True, title="Tren Penjualan Bulanan", color_discrete_sequence=[BLUE])), width="stretch")
        b.plotly_chart(style_fig(px.pie(df.groupby("Category").Line_Total.sum().reset_index(), names="Category", values="Line_Total", hole=.5, title="Distribusi Penjualan per Kategori")), width="stretch")
        a, b = st.columns(2)
        tp = df.groupby("Product_Name").Line_Total.sum().nlargest(10).sort_values().reset_index()
        a.plotly_chart(style_fig(px.bar(tp, x="Line_Total", y="Product_Name", orientation="h", title="Top 10 Produk", color="Line_Total", color_continuous_scale="Blues")), width="stretch")
        b.plotly_chart(style_fig(px.bar(df.groupby("Region").Line_Total.sum().sort_values().reset_index(), x="Line_Total", y="Region", orientation="h", title="Penjualan per Wilayah", color_discrete_sequence=[ORANGE])), width="stretch")
    with t2:
        sup, conf, ml = st.session_state.get("apr", (0.01, 0.15, 2))
        _, _, ru = run_apriori(df, sup, conf, ml)
        st.caption(f"Parameter dari menu Apriori: support={sup}, confidence={conf} · dihitung ulang pada data terfilter")
        if ru.empty: st.info("Tidak ada aturan untuk filter ini (data terlalu sedikit). Longgarkan filter atau turunkan support.")
        else:
            a, b = st.columns([1, 1.2])
            a.dataframe(top_rules(ru).style.format({"Support": "{:.3f}", "Confidence": "{:.1%}", "Lift": "{:.2f}"}), width="stretch", hide_index=True)
            b.plotly_chart(rule_network(ru.head(10)), width="stretch")
    with t3:
        a, b = st.columns(2)
        cnt = seg[seg.Customer_ID.isin(df.Customer_ID)].Segmen.value_counts().reset_index()
        a.plotly_chart(style_fig(px.bar(cnt, x="Segmen", y="count", color="Segmen", title="Jumlah Pelanggan per Klaster"), 360).update_layout(showlegend=False), width="stretch")
        ss = seg[seg.Customer_ID.isin(df.Customer_ID)]
        b.plotly_chart(style_fig(px.scatter(ss, x="PC1", y="PC2", color="Segmen", opacity=.75, title="Visualisasi Segmentasi"), 360), width="stretch")
        st.dataframe(ss.groupby("Segmen")[["Recency", "Frequency", "Monetary"]].mean().round(1).reset_index(), width="stretch", hide_index=True)
    with t4:
        st.caption("Aturan asosiasi dihitung terpisah untuk tiap segmen → produk yang relevan bagi masing-masing kelompok.")
        sl = [s for s in sorted(df.Segmen.unique()) if s != "Tanpa ID"]
        pick = st.selectbox("Pilih segmen", sl)
        sub = df[df.Segmen == pick]
        _, _, r2 = run_apriori(sub, 0.01, 0.15, 2)
        a, b = st.columns(2)
        if r2.empty: a.info("Tidak ada aturan untuk segmen ini.")
        else: a.dataframe(top_rules(r2, 5).style.format({"Support": "{:.3f}", "Confidence": "{:.1%}", "Lift": "{:.2f}"}), width="stretch", hide_index=True)
        b.plotly_chart(style_fig(px.bar(sub.groupby("Product_Name").Line_Total.sum().nlargest(5).sort_values().reset_index(), x="Line_Total", y="Product_Name", orientation="h", title=f"Top 5 Produk — {pick}", color_discrete_sequence=[BLUE]), 320), width="stretch")
    st.markdown("##### 📸 Cara membaca dashboard")
    st.caption("KPI = kondisi bisnis saat ini · Tren = musim/puncak penjualan · Top produk & kategori = fokus stok · Apriori = peluang bundling · K-Means = target pemasaran. Ambil tangkapan layar halaman ini untuk laporan.")


# ════════════════════════════════════════════════════════════════
#  💡  REKOMENDASI
# ════════════════════════════════════════════════════════════════
def page_reco():
    page_head("Rekomendasi Keputusan Bisnis", "Rekomendasi preskriptif berbasis hasil analisis.", "TAHAP 6 · 10%")
    clean = require_data(); seg, _ = get_segments(clean)
    sup, conf, ml = st.session_state.get("apr", (0.01, 0.15, 2))
    _, _, ru = run_apriori(clean, sup, conf, ml)
    prof = seg.groupby("Segmen").agg(n=("Customer_ID", "count"), rev=("Monetary", "sum"), R=("Recency", "mean")).reset_index()
    prof["pct"] = prof.rev / prof.rev.sum() * 100
    t1, t2, t3 = st.tabs(["🧭 Strategi", "🎯 Prioritas & Dampak", "🗓️ Implementasi & KPI"])
    with t1:
        st.markdown("##### a. Strategi Penempatan Produk (hasil Apriori)")
        if ru.empty: st.info("Belum ada aturan Apriori; atur parameter di menu Apriori.")
        else:
            for _, x in top_rules(ru, 3).iterrows():
                a, c = x.Aturan.split("  →  ")
                st.markdown(f"- **{a} ↔ {c}**: tempatkan berdekatan / paket bundling (confidence {x.Confidence:.1%}, lift {x.Lift:.2f}).")
        st.markdown("##### b. Strategi Retensi (hasil K-Means)")
        for _, r in prof.sort_values("rev", ascending=False).iterrows():
            st.markdown(f"- **{r.Segmen}** ({int(r.n)} pelanggan, {r.pct:.1f}% pendapatan): {strategy(r.Segmen)}")
    with t2:
        top = prof.sort_values("rev", ascending=False).iloc[0]
        bad = prof.sort_values("R", ascending=False).iloc[0]
        rows = [("Bundling & penataan rak dari aturan Apriori", 4, 4, 1, 5), (f"Program loyalitas {top.Segmen}", 5, 5, 3, 4),
                (f"Kampanye reaktivasi {bad.Segmen}", 4, 3, 2, 4), ("Pantau dashboard KPI bulanan", 3, 3, 1, 5)]
        pr = pd.DataFrame(rows, columns=["Rekomendasi", "Urgensi (1-5)", "Potensi Manfaat (1-5)", "Biaya (1=murah,5=mahal)", "Kemudahan (1-5)"])
        pr["Skor"] = pr.iloc[:, 1] + pr.iloc[:, 2] + (6 - pr.iloc[:, 3]) + pr.iloc[:, 4]
        st.dataframe(pr.sort_values("Skor", ascending=False), width="stretch", hide_index=True)
        st.caption("Skor = urgensi + manfaat + (6 − biaya) + kemudahan. Nilai 1–5 adalah penilaian analis (asumsi), bukan hasil data.")
        st.markdown("##### d. Estimasi Dampak (simulasi — ubah asumsi)")
        a, b, c = st.columns(3)
        adopt = a.slider("Asumsi: % pembeli antecedent yang menerima tawaran bundling", 1, 30, 10)
        react = b.slider("Asumsi: % pelanggan berisiko yang kembali aktif", 1, 40, 10)
        loyal = c.slider("Asumsi: kenaikan belanja segmen terbaik (%)", 1, 20, 5)
        ntx = clean.Transaction_ID.nunique(); avg_item = clean.Line_Total.mean()
        gain1 = 0
        if not ru.empty:
            x = top_rules(ru, 1).iloc[0]; gain1 = ntx * x.Support / max(x.Confidence, 1e-9) * (1 - x.Confidence) * adopt / 100 * avg_item
        gain2 = bad.rev * react / 100; gain3 = top.rev * loyal / 100
        est = pd.DataFrame({"Sumber": ["Bundling Apriori", f"Reaktivasi {bad.Segmen}", f"Loyalitas {top.Segmen}"], "Estimasi Tambahan Pendapatan": [gain1, gain2, gain3]})
        st.plotly_chart(style_fig(px.bar(est, x="Sumber", y="Estimasi Tambahan Pendapatan", color="Sumber", text_auto=".3s"), 330).update_layout(showlegend=False), width="stretch")
        st.metric("Total estimasi (setahun)", rp(est["Estimasi Tambahan Pendapatan"].sum()), f"{est['Estimasi Tambahan Pendapatan'].sum() / clean.Line_Total.sum() * 100:.1f}% dari pendapatan")
        st.markdown("""<div class="insight"><b>Bedakan tiga jenis informasi:</b><br>
        ✅ <b>Hasil dari data</b> — support, confidence, lift, profil RFM, kontribusi pendapatan segmen.<br>
        🧮 <b>Estimasi berdasarkan asumsi</b> — angka tambahan pendapatan di atas (bergantung slider asumsi).<br>
        ⏳ <b>Dampak aktual</b> — baru diketahui setelah rekomendasi diterapkan dan diukur (A/B test, pembanding periode sebelumnya).</div>""", unsafe_allow_html=True)
    with t3:
        st.markdown("""<div class="card"><h4>e. Rencana Implementasi</h4>
        <b>Jangka pendek (0–3 bulan):</b> pasang bundling untuk 3 aturan teratas, atur ulang rak, kirim voucher reaktivasi ke segmen berisiko, mulai pakai dashboard bulanan.<br>
        <b>Jangka menengah (3–12 bulan):</b> program poin/loyalitas, rekomendasi otomatis di kasir, perbarui model RFM & Apriori tiap kuartal, perluas data (margin, stok).</div>""", unsafe_allow_html=True)
        st.write("")
        st.dataframe(pd.DataFrame({"KPI Evaluasi": ["Rata-rata basket_size", "Attach rate pasangan produk bundling", "Retention / repeat purchase rate", "Reactivation rate segmen berisiko", "Pendapatan per pelanggan (Monetary)", "ROI promosi"],
                                   "Cara Mengukur": ["Rata-rata produk berbeda/nota, sebelum vs sesudah", "% nota berisi pasangan produk", "% pelanggan membeli ulang dalam 90 hari", "% pelanggan berisiko yang bertransaksi lagi", "Total belanja ÷ pelanggan aktif per bulan", "(Tambahan margin − biaya promo) ÷ biaya promo"]}), width="stretch", hide_index=True)


# ════════════════════════════════════════════════════════════════
#  🚀  MAIN
# ════════════════════════════════════════════════════════════════
if "prof" not in st.session_state: st.session_state.prof = dict(PROFILE)
if not st.session_state.get("auth"):
    login_page(); st.stop()

PAGES = {"🏠 Beranda": page_home, "📁 Upload Data": page_upload, "🔍 Business Understanding & EDA": page_eda,
         "🧹 Data Preprocessing": page_prep, "🛒 Apriori": page_apriori, "👥 K-Means": page_kmeans,
         "📊 Dashboard BI": page_dash, "💡 Rekomendasi": page_reco}
with st.sidebar:
    p = st.session_state.prof
    st.markdown(f"<div class='pf'>{avatar_html()}<br><b>{p['nama']}</b><br><small>NPM {p['npm']}<br>{p['kelas']}<br>{p['matkul']}<br>{p['ujian']}</small></div>", unsafe_allow_html=True)
    st.divider()
    choice = st.radio("Navigasi", list(PAGES), label_visibility="collapsed")
    st.divider()
    with st.expander("✏️ Edit Profil"):
        for k_, lab in [("nama", "Nama lengkap"), ("npm", "NPM"), ("kelas", "Kelas"), ("prodi", "Program studi")]:
            st.session_state.prof[k_] = st.text_input(lab, p[k_])
        ph = st.file_uploader("Foto profil (otomatis diubah ke PNG)", type=["png", "jpg", "jpeg", "webp"])
        if ph: st.session_state.foto_bytes = to_png(ph.getvalue())
    if st.button("🚪 Keluar", width="stretch"):
        st.session_state.clear(); st.rerun()
PAGES[choice]()