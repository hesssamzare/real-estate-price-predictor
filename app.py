import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="Real Estate Price Predictor",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;500;600;700;800&display=swap');
:root { --ink:#26352d; --muted:#78847b; --sage:#718b72; --sage-dark:#49634f; --cream:#f7f5ed; }
html, body, [class*="css"], [data-testid="stAppViewContainer"] {
    font-family: 'Vazirmatn', sans-serif;
}
.stApp {
    background:
      radial-gradient(circle at 7% 4%, rgba(191, 207, 177, .27), transparent 27%),
      radial-gradient(circle at 94% 8%, rgba(224, 211, 180, .28), transparent 25%),
      linear-gradient(145deg, #f8f7f1 0%, #f1f4ec 55%, #faf8f1 100%);
    color: var(--ink);
}
.block-container { max-width: 1160px; padding-top: 2rem; padding-bottom: 3rem; }
.hero {
    background: linear-gradient(115deg, #354b3b 0%, #607b62 55%, #8fa28a 100%);
    padding: 34px 38px; border-radius: 28px; color: #fff;
    box-shadow: 0 18px 42px rgba(58, 78, 59, .17);
    margin-bottom: 24px; position: relative; overflow: hidden;
}
.hero:after {
    content:""; position:absolute; width:250px; height:250px; border:1px solid rgba(255,255,255,.18);
    border-radius:50%; right:-60px; top:-115px; box-shadow:0 0 0 28px rgba(255,255,255,.035),0 0 0 58px rgba(255,255,255,.025);
}
.hero .eyebrow { font-size:.78rem; letter-spacing:1.6px; opacity:.82; font-weight:600; }
.hero h1 { font-size:2.15rem; font-weight:800; margin:8px 0 7px; }
.hero p { margin:0; opacity:.88; font-size:.98rem; }
.section-head { color:#405b46; font-size:1.18rem; font-weight:800; margin-bottom:4px; }
.section-sub { color:#818a7f; font-size:.86rem; margin-bottom:17px; }
.panel {
    background:rgba(255,255,255,.82); border:1px solid rgba(100,125,99,.14);
    border-radius:22px; padding:23px 25px; box-shadow:0 9px 28px rgba(50,70,50,.055);
}
.metric {
    border-radius:18px; padding:17px 19px; background:#fff; border:1px solid #e6eadf;
    box-shadow:0 7px 20px rgba(52,72,52,.045); min-height:100px;
}
.metric .label { color:#7c887c; font-size:.78rem; }
.metric .value { color:#3f5945; font-size:1.65rem; font-weight:800; margin-top:6px; }
.result {
    background:linear-gradient(120deg,#49664f,#779477); color:#fff; border-radius:23px;
    padding:25px 28px; box-shadow:0 13px 30px rgba(73,102,79,.17);
    animation: reveal .65s ease-out both;
}
@keyframes reveal { from {opacity:0; transform:translateY(12px) scale(.985)} to {opacity:1; transform:translateY(0) scale(1)} }
.result .caption { opacity:.84; font-size:.88rem; }
.result .price { font-size:2.6rem; line-height:1.3; font-weight:800; margin:7px 0; }
div[data-testid="stNumberInput"] input { border-radius:12px; }
.stButton > button {
    border-radius:13px; min-height:3rem; font-weight:700;
    border:1px solid #6f896f; background:#5e795f; color:#fff;
}
.stButton > button:hover { background:#49634f; border-color:#49634f; color:#fff; }
div[data-testid="stDownloadButton"] button { border-radius:12px; }
div[data-testid="stAlert"] { border-radius:14px; }
[data-testid="stDataFrame"] { border:1px solid #e7eadf; border-radius:14px; overflow:hidden; }
hr { border-color:#e7eadf !important; }
footer { visibility:hidden; }
#MainMenu { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

FEATURES = [
    "X1 transaction date",
    "X2 house age",
    "X3 distance to the nearest MRT station",
    "X4 number of convenience stores",
    "X5 latitude",
    "X6 longitude",
]
TARGET = "Y house price of unit area"

@st.cache_resource
def train_model():
    path = Path(__file__).parent / "Real estate valuation data set.xlsx"
    data = pd.read_excel(path)
    X, y = data[FEATURES], data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    predicted = reg.predict(X_test)
    stats = {
        "mae": mean_absolute_error(y_test, predicted),
        "rmse": float(np.sqrt(mean_squared_error(y_test, predicted))),
        "r2": r2_score(y_test, predicted),
    }
    return reg, stats, data

try:
    model, metrics, dataset = train_model()
except Exception as error:
    st.error("دیتاست کنار فایل app.py پیدا نشد یا بارگذاری آن با مشکل مواجه شد.")
    st.exception(error)
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

def reset_form():
    defaults = {
        "date": 2013.25, "age": 10.0, "distance": 500.0,
        "stores": 5, "latitude": 24.98, "longitude": 121.54
    }
    for key, value in defaults.items():
        st.session_state[key] = value
    st.session_state.prediction_result = None

st.markdown("""
<div class="hero" dir="rtl">
  <div class="eyebrow">PROPERTY INTELLIGENCE · MACHINE LEARNING</div>
  <h1>🏡 Real Estate Price Predictor</h1>
  <p>تخمین هوشمند قیمت ملک بر اساس ویژگی‌های مکانی و مشخصات ملک</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-head" dir="rtl">عملکرد مدل</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub" dir="rtl">ارزیابی مدل رگرسیون خطی روی داده‌های آزمایش (۲۰٪ داده‌ها)</div>', unsafe_allow_html=True)
a, b, c = st.columns(3)
for col, label, value in [
    (a, "میانگین قدر مطلق خطا · MAE", f'{metrics["mae"]:.2f}'),
    (b, "ریشه میانگین مربع خطا · RMSE", f'{metrics["rmse"]:.2f}'),
    (c, "ضریب تعیین · R²", f'{metrics["r2"]:.1%}')
]:
    with col:
        st.markdown(f'<div class="metric" dir="rtl"><div class="label">{label}</div><div class="value">{value}</div></div>', unsafe_allow_html=True)

st.write("")
form_col, info_col = st.columns([1.22, .78], gap="large")
with form_col:
    st.markdown('<div class="panel" dir="rtl">', unsafe_allow_html=True)
    st.markdown('<div class="section-head">مشخصات ملک</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">مقادیر ملک موردنظر را وارد کنید.</div>', unsafe_allow_html=True)
    l, r = st.columns(2)
    with l:
        date = st.number_input("تاریخ معامله (سال اعشاری)", min_value=2012.0, max_value=2014.0, value=2013.25, step=1/12, format="%.6f", key="date",
            help="تاریخ در دیتاست به‌صورت سال اعشاری است؛ برای نمونه 2013.25 معادل آوریل ۲۰۱۳ است.")
        age = st.number_input("سن ملک (سال)", min_value=0.0, max_value=100.0, value=10.0, step=.5, key="age")
        distance = st.number_input("فاصله تا نزدیک‌ترین مترو (متر)", min_value=0.0, max_value=20000.0, value=500.0, step=10.0, key="distance")
    with r:
        stores = st.number_input("تعداد فروشگاه‌های رفاهی", min_value=0, max_value=30, value=5, step=1, key="stores")
        latitude = st.number_input("عرض جغرافیایی (Latitude)", min_value=0.0, max_value=90.0, value=24.98, format="%.5f", key="latitude")
        longitude = st.number_input("طول جغرافیایی (Longitude)", min_value=0.0, max_value=180.0, value=121.54, format="%.5f", key="longitude")
    p1, p2 = st.columns([1.5, 1])
    with p1:
        predict = st.button("محاسبه قیمت تخمینی", use_container_width=True)
    with p2:
        st.button("پاک کردن و شروع مجدد", on_click=reset_form, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with info_col:
    st.markdown('<div class="panel" dir="rtl">', unsafe_allow_html=True)
    st.markdown('<div class="section-head">راهنمای استفاده</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="color:#69776b;line-height:2;font-size:.91rem">
      <b>۱.</b> شش ویژگی ملک را وارد کنید.<br>
      <b>۲.</b> دکمه محاسبه را بزنید.<br>
      <b>۳.</b> تخمین قیمت و اطلاعات تحلیلی را مشاهده کنید.
    </div>
    <hr>
    <div style="color:#7d897e;font-size:.82rem;line-height:1.9">
      <b>واحد خروجی:</b><br>
      ۱۰ هزار دلار تایوان به ازای هر پینگ (واحد مساحت رایج در تایوان).<br><br>
      <b>روش مدل:</b> رگرسیون خطی چندمتغیره
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

if predict:
    row = pd.DataFrame([{
        FEATURES[0]: date, FEATURES[1]: age, FEATURES[2]: distance,
        FEATURES[3]: stores, FEATURES[4]: latitude, FEATURES[5]: longitude
    }], columns=FEATURES)
    estimate = float(model.predict(row)[0])
    st.session_state.prediction_result = {
        "price": estimate, "date": date, "age": age, "distance": distance,
        "stores": stores, "latitude": latitude, "longitude": longitude
    }
    st.session_state.history.insert(0, st.session_state.prediction_result.copy())

result = st.session_state.prediction_result
if result is not None:
    st.write("")
    st.markdown(f"""
    <div class="result" dir="rtl">
      <div class="caption">نتیجه تخمین قیمت ملک</div>
      <div class="price">{result["price"]:,.2f}</div>
      <div class="caption">۱۰ هزار دلار تایوان به ازای هر پینگ</div>
    </div>
    """, unsafe_allow_html=True)
    st.caption("این عدد تخمین مدل است و قیمت قطعی یا تضمین‌شده ملک محسوب نمی‌شود.")
    low, mid, high = st.columns(3)
    with low:
        st.metric("سن ملک", f'{result["age"]:g} سال')
    with mid:
        st.metric("فاصله تا مترو", f'{result["distance"]:,.0f} متر')
    with high:
        st.metric("فروشگاه‌های رفاهی", f'{result["stores"]}')

st.write("")
st.markdown('<div class="section-head" dir="rtl">تحلیل و تاریخچه پیش‌بینی‌ها</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub" dir="rtl">خلاصه عملکرد مدل و نتایج پیش‌بینی‌های همین نشست</div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📊 تحلیل مدل", "🕘 تاریخچه پیش‌بینی‌ها"])
with tab1:
    t1, t2 = st.columns(2)
    with t1:
        st.markdown("**معیارهای ارزیابی**")
        metric_df = pd.DataFrame({
            "معیار": ["MAE", "RMSE", "R²"],
            "مقدار": [metrics["mae"], metrics["rmse"], metrics["r2"]]
        }).set_index("معیار")
        st.bar_chart(metric_df)
    with t2:
        st.markdown("**توزیع قیمت در دیتاست**")
        price_hist = pd.DataFrame({"تعداد املاک": dataset[TARGET].round(0).value_counts().sort_index()})
        st.line_chart(price_hist)
    st.info(f'مدل حدود {metrics["r2"]:.1%} از تغییرات قیمت را در مجموعه آزمایش توضیح می‌دهد. این مقدار به معنی درصد پیش‌بینی‌های درست نیست.')
with tab2:
    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        history_df = history_df.rename(columns={
            "price": "قیمت تخمینی", "date": "تاریخ معامله", "age": "سن ملک",
            "distance": "فاصله تا مترو", "stores": "تعداد فروشگاه",
            "latitude": "عرض جغرافیایی", "longitude": "طول جغرافیایی"
        })
        st.dataframe(history_df, use_container_width=True, hide_index=True)
        csv = history_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("دانلود تاریخچه به‌صورت CSV", data=csv, file_name="prediction_history.csv", mime="text/csv")
        if st.button("پاک کردن تاریخچه"):
            st.session_state.history = []
            st.rerun()
    else:
        st.markdown('<div class="panel" dir="rtl" style="text-align:center;color:#7b887c">هنوز پیش‌بینی‌ای ثبت نشده است. بعد از محاسبه قیمت، نتایج اینجا نمایش داده می‌شوند.</div>', unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center;color:#929b8f;font-size:.76rem;padding:28px 0 0" dir="rtl">
  Real Estate Price Predictor · Powered by Python & Scikit-learn
</div>
""", unsafe_allow_html=True)
