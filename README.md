# Early Warning Decision Engine (Olist Capstone)
## سامانه تصمیم‌گیری و هشدار زودهنگام رضایت مشتریان در بستر اولیست
**دانشجو:** شایان گنجی  
**آزمایشگاه:** فناوری های مالی داتین- ترم سوم  
**پایگاه‌داده هدف:** PostgreSQL 16.4 | دیتابیس `olist_db`  
**محیط اجرا:** Python 3.11, PostgreSQL 16, MLflow 3.18, Scikit-Learn 1.6, LightGBM 4.6  

---

## راهنمای مستندات و فایل‌های تحویلی

- [گزارش فنی معماری سیستم (فارسی)](report.fa.md): تحلیل جامع چرخه حیات سفارش، دلایل فنی انتخاب و رد منابع بیرونی، معماری MLflow و ممیزی نشت داده.
- [Technical Architecture Report (English)](report.md): گزارش کامل مهندسی به زبان انگلیسی شامل تحلیل ریسک و جابه‌جایی توزیع.
- [خلاصه مدیریتی (یک صفحه‌ای)](executive_summary.fa.md): جمع‌بندی غیرفنی برای مدیر عملیات پیرامون تخصیص بهینه سهمیه ۱۰ درصدی پیگیری.
- [ثبت تصمیم‌های معماری (decisions.md)](decisions.md): مستندسازی ده‌گانه تصمیم‌های کلیدی پروژه در قالب استاندارد ADR.
- [نوت‌بوک ۱: تحلیل اکتشافی و ممیزی داده (01_exploration.ipynb)](01_exploration.ipynb): ممیزی زمانی ۴۵ ستون، تحلیل داده‌های بیرونی و بنچ‌مارک نشت داده.
- [نوت‌بوک ۲: پایپ‌لاین تولید و ارزیابی مقید به ظرفیت (02_system.ipynb)](02_system.ipynb): استخراج درون‌پایگاه‌داده، ردگیری با MLflow، ارزیابی ظرفیت و استخراج صف اولویت.

---

## راهنمای راه‌اندازی و اجرای سریع از صفر

```bash
# ۱. نصب پکیج در حالت توسعه (Editable)
pip install -e .

# ۲. اجرای کامل تست‌های خودکار آفلاین (۲۶ تست واحد و یکپارچه)
python -m unittest discover tests -v

# ۳. اجرای تک‌فرمان کل سامانه از صفر تا خروجی نهایی (زیر ۱ دقیقه)
python scripts/run_all.py
# یا در محیط ویندوز پاورشل:
.\run_pipeline.ps1

# فرامین اختصاصی CLI:
early-warning audit     # مشاهده جدول ممیزی زمانی ستون‌ها در لحظه T0
early-warning train     # آموزش مدل‌ها، ارزیابی چنددوره‌ای و لاگ در MLflow
early-warning evaluate  # اجرای بنچ‌مارک تجربی مقایسه مدل سالم و آلوده به نشت
early-warning predict --input reports/priority_followup_orders.csv --capacity 0.10
```

---

## ساختار پوشه‌های پروژه

```text
stage-4-shayan-ganji/
├── 01_exploration.ipynb        # نوت‌بوک اکتشاف، ممیزی زمانی ۴۵ ستون و بنچ‌مارک نشت
├── 02_system.ipynb             # نوت‌بوک سامانه نهایی، اجرای خط لوله و ارزیابی ظرفیت
├── report.fa.md                # گزارش جامع فنی و تصمیم‌های مهندسی (فارسی)
├── report.md                   # گزارش فنی معماری سیستم (انگلیسی)
├── decisions.md                # ۱۰ سند ثبت تصمیم معماری (ADR)
├── executive_summary.fa.md     # خلاصه مدیریتی یک‌صفحه‌ای برای مدیر عملیات
├── run_pipeline.ps1            # اسکریپت تک‌فرمان اجرای کل خط لوله
├── requirements.txt            # فهرست وابستگی‌های قفل‌شده
├── pyproject.toml              # تنظیمات پکیج پایتون و اینترفیس CLI
├── README.md                   # راهنمای تحویل و فرامین اجرایی
├── mlruns.db                   # پایگاه‌داده محلی SQLite برای ردیابی آزمایش‌ها با MLflow
├── data/
│   └── external/               # کش محلی تعطیلات رسمی مصوب برزیل (BrasilAPI)
├── models/                     # آرتیفکت‌های سریال‌شده مدل نهایی تولیدی
├── reports/
│   ├── figures/                # نمودارهای تحلیلی و منحنی‌های ظرفیت
│   └── priority_followup_orders.csv # صف ۱۰ درصد سفارش‌های پرخطر جهت پیگیری
├── scripts/
│   ├── run_all.py              # ارکستراتور اجرای کامل سامانه (اجرا در کمتر از ۴۵ ثانیه)
│   ├── deploy_views.py         # مستقرکننده ویوها در PostgreSQL و ارزیابی زمان کوئری
│   ├── build_exploration_notebook.py
│   └── build_system_notebook.py
├── sql/views/
│   ├── 01_create_feature_views.sql # ویوهای پویا و ماتریالایزد با ایندکس‌های B-Tree
│   └── feature_query_plan.txt      # پلن‌های اجرایی تاییدکننده زمان زیر ۳ میلی‌ثانیه
├── src/early_warning/          # کد اصلی سامانه و ماژول‌های مهندسی
└── tests/                      # ۲۶ آزمون خودکار آفلاین
```
