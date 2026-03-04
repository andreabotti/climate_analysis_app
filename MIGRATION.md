# Migration Guide — Final Complete Structure

## Final folder structure

```
your_project/
├── app.py                              ← Home / EPW upload page
├── libs/
│   ├── __init__.py
│   ├── shared.py                       ← CSS, colour maps, constants (NEW)
│   ├── fn__page_header.py              ← Header + top-nav (REPLACED)
│   ├── fn__libraries.py                ← YOUR FILE — move from root
│   ├── fn__chart_libraries.py          ← YOUR FILE — move from root
│   └── fn_pvwatts.py                   ← PVWatts V8 API wrapper (NEW)
├── pages/
│   ├── 2_📊_View_Summary_Statistics.py ← unchanged logic, imports updated
│   ├── 3_⏱_Hourly_Charts.py           ← unchanged logic, imports updated
│   ├── 4_☀️_PV_Gen_Estimate.py        ← unchanged logic, imports updated (PVGIS)
│   └── 5_🏆_PV_LEED_Estimate.py       ← NEW — PVWatts V8 + LEED reporting
├── data/                               ← auto-created at runtime
├── img/
│   └── logo.png                        ← optional
└── .streamlit/
    └── secrets.toml                    ← NREL_API_KEY goes here
```

---

## Steps to migrate

### 1. Move your existing library files into libs/
```bash
mkdir libs
mv fn__libraries.py       libs/fn__libraries.py
mv fn__chart_libraries.py libs/fn__chart_libraries.py
touch libs/__init__.py
```
> fn__page_header.py at root is replaced by the new libs/fn__page_header.py.

### 2. Copy the provided files into place
All files in this package — drop them in as-is.

### 3. Add your NREL API key (for page 5 / PVWatts)
Create .streamlit/secrets.toml:
```toml
NREL_API_KEY = "your_key_here"
```
Free key at https://developer.nrel.gov/signup/

### 4. Run
```bash
streamlit run app.py
```

---

## What changed in each file

| File | Change |
|------|--------|
| libs/shared.py | NEW — CSS, CLIMATE_COLOR_MAP, .pv-metric-card styles |
| libs/fn__page_header.py | REPLACED — serif header + 5-item top nav |
| libs/fn_pvwatts.py | NEW — PVWatts V8 API wrapper, PVWattsResult dataclass |
| pages/2 | Import paths only (from libs.) |
| pages/3 | Import paths only |
| pages/4 | Import paths only — PVGIS logic untouched |
| pages/5_PV_LEED_Estimate.py | NEW — full PVWatts + LEED page |
| app.py | NEW — EPW/STAT upload + session state seeding |

---

## Top navigation bar

Rendered automatically by create_page_header() — every page gets it for free.

To edit labels or add pages, change _NAV_PAGES in libs/fn__page_header.py:

```python
_NAV_PAGES = [
    ("app",                                 "Home"),
    ("pages/2_View_Summary_Statistics",     "Summary Stats"),
    ("pages/3_Hourly_Charts",               "Hourly Charts"),
    ("pages/4_PV_Gen_Estimate",             "PV - PVGIS"),
    ("pages/5_PV_LEED_Estimate",            "PV - LEED / PVWatts"),
]
```

---

## Page 5 — PVWatts / LEED features

- Location auto-populated from uploaded EPW files (or manual entry)
- System capacity, module type, array/mounting, tilt, azimuth, losses
- Annual AC energy, specific yield (kWh/kWp), performance ratio
- LEED EA Credit compliance table with estimated LEED points badge
- Monthly bar + POA irradiance overlay chart
- Full data table (AC, DC, POA, daily irradiance per month)
- Optional 8760-hour breakdown (AC power, temperature, wind)
- CSV export for monthly and hourly data
- Results cached in st.session_state — sidebar tweaks do not re-query the API
