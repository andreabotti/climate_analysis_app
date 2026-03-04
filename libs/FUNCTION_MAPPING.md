# libs/ — Function Rename & Numbering Reference

## File structure

```
libs/
├── __init__.py        re-exports all four modules
├── fn__header.py      f201–f202   page header (UI series)
├── fn__ui.py          f203–f211   UI helpers  (UI series)
├── fn__data.py        f101–f124   data layer  (data series)
└── fn__charts.py      f301–f324   charts      (charts series)
```

---

## libs/fn__header.py  (f201–f202)

| # | New name | Old name |
|---|---|---|
| f201 | f201__render_logo | _render_logo |
| f202 | f202__create_page_header | create_page_header |

---

## libs/fn__ui.py  (f203–f211)

Absorbs: shared.py + UI helpers from fn__libraries.py

| # | New name | Old name |
|---|---|---|
| f203 | f203__inject_global_css | inject_global_css |
| f204 | f204__custom_divider | custom_divider |
| f205 | f205__custom_hr | custom_hr |
| f206 | f206__tight_hr_spacing | absrd_tight_hr_spacing |
| f207 | f207__custom_filled_region | custom_filled_region |
| f208 | f208__style_df | absrd_style_df_streamlit |
| f209 | f209__highlight_odd_even_cells | highlight_odd_even_cells |
| f210 | f210__row_style | row_style |
| f211 | f211__epw_location_map | absrd__epw_location_map |

---

## libs/fn__data.py  (f101–f124)

Absorbs: fn__libraries.py (data sections) + fn_pvwatts.py

| # | New name | Old name |
|---|---|---|
| f101 | f101__apply_analysis_period | absrd_apply_analysis_period |
| f102 | f102__slice_df_analysis_period | absrd_slice_df_analysis_period |
| f103 | f103__get_colors | get_colors |
| f104 | f104__rgb_to_hex | rgb_to_hex |
| f105 | f105__epw_hash_func | epw_hash_func |
| f106 | f106__hourly_data_hash_func | hourly_data_hash_func |
| f107 | f107__color_hash_func | color_hash_func |
| f108 | f108__list_zip_files_in_s3 | list_zip_files_in_s3 |
| f109 | f109__download_file | download_file |
| f110 | f110__get_epw_from_url_or_path | get_epw_from_url_or_path |
| f111 | f111__add_to_epw_files_list | add_to_epw_files_list |
| f112 | f112__filter_epw_object | filter_epw_object |
| f113 | f113__process_stat_file | absrd_process_stat_file |
| f114 | f114__convert_outputs_to_dataframe | convert_outputs_to_dataframe |
| f115 | f115__fetch_pvgis_hourly_data | fetch_pvgis_hourly_data |
| f116 | f116__fetch_pv_production_data | fetch_pv_production_data |
| f117 | f117__parse_pv_production_data | parse_pv_production_data |
| f118 | f118__extract_meta_monthly_variable_info | extract_meta_monthly_variable_info |
| f119 | f119__extract_meta_variable_info_by_type | extract_meta_variable_info_by_type |
| f120 | f120__split_and_transpose_pv_monthly | split_and_transpose_pv_monthly |
| f121 | f121__iterate_pv_production | iterate_pv_production |
| f122 | f122__define_yaxis_range | define_yaxis_range |
| f123 | f123__get_api_key | get_api_key (fn_pvwatts) |
| f124 | f124__pvwatts_query | pvwatts_query (fn_pvwatts) |

---

## libs/fn__charts.py  (f301–f324)

Absorbs: fn__chart_libraries.py + chart helpers from fn__libraries.py

| # | New name | Old name |
|---|---|---|
| f301 | f301__avg_daily_profile | absrd_avg_daily_profile |
| f302 | f302__plot_hourly_line_chart_with_slider | plot_hourly_line_chart_with_slider |
| f303 | f303__convert_rgba_to_plotly_colorscale | convert_rgba_to_plotly_colorscale |
| f304 | f304__plot_heatmap_from_datetime_index | plot_heatmap_from_datetime_index |
| f305 | f305__slice_data_by_month | slice_data_by_month |
| f306 | f306__bin_timeseries_data | bin_timeseries_data |
| f307 | f307__normalize_data | normalize_data |
| f308 | f308__create_stacked_bar_chart | create_stacked_bar_chart |
| f309 | f309__get_figure_config | get_figure_config |
| f310 | f310__get_fields | get_fields |
| f311 | f311__get_diurnal_average_chart_figure | get_diurnal_average_chart_figure |
| f312 | f312__get_hourly_data_figure | get_hourly_data_figure |
| f313 | f313__get_bar_chart_figure | get_bar_chart_figure |
| f314 | f314__get_hourly_line_chart_figure | get_hourly_line_chart_figure |
| f315 | f315__get_hourly_diurnal_average_chart_figure | get_hourly_diurnal_average_chart_figure |
| f316 | f316__get_daily_chart_figure | get_daily_chart_figure |
| f317 | f317__get_sunpath_figure | get_sunpath_figure |
| f318 | f318__get_degree_days_figure | get_degree_days_figure |
| f319 | f319__get_windrose_figure | get_windrose_figure |
| f320 | f320__get_psy_chart_figure | get_psy_chart_figure |
| f321 | f321__line_chart_daily_range | absrd__line_chart_daily__range |
| f322 | f322__plot_windrose | plot_windrose |
| f323 | f323__daytime_nighttime_scatter | absrd_daytime_nighttime_scatter |
| f324 | f324__plot_pv_monthly_comparison_subplots | plot_pv_monthly_comparison_subplots |

---

## Page import changes summary

| Page | Old import | New import |
|------|-----------|------------|
| 1–5 | from libs.fn__libraries import * | from libs.fn__data import * |
| 1–5 | from libs.fn__chart_libraries import * | from libs.fn__charts import * |
| 1–5 | from libs.fn__page_header import create_page_header | from libs.fn__header import create_page_header |
| 4 | from libs.fn__libraries import fetch_pv_production_data, iterate_pv_production | from libs.fn__data import f116__fetch_pv_production_data as fetch_pv_production_data, f121__iterate_pv_production as iterate_pv_production |
| 5 | from libs.fn_pvwatts import pvwatts_query, get_api_key, ... | from libs.fn__data import pvwatts_query, get_api_key, ... |
| 5 | from libs.shared import CLIMATE_COLOR_MAP, BOX_CHART_MARGINS, custom_divider | from libs.fn__ui import CLIMATE_COLOR_MAP, BOX_CHART_MARGINS, f204__custom_divider as custom_divider |

---

## Notes

All old function names are kept as backward-compatible aliases at the bottom of
each file, so page body code using old names (custom_hr, get_fields, etc.) works
unchanged without any edits to the page bodies.

The five original files can now be deleted:
  shared.py, fn__page_header.py, fn__libraries.py, fn__chart_libraries.py, fn_pvwatts.py
