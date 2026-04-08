from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return '-'
    try:
        value = float(value)
    except Exception:
        return escape(str(value))
    if not np.isfinite(value):
        return '-'
    return f'{value:.{digits}f}'


def build_reason(row: pd.Series) -> str:
    reasons: list[str] = []
    boundary_side = str(row.get('boundary_side', 'general'))

    if boundary_side == 'stable_side_high_voltage':
        reasons.append('稳定标签样本，电压整体仍较高，但功角展宽和角速度已经逼近快失稳区，属于高电压快速失稳边界候选。')
    elif boundary_side == 'stable_side':
        reasons.append('稳定标签样本，但功角尾段仍有明显振荡或电压接近临界恢复区，属于稳定侧边界候选。')
    elif boundary_side == 'unstable_side':
        reasons.append('失稳标签样本，但末段并未完全崩溃，更像反复振荡或局部失稳，属于失稳侧边界候选。')
    elif int(row.get('stable_clear_flag', 0)) == 0 and int(row.get('unstable_clear_flag', 0)) == 0:
        reasons.append('既不属于明显稳定，也不属于明显失稳，属于典型临界候选。')

    tail_spread_mean = float(row.get('tail_spread_mean', np.nan))
    tail_spread_slope_abs = float(row.get('tail_spread_slope_abs', np.nan))
    angle_speed_median = float(row.get('angle_speed_median', np.nan))
    tail_voltage_min = float(row.get('tail_voltage_min', np.nan))
    recover_ratio = float(row.get('final_recovered_ratio_0_9', np.nan))
    high_voltage_fast_flag = int(row.get('high_voltage_fast_flag', 0))

    if np.isfinite(tail_spread_mean) and np.isfinite(angle_speed_median) and tail_spread_mean >= 100.0 and angle_speed_median >= 10.0:
        reasons.append('功角后段振荡较强，展宽和角速度同时落在中间风险带。')

    if high_voltage_fast_flag == 1:
        reasons.append('虽然电压没有明显下跌，但功角动态已经表现出高风险快失稳迹象。')
    elif np.isfinite(tail_voltage_min) and 0.8 <= tail_voltage_min <= 0.93:
        reasons.append('电压末段反复接近低压危险区，但没有彻底崩塌。')
    elif np.isfinite(recover_ratio) and 0.3 <= recover_ratio <= 0.9:
        reasons.append('电压恢复比例处在中间区间，存在临界恢复特征。')

    if np.isfinite(tail_spread_slope_abs) and tail_spread_slope_abs <= 8.0 and np.isfinite(tail_spread_mean) and tail_spread_mean >= 90.0:
        reasons.append('功角展宽斜率不大但尾段仍较宽，更像弱阻尼或慢衰减工况。')

    if int(row.get('stable_clear_flag', 0)) == 1:
        reasons.append('该样本整体仍偏明显稳定，基础分数已按稳定规则下调。')
    if int(row.get('unstable_clear_flag', 0)) == 1:
        reasons.append('该样本整体仍偏明显失稳，且已从 Top-K 候选中排除。')

    if not reasons:
        reasons.append('关键指标整体落在中间区间，建议优先人工看图判断。')

    deduped: list[str] = []
    for reason in reasons:
        if reason not in deduped:
            deduped.append(reason)
    return ''.join(deduped[:3])


def add_reason_column(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if result.empty:
        result['suspicious_reason'] = []
        return result
    result['suspicious_reason'] = result.apply(build_reason, axis=1)
    return result


def _dataset_count_html(series: pd.Series) -> str:
    if series.empty:
        return '<li>None</li>'
    return ''.join(f'<li>{escape(str(idx))}: {int(value)}</li>' for idx, value in series.items())


def _summary_table(all_df: pd.DataFrame) -> str:
    candidate_mask = all_df.get('boundary_candidate_flag', 1).fillna(1).astype(int) == 1 if not all_df.empty else pd.Series(dtype=bool)
    candidate_df = all_df.loc[candidate_mask].copy() if not all_df.empty else all_df
    stats = candidate_df['boundary_score'].describe(percentiles=[0.25, 0.5, 0.75]) if not candidate_df.empty else pd.Series(dtype=float)
    middle_count = int(((candidate_df['stable_clear_flag'] == 0) & (candidate_df['unstable_clear_flag'] == 0)).sum()) if not candidate_df.empty else 0
    stable_side_count = int(candidate_df['stable_side_boundary_flag'].sum()) if 'stable_side_boundary_flag' in candidate_df else 0
    unstable_side_count = int(candidate_df['unstable_side_boundary_flag'].sum()) if 'unstable_side_boundary_flag' in candidate_df else 0
    high_voltage_fast_count = int(candidate_df['high_voltage_fast_flag'].sum()) if 'high_voltage_fast_flag' in candidate_df else 0

    rows = [
        ('Total samples', str(len(all_df))),
        ('Boundary candidates kept for report', str(len(candidate_df))),
        ('Stable-side boundary candidates', str(stable_side_count)),
        ('High-voltage near-instability candidates', str(high_voltage_fast_count)),
        ('Unstable-side boundary candidates', str(unstable_side_count)),
        ('Clear stable samples', str(int(all_df['stable_clear_flag'].sum()) if not all_df.empty else 0)),
        ('Clear unstable samples', str(int(all_df['unstable_clear_flag'].sum()) if not all_df.empty else 0)),
        ('Excluded non-candidate samples', str(int((all_df['boundary_candidate_flag'] == 0).sum()) if 'boundary_candidate_flag' in all_df else 0)),
        ('Middle suspicious samples', str(middle_count)),
        ('Boundary score mean', _fmt(stats.get('mean'))),
        ('Boundary score std', _fmt(stats.get('std'))),
        ('Boundary score min', _fmt(stats.get('min'))),
        ('Boundary score p25', _fmt(stats.get('25%'))),
        ('Boundary score median', _fmt(stats.get('50%'))),
        ('Boundary score p75', _fmt(stats.get('75%'))),
        ('Boundary score max', _fmt(stats.get('max'))),
    ]
    html_rows = ''.join(f'<tr><th>{escape(k)}</th><td>{escape(v)}</td></tr>' for k, v in rows)
    return f'<table class="summary-table">{html_rows}</table>'


def _sample_card(row: pd.Series) -> str:
    plot_rel_path = escape(str(row.get('plot_rel_path', '')))
    metrics_html = ''.join(
        [
            f'<li>boundary_side: {escape(str(row.get("boundary_side", "general")))}</li>',
            f'<li>tail_voltage_min: {_fmt(row.get("tail_voltage_min"))}</li>',
            f'<li>final_recovered_ratio_0_9: {_fmt(row.get("final_recovered_ratio_0_9"))}</li>',
            f'<li>tail_spread_mean: {_fmt(row.get("tail_spread_mean"), 2)}</li>',
            f'<li>tail_spread_slope: {_fmt(row.get("tail_spread_slope"), 2)}</li>',
            f'<li>angle_speed_median: {_fmt(row.get("angle_speed_median"), 2)}</li>',
            f'<li>stable_side_signal: {_fmt(row.get("stable_side_signal"), 3)}</li>',
            f'<li>high_voltage_fast_signal: {_fmt(row.get("high_voltage_fast_signal"), 3)}</li>',
            f'<li>unstable_side_signal: {_fmt(row.get("unstable_side_signal"), 3)}</li>',
            f'<li>stable_side_boundary_flag: {int(row.get("stable_side_boundary_flag", 0))}</li>',
            f'<li>high_voltage_fast_flag: {int(row.get("high_voltage_fast_flag", 0))}</li>',
            f'<li>unstable_side_boundary_flag: {int(row.get("unstable_side_boundary_flag", 0))}</li>',
        ]
    )
    image_html = f'<img loading="lazy" src="{plot_rel_path}" alt="{escape(str(row.get("file", "")))}">' if plot_rel_path else '<div class="missing-plot">No plot available</div>'

    return f"""
    <div class="sample-card">
      <div class="sample-header">
        <h3>#{int(row['boundary_rank'])} | {escape(Path(str(row['file'])).name)}</h3>
        <div class="meta">
          <span>dataset={escape(str(row['dataset_name']))}</span>
          <span>label={escape(str(row['original_label']))}</span>
          <span>side={escape(str(row.get('boundary_side', 'general')))}</span>
          <span>score={_fmt(row['boundary_score'])}</span>
        </div>
      </div>
      <div class="sample-body">
        <div class="sample-image">{image_html}</div>
        <div class="sample-info">
          <p><strong>File:</strong> {escape(str(row['file']))}</p>
          <p><strong>Reason:</strong> {escape(str(row.get('suspicious_reason', '')))}</p>
          <ul>{metrics_html}</ul>
        </div>
      </div>
    </div>
    """


def _build_section(title: str, subset: pd.DataFrame) -> str:
    if subset.empty:
        return f'<section><h2>{escape(title)}</h2><p>No samples available.</p></section>'
    cards = ''.join(_sample_card(row) for _, row in subset.iterrows())
    return f'<section><h2>{escape(title)}</h2>{cards}</section>'


def build_html_report(all_df: pd.DataFrame, top_df: pd.DataFrame, output_dir: str | Path, config: Dict[str, Any]) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / 'report.html'

    dataset_counts = all_df.groupby('dataset_name').size().sort_index() if not all_df.empty else pd.Series(dtype=int)
    dataset_top_counts = top_df.groupby('dataset_name').size().sort_index() if not top_df.empty else pd.Series(dtype=int)
    per_dataset_show = int(config.get('report', {}).get('per_dataset_show', 60))
    side_show = int(config.get('report', {}).get('side_boundary_show', 80))

    top_section = _build_section('Top 可疑样本展示', top_df)
    stable_side_df = top_df[top_df['stable_side_boundary_flag'] == 1].head(side_show) if 'stable_side_boundary_flag' in top_df else top_df.iloc[0:0]
    unstable_side_df = top_df[top_df['unstable_side_boundary_flag'] == 1].head(side_show) if 'unstable_side_boundary_flag' in top_df else top_df.iloc[0:0]
    high_voltage_df = top_df[top_df['high_voltage_fast_flag'] == 1].head(side_show) if 'high_voltage_fast_flag' in top_df else top_df.iloc[0:0]

    grouped_sections = []
    for dataset_name, group in top_df.groupby('dataset_name', sort=True):
        grouped_sections.append(_build_section(f'{dataset_name}data 最可疑样本', group.head(per_dataset_show)))
    grouped_html = ''.join(grouped_sections) if grouped_sections else '<p>No grouped samples available.</p>'

    html = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>Boundary Sample Scan Report</title>
  <style>
    body {{ font-family: 'Microsoft YaHei', 'SimHei', sans-serif; margin: 24px; background: #f7f8fb; color: #1d2433; }}
    h1, h2, h3 {{ color: #152033; }}
    .summary-grid {{ display: grid; grid-template-columns: 1.2fr 1fr 1fr; gap: 16px; align-items: start; }}
    .panel {{ background: #ffffff; border: 1px solid #dbe2ee; border-radius: 12px; padding: 16px; box-shadow: 0 4px 12px rgba(22, 31, 54, 0.04); }}
    .summary-table {{ border-collapse: collapse; width: 100%; }}
    .summary-table th, .summary-table td {{ border-bottom: 1px solid #eef1f6; padding: 8px 10px; text-align: left; }}
    .sample-card {{ background: #ffffff; border: 1px solid #dbe2ee; border-radius: 12px; margin: 16px 0; padding: 16px; box-shadow: 0 4px 12px rgba(22, 31, 54, 0.04); }}
    .sample-header {{ display: flex; justify-content: space-between; gap: 16px; align-items: baseline; flex-wrap: wrap; }}
    .meta span {{ margin-right: 12px; font-size: 0.95rem; color: #41506a; }}
    .sample-body {{ display: grid; grid-template-columns: 1.4fr 1fr; gap: 18px; align-items: start; }}
    .sample-image img {{ width: 100%; border-radius: 10px; border: 1px solid #e6eaf2; background: #fff; }}
    .sample-info ul {{ padding-left: 18px; line-height: 1.65; }}
    .missing-plot {{ min-height: 180px; display: flex; align-items: center; justify-content: center; border: 1px dashed #ccd5e3; border-radius: 10px; background: #fafbfc; color: #607089; }}
    @media (max-width: 1200px) {{ .summary-grid {{ grid-template-columns: 1fr; }} .sample-body {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <h1>可疑边界样本筛选报告</h1>
  <p>本报告会自动排除明显失稳样本，只保留更值得人工看图的边界候选，同时额外强化稳定侧边界、失稳侧边界和高电压快失稳样本的展示。</p>

  <section class="summary-grid">
    <div class="panel">
      <h2>总览统计</h2>
      {_summary_table(all_df)}
    </div>
    <div class="panel">
      <h2>各数据集样本数</h2>
      <ul>{_dataset_count_html(dataset_counts)}</ul>
    </div>
    <div class="panel">
      <h2>Top-K 可疑样本分布</h2>
      <ul>{_dataset_count_html(dataset_top_counts)}</ul>
    </div>
  </section>

  <section>
    <h2>人工查看建议</h2>
    <ul>
      <li>稳定侧边界：重点看 stable 标签样本里，功角尾段是否仍持续振荡，电压是否反复接近 0.85~0.9 p.u.</li>
      <li>高电压快失稳：即便电压整体仍较高，只要功角展宽和角速度明显抬升，也应视为边界候选。</li>
      <li>失稳侧边界：重点看 unstable 标签样本里，是否存在明显震荡但没有完全崩溃的过渡工况。</li>
      <li>stable_side_boundary_flag、high_voltage_fast_flag 或 unstable_side_boundary_flag 为 1 的样本，优先人工看图。</li>
    </ul>
  </section>

  {top_section}
  {_build_section('稳定侧边界候选样本', stable_side_df)}
  {_build_section('高电压快失稳候选样本', high_voltage_df)}
  {_build_section('失稳侧边界候选样本', unstable_side_df)}

  <section>
    <h2>按数据集分组展示</h2>
    {grouped_html}
  </section>
</body>
</html>
"""

    report_path.write_text(html, encoding='utf-8')
    return report_path
