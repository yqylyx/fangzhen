from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

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
    category = str(row.get('category_label', ''))
    label_name = str(row.get('original_label', ''))
    seed_flag = int(row.get('is_seed_boundary', 0))
    reasons: list[str] = []

    if category == 'stable_side_boundary_candidate':
        reasons.append('原始标签为 stable，但尾段行为仍然非常接近边界。')
        if float(row.get('tail_amp_top1', np.nan)) >= 20.0:
            reasons.append('只有少数功角通道振荡特别强，说明存在局部高风险行为。')
        if float(row.get('tail_low_voltage_reentry_count', np.nan)) >= 1:
            reasons.append('电压尾段多次重新进入低压区，说明恢复并不干净。')
    elif category == 'unstable_side_boundary_candidate':
        reasons.append('原始标签为 unstable，但这个样本更像接近边界的失稳侧工况，而不是明显崩溃。')
        if float(row.get('tail_voltage_min', np.nan)) >= 0.78:
            reasons.append('尾段电压没有完全崩塌，仍然保留了明显的临界性。')
        if float(row.get('seed_similarity_score', np.nan)) >= 0.65:
            reasons.append('与已确认 seed 边界样本的相似度较高。')
    elif category == 'central_ambiguous_candidate':
        reasons.append('电压和功角特征同时落在中间风险带，属于中间临界候选。')
        if float(row.get('tail_spread_mean', np.nan)) >= 90.0 and float(row.get('final_recovered_ratio_0_9', np.nan)) <= 0.95:
            reasons.append('这个样本既不是明显恢复，也不是明显崩溃。')
    elif category == 'obvious_unstable':
        reasons.append('这个样本更接近明显失稳，主要作为全表对照保留。')
    else:
        reasons.append('这个样本更接近明显稳定，主要作为全表对照保留。')

    if seed_flag == 1:
        reasons.append('该样本已经在 seed 边界集中。')
    else:
        reasons.append('该样本不在当前 seed 集中，应作为新发现候选重点复查。')

    if label_name == 'stable' and float(row.get('seed_similarity_score', np.nan)) >= 0.7:
        reasons.append('这个 stable 样本与 seed 边界模式高度相似，值得优先人工复查。')
    return ' '.join(dict.fromkeys(reasons))


def add_reason_column(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if result.empty:
        result['suspicious_reason'] = []
        return result
    result['suspicious_reason'] = result.apply(build_reason, axis=1)
    return result


def _summary_rows(all_df: pd.DataFrame, new_df: pd.DataFrame) -> list[tuple[str, str]]:
    dataset_counts = all_df.groupby('dataset_name').size().to_dict() if not all_df.empty else {}
    return [
        ('总样本数', str(len(all_df))),
        ('36 数据集样本数', str(int(dataset_counts.get('36', 0)))),
        ('37 数据集样本数', str(int(dataset_counts.get('37', 0)))),
        ('74 数据集样本数', str(int(dataset_counts.get('74', 0)))),
        ('seed 边界样本数', str(int(all_df['is_seed_boundary'].sum()) if 'is_seed_boundary' in all_df else 0)),
        ('本轮新发现候选数', str(len(new_df))),
        ('稳定侧候选数', str(int((all_df['category_label'] == 'stable_side_boundary_candidate').sum()) if 'category_label' in all_df else 0)),
        ('失稳侧候选数', str(int((all_df['category_label'] == 'unstable_side_boundary_candidate').sum()) if 'category_label' in all_df else 0)),
        ('中间临界候选', str(int((all_df['category_label'] == 'central_ambiguous_candidate').sum()) if 'category_label' in all_df else 0)),
    ]


def _dataset_list_html(df: pd.DataFrame) -> str:
    if df.empty:
        return '<li>None</li>'
    series = df.groupby('dataset_name').size().sort_index()
    return ''.join(f'<li>{escape(str(idx))}: {int(value)}</li>' for idx, value in series.items())


def _sample_card(row: pd.Series) -> str:
    plot_rel_path = escape(str(row.get('plot_rel_path', '')))
    image_html = f'<img loading="lazy" src="{plot_rel_path}" alt="{escape(str(row.get("file", "")))}">' if plot_rel_path else '<div class="missing-plot">No plot available</div>'
    seed_badge = '<span class="badge seed">seed</span>' if int(row.get('is_seed_boundary', 0)) == 1 else '<span class="badge new">new</span>'
    metrics = [
        f'overall={_fmt(row.get("overall_candidate_score"))}',
        f'stable={_fmt(row.get("stable_side_score"))}',
        f'unstable={_fmt(row.get("unstable_side_score"))}',
        f'central={_fmt(row.get("central_ambiguous_score"))}',
        f'seed_sim={_fmt(row.get("seed_similarity_score"))}',
        f'tail_amp_top1={_fmt(row.get("tail_amp_top1"), 2)}',
    ]
    return f"""
    <div class="sample-card">
      <div class="sample-header">
        <h3>{seed_badge} {escape(Path(str(row.get('file', ''))).name)}</h3>
        <div class="meta">
          <span>dataset={escape(str(row.get('dataset_name', '-')))}</span>
          <span>label={escape(str(row.get('original_label', '-')))}</span>
          <span>category={escape(str(row.get('category_label', '-')))}</span>
        </div>
      </div>
      <div class="sample-body">
        <div class="sample-image">{image_html}</div>
        <div class="sample-info">
          <p><strong>文件:</strong> {escape(str(row.get('file', '')))}</p>
          <p><strong>原因:</strong> {escape(str(row.get('suspicious_reason', '')))}</p>
          <p><strong>指标:</strong> {escape(' | '.join(metrics))}</p>
        </div>
      </div>
    </div>
    """


def _section(title: str, df: pd.DataFrame) -> str:
    if df.empty:
        return f'<section><h2>{escape(title)}</h2><p>暂无样本可展示。</p></section>'
    cards = ''.join(_sample_card(row) for _, row in df.iterrows())
    return f'<section><h2>{escape(title)}</h2>{cards}</section>'


def build_reports(all_df: pd.DataFrame, sections: dict[str, Any], output_dir: str | Path, config: dict[str, Any]) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / 'report.html'
    md_path = output_dir / 'report.md'

    new_df = sections.get('new_candidates', all_df.iloc[0:0])
    summary_rows = _summary_rows(all_df, new_df)
    summary_html = ''.join(f'<tr><th>{escape(k)}</th><td>{escape(v)}</td></tr>' for k, v in summary_rows)
    category_sections = ''.join(
        _section(title, df)
        for title, df in [
            ('稳定侧边界候选', sections.get('stable_side', all_df.iloc[0:0])),
            ('失稳侧边界候选', sections.get('unstable_side', all_df.iloc[0:0])),
            ('中间临界候选', sections.get('central_ambiguous', all_df.iloc[0:0])),
        ]
    )
    dataset_sections = ''.join(_section(f'{dataset_name} 数据集最可疑新候选', df) for dataset_name, df in sections.get('dataset_tops', {}).items())

    html = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>边界候选发现报告</title>
  <style>
    body {{ font-family: 'Microsoft YaHei', 'SimHei', sans-serif; margin: 24px; background: #f7f8fb; color: #1d2433; }}
    h1, h2, h3 {{ color: #152033; }}
    .summary-grid {{ display: grid; grid-template-columns: 1.15fr 1fr 1fr; gap: 16px; align-items: start; }}
    .panel {{ background: #ffffff; border: 1px solid #dbe2ee; border-radius: 12px; padding: 16px; box-shadow: 0 4px 12px rgba(22, 31, 54, 0.04); }}
    .summary-table {{ border-collapse: collapse; width: 100%; }}
    .summary-table th, .summary-table td {{ border-bottom: 1px solid #eef1f6; padding: 8px 10px; text-align: left; }}
    .sample-card {{ background: #ffffff; border: 1px solid #dbe2ee; border-radius: 12px; margin: 16px 0; padding: 16px; box-shadow: 0 4px 12px rgba(22, 31, 54, 0.04); }}
    .sample-header {{ display: flex; justify-content: space-between; gap: 16px; align-items: baseline; flex-wrap: wrap; }}
    .meta span {{ margin-right: 12px; font-size: 0.94rem; color: #41506a; }}
    .sample-body {{ display: grid; grid-template-columns: 1.35fr 1fr; gap: 18px; align-items: start; }}
    .sample-image img {{ width: 100%; border-radius: 10px; border: 1px solid #e6eaf2; background: #fff; }}
    .missing-plot {{ min-height: 180px; display: flex; align-items: center; justify-content: center; border: 1px dashed #ccd5e3; border-radius: 10px; background: #fafbfc; color: #607089; }}
    .badge {{ display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 0.82rem; margin-right: 8px; }}
    .badge.seed {{ background: #e9eefb; color: #234e9d; }}
    .badge.new {{ background: #e8f7ee; color: #1d7a46; }}
    @media (max-width: 1200px) {{ .summary-grid {{ grid-template-columns: 1fr; }} .sample-body {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <h1>边界样本补全 / 主动发现报告</h1>
  <p>这份报告的目标是在训练最终边界分类模型之前，先尽量补全边界样本集。当前最重要的是复查本轮新发现的候选，并与现有 seed 集对照。</p>
  <section class="summary-grid">
    <div class="panel">
      <h2>总览</h2>
      <table class="summary-table">{summary_html}</table>
    </div>
    <div class="panel">
      <h2>本轮新候选数据集分布</h2>
      <ul>{_dataset_list_html(new_df)}</ul>
    </div>
    <div class="panel">
      <h2>说明</h2>
      <ul>
        <li><span class="badge seed">seed</span> 已在 seed 边界 CSV 中确认</li>
        <li><span class="badge new">new</span> 本轮新发现候选</li>
        <li>报告会分开展示稳定侧、失稳侧和中间临界候选</li>
      </ul>
    </div>
  </section>
  {_section('总体候选 Top-K', sections.get('overall_top', all_df.iloc[0:0]))}
  {_section('本轮新发现候选 Top-K', new_df)}
  {category_sections}
  <section>
    <h2>按数据集展示</h2>
    {dataset_sections or '<p>暂无可展示的数据集分组。</p>'}
  </section>
</body>
</html>
"""
    html_path.write_text(html, encoding='utf-8')

    md_lines = ['# 边界样本补全 / 主动发现报告', '']
    for key, value in summary_rows:
        md_lines.append(f'- {key}: {value}')
    md_lines.append('')
    md_lines.append('## 本轮新发现候选 Top-K')
    if new_df.empty:
        md_lines.append('- None')
    else:
        for _, row in new_df.iterrows():
            md_lines.append(f"- {row['file']} | dataset={row['dataset_name']} | label={row['original_label']} | category={row['category_label']} | score={_fmt(row.get('overall_candidate_score'))} | seed={int(row.get('is_seed_boundary', 0))}")
            md_lines.append(f"  原因: {row.get('suspicious_reason', '')}")
    md_path.write_text('\n'.join(md_lines) + '\n', encoding='utf-8')
    return html_path, md_path
