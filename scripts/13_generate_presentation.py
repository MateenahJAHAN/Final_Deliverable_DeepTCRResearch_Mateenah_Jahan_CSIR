#!/usr/bin/env python3
"""
Script 13: Generate PowerPoint Presentation
===========================================

Generates a PowerPoint (.pptx) presentation that documents the full work done
in this repository: data loading, preprocessing, DeepTCR training (100-fold
Monte Carlo CV), and post-training analyses (AUC/bootstrapping, attention,
responder vs non-responder, gene enrichment, top sequences, sequence
characteristics, and compute optimization).

Important:
- Images inserted into the deck are **code-generated figures** from this repo
  (or simple plots generated in this script). No AI-generated images are used.
"""

from __future__ import annotations

import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Try to import python-pptx
try:
    from pptx import Presentation
    from pptx.dml.color import RGBColor
    from pptx.enum.shapes import MSO_SHAPE
    from pptx.enum.text import PP_ALIGN
    from pptx.util import Inches, Pt

    PPTX_AVAILABLE = True
except ImportError:
    PPTX_AVAILABLE = False
    print("python-pptx not installed. Installing...")
    os.system("pip install python-pptx")
    try:
        from pptx import Presentation
        from pptx.dml.color import RGBColor
        from pptx.enum.shapes import MSO_SHAPE
        from pptx.enum.text import PP_ALIGN
        from pptx.util import Inches, Pt

        PPTX_AVAILABLE = True
    except Exception:
        print("Failed to install python-pptx. Please install manually: pip install python-pptx")
        PPTX_AVAILABLE = False


print("=" * 80)
print("POWERPOINT PRESENTATION GENERATOR - SCRIPT 13 (UPDATED)")
print("=" * 80)
print(f"Execution started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

if not PPTX_AVAILABLE:
    print("Cannot create PowerPoint - python-pptx not available")
    sys.exit(1)

# ------------------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures", "paper_final")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")

os.makedirs(LOGS_DIR, exist_ok=True)

today = datetime.now().strftime("%Y-%m-%d")
OUTPUT_FILE = os.path.join(RESULTS_DIR, f"DeepTCR_Project_Update_Presentation_{today}.pptx")
LOG_FILE = os.path.join(LOGS_DIR, "13_presentation.log")

# ------------------------------------------------------------------------------
# Styling
# ------------------------------------------------------------------------------

COLORS = {
    "primary": RGBColor(44, 62, 80),  # dark blue
    "accent": RGBColor(39, 174, 96),  # green
    "warning": RGBColor(231, 76, 60),  # red
    "blue": RGBColor(52, 152, 219),  # light blue
    "white": RGBColor(255, 255, 255),
    "gray": RGBColor(127, 140, 141),
    "light_gray": RGBColor(236, 240, 241),
}

SLIDE_WIDTH = Inches(13.333)
SLIDE_HEIGHT = Inches(7.5)


def _safe_read_csv(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _format_sci(x: float) -> str:
    try:
        return f"{x:.2e}"
    except Exception:
        return str(x)


def _add_title_bar(slide, title: str) -> None:
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_WIDTH, Inches(1.05))
    bar.fill.solid()
    bar.fill.fore_color.rgb = COLORS["primary"]
    bar.line.fill.background()

    tb = slide.shapes.add_textbox(Inches(0.6), Inches(0.22), Inches(12.2), Inches(0.8))
    tf = tb.text_frame
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(30)
    p.font.bold = True
    p.font.color.rgb = COLORS["white"]


def add_title_slide(prs: Presentation, title: str, subtitle: str) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_WIDTH, SLIDE_HEIGHT)
    bg.fill.solid()
    bg.fill.fore_color.rgb = COLORS["primary"]
    bg.line.fill.background()

    tb = slide.shapes.add_textbox(Inches(0.7), Inches(2.2), Inches(11.9), Inches(1.6))
    tf = tb.text_frame
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(44)
    p.font.bold = True
    p.font.color.rgb = COLORS["white"]
    p.alignment = PP_ALIGN.CENTER

    sb = slide.shapes.add_textbox(Inches(0.8), Inches(4.0), Inches(11.7), Inches(1.2))
    tf = sb.text_frame
    p = tf.paragraphs[0]
    p.text = subtitle
    p.font.size = Pt(22)
    p.font.color.rgb = COLORS["white"]
    p.alignment = PP_ALIGN.CENTER


def add_bullets_slide(prs: Presentation, title: str, bullets: list[str], note: str | None = None) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _add_title_bar(slide, title)

    box = slide.shapes.add_textbox(Inches(0.75), Inches(1.35), Inches(11.8), Inches(5.8))
    tf = box.text_frame
    tf.word_wrap = True

    for i, b in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = b
        p.font.size = Pt(22 if i == 0 else 20)
        p.font.color.rgb = COLORS["primary"]
        p.space_after = Pt(6)

    if note:
        nb = slide.shapes.add_textbox(Inches(0.75), Inches(6.95), Inches(11.8), Inches(0.5))
        tf = nb.text_frame
        p = tf.paragraphs[0]
        p.text = note
        p.font.size = Pt(12)
        p.font.italic = True
        p.font.color.rgb = COLORS["gray"]


def add_bullets_with_image_slide(
    prs: Presentation, title: str, bullets: list[str], image_path: str, image_caption: str | None = None
) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _add_title_bar(slide, title)

    left = slide.shapes.add_textbox(Inches(0.6), Inches(1.35), Inches(5.7), Inches(5.6))
    tf = left.text_frame
    tf.word_wrap = True
    for i, b in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = b
        p.font.size = Pt(20)
        p.font.color.rgb = COLORS["primary"]
        p.space_after = Pt(6)

    if os.path.exists(image_path):
        slide.shapes.add_picture(image_path, Inches(6.55), Inches(1.35), width=Inches(6.35))
    else:
        missing = slide.shapes.add_textbox(Inches(6.55), Inches(1.35), Inches(6.35), Inches(1))
        missing.text_frame.text = f"[Missing image: {os.path.basename(image_path)}]"

    if image_caption:
        cap = slide.shapes.add_textbox(Inches(6.55), Inches(6.95), Inches(6.35), Inches(0.4))
        p = cap.text_frame.paragraphs[0]
        p.text = image_caption
        p.font.size = Pt(12)
        p.font.italic = True
        p.font.color.rgb = COLORS["gray"]


def add_full_image_slide(prs: Presentation, title: str, image_path: str, caption: str | None = None) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _add_title_bar(slide, title)

    if os.path.exists(image_path):
        slide.shapes.add_picture(image_path, Inches(0.7), Inches(1.25), width=Inches(11.9))
    else:
        missing = slide.shapes.add_textbox(Inches(0.7), Inches(1.25), Inches(11.9), Inches(1))
        missing.text_frame.text = f"[Missing image: {os.path.basename(image_path)}]"

    if caption:
        cap = slide.shapes.add_textbox(Inches(0.75), Inches(7.0), Inches(11.8), Inches(0.4))
        p = cap.text_frame.paragraphs[0]
        p.text = caption
        p.font.size = Pt(12)
        p.font.italic = True
        p.font.color.rgb = COLORS["gray"]


def add_two_images_slide(prs: Presentation, title: str, left_image: str, right_image: str, caption: str | None = None) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _add_title_bar(slide, title)

    if os.path.exists(left_image):
        slide.shapes.add_picture(left_image, Inches(0.65), Inches(1.35), width=Inches(6.1))
    if os.path.exists(right_image):
        slide.shapes.add_picture(right_image, Inches(6.75), Inches(1.35), width=Inches(6.1))

    if caption:
        cap = slide.shapes.add_textbox(Inches(0.75), Inches(7.0), Inches(11.8), Inches(0.4))
        p = cap.text_frame.paragraphs[0]
        p.text = caption
        p.font.size = Pt(12)
        p.font.italic = True
        p.font.color.rgb = COLORS["gray"]


def add_table_slide(prs: Presentation, title: str, dataframe: pd.DataFrame, max_rows: int = 6) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _add_title_bar(slide, title)

    df = dataframe.copy()
    df = df.head(max_rows)

    rows = len(df) + 1
    cols = len(df.columns)

    table_shape = slide.shapes.add_table(rows, cols, Inches(0.7), Inches(1.4), Inches(11.9), Inches(4.8))
    table = table_shape.table

    # Header
    for j, col in enumerate(df.columns):
        cell = table.cell(0, j)
        cell.text = str(col)
        for p in cell.text_frame.paragraphs:
            p.font.bold = True
            p.font.size = Pt(14)
            p.font.color.rgb = COLORS["white"]
        cell.fill.solid()
        cell.fill.fore_color.rgb = COLORS["primary"]

    # Body
    for i in range(len(df)):
        for j, col in enumerate(df.columns):
            val = df.iloc[i, j]
            cell = table.cell(i + 1, j)
            cell.text = str(val)
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(12)
                p.font.color.rgb = COLORS["primary"]
            if i % 2 == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = COLORS["light_gray"]

    note = slide.shapes.add_textbox(Inches(0.75), Inches(6.35), Inches(11.8), Inches(0.6))
    p = note.text_frame.paragraphs[0]
    p.text = f"Showing top {min(max_rows, len(dataframe))} rows."
    p.font.size = Pt(12)
    p.font.italic = True
    p.font.color.rgb = COLORS["gray"]


# ------------------------------------------------------------------------------
# Load computed results (single source of truth for slide numbers)
# ------------------------------------------------------------------------------

print("-" * 80)
print("LOADING RESULTS")
print("-" * 80)

bootstrap_df = _safe_read_csv(os.path.join(RESULTS_DIR, "bootstrap_results.csv"))
patient_df = _safe_read_csv(os.path.join(RESULTS_DIR, "patient_summary_all.csv"))
pred_df = _safe_read_csv(os.path.join(RESULTS_DIR, "predictions_summary.csv"))
attn_summary_df = _safe_read_csv(os.path.join(RESULTS_DIR, "attention_weights_summary.csv"))
responder_stats_df = _safe_read_csv(os.path.join(RESULTS_DIR, "responder_comparison_stats.csv"))
enrichment_df = _safe_read_csv(os.path.join(RESULTS_DIR, "enrichment_summary.csv"))
top_sequences_df = _safe_read_csv(os.path.join(RESULTS_DIR, "top_100_sequences_detailed.csv"))

# Cohort stats (use canonical totals when available)
unique_patients = int(patient_df["patient_id"].nunique()) if patient_df is not None else None
responders = int((patient_df["response_binary"] == 1).sum()) if patient_df is not None else None
non_responders = int((patient_df["response_binary"] == 0).sum()) if patient_df is not None else None

total_sequences = None
if attn_summary_df is not None:
    try:
        total_sequences = int(float(attn_summary_df.loc[attn_summary_df["Metric"] == "Total Sequences", "Value"].iloc[0]))
    except Exception:
        total_sequences = None
if total_sequences is None and patient_df is not None:
    total_sequences = int(patient_df["unique_sequences"].sum())

min_seq = int(patient_df["unique_sequences"].min()) if patient_df is not None else None
max_seq = int(patient_df["unique_sequences"].max()) if patient_df is not None else None
mean_cdr3_len = float(patient_df["avg_len"].mean()) if patient_df is not None else None

# Model performance
mean_auc = float(bootstrap_df.loc[bootstrap_df["Metric"] == "Mean AUC", "Value"].iloc[0]) if bootstrap_df is not None else None
std_auc = float(bootstrap_df.loc[bootstrap_df["Metric"] == "Standard Deviation", "Value"].iloc[0]) if bootstrap_df is not None else None
ci95_low = float(bootstrap_df.loc[bootstrap_df["Metric"] == "95% CI Lower", "Value"].iloc[0]) if bootstrap_df is not None else None
ci95_high = float(bootstrap_df.loc[bootstrap_df["Metric"] == "95% CI Upper", "Value"].iloc[0]) if bootstrap_df is not None else None
auc_min = float(bootstrap_df.loc[bootstrap_df["Metric"] == "Minimum", "Value"].iloc[0]) if bootstrap_df is not None else None
auc_max = float(bootstrap_df.loc[bootstrap_df["Metric"] == "Maximum", "Value"].iloc[0]) if bootstrap_df is not None else None
n_folds = int(float(bootstrap_df.loc[bootstrap_df["Metric"] == "N Folds", "Value"].iloc[0])) if bootstrap_df is not None else None
n_boot = int(float(bootstrap_df.loc[bootstrap_df["Metric"] == "N Bootstrap", "Value"].iloc[0])) if bootstrap_df is not None else None

# Attention summary
high_attention_count = int(attn_summary_df.loc[attn_summary_df["Metric"] == "High Attention Sequences", "Value"].iloc[0]) if attn_summary_df is not None else None
high_attention_pct = float(attn_summary_df.loc[attn_summary_df["Metric"] == "High Attention Threshold (%)", "Value"].iloc[0]) if attn_summary_df is not None else None

# Enrichment
top_vgene = enrichment_df.loc[enrichment_df["Analysis"] == "V-Gene", "Top_Enriched"].iloc[0] if enrichment_df is not None else None
top_vgene_fc = float(enrichment_df.loc[enrichment_df["Analysis"] == "V-Gene", "Max_Fold_Enrichment"].iloc[0]) if enrichment_df is not None else None
top_jgene = enrichment_df.loc[enrichment_df["Analysis"] == "J-Gene", "Top_Enriched"].iloc[0] if enrichment_df is not None else None
top_jgene_fc = float(enrichment_df.loc[enrichment_df["Analysis"] == "J-Gene", "Max_Fold_Enrichment"].iloc[0]) if enrichment_df is not None else None

# Responder stats
mw_p = None
mw_eff = None
if responder_stats_df is not None:
    row = responder_stats_df[responder_stats_df["Test"].str.contains("Attention Weight", na=False)].head(1)
    if len(row) == 1:
        mw_p = float(row["P_Value"].iloc[0])
        mw_eff = float(row["Effect_Size"].iloc[0])

print(f"- Patients: {unique_patients} | Responders: {responders} | Non-responders: {non_responders}")
print(f"- Sequences: {total_sequences} | per patient: {min_seq} - {max_seq}")
print(f"- Mean AUC: {mean_auc:.4f} | 95% CI: [{ci95_low:.4f}, {ci95_high:.4f}] | folds={n_folds} bootstrap={n_boot}")

# ------------------------------------------------------------------------------
# Build presentation
# ------------------------------------------------------------------------------

print("\n" + "-" * 80)
print("CREATING POWERPOINT PRESENTATION")
print("-" * 80)

prs = Presentation()
prs.slide_width = SLIDE_WIDTH
prs.slide_height = SLIDE_HEIGHT

add_title_slide(
    prs,
    "DeepTCR for Immunotherapy Response Prediction",
    f"Relapsed/Refractory B-cell lymphoma • 100-fold Monte Carlo CV\nGenerated {today}",
)

add_bullets_slide(
    prs,
    "What’s in this talk",
    [
        "Project goal & dataset",
        "Pipeline (scripts 01–13): preprocessing → training → analysis",
        "Model performance (AUC + bootstrap confidence intervals)",
        "Attention-based interpretability + biological insights",
        "Key findings + next steps",
    ],
    note="All figures are generated by repository code (PNG exports), not AI-generated images.",
)

add_bullets_with_image_slide(
    prs,
    "Project Overview",
    [
        "Goal: predict immunotherapy response from TCRβ repertoires using DeepTCR (MIL + attention).",
        f"Dataset: {unique_patients} patients; {total_sequences:,} TCRβ sequences.",
        f"Labels: {responders} responders vs {non_responders} non-responders.",
        "Approach: 100-fold Monte Carlo cross-validation + post-training interpretation.",
    ],
    os.path.join(FIGURES_DIR, "figure3_cohort_overview.png"),
    image_caption="Cohort overview generated by analysis scripts.",
)

add_full_image_slide(
    prs,
    "End-to-end pipeline",
    os.path.join(FIGURES_DIR, "figure1_pipeline.png"),
    caption="Data loading → preprocessing/encoding → DeepTCR training → post-training analysis → figures/presentation.",
)

add_bullets_slide(
    prs,
    "Repository work completed (scripts 01–13)",
    [
        "01: environment verification (Python/CUDA/TensorFlow/DeepTCR).",
        "02: load + clean dataset → `data_processed/deeptcr_trb_ready.csv`.",
        "03: EDA → cohort/sequence figures.",
        "04: one-hot encoding → `X_onehot.npy`, labels, patient IDs.",
        "06: 100-fold Monte Carlo training (patient-level splits).",
        "07–12: AUC + bootstrap, attention extraction/plots, R vs NR stats, gene enrichment, top sequences, AA characteristics.",
        "13: generate this PowerPoint.",
    ],
)

add_bullets_with_image_slide(
    prs,
    "DeepTCR model architecture",
    [
        "Multiple Instance Learning: a patient = bag of TCR sequences.",
        "CNN embedding of CDR3 sequences.",
        "Attention learns which sequences drive the prediction (interpretability).",
        "Output: binary responder vs non-responder.",
    ],
    os.path.join(FIGURES_DIR, "figure2_architecture.png"),
)

add_bullets_with_image_slide(
    prs,
    "Training setup: 100-fold Monte Carlo CV",
    [
        f"{n_folds} random patient-level splits (train 75% / test 25%).",
        "Re-estimate performance over many splits to reduce variance from any one partition.",
        "Saved models + logs for reproducibility.",
    ],
    os.path.join(FIGURES_DIR, "figure5_training_dynamics.png"),
    image_caption="Training dynamics from training logs/exports.",
)

add_bullets_with_image_slide(
    prs,
    "Performance summary (AUC)",
    [
        f"Mean AUC = {mean_auc:.3f} ± {std_auc:.3f} (SD across folds).",
        f"95% bootstrap CI = [{ci95_low:.3f}, {ci95_high:.3f}] (n={n_boot}).",
        f"AUC range across folds = [{auc_min:.3f}, {auc_max:.3f}].",
    ],
    os.path.join(FIGURES_DIR, "figure4_model_performance.png"),
)

add_full_image_slide(
    prs,
    "AUC distribution across folds",
    os.path.join(FIGURES_DIR, "figureS1_auc_details.png"),
    caption="Fold-wise AUC distribution; generated by post-training analysis scripts.",
)

add_bullets_with_image_slide(
    prs,
    "Attention analysis (interpretability)",
    [
        f"High-attention sequences: {high_attention_count:,} (top {high_attention_pct:.0f}%).",
        "Attention is sparse: a small subset of sequences explains predictions.",
        "Enables biological follow-up: enriched V/J genes and high-attention clonotypes.",
    ],
    os.path.join(FIGURES_DIR, "figure6_attention_analysis.png"),
)

add_two_images_slide(
    prs,
    "Attention distributions",
    os.path.join(FIGURES_DIR, "figure_attention_distribution.png"),
    os.path.join(FIGURES_DIR, "figure_attention_by_response.png"),
    caption="Left: overall attention distribution. Right: attention by response label.",
)

add_bullets_with_image_slide(
    prs,
    "Responder vs non-responder comparison",
    [
        "Statistical tests on attention/sequence properties (see `results/responder_comparison_stats.csv`).",
        f"Attention Mann–Whitney p={_format_sci(mw_p)} (effect size={mw_eff:.3f})" if mw_p is not None else "Attention Mann–Whitney computed in results.",
        "Additional: CDR3 length distribution tests (KS/t-test) included in results.",
    ],
    os.path.join(FIGURES_DIR, "figure_responder_comparison.png"),
)

add_bullets_with_image_slide(
    prs,
    "V/J gene usage and enrichment",
    [
        f"Top enriched V-gene in high-attention sequences: {top_vgene} ({top_vgene_fc:.2f}×).",
        f"Top enriched J-gene in high-attention sequences: {top_jgene} ({top_jgene_fc:.2f}×).",
        "Gene usage patterns provide interpretable biological signals.",
    ],
    os.path.join(FIGURES_DIR, "figure7_gene_usage.png"),
)

if top_sequences_df is not None:
    cols = ["rank", "aminoAcid", "vGeneName", "jGeneName", "response_label", "attention_weight"]
    shown = top_sequences_df.sort_values("rank").loc[:, cols].head(5).copy()
    shown["attention_weight"] = shown["attention_weight"].map(lambda x: f"{x:.4f}")
    shown.rename(
        columns={
            "rank": "Rank",
            "aminoAcid": "CDR3",
            "vGeneName": "V gene",
            "jGeneName": "J gene",
            "response_label": "Label",
            "attention_weight": "Attention",
        },
        inplace=True,
    )
    add_table_slide(prs, "Top predictive sequences (examples)", shown, max_rows=5)

add_full_image_slide(
    prs,
    "Top sequence patterns (top-100 analysis)",
    os.path.join(FIGURES_DIR, "figure_top_sequences.png"),
    caption="Summary visualization of the top 100 high-attention sequences.",
)

add_full_image_slide(
    prs,
    "Sequence characteristics",
    os.path.join(FIGURES_DIR, "figure_sequence_characteristics.png"),
    caption=f"CDR3 length & amino-acid properties (mean length ~{mean_cdr3_len:.2f}).",
)

add_full_image_slide(
    prs,
    "Compute optimization (GPU/H100)",
    os.path.join(FIGURES_DIR, "figureS4_computational.png"),
    caption="Runtime/memory benchmarking and optimized training settings.",
)

add_bullets_slide(
    prs,
    "Key takeaways",
    [
        f"DeepTCR achieved mean AUC {mean_auc:.3f} with tight bootstrap CI [{ci95_low:.3f}, {ci95_high:.3f}].",
        "Attention provides interpretability: highlights a small subset of predictive sequences.",
        "Enriched V/J gene usage and top clonotypes are candidates for biological validation.",
        "End-to-end pipeline is automated and reproducible (scripts 01–13, logs, figures, paper).",
    ],
)

add_title_slide(prs, "Thank you", "Questions?")

print("\n" + "-" * 80)
print("SAVING PRESENTATION")
print("-" * 80)

prs.save(OUTPUT_FILE)
print(f"Saved: {OUTPUT_FILE}")

with open(LOG_FILE, "w") as f:
    f.write("=" * 80 + "\n")
    f.write("PRESENTATION GENERATION LOG\n")
    f.write("=" * 80 + "\n")
    f.write(f"Execution time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write(f"Output file: {OUTPUT_FILE}\n")
    f.write(f"Slides: {len(prs.slides)}\n")

print(f"Log saved to: {LOG_FILE}")

print("\n" + "=" * 80)
print("PRESENTATION GENERATION COMPLETE!")
print("=" * 80)
