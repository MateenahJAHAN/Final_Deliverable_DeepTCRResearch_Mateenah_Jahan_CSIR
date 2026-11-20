"""Build the correction PDFs and their figures from verified aggregate outputs."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Table, TableStyle

root = Path(__file__).resolve().parents[1]
evidence = root / 'validation/correction-rerun'
figures = root / 'figures/correction'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.labelcolor': '#24384c',
                     'text.color': '#162b40', 'savefig.facecolor': 'white'})
results = json.loads((evidence / 'aggregate_results.json').read_text())
descriptive = json.loads((evidence / 'checked_descriptives.json').read_text())
folds = results['folds']
control = results['shuffled_training_control']
assert results['status'] == 'complete' and len(folds) == 5
assert all(len(item['epochs']) == 10 for item in [*folds, control])
assert results['patients'] == sum(fold['test_patients'] for fold in folds) == 34
assert results['retained_rows'] == sum(fold['test_rows'] for fold in folds) == 239634
auroc = results['within_fold_patient_auroc']
lo, hi = results['within_fold_patient_auroc_bootstrap_ci95']
pooled = results['pooled_oof_patient_auroc']
ap = results['pooled_oof_patient_average_precision']
wins = results['patient_pairs_concordant_ties_half']
pairs = results['patient_pairs']
users = results['patients']
responders = results['positive_patients']
teal = '#12796f'
blue = '#366c90'
rust = '#a45135'
ink = colors.HexColor('#162b40')
muted = colors.HexColor('#3f5365')
rule = colors.HexColor('#d5dfe7')
wash = colors.HexColor('#edf5f4')
base_url = 'https://github.com/MateenahJAHAN/Final_Deliverable_DeepTCRResearch_Mateenah_Jahan_CSIR'
archive = base_url + '/blob/ce1fa3a421edb9f4025a45cb79dc7b014867ad83'
pin = archive + '/validation/patient-grouped-deeptcr'

def save(fig, name):
    fig.savefig(figures / (name + '.png'), dpi=215)
    plt.close(fig)

fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
a.axvline(.5, color='#9daab4', ls='--', lw=1.2)
a.errorbar(auroc, 6, xerr=[[auroc-lo], [hi-auroc]], fmt='D', color=teal, capsize=4, lw=2)
a.set_yticks(range(7), ['Shuffled control', 'Fold 5', 'Fold 4', 'Fold 3', 'Fold 2', 'Fold 1', 'Within-fold aggregate'])
for fold in folds:
    a.scatter(fold['patient_auroc'], 6-fold['fold'], color=blue, s=35)
a.scatter(control['patient_auroc'], 0, color=rust, marker='s', s=38)
a.set(xlim=(-.035, 1.035), ylim=(-.6, 6.6), xlabel='Patient AUROC')
a.set_title('A   Held-out patient discrimination', loc='left', weight='bold')
for fold in [*folds, control]:
    is_control = fold is control
    b.plot([e['epoch'] for e in fold['epochs']], [e['batch_mean_loss'] for e in fold['epochs']],
           marker=None if is_control else 'o', markersize=2.4, lw=1.4,
           linestyle='--' if is_control else '-', color=rust if is_control else None,
           label='Shuffled control' if is_control else f"Fold {fold['fold']}")
b.legend(frameon=False, fontsize=7, ncols=2)
b.set(xlabel='Training epoch', ylabel='Mean minibatch training loss', xticks=range(1, 11))
b.set_title('B   Observed training loss', loc='left', weight='bold')
save(fig, 'held-out-performance')

fig, (a, b, c) = plt.subplots(1, 3, figsize=(11, 3.7), layout='constrained', gridspec_kw={'width_ratios':[1, 1, 1.45]})
groups = descriptive['groups']
for axis, key, name, unit in ((a, 'record_quantiles', 'A   Repertoire size', 'Eligible records per patient'),
                              (b, 'mean_length_quantiles', 'B   Mean CDR3 length', 'Amino acids')):
    for i, group in enumerate(groups):
        q1, median, q3 = group[key]
        axis.errorbar(median, i, xerr=[[median-q1], [q3-median]], fmt='o', capsize=4,
                      color=[blue, teal][i], lw=2)
    axis.set_yticks([0,1], ['Nonresp.', 'Resp.'])
    axis.set_ylim(-.5, 1.5)
    axis.set_xlabel(unit)
    axis.set_title(name, loc='left', weight='bold')
bins = descriptive['size_bins']
heights = [group['responders']/group['patients']*100 for group in bins]
c.bar(range(len(bins)), heights, color=teal, width=.58)
for i, (group, height) in enumerate(zip(bins, heights)):
    c.text(i, height + 3, f"{group['responders']}/{group['patients']}", ha='center')
c.set_xticks(range(len(bins)), [g['bin'] for g in bins], rotation=21, ha='right', fontsize=8)
c.set(xlabel='Eligible records per patient', ylabel='Recorded responders (%)', ylim=(0,118))
c.set_title('C   Response by repertoire size', loc='left', weight='bold')
save(fig, 's2-cohort-description')

fig, ax = plt.subplots(figsize=(10.5, 3.7), layout='constrained')
x = np.arange(10)
ax.bar(x-.18, descriptive['cluster_means']['Responder'], .35, label='Responders (18)', color=teal)
ax.bar(x+.18, descriptive['cluster_means']['Nonresponder'], .35, label='Nonresponders (16)', color=blue)
ax.set(xlabel='K-means cluster', ylabel='Mean within-patient share (%)', xticks=x)
ax.legend(frameon=False)
save(fig, 's12-cluster-occupancy')

font_dir = Path(matplotlib.get_data_path()) / 'fonts/ttf'
for name, file in [('DV', 'DejaVuSans.ttf'), ('DV-Bold','DejaVuSans-Bold.ttf')]:
    pdfmetrics.registerFont(TTFont(name, str(font_dir / file)))
pdfmetrics.registerFontFamily('DV', normal='DV', bold='DV-Bold')
styles = {
    'title': ParagraphStyle('title', fontName='DV-Bold', fontSize=25, leading=29, textColor=ink, spaceAfter=12),
    'section': ParagraphStyle('section', fontName='DV-Bold', fontSize=14, leading=18, textColor=ink, spaceBefore=14, spaceAfter=6),
    'small': ParagraphStyle('small', fontName='DV', fontSize=8.9, leading=13, textColor=muted, spaceAfter=8),
    'body': ParagraphStyle('body', fontName='DV', fontSize=10.2, leading=15.7, textColor=ink, spaceAfter=9),
}

def p(text, style='body'):
    return Paragraph(text, styles[style])

def heading(text):
    return p(text, 'section')

def graphic(name, width, height):
    return Image(str(figures / (name + '.png')), width, height)

def table(rows, widths):
    tab = Table(rows, colWidths=widths, hAlign='LEFT', repeatRows=1)
    tab.setStyle(TableStyle([
        ('FONTNAME',(0,0),(-1,0),'DV-Bold'), ('FONTNAME',(0,1),(-1,-1),'DV'),
        ('FONTSIZE',(0,0),(-1,-1),8.6), ('TEXTCOLOR',(0,0),(-1,-1),ink),
        ('BACKGROUND',(0,0),(-1,0),wash), ('LINEBELOW',(0,0),(-1,0),.75,rule),
        ('LINEBELOW',(0,-1),(-1,-1),.75,rule), ('TOPPADDING',(0,0),(-1,-1),7),
        ('BOTTOMPADDING',(0,0),(-1,-1),7), ('ALIGN',(1,1),(-1,-1),'RIGHT'),
    ]))
    tab.spaceAfter = 9
    return tab

def link(url, label):
    return f'<link href="{url}" color="#12796f"><u>{label}</u></link>'

def footer(canvas, doc):
    canvas.setStrokeColor(rule)
    canvas.line(52, 45, 560, 45)
    canvas.setFillColor(muted)
    canvas.setFont('DV', 8)
    canvas.drawString(52, 31, 'DeepTCR correction · 20 November 2026')
    canvas.drawRightString(560, 31, f'{doc.page}')

def write(filename, title, story):
    doc = SimpleDocTemplate(str(root / 'paper' / filename), pagesize=letter, title=title,
                            leftMargin=52, rightMargin=52, topMargin=48, bottomMargin=58)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)

def supplement():
    cohort = [['Recorded response', 'Patients', 'Records, median [IQR]', 'Mean CDR3, aa [IQR]']]
    for group in reversed(groups):
        q1, med, q3 = group['record_quantiles']
        a1, amed, a3 = group['mean_length_quantiles']
        cohort.append([group['response'], str(group['patients']),
                       f'{med:,.0f} [{q1:,.0f}, {q3:,.0f}]', f'{amed:.2f} [{a1:.2f}, {a3:.2f}]'])
    return [
        p('Supplementary material', 'title'),
        p('Patient-held-out evaluation of DeepTCR for CD19 CAR-T response', 'small'),
        heading('Cohort descriptions'),
        p('Summaries use all 239,634 eligible records from 34 patients. Repertoire size is the number of eligible records per patient. CDR3 length was averaged within each patient before summarizing each response group. Points and bars in panels A and B show medians and interquartile ranges.'),
        table(cohort, [143, 67, 147, 139]),
        graphic('s2-cohort-description', 494, 166.2),
        p('<b>Figure S2.</b> (A) Median and interquartile range of repertoire size. (B) Median and interquartile range of per-patient mean CDR3 length. (C) Responders divided by patients in each repertoire-size bin. These are cohort descriptions. Small bin counts limit interpretation.', 'small'),
        heading('Fixed-score shuffle'),
        p(f'With the fitted predictions held fixed, {results["fixed_score_shuffles_at_least_as_high"]:,} of {results["fixed_score_shuffle_draws"]:,} within-fold label shuffles met or exceeded the observed pair concordance. This checks the rank statistic against rearranged labels. Predictions from different folds share training data, which limits the inference from this diagnostic.'),
        PageBreak(),
        heading('Exploratory cluster occupancy'),
        p('The archived exploratory analysis capped its input at 30,000 TCR records across the 34 patients. It standardized amino-acid 3-mer frequencies and one-hot V/J features, projected them onto 128 principal components, and fit K-means with ten clusters. The figure recomputes response-group means from all archived per-patient cluster shares.'),
        graphic('s12-cluster-occupancy', 494, 174),
        p('<b>Figure S12.</b> For each cluster, bars show the mean of each patient’s percentage of sampled TCR records assigned to that cluster. Patients are weighted equally within each response group. The cluster labels are arbitrary. The groups describe occupancy in this cohort.', 'small'),
        heading('Changes to the historical figure set'),
        p('The observed fold results and training losses supersede the formula-generated ROC, simulated training curves, and epoch-based fold summaries. The corrected report omits the attention figures and derived sequence rankings because the saved scores came from a random fallback.'),
        p('Figures S2 and S12 retain their identifiers for comparison with the original submission. S2 has been rebuilt as an aggregate cohort figure. S12 has been relabeled and reaggregated from the archived clustering outputs.'),
    ]

def report():
    rows = [['Fold', 'Train patients', 'Test patients', 'Responders', 'Test records', 'AUROC']]
    for fold in folds:
        rows.append([str(fold['fold']), str(fold['train_patients']), str(fold['test_patients']),
                     str(fold['test_positive_patients']), f"{fold['test_rows']:,}", f"{fold['patient_auroc']:.3f}"])
    return [
        p('Patient-held-out evaluation of DeepTCR for CD19 CAR-T response', 'title'),
        p('Corrected trainee report · 20 November 2026<br/>Original CSIR-IGIB trainee submission by Mateenah Jahan and Sherry Bhalla · 11 January 2026', 'small'),
        heading('Summary'),
        p(f'The full repeat trained the DeepTCR sequence classifier across five patient-held-out folds in the archived CD19 CAR-T cohort. All {users} patients were scored once using {results["retained_rows"]:,} eligible TCRβ records. Within-fold patient AUROC was <b>{auroc:.3f}</b> ({wins:g}/{pairs} concordant-pair equivalents, conditional 95% bootstrap interval {lo:.3f}–{hi:.3f}). The test gave no clear evidence of patient-level discrimination in this cohort.'),
        heading('Why the original estimate was corrected'),
        p('The reported 0.754 AUROC summarized the first 100 logged test-epoch scores. The sequence-level split replay placed patients in both training and test sets. The archived code called the DeepTCR sequence model, and the saved attention values reproduce a seeded random fallback. The earlier score and attention-ranked findings therefore cannot support prediction or biomarker claims.'),
        heading('Patient-separated analysis'),
        p('The study used the archived binary response labels for 18 responders and 16 nonresponders. The package’s 40-residue input limit excluded three of 239,637 sequence records. Five stratified folds kept each patient’s repertoire together. The V/J vocabulary was fitted on the training patients within each fold.'),
        p('Each fold used the official DeepTCR_SS 2.1.29 large sequence CNN with a two-class head, Adam, a batch size of 1,024 and ten epochs. The score for each held-out patient was the mean responder-class score across that patient’s eligible records.'),
        p('The primary AUROC weights each eligible responder–nonresponder pair within a fold equally, with half credit for ties. The interval uses 2,000 patient bootstrap resamples stratified by fold and response. It is conditional on this cohort, the fixed splits and the fitted predictions. It does not quantify variability from retraining.'),
        PageBreak(),
        heading('Held-out results'),
        graphic('held-out-performance', 494, 202.1),
        p(f'<b>Figure 1.</b> (A) Held-out patient AUROCs by fold, the within-fold aggregate with its conditional 95% interval, and the fold-1 shuffled-label control. The dashed vertical line marks AUROC 0.50. (B) Observed mean minibatch training loss over all ten epochs for each fit.', 'small'),
        table(rows, [48, 88, 87, 87, 94, 92]),
        p(f'Pooled out-of-fold AUROC was {pooled:.3f} and average precision was {ap:.3f} (18/34 responders). At a threshold of 0.5, {results["correct_at_threshold_0_5"]}/34 patients were correctly classified. The pooled metrics compare predictions from different fitted models.'),
        p(f'Retraining fold 1 with shuffled training-patient labels scored {control["patient_auroc"]:.3f} versus {folds[0]["patient_auroc"]:.3f} on the same seven held-out patients. The earlier full run yielded within-fold AUROC 0.316 (conditional interval 0.140–0.509). The two runs used the same cohort and split protocol. Their different estimates illustrate run-to-run variability.'),
        p(f'One retrospective cohort and small test folds limit generalization. {link(pin + "/REPORT.md", "Earlier full run and reproduction materials")} · {link(archive + "/paper/main.pdf", "Historical trainee submission")} · {link("https://pmc.ncbi.nlm.nih.gov/articles/PMC7952906/", "DeepTCR framework paper")}', 'small'),
    ]

write('main.pdf', 'Patient-held-out evaluation of DeepTCR for CD19 CAR-T response', report() + [PageBreak()] + supplement())
write('supplement.pdf', 'Supplement to the DeepTCR correction', supplement())
