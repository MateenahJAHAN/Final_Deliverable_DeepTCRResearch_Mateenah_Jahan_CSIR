"""Refresh the editable study presentation in its existing layout."""

import argparse
import io
import json
import posixpath
from pathlib import Path
from zipfile import ZipFile

from lxml import etree as ET
from PIL import Image

NS = {
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Study repository root",
    )
    ap.add_argument(
        "--template",
        type=Path,
        help="Optional copy of the eight-slide study presentation",
    )
    args = ap.parse_args()
    r = args.root

    def load(p):
        return json.loads((r / p).read_text())

    repeated = load("validation/repeated-patient-cv/repeat_results.json")
    fixed = load("validation/patient-grouped-deeptcr/aggregate_results.json")
    st = repeated["summary"]
    deep = st["deeptcr_within_fold"]
    base = st["baseline_within_fold"]
    ctr = fixed["shuffled_training_control"]
    original = fixed["folds"][ctr["fold"] - 1]
    low, hi = fixed["within_fold_patient_auroc_bootstrap_ci95"]
    sensitivity = load("validation/source-cohort/composition-sensitivity.json")
    pairs = sensitivity["archived_pair_rows"]["primary_median_within_fold_auroc"]
    paired_beta = sensitivity["paired_cells"]["primary_median_within_fold_auroc"]
    text = {
        1: [
            ["T-cell receptor repertoires", "and CD19 CAR-T response"],
            [f"{deep['median']:.3f}"],
            ["Median within-fold patient AUROC"],
            [
                f"Ten evaluations across 34 patients (range {deep['minimum']:.3f}–{deep['maximum']:.3f})"
            ],
            ["Mateenah Jahan and Sherry Bhalla · CSIR-IGIB"],
        ],
        2: [
            ["Study design"],
            ["Cohort"],
            [
                "34 patients with large B-cell lymphoma",
                "Three-month PET/CT: 18 responders and 16 nonresponders",
            ],
            ["Patients"],
            [
                "Each product stays in one fold.",
                "Five patient partitions and two training seeds give ten evaluations.",
            ],
            ["Model"],
            [
                "DeepTCR scores each eligible beta CDR3 and its V/J genes.",
                "We average the sequence scores within each held-out product.",
            ],
            ["Comparator"],
            [
                "Logistic regression on length and amino-acid composition.",
                "Both models use the same training and test patients.",
            ],
        ],
        3: [
            ["Discrimination in held-out patients"],
            [
                f"Median AUROC: DeepTCR {deep['median']:.3f} and composition {base['median']:.3f}"
            ],
            [
                f"DeepTCR scored higher in {st['deeptcr_above_baseline']} of ten matched evaluations."
            ],
        ],
        4: [
            ["Fixed-partition evaluation"],
            [
                f"{fixed['patient_pairs_concordant_ties_half']:g} of {fixed['patient_pairs']} within-fold pair equivalents were concordant."
            ],
        ],
        5: [
            ["Controls and reproducibility"],
            ["Fixed split"],
            [
                f"Within-fold AUROC {fixed['within_fold_patient_auroc']:.3f} (conditional 95% interval {low:.3f}–{hi:.3f}).",
                f"Pooled out-of-fold AUROC {fixed['pooled_oof_patient_auroc']:.3f}.",
            ],
            ["Shuffle control"],
            [
                f"Recorded training labels: AUROC {original['patient_auroc']:.3f}.",
                f"Shuffled training labels: AUROC {ctr['patient_auroc']:.3f} on the same {ctr['test_patients']} patients.",
            ],
            ["Repeatability"],
            [
                "An independent process repeated one ten-epoch primary fit.",
                "All seven held-out probabilities matched exactly.",
            ],
            ["The conditional interval holds the fitted patient scores fixed."],
        ],
        6: [
            ["Beta-chain length by clinical response"],
            ["Each patient contributes one repertoire to the group summary."],
        ],
        7: [
            ["Exploratory repertoire clusters"],
            [
                "Thirty thousand pair rows, 128 principal components and ten K-means clusters."
            ],
            ["Bars show mean within-patient occupancy for each response group."],
        ],
        8: [
            ["Interpretation"],
            ["Patient scores"],
            [
                f"Median AUROC was {deep['median']:.3f} across ten five-fold evaluations.",
                "The 34 patients recur across partitions.",
            ],
            ["Chain pairing"],
            [
                "Counting each beta contig once in paired cells changed the",
                f"median composition AUROC from {pairs:.3f} to {paired_beta:.3f}.",
            ],
            ["Clinical scope"],
            [
                "This cohort measures internal discrimination. An independent",
                "cohort is needed to evaluate clinical prediction.",
            ],
        ],
    }
    charts = {
        3: r / "figures/correction/repeated-patient-auroc.png",
        4: r / "validation/patient-grouped-deeptcr/deeptcr_patient_run.png",
        6: r / "figures/manuscript/response-length.png",
        7: r / "figures/manuscript/cluster-occupancy.png",
    }
    target = r / "results/DeepTCR_Results_Presentation.pptx"
    changes = {}
    with ZipFile(args.template or target) as src:
        for i, replacements in text.items():
            entry = f"ppt/slides/slide{i}.xml"
            slide = ET.fromstring(src.read(entry))
            shapes = [
                s
                for s in slide.xpath("//p:sp", namespaces=NS)
                if s.xpath(".//a:t/text()", namespaces=NS)
            ]
            if len(shapes) != len(replacements):
                raise ValueError(f"Slide {i}: template text boxes changed")
            for shape, paragraphs in zip(shapes, replacements):
                nodes = shape.xpath("p:txBody/a:p", namespaces=NS)
                if len(nodes) != len(paragraphs):
                    raise ValueError(
                        f"Slide {i}: paragraph count {len(nodes)} != {len(paragraphs)}"
                    )
                for node, value in zip(nodes, paragraphs):
                    runs = node.xpath(".//a:t", namespaces=NS)
                    runs[0].text = value
                    for run in runs[1:]:
                        run.text = ""
                shape.find("p:nvSpPr/p:cNvPr", NS).set("name", paragraphs[0])
            if i in charts:
                (pic,) = slide.xpath("//p:pic", namespaces=NS)
                (blip,) = pic.xpath(".//a:blip", namespaces=NS)
                rels = ET.fromstring(src.read(f"ppt/slides/_rels/slide{i}.xml.rels"))
                rid = blip.get("{" + NS["r"] + "}embed")
                href = next(x.get("Target") for x in rels if x.get("Id") == rid)
                media = (
                    href.lstrip("/")
                    if href.startswith("/")
                    else posixpath.normpath(posixpath.join("ppt/slides", href))
                )
                bounds = pic.find("p:spPr/a:xfrm/a:ext", NS)
                ratio = int(bounds.get("cx")) / int(bounds.get("cy"))
                # Fit the scientific panel in the original image frame without cropping or stretching it.
                with Image.open(charts[i]).convert("RGB") as panel:
                    w = max(panel.width, round(panel.height * ratio))
                    h = round(w / ratio)
                    canvas = Image.new("RGB", (w, h), "white")
                    canvas.paste(
                        panel, ((w - panel.width) // 2, (h - panel.height) // 2)
                    )
                    out = io.BytesIO()
                    canvas.save(out, format="PNG")
                    changes[media] = out.getvalue()
                crop = pic.find("p:blipFill/a:srcRect", NS)
                if crop is not None:
                    crop.attrib.clear()
                pic.find("p:nvPicPr/p:cNvPr", NS).set("descr", replacements[0][0])
            changes[entry] = ET.tostring(
                slide, encoding="UTF-8", xml_declaration=True, standalone=True
            )
        tmp = target.with_suffix(".tmp")
        with ZipFile(tmp, "w") as dest:
            for entry in src.infolist():
                dest.writestr(
                    entry,
                    changes[entry.filename]
                    if entry.filename in changes
                    else src.read(entry),
                )
    tmp.replace(target)


if __name__ == "__main__":
    main()
