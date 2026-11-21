# -*- coding: utf-8 -*-
"""Assemble the PLOS Computational Biology submission manuscript.

Source: paper/PAPER_EN_final.md (canonical EN, numbered [N] citations - PLOS uses numbered refs).
Inputs: paper/PAPER_EN_final.md, paper/submission/plos/ABSTRACT-PLOSCB.md,
        paper/submission/plos/AUTHOR-SUMMARY-PLOSCB.md, paper/submission/DATA-AVAILABILITY.md,
        audit/SHA256SUMS.txt (manifest count for the data statement).
Outputs: paper/submission/plos/MANUSCRIPT-PLOSCB.md and S1_TEXT..S9_TEXT.md.

Section mapping (PLOS CB order):
  Title page / Abstract / Author Summary
  Introduction(S1) -> Related work(S2) -> The model(S3) -> Results(S4)
  Discussion(S5 + S6 as subsection) -> Materials and methods(S8) -> Data availability
  -> References (strip tier tags) -> Figure legends (5, expanded) -> Supporting information list.
  SI: S7 supercomputing protocol -> S1 Text; Appendices A-G -> S2-S8 Text;
      S9(+9.10/9.11) revision record -> S9 Text.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EN = ROOT / "paper" / "PAPER_EN_final.md"
OUT = ROOT / "paper" / "submission" / "plos"

TITLE = (
    "A log-N collapse law in attractor memory: a reproducible, dual-readout "
    "hippocampal-cortical complementary memory model")
RUNNING_HEAD = "The log-N collapse law in attractor memory"
AUTHOR_LINE = "Haowei Wang^1^*"
AFFILIATION = "^1^Undergraduate Student, Shandong University of Finance and Economics, Jinan, China"
CORRESP = "* Corresponding author: Haowei Wang, 202529640232@stu.sdufe.edu.cn (ORCID: 0009-0004-2241-5511)"
CONTRIB = ("H. Wang: Conceptualization, Supervision, Project administration, Software, "
           "Investigation, Data curation, Formal analysis, Writing - original draft "
           "preparation, Writing - review & editing.")
COMPETING = "The authors have declared that no competing interests exist."
ETHICS = ("Ethics statement. The real-corpus study (Section 4.13) is a retrospective, purely observational analysis of the corresponding author's own exported instant-messaging records, including messages sent or received by the author within his own account. No experimental intervention or manipulation was performed on any participant, and no health-related or special-category data were analysed. The study proceeded without prospective ethical-review-board approval or individual participant consent for three reasons: the analysis is retrospective and purely observational of records the author is entitled to export and hold; re-contacting every past participant of group conversations years later to obtain individual consent is not feasible; and only aggregate-level statistics derived from pseudonymised text are reported. Speaker identities are pseudonymised (SHA-256) before any processing; message content is processed only on compute infrastructure under the corresponding author's control and is never redistributed or shared with third parties. All conclusions that do not require the restricted corpus are additionally validated on the public DailyDialog corpus (CC BY 4.0, Section 4.20).")
FUNDING = "The author received no specific funding for this work."

LEGENDS = {
    1: (
        "**Figure 1. The nine operators of HCM and their data flow.** Input/context "
        "space X streams observations to the encoder \u03a6, which maintains a slow "
        "expectation bank \u03bc and encodes the expectation residual through expansion and "
        "sparsification. The episodic fast store E writes one-shot and retrieves via a "
        "modern Hopfield attractor; the semantic slow store S is a low-rank feedforward "
        "map updated by cascade plasticity; W is a working-memory buffer. Consolidation "
        "\u03a8 replays from E into S during quiet periods; retrieval control \u03a0 schedules "
        "recall; forgetting \u0393 prunes and suppresses traces; neuromodulatory gating G "
        "(novelty, reward prediction error, error, abstention) decides writes. The "
        "operator-to-biology anchor table is given in S2 Text (Appendix A)."
    ),
    2: (
        "**Figure 2. Three-way relation between overlap, residual, and recall.** With "
        "the expectation-bank residual disabled, inter-item code cosine rises with input "
        "overlap toward saturation and exact recall at 90% overlap is 1/30; enabling the "
        "residual lowers inter-item code cosine from 0.872 to 0.308 at the same overlap "
        "and raises exact recall to 25/30 (Section 4.2)."
    ),
    3: (
        "**Figure 3. Recall as a function of library size for two cue types.** Exact "
        "cues remain at 20/20 up to 16,000 stored items (d_h=1024, k=20), while 25%-"
        "dropout cues degrade earlier: the capacity bottleneck is set by denoising "
        "rather than by item count (Section 4.3)."
    ),
    4: (
        "**Figure 4. Interleaved replay and continual learning.** Old-task accuracy "
        "over training cycles (20 seeds, mean +/- sd): without replay accuracy collapses "
        "to 0.0/24; interleaved replay retains 16.0/24; a prioritised variant (recency x "
        "surprise, sampled from that distribution) scores 15.3/24 and does not beat "
        "uniform interleaving (Section 4.4)."
    ),
    5: (
        "**Figure 5. Working-memory capacity knee and the risk-coverage curve.** Left: "
        "recall as a function of working-memory buffer length, showing the capacity "
        "knee. Right: risk-coverage obtained from the abstention gate driven by "
        "attractor-reconstruction match; d-prime = 7.12 separates in-store from novel "
        "cues, converting a 0.500 error rate at full coverage into 0.000 at 0.50 "
        "coverage (Sections 4.5-4.6)."
    ),
}


def read_lines(path):
    text = path.read_text(encoding="utf-8")
    return [l.rstrip("\r") for l in text.split("\n")]


def split_sections(lines):
    secs, order, cur = {}, [], None
    for l in lines:
        if l.startswith("## "):
            cur = l[3:].strip()
            secs[cur] = []
            order.append(cur)
        elif cur is not None:
            secs[cur].append(l)
    return secs, order


def get_section(secs, order, prefix):
    for t in order:
        if t.startswith(prefix):
            return "\n".join(secs[t]).strip("\n")
    raise KeyError(prefix)


def strip_figures(block):
    out_lines = []
    for l in block.split("\n"):
        if l.startswith("![Figure"):
            continue
        if re.match(r"^\*Figure \d\.", l):
            continue
        out_lines.append(l)
    return "\n".join(out_lines)


SI_MAP_NOTE = "rewrite body pointers to Supporting-information labels"


def remap_si(text):
    # Appendix letters -> S2..S8 Text; supercomputing protocol -> S1 Text; revision record -> S9 Text
    text = re.sub(r"Appendices ([A-G])/([A-G])", lambda m: "S" + str(ord(m.group(1)) - 64 + 1) + "/S" + str(ord(m.group(2)) - 64 + 1) + " Text", text)
    text = re.sub(r"Appendix ([A-G])\b", lambda m: "S" + str(ord(m.group(1)) - 64 + 1) + " Text", text)
    text = re.sub(r"\u00a77\b", "S1 Text", text)
    text = re.sub(r"Section 7\b", "S1 Text", text)
    text = re.sub(r"§9\.(\d)", "S9 Text §9.\\1", text)
    text = re.sub(r"§9(?!\.\d)", "S9 Text", text)
    return text

def main():
    lines = read_lines(EN)
    secs, order = split_sections(lines)

    intro = remap_si(get_section(secs, order, "1 Introduction"))
    related = remap_si(get_section(secs, order, "2 Related Work"))
    model = remap_si(get_section(secs, order, "3 The Model"))
    results = remap_si(get_section(secs, order, "4 Experiments"))
    disc = remap_si(get_section(secs, order, "5 Discussion"))
    limits = remap_si(get_section(secs, order, "6 Limitations"))
    methods = remap_si(get_section(secs, order, "8 Methods"))
    si_protocol = get_section(secs, order, "7 Supercomputing")
    rev_parts = ["9 Revision record", "9.10 Fourth-round", "9.11 Fifth-round", "9.12 Neighbouring-literature", "9.13 External-review response"]
    revision = "\n\n".join(get_section(secs, order, p) for p in rev_parts)
    refs = get_section(secs, order, "References")

    appendices = {}
    for t in order:
        m = re.match(r"Appendix ([A-G]):(.*)", t)
        if m:
            appendices[m.group(1)] = (t, "\n".join(secs[t]).strip("\n"))
    if len(appendices) != 7:
        sys.exit("FATAL: expected 7 appendices, got " + str(sorted(appendices)))

    kw_line = next((l for l in lines if l.startswith("**Keywords.**")), "")
    keywords = kw_line.replace("**Keywords.**", "").strip()

    abs_text = (OUT / "ABSTRACT-PLOSCB.md").read_text(encoding="utf-8").split("\n", 1)[1].strip()
    asum_text = (OUT / "AUTHOR-SUMMARY-PLOSCB.md").read_text(encoding="utf-8").split("\n", 1)[1].strip()

    da = (ROOT / "paper" / "submission" / "DATA-AVAILABILITY.md").read_text(encoding="utf-8")
    da_lines = [l for l in da.split("\n") if not l.startswith("# ")]
    da = "\n".join(da_lines).strip("\n")
    da = da.replace("\u3010\u6295\u7a3f\u65f6\u751f\u6210\u3011", "[to be assigned at submission]")
    da = da.replace("\u3010GitHub URL\u3011", "[GitHub repository URL]")
    manifest = ROOT / "audit" / "SHA256SUMS.txt"
    if manifest.exists():
        n_files = len(manifest.read_text(encoding="utf-8").strip().splitlines())
        da = re.sub(r"SHA256SUMS\.txt`, \d+ files", "SHA256SUMS.txt`, " + str(n_files) + " files", da)

    refs = re.sub(r"\s*\[T[123]\]", "", refs)

    results = strip_figures(results)

    legends = "\n\n".join(LEGENDS[i] for i in sorted(LEGENDS))

    si_list = (
        "S1 Text. Supercomputing validation protocol (L0-L3).\n"
        "S2 Text. Operator-to-biology mapping (Appendix A).\n"
        "S3 Text. Complete results-file index (Appendix B).\n"
        "S4 Text. Citation-verification report (Appendix C).\n"
        "S5 Text. Engineering feasibility - sparse storage and inverted index (Appendix D).\n"
        "S6 Text. Analytic false-alarm bound for k-sparse codes (Appendix E).\n"
        "S7 Text. External replication - the collapse law does not depend on the HCM code (Appendix F).\n"
        "S8 Text. Module evidence base (Appendix G).\n"
        "S9 Text. Revision record and review response (internal provenance; not part of the review narrative)."
    )

    parts = []
    parts.append("# " + TITLE)
    parts.append("**Running head:** " + RUNNING_HEAD)
    parts.append(AUTHOR_LINE + "\n" + AFFILIATION + "\n" + CORRESP)
    parts.append("**Author contributions (CRediT):** " + CONTRIB)
    parts.append("**Funding:** " + FUNDING)
    parts.append("**Competing interests:** " + COMPETING)
    parts.append("**Ethics statement:** " + ETHICS[len("Ethics statement. "):])
    parts.append("---")
    parts.append("**Abstract.** " + abs_text)
    parts.append("**Keywords.** " + keywords)
    parts.append("**Author Summary.** " + asum_text)
    parts.append("---")
    parts.append("## Introduction\n\n" + intro)
    parts.append("## Related work\n\n" + related)
    parts.append("## The model\n\n" + model)
    parts.append("## Results\n\n" + results)
    parts.append("## Discussion\n\n" + disc + "\n\n### Limitations and honesty statement\n\n" + limits)
    parts.append("## Materials and methods\n\n" + methods)
    parts.append("## Data availability\n\n" + da)
    parts.append("## References\n\n" + refs)
    parts.append("## Figure legends\n\n" + legends)
    parts.append("## Supporting information\n\n" + si_list)

    manuscript = "\n\n".join(parts) + "\n"
    out_path = OUT / "MANUSCRIPT-PLOSCB.md"
    out_path.write_text(manuscript, encoding="utf-8", newline="\n")

    si_items = [
        ("S1_TEXT", "S1 Text. Supercomputing validation protocol (L0-L3)", si_protocol),
        ("S9_TEXT", "S9 Text. Revision record and review response", revision),
    ]
    letters = "ABCDEFG"
    for idx, letter in enumerate(letters, start=2):
        title, body = appendices[letter]
        si_items.append(("S" + str(idx) + "_TEXT", "S" + str(idx) + " Text. " + title, body))
    for stem, head, body in si_items:
        (OUT / (stem + ".md")).write_text("## " + head + "\n\n" + body + "\n", encoding="utf-8", newline="\n")

    body_cite = set()
    pre = manuscript.split("## References")[0]
    for m in re.finditer(r"\[(\d+)\]", pre):
        body_cite.add(int(m.group(1)))
    ref_ids = set(int(m.group(1)) for m in re.finditer(r"^\[(\d+)\]", refs, re.M))
    missing_refs = sorted(body_cite - ref_ids)
    unused_refs = sorted(ref_ids - body_cite)
    cjk = [ch for ch in manuscript if "\u4e00" <= ch <= "\u9fff"]
    main_words = len(re.sub(r"\s+", " ", pre.replace("#", "")).split())
    print("manuscript:", len(manuscript), "chars,", manuscript.count("\n") + 1, "lines")
    print("abstract words:", len(abs_text.split()), ", author summary words:", len(asum_text.split()))
    print("main-text words (title..data availability):", main_words)
    print("body citations:", len(body_cite), "unique [N]; refs entries:", len(ref_ids))
    print("missing refs (cited, no entry):", missing_refs)
    print("unused refs (entry, never cited):", unused_refs)
    print("CJK chars in manuscript:", len(cjk), ("".join(cjk[:20]) if cjk else ""))
    print("figure legends:", len(LEGENDS), "; SI files:", len(si_items))
    post = manuscript.split("## Figure legends")[0]
    print("figure callouts in body:", ", ".join("F" + str(i) + "=" + str(len(re.findall(r"Figure " + str(i) + r"[^\.:]", post))) for i in range(1, 6)))


if __name__ == "__main__":
    main()