"""AN2: mechanical traceability sweep of every numeric literal printed in
Sections 3-8 of the manuscript.

For each number found in main.tex it searches every JSON file under
`code/bundle/code/experiments/`, plus the per-figure value files written by the
figure scripts, for a stored value that rounds to the same printed
representation.  A number with no hit is not necessarily wrong (it may be
arithmetic, or a rounded combination), but it is a candidate for manual
checking.

W6.2 changes, all of them widening or sharpening the sweep:

  * the window is no longer two hard-coded line numbers.  The script finds the
    numbered \\section commands itself and audits from the third to the eighth,
    i.e. "The readout floor" through "Discussion".  Line numbers move whenever
    the manuscript is edited; section boundaries do not.
  * literals that are not measurements are dropped before matching instead of
    being reported as misses: citation years inside \\cite keys, cross-reference
    keys, equation and section labels, and LaTeX geometry (column widths,
    lengths, graphics options).
  * thousands separators are normalised, so 35{,}760 is one number and not the
    two numbers 35 and 760.
  * matching is on magnitude, so a value the manuscript prints as a percentage
    drop or an absolute shift still matches a signed field in a JSON file.
  * the audit_numbers outputs themselves are searched (they are data files like
    any other), except this script's own output, which would match everything.
  * misses are classified and counted in an `_summary` record at the head of
    the report, so the state of the audit is one field and not a manual count.

Usage: python an2_traceability.py [--sections 3-8]
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.normpath(os.path.join(HERE, "..", "..", "..", "..", "..",
                                    "article", "main.tex"))
EXP = os.path.normpath(os.path.join(HERE, ".."))
FIGVALS = os.path.normpath(os.path.join(HERE, "..", "..", "..", "..", "..",
                                        "article", "figures"))
SELF = "an2_traceability.json"

FIRST_SEC, LAST_SEC = 3, 8
for i, a in enumerate(sys.argv):
    if a == "--sections" and i + 1 < len(sys.argv):
        FIRST_SEC, LAST_SEC = (int(x) for x in sys.argv[i + 1].split("-"))

# ---------------------------------------------------------------- gather
vals = []


def collect(path):
    try:
        d = json.load(open(path, encoding="utf-8"))
    except Exception:
        return

    def walk(o, p):
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, p + "/" + str(k))
        elif isinstance(o, list):
            for i, v in enumerate(o):
                walk(v, p + "[%d]" % i)
        elif isinstance(o, (int, float)) and not isinstance(o, bool):
            vals.append((float(o), path, p))
    walk(d, "")


for root, _, files in os.walk(EXP):
    for f in files:
        if f.endswith(".json") and f != SELF:
            collect(os.path.join(root, f))
if os.path.isdir(FIGVALS):
    for f in sorted(os.listdir(FIGVALS)):
        if f.endswith("_values.json"):
            collect(os.path.join(FIGVALS, f))

print("scanned %d numeric leaves" % len(vals), file=sys.stderr)

# ---------------------------------------------------------------- parse tex
if not os.path.exists(ART):
    # The bundle is designed to stand on its own once it is zipped, and the
    # manuscript source is not part of it.  Every other script here works
    # without main.tex; this one cannot, so it says so and stops rather than
    # failing the build.
    print("main.tex not found at %s -- this sweep needs the manuscript "
          "source and is skipped." % ART)
    sys.exit(0)
lines = open(ART, encoding="utf-8").read().splitlines()

sec_lines = [i for i, s in enumerate(lines)
             if re.match(r"\s*\\section\{", s)]
if len(sec_lines) < LAST_SEC:
    sys.exit("main.tex has %d numbered sections, need %d"
             % (len(sec_lines), LAST_SEC))
LO = sec_lines[FIRST_SEC - 1] + 1
# The last audited section runs to the next numbered section, or, if it is the
# last one, to the first unnumbered closing block (\section*{...}).
if LAST_SEC < len(sec_lines):
    HI = sec_lines[LAST_SEC]
else:
    star = [i for i, s in enumerate(lines) if re.match(r"\s*\\section\*\{", s)
            and i > sec_lines[-1]]
    HI = star[0] if star else len(lines)
print("auditing sections %d-%d, lines %d-%d" % (FIRST_SEC, LAST_SEC, LO, HI),
      file=sys.stderr)

# Constructs whose digits are never measurements.
DROP = [
    re.compile(r"\\(?:cite[a-z]*|ref|eqref|label|autoref|nameref)\s*\{[^}]*\}"),
    re.compile(r"\\includegraphics\s*(?:\[[^\]]*\])?\s*\{[^}]*\}"),
    re.compile(r"\\(?:setlength|addtolength|settowidth)\s*\{[^}]*\}\s*\{[^}]*\}"),
    re.compile(r"\\(?:hspace|vspace|hskip|vskip|arraystretch|tabcolsep)\*?\s*"
               r"\{?[^}\s]*\}?"),
    re.compile(r"[pmb]\{[\d.]+\s*(?:cm|mm|pt|in|em|ex|\\[a-z]+)\}"),
    re.compile(r"\d*\.?\d+\s*(?:cm|mm|pt|in|em|ex)\b"),
    re.compile(r"\\begin\{[^}]*\}(?:\[[^\]]*\])?(?:\{[^}]*\})?"),
    re.compile(r"\\end\{[^}]*\}"),
]

num_re = re.compile(
    r"(\d[\d,]*(?:\{,\}\d{3})*\.?\d*)\s*"
    r"(?:\\times\s*10\^\{?(-?\d+)\}?)?"
    r"(\s*(?:\\[,;: ])?\\?%)?")

found = []
for ln in range(LO - 1, HI):
    s = lines[ln]
    if s.lstrip().startswith("%"):
        continue
    s = s.split("  %")[0]
    for rx in DROP:
        s = rx.sub(" ", s)
    for m in num_re.finditer(s):
        raw = m.group(1).replace("{,}", "").replace(",", "")
        if raw in ("", "."):
            continue
        try:
            x = float(raw)
        except ValueError:
            continue
        if m.group(2) is not None:
            x *= 10.0 ** int(m.group(2))
        digits = len(raw.replace(".", "").lstrip("0")) or 1
        pct = m.group(3) is not None
        found.append((ln + 1, m.group(0).strip(), x, digits, pct))


try:
    import numpy as _np
    _ARR = _np.array([abs(v) for v, _, _ in vals])
except Exception:                                 # numpy is optional here
    _np = None
    _ARR = None


def matches(x, digits, pct):
    """The three stored values closest to the printed one, in relative terms.

    Ranking by closeness rather than by scan order matters: at two or three
    significant digits a value like 0.51 has hundreds of coincidental
    neighbours in 16000 numeric leaves, and the first one os.walk happens to
    reach says nothing.  The nearest one usually is the source, and the
    distance printed beside it says how much to believe that.
    """
    if x == 0:
        return []
    tol = 0.5 * 10.0 ** (-(digits - 1))          # relative half-ulp
    # A percentage in the prose may be stored either as the percentage or as
    # the ratio it stands for; both readings are accepted.
    targets = [x] + ([x / 100.0] if pct else [])
    best = {}
    for t in targets:
        at = abs(t)
        if _ARR is not None:
            idx = _np.nonzero(_np.abs(_ARR - at) <= tol * at * 1.0000001)[0]
            cand = [(abs(_ARR[i] - at) / at, int(i)) for i in idx]
        else:
            cand = [(abs(abs(v) - at) / at, i) for i, (v, _, _) in enumerate(vals)
                    if v != 0 and abs(abs(v) - at) <= tol * at * 1.0000001]
        for d, i in cand:
            if i not in best or d < best[i]:
                best[i] = d
    out = []
    for i, d in sorted(best.items(), key=lambda kv: kv[1])[:3]:
        v, path, key = vals[i]
        out.append((v, os.path.relpath(path, EXP), key, d))
    return out


report = []
for ln, txt, x, digits, pct in found:
    if digits <= 1 and abs(x) < 100:
        continue                                  # skip 2, 3, 6 ...
    hits = matches(x, digits, pct)
    report.append({"line": ln, "text": txt, "value": x,
                   "sig": digits, "n_hits": len(hits),
                   "hits": [{"v": h[0], "file": h[1], "key": h[2],
                             "rel_dist": h[3]}
                            for h in hits]})

# ---------------------------------------------------------------- classify
YEAR = re.compile(r"^(1[89]|20)\d\d$")


def classify(r):
    if YEAR.match(r["text"].rstrip(",").rstrip(".")) and 1800 <= r["value"] <= 2100:
        return "publication year"
    return "UNTRACED"


miss = [r for r in report if r["n_hits"] == 0]
for r in miss:
    r["miss_class"] = classify(r)
untraced = [r for r in miss if r["miss_class"] == "UNTRACED"]

summary = {
    "_summary": {
        "sections_audited": "%d-%d" % (FIRST_SEC, LAST_SEC),
        "tex_lines": [LO, HI],
        "json_leaves_searched": len(vals),
        "literals_tested": len(report),
        "literals_matched": len(report) - len(miss),
        "misses_total": len(miss),
        "misses_publication_years": len(miss) - len(untraced),
        "misses_untraced": len(untraced),
        "untraced": [{"line": r["line"], "text": r["text"], "value": r["value"]}
                     for r in untraced],
    }
}

print("\n==== numbers with NO json match (%d of %d) ====" % (len(miss),
                                                            len(report)))
for r in miss:
    print("L%-5d %-28s %-14g %s" % (r["line"], r["text"], r["value"],
                                    r["miss_class"]))
print("\nmatched %d / %d; %d publication years; %d UNTRACED"
      % (len(report) - len(miss), len(report),
         len(miss) - len(untraced), len(untraced)))

json.dump([summary] + report,
          open(os.path.join(HERE, SELF), "w", encoding="utf-8"), indent=1)
