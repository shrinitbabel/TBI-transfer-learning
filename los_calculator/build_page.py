"""Inject model.json and validation.json into template.html -> page.html (body only) and index.html (standalone)."""
import os, json
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
model = json.load(open(os.path.join(HERE, "model.json"), encoding="utf-8"))
# observed cumulative incidence of live discharge in the pooled training data (complete follow-up,
# so it is the proportion discharged alive by each day) -> reference line on the chart
coh = os.path.join(HERE, "..", "revision2", "outputs", "cohorts")
df = pd.concat([pd.read_csv(os.path.join(coh, f"{n.lower()}_r2.csv")) for n in model["trained_on"]])
model["observed_cif"] = [round(float(((df.discharged == 1) & (df.los <= d)).mean()), 5) for d in model["days"]]
page = open(os.path.join(HERE, "template.html"), encoding="utf-8").read()
page = page.replace("/*MODEL_JSON*/null", json.dumps(model, separators=(",", ":")))
page = page.replace("/*VALIDATION_JSON*/null", open(os.path.join(HERE, "validation.json"), encoding="utf-8").read())

# page.html: page body only (for hosts that add the document skeleton themselves)
open(os.path.join(HERE, "page.html"), "w", encoding="utf-8").write(page)

# index.html: complete standalone document (GitHub Pages, opening the file locally)
head = "\n".join([
    "<!doctype html>",
    '<html lang="en">',
    "<head>",
    '<meta charset="utf-8">',
    '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">',
    "</head>",
    "<body>",
])
open(os.path.join(HERE, "index.html"), "w", encoding="utf-8").write(head + "\n" + page + "\n</body>\n</html>\n")
print("index.html:", os.path.getsize(os.path.join(HERE, "index.html")) // 1024, "KB")
