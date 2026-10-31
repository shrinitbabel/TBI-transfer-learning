"""Inject model.json and validation.json into template.html -> page.html (body only) and index.html (standalone)."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
read = lambda name: open(os.path.join(HERE, name), encoding="utf-8").read()
page = read("template.html").replace("/*MODEL_JSON*/null", read("model.json")).replace("/*VALIDATION_JSON*/null", read("validation.json"))

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
