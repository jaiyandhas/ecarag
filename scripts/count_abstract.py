import re

with open("ECA-RAG_final.tex") as f:
    text = f.read()

start_tag = r"\begin{abstract}"
end_tag = r"\end{abstract}"
start = text.find(start_tag) + len(start_tag)
end = text.find(end_tag)
abstract = text[start:end].strip()

# Stripping LaTeX markup
clean = abstract.replace(r"\Delta", "Delta").replace(r"\%", "%")
clean = re.sub(r"\$([^\$]+)\$", r"\1", clean)
clean = re.sub(r"\\[a-zA-Z]+", "", clean)
clean = clean.replace("{", "").replace("}", "")
words = clean.split()

print(f"Word count: {len(words)}")
print("\nFinal abstract (LaTeX stripped):")
print(" ".join(words))
