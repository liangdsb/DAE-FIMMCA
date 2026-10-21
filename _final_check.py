"""全文一致性检查（正确处理 \\% 转义，避免误判）"""
import re

FN = r"E:\My LaTex\Springer_Nature_LaTeX_Template\sn-article.tex"
src = open(FN, encoding="utf-8", errors="ignore").read()

# 去注释：只把未被反斜杠转义的 % 当作注释起始
def strip_comment(line):
    out, i = [], 0
    while i < len(line):
        if line[i] == "\\" and i + 1 < len(line):
            out.append(line[i:i+2]); i += 2; continue
        if line[i] == "%":
            break
        out.append(line[i]); i += 1
    return "".join(out)

nc = "\n".join(strip_comment(l) for l in src.split("\n"))

labs = set(re.findall(r"\\label\{([^}]*)\}", nc))
refs = set(re.findall(r"\\(?:ref|eqref|autoref)\{([^}]*)\}", nc))

print("【引用但未定义】", sorted(refs - labs) or "无")
print("【定义了但未引用】", sorted(labs - refs) or "无")
print()

print("【重复 label】", sorted({l for l in re.findall(r"\\label\{([^}]*)\}", nc)
                              if re.findall(r"\\label\{([^}]*)\}", nc).count(l) > 1}) or "无")
print()

# 术语一致性
print("【术语】")
print("  'loss function' 出现:", len(re.findall(r"loss function", nc)))
print("  'objective'     出现:", len(re.findall(r"objective", nc)))
print("  'density of'    出现:", len(re.findall(r"density of", nc)))
print("  'corruption ratio' 出现:", len(re.findall(r"corruption ratio", nc)))
print()

# 维度符号
print("【第3章残留的 p（应为 d）】")
found = False
for i, ln in enumerate(nc.split("\n"), 1):
    if i >= 490 and re.search(r"\\mathbb\{R\}\^\{p|\\lfloor p/2|dimension \$p\$", ln):
        print(f"  行{i}: {ln.strip()[:90]}"); found = True
if not found:
    print("  无")
