#!/usr/bin/env python3
import re
from pathlib import Path

teaching_dir = Path("teaching")
md_files = sorted(list(teaching_dir.glob("**/*.md")))

print(f"Auditing {len(md_files)} markdown files in teaching/ ...\n")

findings = {
    "html_collision": [],
    "arabic_in_math": [],
    "pipe_in_table_math": [],
    "indented_display_math": [],
    "inline_display_math": [],
    "missing_blank_lines_display": [],
    "spaces_inside_delimiters": []
}

for f in md_files:
    content = f.read_text(encoding="utf-8")
    lines = content.splitlines()
    in_code_block = False
    
    for i, line in enumerate(lines):
        line_num = i + 1
        stripped = line.strip()
        
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue
            
        # 1. HTML collision: <letter inside math
        # Look for $...<[a-zA-Z]...$ or $$...<[a-zA-Z]...$$
        for m in re.finditer(r'(?<!\\)\$([^\$]+)(?<!\\)\$', line):
            span_text = m.group(1)
            if re.search(r'<[a-zA-Z]', span_text):
                findings["html_collision"].append((f, line_num, span_text, line))
                
        # 2. Arabic characters inside math
        for m in re.finditer(r'(?<!\\)\$([^\$]+)(?<!\\)\$', line):
            span_text = m.group(1)
            if re.search(r'[\u0600-\u06FF]', span_text):
                findings["arabic_in_math"].append((f, line_num, span_text, line))
                
        # 3. Pipe in table math
        if stripped.startswith("|") and stripped.endswith("|"):
            # Check cells
            cells = stripped.split("|")[1:-1]
            for cell in cells:
                for m in re.finditer(r'(?<!\\)\$([^\$]+)(?<!\\)\$', cell):
                    math_content = m.group(1)
                    # if unescaped pipe inside math
                    if re.search(r'(?<!\\)\|', math_content):
                        findings["pipe_in_table_math"].append((f, line_num, math_content, line))
                        
        # 4. Indented display math
        if line.startswith(" ") and stripped.startswith("$$"):
            findings["indented_display_math"].append((f, line_num, stripped, line))
            
        # 5. Inline display math: e.g. text $$math$$ or $$math$$ text
        if "$$" in line and not stripped.startswith("$$"):
            findings["inline_display_math"].append((f, line_num, stripped, line))
            
        # 6. Display math without blank line before/after
        if stripped.startswith("$$") and not line.startswith(" "):
            # check line before
            if i > 0 and lines[i-1].strip() != "" and not lines[i-1].strip().startswith("$$") and not lines[i-1].strip().startswith("#"):
                findings["missing_blank_lines_display"].append((f, line_num, "No blank line before $$", line))
            # check line after
            if i < len(lines) - 1 and lines[i+1].strip() != "" and not lines[i+1].strip().startswith("$$"):
                findings["missing_blank_lines_display"].append((f, line_num, "No blank line after $$", line))

print("=== SUMMARY OF REAL LATEX RENDERING PITFALLS ===")
for category, items in findings.items():
    print(f"{category:30s}: {len(items)} issues")
    for item in items[:5]:
        rel = str(item[0])
        print(f"   -> {rel}:{item[1]} : {item[2]}")
    if len(items) > 5:
        print(f"      ... and {len(items)-5} more")
    print()
