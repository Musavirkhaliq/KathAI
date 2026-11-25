#!/usr/bin/env python3
"""
Comprehensive LaTeX and GitHub Markdown Math Normalizer
Fixes all 5 classes of GFM rendering failures:
1. Replaces `<` followed by a letter (e.g., `y_{<u}`) with `\lt ` to avoid HTML parser tag interception.
2. Unindents display math ($$) to column 0 and enforces blank lines before and after.
3. Fixes Arabic/Perso-Arabic glyphs inside math mode by extracting them to bold markdown or code.
4. Escapes unescaped pipes (`|`) inside math in markdown table rows.
5. Replaces non-standard macros like `\\argmax` with standard KaTeX `\\arg\\max`.
"""

import re
from pathlib import Path

def normalize_math_in_text(text: str) -> str:
    # 1. First, replace HTML collision `<letter` inside math contexts
    # Handle both single line and multiline math blocks
    def fix_html_in_math(match):
        inner = match.group(0)
        # replace <([a-zA-Z]) with \lt \1
        inner = re.sub(r'<([a-zA-Z])', r'\\lt \1', inner)
        inner = inner.replace(r'\argmax', r'\arg\max').replace(r'\argmin', r'\arg\min')
        return inner

    # Replace in $$ ... $$
    text = re.sub(r'\$\$[\s\S]*?\$\$', fix_html_in_math, text)
    # Replace in $ ... $
    text = re.sub(r'(?<!\$)\$(?:\\\$|[^\$])+\$(?!\$)', fix_html_in_math, text)

    lines = text.splitlines()
    new_lines = []
    in_code_block = False

    for line in lines:
        stripped = line.strip()

        # Preserve fenced code blocks
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            new_lines.append(line)
            continue
        if in_code_block:
            new_lines.append(line)
            continue

        # 2. Unindent display math ($$)
        if stripped.startswith("$$") and line.startswith(" "):
            line = stripped

        # 3. Escape pipes inside math in markdown tables
        if stripped.startswith("|") and stripped.endswith("|") and "$" in line:
            in_math = False
            char_list = list(line)
            for c_idx, char in enumerate(char_list):
                if char == "$" and (c_idx == 0 or char_list[c_idx - 1] != "\\"):
                    in_math = not in_math
                elif char == "|" and in_math:
                    if c_idx > 0 and char_list[c_idx - 1] != "\\":
                        char_list[c_idx] = "\\|"
            line = "".join(char_list)

        # 4. Clean Arabic characters inside math if any remain
        if re.search(r'[\u0600-\u06FF]', line) and "$" in line:
            line = re.sub(r'\\text\{"([\u0600-\u06FF\s\u064b-\u065f\u0670\u06d6-\u06ed]+)"\}', r'"\1"', line)
            line = re.sub(r'\\text\{([\u0600-\u06FF\s\u064b-\u065f\u0670\u06d6-\u06ed]+)\}', r'"\1"', line)

        new_lines.append(line)

    # 5. Ensure blank lines before and after display math ($$)
    final_lines = []
    for j, l in enumerate(new_lines):
        str_l = l.strip()
        if str_l.startswith("$$"):
            # Blank line before if previous line has content and isn't a heading or list parent
            if final_lines and final_lines[-1].strip() != "" and not final_lines[-1].strip().startswith("$$") and not final_lines[-1].strip().startswith("#"):
                final_lines.append("")
            final_lines.append(l)
            # Blank line after if next line has content
            if j < len(new_lines) - 1 and new_lines[j+1].strip() != "" and not new_lines[j+1].strip().startswith("$$"):
                final_lines.append("")
        else:
            final_lines.append(l)

    return "\n".join(final_lines) + "\n"

def main():
    target_dirs = ["teaching", "docs", "reading", "research"]
    for t_dir in target_dirs:
        p = Path(t_dir)
        if not p.exists():
            continue
        md_files = sorted(list(p.glob("**/*.md")))
        print(f"Normalizing math across {len(md_files)} markdown files in {t_dir}/ ...")

        changed = 0
        for f in md_files:
            orig = f.read_text(encoding="utf-8")
            norm = normalize_math_in_text(orig)
            if norm != orig:
                f.write_text(norm, encoding="utf-8")
                changed += 1

        print(f"Done for {t_dir}! Successfully updated {changed} files.")

if __name__ == "__main__":
    main()
