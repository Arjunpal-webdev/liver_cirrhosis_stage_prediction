"""
Scan every st.markdown block in all Python files for unclosed/orphaned HTML divs.
"""
import re, os, glob

def audit_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    issues = []
    
    # Find all st.markdown(...) calls - handle multi-line
    # Strategy: find 'st.markdown(' then collect until balanced parens
    i = 0
    while i < len(content):
        idx = content.find('st.markdown(', i)
        if idx == -1:
            break
        
        # Collect the full call by tracking paren depth
        start = idx
        pos = idx + len('st.markdown(')
        depth = 1
        in_str = False
        str_char = None
        
        while pos < len(content) and depth > 0:
            ch = content[pos]
            if in_str:
                if ch == str_char and content[pos-1] != '\\':
                    in_str = False
            else:
                if ch in ('"', "'"):
                    in_str = True
                    str_char = ch
                elif ch == '(':
                    depth += 1
                elif ch == ')':
                    depth -= 1
            pos += 1
        
        block = content[start:pos]
        line_no = content[:start].count('\n') + 1
        
        # Count div opens and closes in this block
        opens  = len(re.findall(r'<div[\s>]', block))
        closes = len(re.findall(r'</div>', block))
        
        if opens != closes:
            issues.append((line_no, opens, closes, block[:200].replace('\n',' ')))
        
        i = pos
    
    return issues


files = glob.glob('app/**/*.py', recursive=True) + glob.glob('services/*.py') + ['main.py']

any_issues = False
for f in sorted(files):
    issues = audit_file(f)
    if issues:
        any_issues = True
        print(f"\n{'='*60}")
        print(f"FILE: {f}")
        for line_no, opens, closes, snippet in issues:
            print(f"  Line {line_no}: OPENS={opens} CLOSES={closes}")
            print(f"  Snippet: {snippet[:150]}...")

if not any_issues:
    print("ALL OK - No HTML div mismatches found in any file.")
