"""
Deep audit: find every st.markdown block and count unclosed HTML tags.
"""
import re, os

files = [
    'app/pages/single_prediction.py',
    'app/pages/batch_prediction.py',
    'app/pages/model_analytics.py',
    'app/components/ui_components.py',
]

for filepath in files:
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find every st.markdown(..., unsafe_allow_html=True) block
    # using a simple line-by-line approach
    lines = content.split('\n')
    in_markdown = False
    html_buf = []
    start_line = 0
    paren_depth = 0

    for i, line in enumerate(lines, 1):
        if not in_markdown:
            if 'st.markdown(' in line:
                in_markdown = True
                start_line = i
                html_buf = [line]
                paren_depth = line.count('(') - line.count(')')
                if paren_depth <= 0:
                    in_markdown = False
                    block = '\n'.join(html_buf)
                    check_block(filepath, start_line, block)
                    html_buf = []
        else:
            html_buf.append(line)
            paren_depth += line.count('(') - line.count(')')
            if paren_depth <= 0:
                in_markdown = False
                block = '\n'.join(html_buf)
                # Extract the HTML content
                html_content = ''.join(html_buf)
                # Count div opens vs closes
                opens = html_content.count('<div') + html_content.count('<div\n')
                closes = html_content.count('</div>')
                if opens != closes:
                    print(f"\n{'='*60}")
                    print(f"FILE: {filepath}  LINE: {start_line}")
                    print(f"  OPENS={opens}  CLOSES={closes}  MISMATCH!")
                    # Print first 3 lines
                    for ln in html_buf[:5]:
                        print(f"    {ln}")
                html_buf = []

print("\nAudit complete.")
