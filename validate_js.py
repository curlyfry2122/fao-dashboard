#!/usr/bin/env python3
"""Standalone JavaScript validation for Python files."""

import re
from pathlib import Path
from typing import List, Tuple


def extract_js_code(file_content: str) -> List[Tuple[str, int]]:
    """Extract JavaScript code blocks from Python file."""
    js_blocks = []
    
    patterns = [
        r'JsCode\s*\(\s*["\']([^"\']+)["\']',  # Single line JsCode
        r'JsCode\s*\(\s*"""([^"]+)"""',        # Multi-line JsCode with """
        r'JsCode\s*\(\s*\'\'\'([^\']+)\'\'\'', # Multi-line JsCode with '''
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, file_content, re.MULTILINE | re.DOTALL)
        for match in matches:
            js_code = match.group(1).strip()
            line_num = file_content[:match.start()].count('\n') + 1
            js_blocks.append((js_code, line_num))
    
    return js_blocks


def validate_js_syntax(js_code: str) -> List[str]:
    """Basic JavaScript syntax validation."""
    errors = []
    
    # Check for common issues
    if 'function(' not in js_code and '=>' not in js_code and 'function ' not in js_code:
        if any(var in js_code for var in ['x', 'params', 'value', 'data']):
            errors.append("Possible undefined variable - missing function wrapper")
    
    # Check for unclosed braces/brackets
    open_braces = js_code.count('{')
    close_braces = js_code.count('}')
    if open_braces != close_braces:
        errors.append(f"Mismatched braces: {open_braces} open, {close_braces} close")
    
    open_parens = js_code.count('(')
    close_parens = js_code.count(')')
    if open_parens != close_parens:
        errors.append(f"Mismatched parentheses: {open_parens} open, {close_parens} close")
    
    return errors


def validate_file(file_path: Path) -> dict:
    """Validate JavaScript code in a Python file."""
    try:
        content = file_path.read_text()
        js_blocks = extract_js_code(content)
        
        issues = []
        for js_code, line_num in js_blocks:
            js_errors = validate_js_syntax(js_code)
            if js_errors:
                issues.append({
                    'line': line_num,
                    'code': js_code[:100] + '...' if len(js_code) > 100 else js_code,
                    'errors': js_errors
                })
        
        return {
            'file': str(file_path),
            'js_blocks_found': len(js_blocks),
            'issues': issues,
            'valid': len(issues) == 0
        }
        
    except Exception as e:
        return {
            'file': str(file_path),
            'js_blocks_found': 0,
            'issues': [{'line': 0, 'code': '', 'errors': [f"File read error: {e}"]}],
            'valid': False
        }


def main():
    """Validate all Python files in current directory."""
    print("🔍 JavaScript Validation Report")
    print("="*50)
    
    python_files = list(Path('.').glob('*.py'))
    total_issues = 0
    files_with_js = 0
    
    for file_path in python_files:
        result = validate_file(file_path)
        
        if result['js_blocks_found'] > 0:
            files_with_js += 1
            status = "✅" if result['valid'] else "❌"
            print(f"\n{status} {file_path.name}")
            print(f"   JS blocks: {result['js_blocks_found']}")
            
            if result['issues']:
                total_issues += len(result['issues'])
                for issue in result['issues']:
                    print(f"   ⚠️  Line {issue['line']}: {', '.join(issue['errors'])}")
                    print(f"      Code: {issue['code']}")
    
    print(f"\n📊 Summary:")
    print(f"   Files checked: {len(python_files)}")
    print(f"   Files with JS: {files_with_js}")
    print(f"   Total issues: {total_issues}")
    
    if total_issues == 0:
        print("🎉 All JavaScript code is valid!")
        return 0
    else:
        print("❌ JavaScript validation failed!")
        return 1


if __name__ == "__main__":
    exit(main())