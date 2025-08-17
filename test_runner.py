#!/usr/bin/env python3
"""
Comprehensive Test Runner for FAO Dashboard

This script orchestrates all testing and validation activities to ensure
code quality and prevent deployment of broken features.
"""

import ast
import importlib
import inspect
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import traceback


class JavaScriptValidator:
    """Validates JavaScript code embedded in Python files."""
    
    def __init__(self):
        self.js_patterns = [
            r'JsCode\s*\(\s*["\']([^"\']+)["\']',  # Single line JsCode
            r'JsCode\s*\(\s*"""([^"]+)"""',        # Multi-line JsCode with """
            r'JsCode\s*\(\s*\'\'\'([^\']+)\'\'\'', # Multi-line JsCode with '''
        ]
    
    def extract_js_code(self, file_content: str) -> List[Tuple[str, int]]:
        """Extract JavaScript code blocks from Python file."""
        js_blocks = []
        
        for pattern in self.js_patterns:
            matches = re.finditer(pattern, file_content, re.MULTILINE | re.DOTALL)
            for match in matches:
                js_code = match.group(1).strip()
                line_num = file_content[:match.start()].count('\n') + 1
                js_blocks.append((js_code, line_num))
        
        return js_blocks
    
    def validate_js_syntax(self, js_code: str) -> List[str]:
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
        
        # Check for missing semicolons in return statements
        if 'return ' in js_code and not js_code.strip().endswith(';') and not js_code.strip().endswith('}'):
            errors.append("Missing semicolon in return statement")
        
        return errors
    
    def validate_file(self, file_path: Path) -> Dict[str, Any]:
        """Validate JavaScript code in a Python file."""
        try:
            content = file_path.read_text()
            js_blocks = self.extract_js_code(content)
            
            issues = []
            for js_code, line_num in js_blocks:
                js_errors = self.validate_js_syntax(js_code)
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


class ImportValidator:
    """Validates that all imports can be resolved."""
    
    def check_imports(self, file_path: Path) -> Dict[str, Any]:
        """Check if all imports in a file can be resolved."""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Parse the AST to find imports
            tree = ast.parse(content)
            imports = []
            issues = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    for alias in node.names:
                        imports.append(f"{module}.{alias.name}" if module else alias.name)
            
            # Test each import
            for imp in imports:
                try:
                    # Extract the base module name
                    base_module = imp.split('.')[0]
                    importlib.import_module(base_module)
                except ImportError as e:
                    issues.append({
                        'import': imp,
                        'error': str(e)
                    })
            
            return {
                'file': str(file_path),
                'imports_found': len(imports),
                'issues': issues,
                'valid': len(issues) == 0
            }
            
        except Exception as e:
            return {
                'file': str(file_path),
                'imports_found': 0,
                'issues': [{'import': 'file_parse', 'error': str(e)}],
                'valid': False
            }


class ComponentTester:
    """Tests individual components and functions."""
    
    def test_function_signatures(self, file_path: Path) -> Dict[str, Any]:
        """Test that functions have valid signatures and can be imported."""
        try:
            # Import the module
            module_name = file_path.stem
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            
            # Add current directory to path temporarily
            sys.path.insert(0, str(file_path.parent))
            
            try:
                spec.loader.exec_module(module)
            except Exception as e:
                return {
                    'file': str(file_path),
                    'functions_found': 0,
                    'issues': [{'function': 'module_load', 'error': str(e)}],
                    'valid': False
                }
            finally:
                sys.path.pop(0)
            
            # Get all functions
            functions = []
            issues = []
            
            for name, obj in inspect.getmembers(module, inspect.isfunction):
                if not name.startswith('_'):  # Skip private functions
                    functions.append(name)
                    
                    try:
                        # Get function signature
                        sig = inspect.signature(obj)
                        # Basic validation - function has parameters and signature is valid
                        str(sig)  # This will raise if signature is invalid
                    except Exception as e:
                        issues.append({
                            'function': name,
                            'error': f"Invalid signature: {e}"
                        })
            
            return {
                'file': str(file_path),
                'functions_found': len(functions),
                'issues': issues,
                'valid': len(issues) == 0
            }
            
        except Exception as e:
            return {
                'file': str(file_path),
                'functions_found': 0,
                'issues': [{'function': 'file_load', 'error': str(e)}],
                'valid': False
            }


class TestRunner:
    """Main test runner that orchestrates all validation activities."""
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path.cwd()
        self.js_validator = JavaScriptValidator()
        self.import_validator = ImportValidator()
        self.component_tester = ComponentTester()
        
        self.results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'project_root': str(self.project_root),
            'pytest_results': {},
            'javascript_validation': {},
            'import_validation': {},
            'component_validation': {},
            'overall_status': 'unknown'
        }
    
    def find_python_files(self) -> List[Path]:
        """Find all Python files in the project."""
        python_files = []
        
        for pattern in ['*.py']:
            python_files.extend(self.project_root.glob(pattern))
        
        # Exclude virtual environment and cache directories
        excluded_dirs = {'venv', '__pycache__', '.git', 'cache', 'automation/logs'}
        
        filtered_files = []
        for file_path in python_files:
            if not any(excluded_dir in file_path.parts for excluded_dir in excluded_dirs):
                filtered_files.append(file_path)
        
        return filtered_files
    
    def run_pytest(self) -> Dict[str, Any]:
        """Run pytest and capture results."""
        print("Running pytest...")
        
        try:
            # Run pytest with JSON output
            result = subprocess.run([
                'python3', '-m', 'pytest', 
                '--tb=short', 
                '-v',
                '--disable-warnings'
            ], capture_output=True, text=True, cwd=self.project_root)
            
            return {
                'exit_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'passed': result.returncode == 0
            }
            
        except Exception as e:
            return {
                'exit_code': -1,
                'stdout': '',
                'stderr': str(e),
                'passed': False
            }
    
    def validate_javascript(self) -> Dict[str, Any]:
        """Validate JavaScript code in all Python files."""
        print("Validating JavaScript code...")
        
        python_files = self.find_python_files()
        results = {}
        total_issues = 0
        
        for file_path in python_files:
            result = self.js_validator.validate_file(file_path)
            if result['issues']:
                results[str(file_path)] = result
                total_issues += len(result['issues'])
        
        return {
            'files_checked': len(python_files),
            'files_with_issues': len(results),
            'total_issues': total_issues,
            'file_results': results,
            'passed': total_issues == 0
        }
    
    def validate_imports(self) -> Dict[str, Any]:
        """Validate imports in all Python files."""
        print("Validating imports...")
        
        python_files = self.find_python_files()
        results = {}
        total_issues = 0
        
        for file_path in python_files:
            result = self.import_validator.check_imports(file_path)
            if result['issues']:
                results[str(file_path)] = result
                total_issues += len(result['issues'])
        
        return {
            'files_checked': len(python_files),
            'files_with_issues': len(results),
            'total_issues': total_issues,
            'file_results': results,
            'passed': total_issues == 0
        }
    
    def validate_components(self) -> Dict[str, Any]:
        """Validate component functions."""
        print("Validating components...")
        
        # Focus on main application files
        main_files = [
            'app.py', 'pivot_builder.py', 'chart_builder.py', 
            'data_pipeline.py', 'kpi_calculator.py'
        ]
        
        results = {}
        total_issues = 0
        
        for filename in main_files:
            file_path = self.project_root / filename
            if file_path.exists():
                result = self.component_tester.test_function_signatures(file_path)
                if result['issues']:
                    results[str(file_path)] = result
                    total_issues += len(result['issues'])
        
        return {
            'files_checked': len([f for f in main_files if (self.project_root / f).exists()]),
            'files_with_issues': len(results),
            'total_issues': total_issues,
            'file_results': results,
            'passed': total_issues == 0
        }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all validation tests."""
        print("🧪 Starting comprehensive validation...\n")
        
        # Run pytest
        self.results['pytest_results'] = self.run_pytest()
        
        # Validate JavaScript
        self.results['javascript_validation'] = self.validate_javascript()
        
        # Validate imports
        self.results['import_validation'] = self.validate_imports()
        
        # Validate components
        self.results['component_validation'] = self.validate_components()
        
        # Determine overall status
        all_passed = all([
            self.results['pytest_results']['passed'],
            self.results['javascript_validation']['passed'],
            self.results['import_validation']['passed'],
            self.results['component_validation']['passed']
        ])
        
        self.results['overall_status'] = 'passed' if all_passed else 'failed'
        
        return self.results
    
    def print_summary(self):
        """Print a human-readable summary of results."""
        print("\n" + "="*80)
        print("COMPREHENSIVE TEST RESULTS")
        print("="*80)
        
        # Overall status
        status_emoji = "✅" if self.results['overall_status'] == 'passed' else "❌"
        print(f"\nOverall Status: {status_emoji} {self.results['overall_status'].upper()}")
        
        # Individual test results
        tests = [
            ("Pytest Tests", self.results['pytest_results']),
            ("JavaScript Validation", self.results['javascript_validation']),
            ("Import Validation", self.results['import_validation']),
            ("Component Validation", self.results['component_validation'])
        ]
        
        print(f"\nIndividual Results:")
        for test_name, result in tests:
            status = "✅ PASSED" if result['passed'] else "❌ FAILED"
            print(f"  {test_name:.<50} {status}")
            
            # Show details for failed tests
            if not result['passed']:
                if 'total_issues' in result:
                    print(f"    Issues found: {result['total_issues']}")
                if 'exit_code' in result and result['exit_code'] != 0:
                    print(f"    Exit code: {result['exit_code']}")
        
        # Detailed error reporting
        if self.results['overall_status'] == 'failed':
            print(f"\n🔍 DETAILED ERROR REPORT:")
            self._print_detailed_errors()
        
        print("="*80)
    
    def _print_detailed_errors(self):
        """Print detailed error information."""
        
        # JavaScript errors
        js_results = self.results['javascript_validation']
        if not js_results['passed']:
            print(f"\n📜 JavaScript Issues ({js_results['total_issues']} total):")
            for file_path, file_result in js_results['file_results'].items():
                print(f"  📁 {file_path}:")
                for issue in file_result['issues']:
                    print(f"    Line {issue['line']}: {', '.join(issue['errors'])}")
                    print(f"    Code: {issue['code']}")
        
        # Import errors
        import_results = self.results['import_validation']
        if not import_results['passed']:
            print(f"\n📦 Import Issues ({import_results['total_issues']} total):")
            for file_path, file_result in import_results['file_results'].items():
                print(f"  📁 {file_path}:")
                for issue in file_result['issues']:
                    print(f"    Import '{issue['import']}': {issue['error']}")
        
        # Component errors
        comp_results = self.results['component_validation']
        if not comp_results['passed']:
            print(f"\n🔧 Component Issues ({comp_results['total_issues']} total):")
            for file_path, file_result in comp_results['file_results'].items():
                print(f"  📁 {file_path}:")
                for issue in file_result['issues']:
                    print(f"    Function '{issue['function']}': {issue['error']}")
        
        # Pytest errors
        pytest_results = self.results['pytest_results']
        if not pytest_results['passed']:
            print(f"\n🧪 Pytest Output:")
            if pytest_results['stderr']:
                print(f"  Errors:\n{pytest_results['stderr']}")
            if pytest_results['stdout']:
                print(f"  Output:\n{pytest_results['stdout']}")
    
    def save_results(self, output_file: Optional[Path] = None):
        """Save results to JSON file."""
        if output_file is None:
            output_file = self.project_root / f"test_results_{int(time.time())}.json"
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n📄 Results saved to: {output_file}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive Test Runner for FAO Dashboard")
    parser.add_argument('--output', '-o', help='Output file for JSON results')
    parser.add_argument('--project-root', '-p', help='Project root directory')
    parser.add_argument('--save-results', '-s', action='store_true', help='Save results to file')
    
    args = parser.parse_args()
    
    # Set up project root
    project_root = Path(args.project_root) if args.project_root else Path.cwd()
    
    # Create and run test runner
    runner = TestRunner(project_root)
    
    try:
        results = runner.run_all_tests()
        runner.print_summary()
        
        # Save results if requested
        if args.save_results or args.output:
            output_file = Path(args.output) if args.output else None
            runner.save_results(output_file)
        
        # Exit with appropriate code
        exit_code = 0 if results['overall_status'] == 'passed' else 1
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        print("\n🛑 Test run interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test runner failed with error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()