#!/usr/bin/env python3
"""
Quick health check script for FAO Dashboard.

This provides a fast way to check if the project is in a good state
before commits or deployments.
"""

import subprocess
import sys
import time
from pathlib import Path
from typing import List, Dict, Any


class QuickHealthChecker:
    """Performs quick health checks on the project."""
    
    def __init__(self):
        self.project_root = Path.cwd()
        self.results = []
    
    def check_required_files(self) -> Dict[str, Any]:
        """Check that all required files exist."""
        required_files = [
            'app.py',
            'requirements.txt',
            'pivot_builder.py',
            'data_pipeline.py',
            'chart_builder.py'
        ]
        
        missing_files = []
        for file_name in required_files:
            if not (self.project_root / file_name).exists():
                missing_files.append(file_name)
        
        return {
            'test': 'required_files',
            'passed': len(missing_files) == 0,
            'missing_files': missing_files,
            'total_required': len(required_files)
        }
    
    def check_python_syntax(self) -> Dict[str, Any]:
        """Check Python syntax for main files."""
        main_files = [
            'app.py',
            'pivot_builder.py', 
            'data_pipeline.py',
            'chart_builder.py',
            'kpi_calculator.py'
        ]
        
        syntax_errors = []
        
        for file_name in main_files:
            file_path = self.project_root / file_name
            if file_path.exists():
                try:
                    # Try to compile the file
                    with open(file_path) as f:
                        compile(f.read(), file_path, 'exec')
                except SyntaxError as e:
                    syntax_errors.append({
                        'file': file_name,
                        'line': e.lineno,
                        'error': str(e)
                    })
        
        return {
            'test': 'python_syntax',
            'passed': len(syntax_errors) == 0,
            'files_checked': len(main_files),
            'syntax_errors': syntax_errors
        }
    
    def check_javascript_code(self) -> Dict[str, Any]:
        """Quick JavaScript validation."""
        try:
            result = subprocess.run([
                'python3', 'validate_js.py'
            ], capture_output=True, text=True, timeout=10)
            
            return {
                'test': 'javascript_validation',
                'passed': result.returncode == 0,
                'exit_code': result.returncode,
                'output': result.stdout if result.returncode != 0 else "All JS code valid"
            }
            
        except subprocess.TimeoutExpired:
            return {
                'test': 'javascript_validation',
                'passed': False,
                'error': 'JavaScript validation timed out'
            }
        except FileNotFoundError:
            return {
                'test': 'javascript_validation',
                'passed': False,
                'error': 'validate_js.py not found'
            }
    
    def check_imports(self) -> Dict[str, Any]:
        """Check critical imports work."""
        critical_imports = [
            'streamlit',
            'pandas',
            'plotly',
            'st_aggrid'
        ]
        
        import_failures = []
        
        for module_name in critical_imports:
            try:
                __import__(module_name)
            except ImportError as e:
                import_failures.append({
                    'module': module_name,
                    'error': str(e)
                })
        
        return {
            'test': 'critical_imports',
            'passed': len(import_failures) == 0,
            'modules_checked': len(critical_imports),
            'import_failures': import_failures
        }
    
    def check_pivot_functionality(self) -> Dict[str, Any]:
        """Quick check of pivot functionality."""
        try:
            # Add current directory to Python path
            if str(self.project_root) not in sys.path:
                sys.path.insert(0, str(self.project_root))
            
            from pivot_builder import create_pivot_table, validate_pivot_size
            import pandas as pd
            
            # Create minimal test data
            test_data = {
                'date': pd.date_range('2023-01-01', periods=12, freq='ME'),
                'food_price_index': range(100, 112),
                'meat': range(95, 107)
            }
            df = pd.DataFrame(test_data)
            
            index_mapping = {
                'Food Price Index': 'food_price_index',
                'Meat': 'meat'
            }
            
            # Test pivot creation
            pivot_df = create_pivot_table(df, 'Year', ['Food Price Index'], 'mean', index_mapping)
            
            # Test size validation
            is_valid, _, _ = validate_pivot_size(df, 'Year', ['Food Price Index'], 1000)
            
            return {
                'test': 'pivot_functionality',
                'passed': not pivot_df.empty and is_valid,
                'pivot_rows': len(pivot_df),
                'size_validation': is_valid
            }
            
        except Exception as e:
            return {
                'test': 'pivot_functionality',
                'passed': False,
                'error': str(e)
            }
    
    def run_quick_checks(self) -> Dict[str, Any]:
        """Run all quick health checks."""
        print("⚡ Running quick health checks...\n")
        
        # Run all checks
        checks = [
            self.check_required_files(),
            self.check_python_syntax(),
            self.check_javascript_code(),
            self.check_imports(),
            self.check_pivot_functionality()
        ]
        
        self.results = checks
        
        # Calculate overall status
        passed_checks = sum(1 for check in checks if check['passed'])
        total_checks = len(checks)
        
        return {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_checks': total_checks,
            'passed_checks': passed_checks,
            'overall_passed': passed_checks == total_checks,
            'check_results': checks
        }
    
    def print_summary(self, results: Dict[str, Any]):
        """Print health check summary."""
        print("="*50)
        print("QUICK HEALTH CHECK SUMMARY")
        print("="*50)
        
        overall_status = "✅ HEALTHY" if results['overall_passed'] else "❌ ISSUES FOUND"
        print(f"\nOverall Status: {overall_status} ({results['passed_checks']}/{results['total_checks']})")
        
        print(f"\nCheck Results:")
        for check in results['check_results']:
            test_name = check['test'].replace('_', ' ').title()
            status = "✅" if check['passed'] else "❌"
            print(f"  {status} {test_name}")
            
            # Show additional details for failures
            if not check['passed']:
                if 'error' in check:
                    print(f"    Error: {check['error']}")
                elif 'syntax_errors' in check:
                    for error in check['syntax_errors']:
                        print(f"    {error['file']} line {error['line']}: {error['error']}")
                elif 'missing_files' in check:
                    print(f"    Missing: {', '.join(check['missing_files'])}")
                elif 'import_failures' in check:
                    for failure in check['import_failures']:
                        print(f"    {failure['module']}: {failure['error']}")
        
        print("="*50)
        
        # Show next steps
        if not results['overall_passed']:
            print("\n💡 Recommended Actions:")
            for check in results['check_results']:
                if not check['passed']:
                    if check['test'] == 'required_files':
                        print("  - Check that all required files are present")
                    elif check['test'] == 'python_syntax':
                        print("  - Fix Python syntax errors before proceeding")
                    elif check['test'] == 'javascript_validation':
                        print("  - Run 'python3 validate_js.py' for detailed JS issues")
                    elif check['test'] == 'critical_imports':
                        print("  - Install missing dependencies: pip3 install -r requirements.txt")
                    elif check['test'] == 'pivot_functionality':
                        print("  - Check pivot_builder.py for implementation issues")
            print("")


def main():
    """Main entry point for health checks."""
    checker = QuickHealthChecker()
    
    try:
        results = checker.run_quick_checks()
        checker.print_summary(results)
        
        # Exit with appropriate code
        sys.exit(0 if results['overall_passed'] else 1)
        
    except KeyboardInterrupt:
        print("\n🛑 Health check interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Health check failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()