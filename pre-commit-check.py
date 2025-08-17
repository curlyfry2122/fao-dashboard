#!/usr/bin/env python3
"""
Pre-commit check script for FAO Dashboard.

This script should be run before committing code to catch issues early.
Can be installed as a git hook or run manually.
"""

import subprocess
import sys
import time
from pathlib import Path


def run_command(command: list, description: str, timeout: int = 30) -> bool:
    """Run a command and return success status."""
    print(f"🔍 {description}...")
    
    try:
        result = subprocess.run(
            command, 
            capture_output=True, 
            text=True, 
            timeout=timeout
        )
        
        if result.returncode == 0:
            print(f"✅ {description} passed")
            return True
        else:
            print(f"❌ {description} failed")
            if result.stdout:
                print(f"Output: {result.stdout}")
            if result.stderr:
                print(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} timed out")
        return False
    except FileNotFoundError:
        print(f"🚫 {description} - command not found")
        return False


def check_modified_files() -> list:
    """Get list of modified Python files."""
    try:
        result = subprocess.run([
            'git', 'diff', '--cached', '--name-only', '--diff-filter=ACM'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            files = result.stdout.strip().split('\n')
            python_files = [f for f in files if f.endswith('.py') and f]
            return python_files
        else:
            # Fallback to all Python files if git command fails
            return [str(p) for p in Path('.').glob('*.py')]
            
    except FileNotFoundError:
        # Git not available, check all Python files
        return [str(p) for p in Path('.').glob('*.py')]


def main():
    """Run pre-commit checks."""
    print("🚀 FAO Dashboard Pre-Commit Checks")
    print("="*50)
    
    start_time = time.time()
    failed_checks = []
    
    # Check 1: Python syntax for modified files
    modified_files = check_modified_files()
    if modified_files:
        print(f"\n📝 Modified Python files: {', '.join(modified_files)}")
        
        for file_path in modified_files:
            if Path(file_path).exists():
                success = run_command([
                    'python3', '-m', 'py_compile', file_path
                ], f"Python syntax check for {file_path}", 10)
                
                if not success:
                    failed_checks.append(f"Syntax error in {file_path}")
    else:
        print("\n📝 No Python files modified")
    
    # Check 2: JavaScript validation
    if Path('validate_js.py').exists():
        success = run_command([
            'python3', 'validate_js.py'
        ], "JavaScript validation", 15)
        
        if not success:
            failed_checks.append("JavaScript validation failed")
    
    # Check 3: Quick health check
    if Path('check_health.py').exists():
        success = run_command([
            'python3', 'check_health.py'
        ], "Quick health check", 20)
        
        if not success:
            failed_checks.append("Health check failed")
    
    # Check 4: Critical tests (fast subset)
    success = run_command([
        'python3', '-m', 'pytest', 
        'test_pivot_interface.py', 
        '--tb=line', '-q'
    ], "Critical pivot tests", 25)
    
    if not success:
        failed_checks.append("Critical tests failed")
    
    # Summary
    elapsed_time = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"⏱️  Checks completed in {elapsed_time:.1f} seconds")
    
    if failed_checks:
        print(f"❌ {len(failed_checks)} check(s) failed:")
        for check in failed_checks:
            print(f"   • {check}")
        print(f"\n🚫 Commit blocked - please fix issues above")
        print(f"💡 Run individual scripts for more details:")
        print(f"   python3 validate_js.py")
        print(f"   python3 check_health.py")
        print(f"   python3 -m pytest test_pivot_interface.py -v")
        return 1
    else:
        print(f"✅ All checks passed - ready to commit!")
        return 0


if __name__ == "__main__":
    sys.exit(main())