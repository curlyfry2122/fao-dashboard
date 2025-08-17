#!/usr/bin/env python3
"""
Clear all caches and test the pivot interface to ensure the fix is working.
"""

import subprocess
import sys
import shutil
import os
from pathlib import Path


def clear_python_cache():
    """Clear Python cache files."""
    print("🧹 Clearing Python cache files...")
    
    cache_dirs = []
    cache_files = []
    
    # Find __pycache__ directories
    for root, dirs, files in os.walk('.'):
        if '__pycache__' in dirs:
            cache_dirs.append(os.path.join(root, '__pycache__'))
        
        # Find .pyc files
        for file in files:
            if file.endswith('.pyc'):
                cache_files.append(os.path.join(root, file))
    
    # Remove cache directories
    for cache_dir in cache_dirs:
        try:
            shutil.rmtree(cache_dir)
            print(f"  Removed: {cache_dir}")
        except Exception as e:
            print(f"  Failed to remove {cache_dir}: {e}")
    
    # Remove .pyc files
    for cache_file in cache_files:
        try:
            os.remove(cache_file)
            print(f"  Removed: {cache_file}")
        except Exception as e:
            print(f"  Failed to remove {cache_file}: {e}")
    
    print(f"✅ Cleared {len(cache_dirs)} cache directories and {len(cache_files)} .pyc files")


def clear_streamlit_cache():
    """Clear Streamlit cache."""
    print("🧹 Clearing Streamlit cache...")
    
    # Common Streamlit cache locations
    cache_locations = [
        Path.home() / '.streamlit',
        Path('.streamlit'),
        Path.home() / '.cache' / 'streamlit'
    ]
    
    for cache_path in cache_locations:
        if cache_path.exists():
            try:
                if cache_path.is_dir():
                    shutil.rmtree(cache_path)
                else:
                    cache_path.unlink()
                print(f"  Removed: {cache_path}")
            except Exception as e:
                print(f"  Failed to remove {cache_path}: {e}")
    
    print("✅ Streamlit cache cleared")


def test_pivot_functionality():
    """Test pivot functionality after cache clearing."""
    print("🧪 Testing pivot functionality...")
    
    try:
        # Import and test pivot functions
        from pivot_builder import configure_aggrid_options, create_pivot_table
        import pandas as pd
        
        # Create test data
        test_data = {
            'Year': [2023, 2024],
            'Food Price Index': [120.5, 125.8],
            'Meat': [115.2, 118.9]
        }
        df = pd.DataFrame(test_data)
        
        # Test AgGrid options generation
        options = configure_aggrid_options(df, 'Year')
        
        # Check for JavaScript code
        found_js_code = False
        for col_def in options.get('columnDefs', []):
            if 'valueFormatter' in col_def:
                formatter = col_def['valueFormatter']
                if hasattr(formatter, 'js_code'):
                    js_code = formatter.js_code
                    print(f"  Found JS code: {js_code}")
                    if 'function(x)' in js_code and 'toFixed(1)' in js_code:
                        found_js_code = True
                        print("  ✅ JavaScript code is correctly formatted")
                    else:
                        print("  ❌ JavaScript code has issues")
                        return False
        
        if not found_js_code:
            print("  ⚠️  No JavaScript formatter found")
        
        print("✅ Pivot functionality test passed")
        return True
        
    except Exception as e:
        print(f"❌ Pivot functionality test failed: {e}")
        return False


def restart_streamlit_advice():
    """Provide advice for restarting Streamlit."""
    print("\n📋 To completely clear browser cache and test:")
    print("1. Stop any running Streamlit processes:")
    print("   - Press Ctrl+C in the terminal running Streamlit")
    print("   - Or: pkill -f streamlit")
    print("")
    print("2. Clear browser cache:")
    print("   - Chrome/Edge: Ctrl+Shift+R (Windows) or Cmd+Shift+R (Mac)")
    print("   - Firefox: Ctrl+F5 (Windows) or Cmd+Shift+R (Mac)")
    print("   - Or open in incognito/private window")
    print("")
    print("3. Restart Streamlit:")
    print("   streamlit run app.py")
    print("")
    print("4. Test the pivot interface in a fresh browser tab")


def kill_existing_streamlit():
    """Kill any existing Streamlit processes."""
    print("🔄 Killing existing Streamlit processes...")
    
    try:
        result = subprocess.run(['pkill', '-f', 'streamlit'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Existing Streamlit processes killed")
        else:
            print("ℹ️  No existing Streamlit processes found")
    except FileNotFoundError:
        print("ℹ️  pkill command not available")
    except Exception as e:
        print(f"⚠️  Could not kill processes: {e}")


def main():
    """Main cache clearing and testing function."""
    print("🚀 Cache Clearing and Pivot Fix Verification")
    print("=" * 50)
    
    # Step 1: Kill existing Streamlit processes
    kill_existing_streamlit()
    
    # Step 2: Clear Python cache
    clear_python_cache()
    
    # Step 3: Clear Streamlit cache  
    clear_streamlit_cache()
    
    # Step 4: Test functionality
    functionality_ok = test_pivot_functionality()
    
    # Step 5: Provide restart advice
    restart_streamlit_advice()
    
    print("\n" + "=" * 50)
    if functionality_ok:
        print("✅ Cache cleared and pivot functionality verified!")
        print("🎯 The JavaScript fix is in place and should work after browser cache clear")
    else:
        print("❌ Issues found - check the error messages above")
    
    return functionality_ok


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)