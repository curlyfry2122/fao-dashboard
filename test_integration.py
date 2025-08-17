#!/usr/bin/env python3
"""
Integration tests for FAO Dashboard components.

This module provides end-to-end testing of the dashboard to ensure
all components work together correctly and catch integration issues.
"""

import subprocess
import sys
import time
import requests
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
import tempfile
import os


class StreamlitAppTester:
    """Test Streamlit app integration."""
    
    def __init__(self, app_file: str = "app.py", port: int = 8506):
        self.app_file = app_file
        self.port = port
        self.process = None
        
    def start_app(self, timeout: int = 15) -> bool:
        """Start the Streamlit app."""
        try:
            self.process = subprocess.Popen([
                'streamlit', 'run', self.app_file,
                '--server.headless', 'true',
                '--server.port', str(self.port),
                '--logger.level', 'error'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Wait for app to start
            for _ in range(timeout):
                try:
                    response = requests.get(f'http://localhost:{self.port}', timeout=2)
                    if response.status_code == 200:
                        return True
                except requests.RequestException:
                    pass
                time.sleep(1)
            
            return False
            
        except Exception as e:
            print(f"Failed to start app: {e}")
            return False
    
    def stop_app(self):
        """Stop the Streamlit app."""
        if self.process:
            self.process.terminate()
            self.process.wait(timeout=5)
            self.process = None
    
    def test_app_startup(self) -> Dict[str, Any]:
        """Test that the app starts without errors."""
        print("Testing Streamlit app startup...")
        
        started = self.start_app()
        
        if not started:
            stderr_output = ""
            if self.process and self.process.stderr:
                stderr_output = self.process.stderr.read().decode()
            
            return {
                'test': 'app_startup',
                'passed': False,
                'error': 'App failed to start within timeout',
                'stderr': stderr_output
            }
        
        # Test basic endpoints
        try:
            response = requests.get(f'http://localhost:{self.port}', timeout=5)
            html_content = response.text
            
            # Check for basic app elements
            checks = {
                'has_title': 'FAO Food Price Index Dashboard' in html_content,
                'no_error_traces': 'Traceback' not in html_content,
                'has_streamlit_content': 'streamlit' in html_content.lower()
            }
            
            all_passed = all(checks.values())
            
            return {
                'test': 'app_startup',
                'passed': all_passed,
                'checks': checks,
                'status_code': response.status_code
            }
            
        except Exception as e:
            return {
                'test': 'app_startup',
                'passed': False,
                'error': str(e)
            }
        finally:
            self.stop_app()


class ComponentImportTester:
    """Test that all components can be imported without errors."""
    
    def __init__(self):
        self.components = [
            'app',
            'pivot_builder', 
            'chart_builder',
            'data_pipeline',
            'kpi_calculator',
            'excel_exporter'
        ]
    
    def test_component_imports(self) -> Dict[str, Any]:
        """Test importing all main components."""
        print("Testing component imports...")
        
        results = {}
        
        for component in self.components:
            try:
                # Add current directory to path
                if str(Path.cwd()) not in sys.path:
                    sys.path.insert(0, str(Path.cwd()))
                
                # Try to import the component
                __import__(component)
                
                results[component] = {
                    'imported': True,
                    'error': None
                }
                
            except Exception as e:
                results[component] = {
                    'imported': False,
                    'error': str(e)
                }
        
        # Calculate overall result
        imported_count = sum(1 for r in results.values() if r['imported'])
        total_count = len(results)
        
        return {
            'test': 'component_imports',
            'passed': imported_count == total_count,
            'imported': imported_count,
            'total': total_count,
            'results': results
        }


class DataPipelineTester:
    """Test data pipeline integration."""
    
    def test_data_pipeline(self) -> Dict[str, Any]:
        """Test the data pipeline can run without errors."""
        print("Testing data pipeline integration...")
        
        try:
            from data_pipeline import DataPipeline
            
            # Test with minimal configuration
            pipeline = DataPipeline(
                sheet_name='Monthly',
                cache_ttl_hours=0.1,  # Very short TTL for testing
                fetcher=self._create_mock_fetcher()
            )
            
            # Try to run the pipeline
            result_df = pipeline.run()
            
            if result_df is None:
                return {
                    'test': 'data_pipeline',
                    'passed': False,
                    'error': 'Pipeline returned None'
                }
            
            # Validate result structure
            expected_columns = ['date', 'food_price_index', 'meat', 'dairy', 'cereals', 'oils', 'sugar']
            has_required_columns = all(col in result_df.columns for col in expected_columns)
            
            return {
                'test': 'data_pipeline',
                'passed': has_required_columns and len(result_df) > 0,
                'columns_found': list(result_df.columns),
                'rows_found': len(result_df),
                'has_required_columns': has_required_columns
            }
            
        except Exception as e:
            return {
                'test': 'data_pipeline',
                'passed': False,
                'error': str(e)
            }
    
    def _create_mock_fetcher(self):
        """Create a mock data fetcher for testing."""
        def mock_fetcher():
            # Create minimal test Excel data
            from io import BytesIO
            import pandas as pd
            
            # Create test data
            test_data = {
                'Date': pd.date_range('2023-01-01', periods=12, freq='ME'),
                'Food Price Index': range(100, 112),
                'Meat Price Index': range(95, 107),
                'Dairy Products': range(105, 117),
                'Cereals Index': range(90, 102),
                'Vegetable Oils': range(110, 122),
                'Sugar Index': range(85, 97)
            }
            
            df = pd.DataFrame(test_data)
            
            # Create Excel file in memory
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Monthly', index=False)
            
            output.seek(0)
            return output
        
        return mock_fetcher


class PivotIntegrationTester:
    """Test pivot functionality integration."""
    
    def test_pivot_integration(self) -> Dict[str, Any]:
        """Test pivot functionality end-to-end."""
        print("Testing pivot integration...")
        
        try:
            from pivot_builder import (
                prepare_temporal_dimensions, 
                create_pivot_table, 
                configure_aggrid_options,
                validate_pivot_size
            )
            
            # Create test data
            test_data = {
                'date': pd.date_range('2023-01-01', periods=24, freq='ME'),
                'food_price_index': range(100, 124),
                'meat': range(95, 119),
                'dairy': range(105, 129)
            }
            df = pd.DataFrame(test_data)
            
            index_mapping = {
                'Food Price Index': 'food_price_index',
                'Meat': 'meat',
                'Dairy': 'dairy'
            }
            
            # Test temporal dimensions
            df_with_dims = prepare_temporal_dimensions(df)
            if not all(col in df_with_dims.columns for col in ['Year', 'Quarter', 'Month']):
                return {
                    'test': 'pivot_integration',
                    'passed': False,
                    'error': 'Temporal dimensions not created properly'
                }
            
            # Test pivot creation
            pivot_df = create_pivot_table(df, 'Year', ['Food Price Index', 'Meat'], 'mean', index_mapping)
            if pivot_df.empty:
                return {
                    'test': 'pivot_integration', 
                    'passed': False,
                    'error': 'Pivot table creation failed'
                }
            
            # Test AgGrid configuration
            grid_options = configure_aggrid_options(pivot_df, 'Year')
            if not isinstance(grid_options, dict) or 'columnDefs' not in grid_options:
                return {
                    'test': 'pivot_integration',
                    'passed': False,
                    'error': 'AgGrid configuration failed'
                }
            
            # Test size validation
            is_valid, cell_count, msg = validate_pivot_size(df, 'Year', ['Food Price Index'], 1000)
            if not is_valid:
                return {
                    'test': 'pivot_integration',
                    'passed': False,
                    'error': f'Size validation failed: {msg}'
                }
            
            return {
                'test': 'pivot_integration',
                'passed': True,
                'pivot_rows': len(pivot_df),
                'pivot_columns': len(pivot_df.columns),
                'cell_count': cell_count
            }
            
        except Exception as e:
            return {
                'test': 'pivot_integration',
                'passed': False,
                'error': str(e)
            }


class IntegrationTestRunner:
    """Main integration test runner."""
    
    def __init__(self):
        self.testers = [
            ComponentImportTester(),
            DataPipelineTester(),
            PivotIntegrationTester(),
            StreamlitAppTester()
        ]
        
        self.results = []
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all integration tests."""
        print("🔗 Starting integration tests...\n")
        
        # Component imports
        result = self.testers[0].test_component_imports()
        self.results.append(result)
        
        # Data pipeline
        result = self.testers[1].test_data_pipeline()
        self.results.append(result)
        
        # Pivot integration
        result = self.testers[2].test_pivot_integration()
        self.results.append(result)
        
        # Streamlit app (last, as it's most resource intensive)
        result = self.testers[3].test_app_startup()
        self.results.append(result)
        
        # Calculate overall result
        passed_tests = sum(1 for r in self.results if r['passed'])
        total_tests = len(self.results)
        
        return {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'overall_passed': passed_tests == total_tests,
            'test_results': self.results
        }
    
    def print_summary(self, results: Dict[str, Any]):
        """Print integration test summary."""
        print("\n" + "="*60)
        print("INTEGRATION TEST SUMMARY")
        print("="*60)
        
        overall_status = "✅ PASSED" if results['overall_passed'] else "❌ FAILED"
        print(f"\nOverall: {overall_status} ({results['passed_tests']}/{results['total_tests']})")
        
        print(f"\nDetailed Results:")
        for result in results['test_results']:
            test_name = result['test'].replace('_', ' ').title()
            status = "✅ PASSED" if result['passed'] else "❌ FAILED"
            print(f"  {test_name:.<40} {status}")
            
            if not result['passed'] and 'error' in result:
                print(f"    Error: {result['error']}")
        
        print("="*60)


def main():
    """Main entry point for integration tests."""
    runner = IntegrationTestRunner()
    
    try:
        results = runner.run_all_tests()
        runner.print_summary(results)
        
        # Exit with appropriate code
        sys.exit(0 if results['overall_passed'] else 1)
        
    except KeyboardInterrupt:
        print("\n🛑 Integration tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Integration tests failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()