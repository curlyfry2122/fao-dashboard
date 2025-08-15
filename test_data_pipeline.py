"""Tests for data_pipeline module."""

import os
import pickle
import tempfile
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pandas as pd
import pytest

from data_pipeline import DataPipeline


class TestDataPipeline:
    """Test cases for the data pipeline orchestration."""

    def create_mock_excel_data(self):
        """Create mock Excel data for testing."""
        # Create a simple DataFrame that mimics FAO structure
        df_annual = pd.DataFrame({
            'Year': [2020, 2021, 2022],
            'Food Price Index': [100.0, 105.0, 110.0],
            'Meat': [95.0, 98.0, 102.0],
            'Dairy': [105.0, 108.0, 112.0],
            'Cereals': [98.0, 101.0, 104.0],
            'Oils': [110.0, 115.0, 120.0],
            'Sugar': [85.0, 87.0, 89.0]
        })
        
        df_monthly = pd.DataFrame({
            'Date': pd.date_range('2022-01-01', periods=12, freq='MS'),
            'Food Price Index': [100.0 + i for i in range(12)],
            'Meat': [95.0 + i*0.8 for i in range(12)],
            'Dairy': [105.0 + i*0.9 for i in range(12)],
            'Cereals': [98.0 + i*0.7 for i in range(12)],
            'Oils': [110.0 + i*1.2 for i in range(12)],
            'Sugar': [85.0 + i*0.5 for i in range(12)]
        })
        
        # Create Excel file in memory
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            df_annual.to_excel(writer, sheet_name='Annual', index=False)
            df_monthly.to_excel(writer, sheet_name='Monthly', index=False)
        
        excel_buffer.seek(0)
        return excel_buffer

    @patch('data_pipeline.download_fao_fpi_data')
    @patch('data_pipeline.validate_excel_structure')
    def test_run_basic_pipeline(self, mock_validate, mock_download):
        """Test basic pipeline execution with mocked fetcher."""
        # Setup mocks
        mock_excel_data = self.create_mock_excel_data()
        mock_download.return_value = mock_excel_data
        mock_validate.return_value = (True, "")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create pipeline with temp cache directory
            pipeline = DataPipeline(
                sheet_name='Monthly',
                metrics=['yoy_change', 'mom_change'],
                cache_dir=temp_dir
            )
            
            # Run pipeline
            result_df = pipeline.run()
            
            # Verify the result
            assert isinstance(result_df, pd.DataFrame)
            assert 'date' in result_df.columns
            assert 'food_price_index' in result_df.columns
            assert 'food_price_index_yoy_change' in result_df.columns
            assert 'food_price_index_mom_change' in result_df.columns
            assert len(result_df) == 12  # 12 months of data
            
            # Verify mocks were called
            mock_download.assert_called_once()
            mock_validate.assert_called_once()

    def test_custom_fetcher_injection(self):
        """Test pipeline with custom fetcher injection."""
        # Create custom fetcher
        mock_excel_data = self.create_mock_excel_data()
        custom_fetcher = Mock(return_value=mock_excel_data)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create pipeline with custom fetcher
            pipeline = DataPipeline(
                sheet_name='Annual',
                metrics=['yoy_change'],
                cache_dir=temp_dir,
                fetcher=custom_fetcher
            )
            
            # Run pipeline
            result_df = pipeline.run()
            
            # Verify custom fetcher was used
            custom_fetcher.assert_called_once()
            assert len(result_df) == 3  # 3 years of annual data

    def test_cache_ttl_valid(self):
        """Test that valid cache is used instead of fetching."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create pipeline
            pipeline = DataPipeline(
                sheet_name='Monthly',
                cache_dir=temp_dir,
                cache_ttl_hours=1
            )
            
            # Create a recent cache file
            cache_data = {
                'timestamp': datetime.now() - timedelta(minutes=30),  # 30 minutes old
                'data': pd.DataFrame({
                    'date': pd.date_range('2022-01-01', periods=5, freq='MS'),
                    'food_price_index': [100, 101, 102, 103, 104],
                    'meat': [95, 96, 97, 98, 99],
                    'dairy': [105, 106, 107, 108, 109],
                    'cereals': [98, 99, 100, 101, 102],
                    'oils': [110, 111, 112, 113, 114],
                    'sugar': [85, 86, 87, 88, 89]
                })
            }
            
            cache_file = Path(temp_dir) / 'monthly_cache.pkl'
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            # Mock the fetcher to track if it's called
            with patch('data_pipeline.download_fao_fpi_data') as mock_download:
                result_df = pipeline.run()
                
                # Fetcher should NOT be called since cache is valid
                mock_download.assert_not_called()
                
                # Should return cached data
                assert len(result_df) == 5
                assert result_df['food_price_index'].iloc[0] == 100

    def test_cache_ttl_expired(self):
        """Test that expired cache triggers new fetch."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock fetcher
            mock_excel_data = self.create_mock_excel_data()
            mock_fetcher = Mock(return_value=mock_excel_data)
            
            # Create pipeline with mock fetcher
            pipeline = DataPipeline(
                sheet_name='Monthly',
                cache_dir=temp_dir,
                cache_ttl_hours=1,
                fetcher=mock_fetcher
            )
            
            # Create an expired cache file
            cache_data = {
                'timestamp': datetime.now() - timedelta(hours=2),  # 2 hours old (expired)
                'data': pd.DataFrame({
                    'date': pd.date_range('2022-01-01', periods=3, freq='MS'),
                    'food_price_index': [100, 101, 102],
                    'meat': [95, 96, 97],
                    'dairy': [105, 106, 107],
                    'cereals': [98, 99, 100],
                    'oils': [110, 111, 112],
                    'sugar': [85, 86, 87]
                })
            }
            
            cache_file = Path(temp_dir) / 'monthly_cache.pkl'
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            result_df = pipeline.run()
            
            # Fetcher SHOULD be called since cache is expired
            mock_fetcher.assert_called_once()
            
            # Should return new data, not cached
            assert len(result_df) == 12  # New data has 12 months

    def test_get_latest_update(self):
        """Test getting the timestamp of latest update."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock fetcher
            mock_excel_data = self.create_mock_excel_data()
            mock_fetcher = Mock(return_value=mock_excel_data)
            
            pipeline = DataPipeline(cache_dir=temp_dir, fetcher=mock_fetcher)
            
            # Initially should return None
            assert pipeline.get_latest_update() is None
            
            before_run = datetime.now()
            pipeline.run()
            after_run = datetime.now()
            
            # Should have a timestamp between before and after
            update_time = pipeline.get_latest_update()
            assert update_time is not None
            assert before_run <= update_time <= after_run

    def test_clear_cache(self):
        """Test cache clearing functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = DataPipeline(cache_dir=temp_dir)
            
            # Create a cache file
            cache_data = {
                'timestamp': datetime.now(),
                'data': pd.DataFrame({'col': [1, 2, 3]})
            }
            
            cache_file = Path(temp_dir) / 'monthly_cache.pkl'
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            assert cache_file.exists()
            
            # Clear cache
            pipeline.clear_cache()
            
            # Cache file should be removed
            assert not cache_file.exists()

    @patch('data_pipeline.download_fao_fpi_data')
    def test_error_fallback_to_cache(self, mock_download):
        """Test fallback to cache when fetch fails."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = DataPipeline(cache_dir=temp_dir)
            
            # Create a valid cache file
            cached_df = pd.DataFrame({
                'date': pd.date_range('2022-01-01', periods=3, freq='MS'),
                'food_price_index': [100, 101, 102],
                'meat': [95, 96, 97],
                'dairy': [105, 106, 107],
                'cereals': [98, 99, 100],
                'oils': [110, 111, 112],
                'sugar': [85, 86, 87]
            })
            
            cache_data = {
                'timestamp': datetime.now() - timedelta(hours=2),  # Expired but available
                'data': cached_df
            }
            
            cache_file = Path(temp_dir) / 'monthly_cache.pkl'
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            # Make download fail
            mock_download.side_effect = Exception("Network error")
            
            # Should fall back to cache
            with patch('data_pipeline.logger') as mock_logger:
                result_df = pipeline.run()
                
                # Should log warning about fallback
                mock_logger.warning.assert_called()
                
                # Should return cached data
                assert len(result_df) == 3
                pd.testing.assert_frame_equal(result_df, cached_df)

    @patch('data_pipeline.download_fao_fpi_data')
    def test_error_no_cache_available(self, mock_download):
        """Test that error is raised when fetch fails and no cache exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = DataPipeline(cache_dir=temp_dir)
            
            # Make download fail
            mock_download.side_effect = Exception("Network error")
            
            # Should raise exception since no cache available
            with pytest.raises(Exception, match="Network error"):
                pipeline.run()

    def test_pipeline_with_all_metrics(self):
        """Test pipeline with all available metrics."""
        mock_excel_data = self.create_mock_excel_data()
        custom_fetcher = Mock(return_value=mock_excel_data)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = DataPipeline(
                sheet_name='Monthly',
                metrics=['yoy_change', 'mom_change', '12m_avg'],
                cache_dir=temp_dir,
                fetcher=custom_fetcher
            )
            
            result_df = pipeline.run()
            
            # Check all metric columns exist
            expected_metrics = [
                'food_price_index_yoy_change', 'food_price_index_mom_change', 'food_price_index_12m_avg',
                'meat_yoy_change', 'meat_mom_change', 'meat_12m_avg',
                'dairy_yoy_change', 'dairy_mom_change', 'dairy_12m_avg',
                'cereals_yoy_change', 'cereals_mom_change', 'cereals_12m_avg',
                'oils_yoy_change', 'oils_mom_change', 'oils_12m_avg',
                'sugar_yoy_change', 'sugar_mom_change', 'sugar_12m_avg'
            ]
            
            for metric_col in expected_metrics:
                assert metric_col in result_df.columns

    @patch('data_pipeline.validate_excel_structure')
    def test_validation_failure_handling(self, mock_validate):
        """Test handling of validation failures."""
        mock_excel_data = self.create_mock_excel_data()
        custom_fetcher = Mock(return_value=mock_excel_data)
        
        # Make validation fail
        mock_validate.return_value = (False, "Missing required sheet: Annual")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = DataPipeline(
                cache_dir=temp_dir,
                fetcher=custom_fetcher
            )
            
            with pytest.raises(ValueError, match="Missing required sheet"):
                pipeline.run()

    def test_get_cache_status(self):
        """Test cache status information retrieval."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = DataPipeline(
                cache_dir=temp_dir,
                cache_ttl_hours=1
            )
            
            # No cache initially
            status = pipeline.get_cache_status()
            assert status['exists'] is False
            assert status['age'] is None
            assert status['ttl_remaining'] is None
            
            # Create cache
            cache_data = {
                'timestamp': datetime.now() - timedelta(minutes=30),
                'data': pd.DataFrame({'col': [1, 2, 3]})
            }
            
            cache_file = Path(temp_dir) / 'monthly_cache.pkl'
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            # Check status with cache
            status = pipeline.get_cache_status()
            assert status['exists'] is True
            assert 29 <= status['age'].total_seconds() / 60 <= 31  # Around 30 minutes
            assert 29 <= status['ttl_remaining'].total_seconds() / 60 <= 31  # Around 30 minutes remaining

    def test_concurrent_pipeline_runs(self):
        """Test that concurrent runs don't cause issues."""
        import threading
        
        mock_excel_data = self.create_mock_excel_data()
        custom_fetcher = Mock(return_value=mock_excel_data)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = DataPipeline(
                cache_dir=temp_dir,
                fetcher=custom_fetcher
            )
            
            results = []
            errors = []
            
            def run_pipeline():
                try:
                    df = pipeline.run()
                    results.append(len(df))
                except Exception as e:
                    errors.append(str(e))
            
            # Run multiple threads
            threads = [threading.Thread(target=run_pipeline) for _ in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            
            # All should succeed
            assert len(errors) == 0
            assert len(results) == 5
            # All should get same result (12 rows for monthly data)
            assert all(r == 12 for r in results)

    def test_empty_dataframe_handling(self):
        """Test pipeline with empty Excel data."""
        # Create empty Excel data
        df_empty = pd.DataFrame(columns=['Date', 'Food Price Index', 'Meat', 'Dairy', 'Cereals', 'Oils', 'Sugar'])
        
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            df_empty.to_excel(writer, sheet_name='Annual', index=False)
            df_empty.to_excel(writer, sheet_name='Monthly', index=False)
        
        excel_buffer.seek(0)
        custom_fetcher = Mock(return_value=excel_buffer)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = DataPipeline(
                cache_dir=temp_dir,
                fetcher=custom_fetcher
            )
            
            result_df = pipeline.run()
            
            # Should handle empty data gracefully
            assert len(result_df) == 0
            assert 'date' in result_df.columns
            assert 'food_price_index' in result_df.columns