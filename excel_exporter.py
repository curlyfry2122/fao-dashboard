"""
Excel Export functionality for FAO Food Price Index Dashboard.

Provides professional Excel export capabilities with advanced formatting including
bold headers, frozen panes, and proper number formatting for data analysis.
"""

from typing import Optional, Dict, Any, TYPE_CHECKING, Tuple, Union
import pandas as pd
import xlsxwriter
from io import BytesIO
import logging
import tempfile
import os
from performance_monitor import performance_monitor, performance_context

if TYPE_CHECKING:
    from xlsxwriter.workbook import Workbook
    from xlsxwriter.worksheet import Worksheet
    import plotly.graph_objects as go

logger = logging.getLogger(__name__)


class ExcelExporter:
    """
    Excel export utility with professional formatting capabilities.
    
    Provides methods to export pandas DataFrames to Excel format with
    advanced formatting including bold headers, frozen panes, number
    formatting, and professional styling.
    
    Example:
        >>> exporter = ExcelExporter()
        >>> df = pd.DataFrame({'Price Index': [100, 101, 102]})
        >>> workbook = exporter.generate_data_sheet(df, 'FAO Data')
        >>> workbook.close()
    """
    
    def __init__(self):
        """Initialize ExcelExporter with default formatting options."""
        self.default_formats = {
            'header': {
                'bold': True,
                'font_color': '#FFFFFF',
                'bg_color': '#4472C4',
                'border': 1,
                'align': 'center',
                'valign': 'vcenter',
                'text_wrap': True
            },
            'number': {
                'num_format': '#,##0.0',
                'align': 'right'
            },
            'integer': {
                'num_format': '#,##0',
                'align': 'right'
            },
            'percentage': {
                'num_format': '0.0%',
                'align': 'right'
            },
            'date': {
                'num_format': 'yyyy-mm-dd',
                'align': 'center'
            }
        }
    
    @performance_monitor('excel_export', include_args=True)
    def generate_data_sheet(
        self, 
        df: pd.DataFrame, 
        sheet_name: str,
        freeze_panes: bool = True,
        auto_filter: bool = True
    ) -> 'Workbook':
        """
        Generate Excel sheet with formatted DataFrame data.
        
        Creates an Excel workbook with professional formatting including
        bold headers, frozen panes, and appropriate number formatting
        for different data types.
        
        Args:
            df: DataFrame containing the data to export
            sheet_name: Name for the Excel worksheet
            freeze_panes: Whether to freeze the header row (default: True)
            auto_filter: Whether to add auto-filter to headers (default: True)
            
        Returns:
            xlsxwriter.Workbook: Configured workbook object with formatted data
            
        Raises:
            ValueError: If sheet_name is empty or invalid
            TypeError: If df is not a pandas DataFrame
            
        Example:
            >>> exporter = ExcelExporter()
            >>> df = pd.DataFrame({
            ...     'Date': pd.date_range('2020-01-01', periods=3, freq='MS'),
            ...     'Price Index': [100.5, 101.2, 102.8],
            ...     'Change %': [0.5, 0.7, 1.6]
            ... })
            >>> workbook = exporter.generate_data_sheet(df, 'Monthly Data')
            >>> # Save to file or return for further processing
            >>> workbook.close()
        """
        # Input validation
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        
        if not sheet_name or not isinstance(sheet_name, str):
            raise ValueError("sheet_name must be a non-empty string")
        
        # Sanitize sheet name for Excel compatibility
        sheet_name = self._sanitize_sheet_name(sheet_name)
        
        try:
            # Create workbook and worksheet using BytesIO for in-memory handling
            output = BytesIO()
            workbook = xlsxwriter.Workbook(output, {'in_memory': True})
            worksheet = workbook.add_worksheet(sheet_name)
            
            # Create format objects
            formats = self._create_formats(workbook)
            
            # Handle empty DataFrame
            if df.empty:
                self._write_empty_sheet(worksheet, formats)
                return workbook
            
            # Write data with formatting
            self._write_data_with_formatting(worksheet, df, formats)
            
            # Apply header formatting
            self._format_headers(worksheet, df, formats['header'])
            
            # Apply column formatting based on data types
            self._format_columns(worksheet, df, formats)
            
            # Set column widths
            self._adjust_column_widths(worksheet, df)
            
            # Apply freeze panes if requested
            if freeze_panes:
                worksheet.freeze_panes(1, 0)  # Freeze first row (headers)
            
            # Add auto-filter if requested
            if auto_filter and not df.empty:
                worksheet.autofilter(0, 0, len(df), len(df.columns) - 1)
            
            logger.info(f"Successfully created Excel sheet '{sheet_name}' with {len(df)} rows")
            return workbook
            
        except Exception as e:
            logger.error(f"Error creating Excel sheet: {str(e)}")
            raise
    
    def _create_formats(self, workbook: 'Workbook') -> Dict[str, Any]:
        """Create format objects for different data types and styling."""
        formats = {}
        
        for format_name, format_props in self.default_formats.items():
            formats[format_name] = workbook.add_format(format_props)
        
        return formats
    
    def _write_empty_sheet(self, worksheet: 'Worksheet', formats: Dict[str, Any]) -> None:
        """Write message for empty DataFrame."""
        worksheet.write(0, 0, "No data available", formats['header'])
        worksheet.set_column(0, 0, 20)
    
    def _write_data_with_formatting(
        self, 
        worksheet: 'Worksheet', 
        df: pd.DataFrame, 
        formats: Dict[str, Any]
    ) -> None:
        """Write DataFrame data to worksheet with basic formatting."""
        # Write headers
        for col_idx, column_name in enumerate(df.columns):
            worksheet.write(0, col_idx, str(column_name), formats['header'])
        
        # Write data rows
        for row_idx, (_, row) in enumerate(df.iterrows(), start=1):
            for col_idx, value in enumerate(row):
                # Handle different data types
                if pd.isna(value):
                    worksheet.write(row_idx, col_idx, "")
                else:
                    worksheet.write(row_idx, col_idx, value)
    
    def _format_headers(
        self, 
        worksheet: 'Worksheet', 
        df: pd.DataFrame, 
        header_format: Any
    ) -> None:
        """Apply header formatting to the first row."""
        for col_idx in range(len(df.columns)):
            worksheet.write(0, col_idx, df.columns[col_idx], header_format)
    
    def _format_columns(
        self, 
        worksheet: 'Worksheet', 
        df: pd.DataFrame, 
        formats: Dict[str, Any]
    ) -> None:
        """Apply column-specific formatting based on data types."""
        for col_idx, (column_name, series) in enumerate(df.items()):
            # Determine appropriate format based on data type and column name
            format_to_use = self._get_column_format(series, column_name, formats)
            
            if format_to_use:
                # Apply format to entire column (excluding header)
                worksheet.set_column(col_idx, col_idx, None, format_to_use)
    
    def _get_column_format(
        self, 
        series: pd.Series, 
        column_name: str, 
        formats: Dict[str, Any]
    ) -> Optional[Any]:
        """Determine appropriate format for a column based on its data type and name."""
        # Check for percentage columns by name
        if any(keyword in column_name.lower() for keyword in ['%', 'percent', 'change', 'pct']):
            return formats['percentage']
        
        # Check for date columns
        if pd.api.types.is_datetime64_any_dtype(series):
            return formats['date']
        
        # Check for numeric columns
        if pd.api.types.is_numeric_dtype(series):
            # Check if values are primarily integers
            non_null_values = series.dropna()
            if len(non_null_values) > 0:
                # Use integer format if all values are close to integers
                if non_null_values.apply(lambda x: abs(x - round(x)) < 0.001).all():
                    return formats['integer']
                else:
                    return formats['number']
        
        return None
    
    def _adjust_column_widths(self, worksheet: 'Worksheet', df: pd.DataFrame) -> None:
        """Auto-adjust column widths based on content."""
        for col_idx, column_name in enumerate(df.columns):
            # Calculate width based on column name and sample data
            header_width = len(str(column_name))
            
            # Sample first few rows to estimate content width
            sample_size = min(10, len(df))
            if sample_size > 0:
                sample_data = df.iloc[:sample_size, col_idx]
                max_content_width = max(
                    len(str(value)) for value in sample_data if pd.notna(value)
                ) if not sample_data.isna().all() else 0
            else:
                max_content_width = 0
            
            # Set width with reasonable bounds
            optimal_width = max(header_width, max_content_width) + 2
            final_width = min(max(optimal_width, 8), 50)  # Between 8 and 50 characters
            
            worksheet.set_column(col_idx, col_idx, final_width)
    
    def _sanitize_sheet_name(self, sheet_name: str) -> str:
        """
        Sanitize sheet name for Excel compatibility.
        
        Excel sheet names cannot contain: [ ] : * ? / \\
        and must be 31 characters or less.
        """
        # Remove invalid characters
        invalid_chars = ['[', ']', ':', '*', '?', '/', '\\']
        sanitized = sheet_name
        for char in invalid_chars:
            sanitized = sanitized.replace(char, '_')
        
        # Truncate to 31 characters
        if len(sanitized) > 31:
            sanitized = sanitized[:31]
        
        return sanitized
    
    def add_chart_sheet(
        self,
        workbook: 'Workbook',
        figure: 'go.Figure',
        sheet_name: str,
        image_size: Tuple[int, int] = (800, 600),
        position: str = 'B2'
    ) -> bool:
        """
        Add a new worksheet with embedded Plotly chart as PNG image.
        
        Converts Plotly figure to PNG using kaleido and embeds it in Excel worksheet.
        Falls back to data-only sheet if PNG conversion fails.
        
        Args:
            workbook: xlsxwriter workbook to add sheet to
            figure: Plotly figure object to convert and embed
            sheet_name: Name for the new worksheet
            image_size: Tuple of (width, height) for PNG image, max (800, 600)
            position: Excel cell position for image placement (e.g., 'B2')
            
        Returns:
            bool: True if chart was successfully embedded, False if fell back to data-only
            
        Raises:
            ValueError: If workbook is None or sheet_name is invalid
            
        Example:
            >>> import plotly.graph_objects as go
            >>> fig = go.Figure(go.Scatter(x=[1, 2, 3], y=[4, 5, 6]))
            >>> exporter = ExcelExporter()
            >>> workbook = xlsxwriter.Workbook('test.xlsx')
            >>> success = exporter.add_chart_sheet(workbook, fig, 'Chart')
            >>> workbook.close()
        """
        if workbook is None:
            raise ValueError("Workbook cannot be None")
        
        # Sanitize sheet name
        sheet_name = self._sanitize_sheet_name(sheet_name)
        
        # Enforce size constraints
        width, height = image_size
        width = min(width, 800)
        height = min(height, 600)
        constrained_size = (width, height)
        
        try:
            # Attempt to convert Plotly figure to PNG
            png_success, temp_png_path = self._convert_plotly_to_png(figure, constrained_size)
            
            if png_success and temp_png_path:
                # Create worksheet and embed PNG (don't clean up temp file yet)
                success = self._embed_png_in_worksheet(workbook, temp_png_path, sheet_name, position)
                
                if success:
                    logger.info(f"Successfully created chart sheet '{sheet_name}' with embedded PNG")
                    # Note: temp file will be cleaned up when workbook is closed
                    return True
                else:
                    # Clean up temp file if embedding failed
                    try:
                        if os.path.exists(temp_png_path):
                            os.remove(temp_png_path)
                    except Exception as e:
                        logger.warning(f"Failed to remove temporary PNG file: {e}")
            
        except Exception as e:
            logger.warning(f"Chart embedding failed: {e}")
        
        # Fallback to data-only sheet
        logger.info(f"Falling back to data-only sheet for '{sheet_name}'")
        self._create_fallback_data_sheet(workbook, figure, sheet_name)
        return False
    
    def _convert_plotly_to_png(
        self, 
        figure: 'go.Figure', 
        image_size: Tuple[int, int]
    ) -> Tuple[bool, Optional[str]]:
        """
        Convert Plotly figure to PNG using kaleido.
        
        Args:
            figure: Plotly figure to convert
            image_size: Tuple of (width, height) for PNG
            
        Returns:
            Tuple[bool, Optional[str]]: (success, temp_file_path)
        """
        try:
            # Import plotly here to handle missing dependency gracefully
            import plotly.graph_objects as go
            
            # Check if figure has the required attributes instead of strict type checking
            if not hasattr(figure, 'write_image') or not hasattr(figure, 'data'):
                logger.error(f"Invalid figure object provided: {type(figure)}")
                return False, None
            
            # Create temporary file for PNG
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                temp_png_path = tmp_file.name
            
            # Convert to PNG using kaleido with optimized settings for Excel export
            width, height = image_size
            
            # Create a copy of the figure to avoid modifying the original
            import copy
            figure_copy = copy.deepcopy(figure)
            
            # Update layout for better static export
            figure_copy.update_layout(
                font=dict(size=12),  # Larger font for readability
                margin=dict(l=60, r=40, t=60, b=60),  # Better margins
                legend=dict(
                    orientation="h",
                    yanchor="bottom", 
                    y=-0.15,  # Move legend below chart
                    xanchor="center",
                    x=0.5
                ),
                title=dict(x=0.5),  # Center title
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            
            figure_copy.write_image(
                temp_png_path,
                format='png',
                width=width,
                height=height,
                scale=2  # Higher resolution for crisp image
            )
            
            # Verify PNG was created successfully
            if os.path.exists(temp_png_path) and os.path.getsize(temp_png_path) > 0:
                logger.info(f"Successfully converted Plotly figure to PNG: {temp_png_path}")
                return True, temp_png_path
            else:
                logger.error("PNG file was not created or is empty")
                return False, None
                
        except ImportError as e:
            logger.error(f"Missing dependency for PNG conversion: {e}")
            return False, None
        except Exception as e:
            logger.error(f"Failed to convert Plotly figure to PNG: {e}")
            return False, None
    
    def _embed_png_in_worksheet(
        self, 
        workbook: 'Workbook', 
        png_path: str, 
        sheet_name: str, 
        position: str
    ) -> bool:
        """
        Create worksheet and embed PNG image.
        
        Args:
            workbook: xlsxwriter workbook
            png_path: Path to PNG file
            sheet_name: Name for worksheet
            position: Cell position for image
            
        Returns:
            bool: True if successful
        """
        try:
            # Create new worksheet
            worksheet = workbook.add_worksheet(sheet_name)
            
            # Insert PNG image at specified position with better sizing
            worksheet.insert_image(position, png_path, {
                'x_scale': 0.8,  # Scale down slightly for better fit
                'y_scale': 0.8
            })
            
            # Add title above the chart
            title_format = workbook.add_format({
                'bold': True,
                'font_size': 16,
                'align': 'center',
                'font_color': '#1f77b4'
            })
            worksheet.write('B1', f'Food Price Index Chart', title_format)
            
            # Set reasonable column width for image display
            worksheet.set_column('A:A', 2)  # Narrow left margin
            worksheet.set_column('B:F', 20)  # Wide columns for chart
            worksheet.set_row(1, 480)  # Make row tall enough for scaled image
            
            logger.info(f"Successfully embedded PNG in worksheet '{sheet_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to embed PNG in worksheet: {e}")
            return False
    
    def _create_fallback_data_sheet(
        self, 
        workbook: 'Workbook', 
        figure: 'go.Figure', 
        sheet_name: str
    ) -> None:
        """
        Create data-only worksheet as fallback when PNG embedding fails.
        
        Extracts data from Plotly figure traces and creates formatted data table.
        
        Args:
            workbook: xlsxwriter workbook
            figure: Plotly figure to extract data from
            sheet_name: Name for worksheet
        """
        try:
            # Extract data from figure traces
            data_dict = {'x': [], 'y': [], 'series': []}
            
            if hasattr(figure, 'data') and figure.data:
                for i, trace in enumerate(figure.data):
                    trace_name = getattr(trace, 'name', f'Series {i+1}')
                    
                    # Extract x and y data
                    if hasattr(trace, 'x') and hasattr(trace, 'y'):
                        x_data = list(trace.x) if trace.x is not None else []
                        y_data = list(trace.y) if trace.y is not None else []
                        
                        # Ensure both arrays have same length
                        min_length = min(len(x_data), len(y_data))
                        
                        for j in range(min_length):
                            data_dict['x'].append(x_data[j])
                            data_dict['y'].append(y_data[j])
                            data_dict['series'].append(trace_name)
            
            # Convert to DataFrame
            if data_dict['x']:
                fallback_df = pd.DataFrame(data_dict)
                fallback_df.columns = ['X Value', 'Y Value', 'Series']
            else:
                # Empty fallback
                fallback_df = pd.DataFrame({
                    'Note': ['Chart data could not be extracted'],
                    'Sheet': [sheet_name]
                })
            
            # Create worksheet using existing data sheet method
            worksheet = workbook.add_worksheet(sheet_name)
            
            # Create formats
            formats = self._create_formats(workbook)
            
            # Write title
            title_format = workbook.add_format({
                'bold': True,
                'font_size': 14,
                'bg_color': '#FFF2CC',
                'border': 1
            })
            worksheet.write('A1', f'Chart Data: {sheet_name}', title_format)
            worksheet.write('A2', '(Chart image could not be generated)', 
                          workbook.add_format({'italic': True, 'font_color': '#666666'}))
            
            # Write data starting from row 4
            start_row = 3
            for col_idx, column_name in enumerate(fallback_df.columns):
                worksheet.write(start_row, col_idx, str(column_name), formats['header'])
            
            # Write data rows
            for row_idx, (_, row) in enumerate(fallback_df.iterrows(), start=start_row + 1):
                for col_idx, value in enumerate(row):
                    if pd.isna(value):
                        worksheet.write(row_idx, col_idx, "")
                    else:
                        worksheet.write(row_idx, col_idx, value)
            
            # Auto-adjust column widths
            for col_idx, column_name in enumerate(fallback_df.columns):
                max_length = max(len(str(column_name)), 15)
                worksheet.set_column(col_idx, col_idx, max_length)
            
            logger.info(f"Created fallback data sheet '{sheet_name}' with {len(fallback_df)} rows")
            
        except Exception as e:
            logger.error(f"Failed to create fallback data sheet: {e}")
            
            # Final fallback - empty sheet with error message
            try:
                worksheet = workbook.add_worksheet(sheet_name)
                error_format = workbook.add_format({'font_color': 'red', 'bold': True})
                worksheet.write('A1', f"Error: Could not create chart or data for '{sheet_name}'", error_format)
                worksheet.write('A2', f"Technical details: {str(e)}")
            except Exception:
                pass  # Give up gracefully