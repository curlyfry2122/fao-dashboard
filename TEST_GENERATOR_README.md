# Test Generator Agent

An intelligent test generator that analyzes Python modules and automatically generates comprehensive pytest-compatible test suites.

## Features

- **AST-based Analysis**: Extracts classes, methods, functions, and their signatures using Abstract Syntax Tree parsing
- **Context-Aware Mock Generation**: Creates realistic test data based on function names and type hints
- **Pattern Recognition**: Learns from existing test patterns in the codebase
- **Smart Dependency Detection**: Automatically identifies external dependencies that need mocking
- **Type-Aware Assertions**: Generates appropriate assertions based on return type annotations
- **Customizable Configuration**: Supports project-specific test templates and patterns

## Quick Start

### Basic Usage

```bash
# Generate tests for a single module
python3 generate_tests.py module_name.py

# Generate tests with custom output path
python3 generate_tests.py module_name.py -o test_custom_name.py

# Use custom configuration
python3 generate_tests.py module_name.py -c test_generator_config.json

# Analyze module structure only (no test generation)
python3 generate_tests.py module_name.py --analyze-only
```

### Integration with Git Workflow

The test generator is integrated with the pre-commit hook to automatically suggest test generation for new Python files:

```bash
# When committing new Python files without tests, you'll see:
⚠️  No test file found for new_module.py
💡 Consider running: python3 generate_tests.py new_module.py
```

## Configuration

### Default Configuration

The generator uses sensible defaults, but can be customized via `test_generator_config.json`:

```json
{
  "test_file_prefix": "test_",
  "test_class_prefix": "Test",
  "mock_external_deps": true,
  "generate_fixtures": true,
  "include_edge_cases": true,
  "mock_patterns": [
    "requests.",
    "open(",
    "datetime.now",
    "download_",
    "fetch_"
  ]
}
```

### FAO Project-Specific Configuration

For this project, the configuration includes FAO-specific patterns:

- **Mock Data Patterns**: Realistic FAO food price index data
- **Common Fixtures**: Pre-configured fixtures for FAO data structures
- **Test Scenarios**: Standard test cases for data fetching, caching, filtering

## Generated Test Quality

### What the Generator Creates

1. **Test Structure**
   - Proper pytest imports and fixtures
   - Test classes mirroring source code structure
   - Individual test methods for each public function/method

2. **Realistic Test Data**
   - Context-aware mock arguments (e.g., realistic dates for date functions)
   - Appropriate pandas DataFrames for data processing functions
   - Type-appropriate assertions based on return type hints

3. **Edge Case Scaffolding**
   - TODO comments for common edge cases
   - Exception handling test stubs
   - Input validation test placeholders

### Example Generated Test

```python
def test_filter_by_date_range():
    """Test filter_by_date_range function."""
    # Test filter_by_date_range function
    result = filter_by_date_range(pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=5, freq='MS'),
        'food_price_index': [100.0, 101.5, 102.0, 103.2, 104.1],
        'meat': [95.0, 96.2, 97.1, 98.0, 99.3]
    }), '2020-01-01', '2020-12-31', 'date')
    assert isinstance(result, pd.DataFrame)
    # TODO: Add specific assertions based on expected behavior
    
    # Edge cases
    # TODO: Test with None inputs
    # TODO: Test with empty inputs
    # TODO: Test exception handling
```

## Project Integration

### File Structure

```
fao-dash/
├── generate_tests.py              # Main test generator script
├── test_generator_config.json     # Configuration file
├── .git/hooks/pre-commit          # Enhanced with test generation suggestions
├── requirements.txt               # Updated with dependencies
└── TEST_GENERATOR_README.md       # This documentation
```

### Generated Files

- `test_<module_name>.py`: Standard generated test files
- `test_<module_name>_generated.py`: Explicitly generated files (tracked by pre-commit hook)

### Pre-commit Hook Integration

The enhanced pre-commit hook:

1. **Detects** new Python files without corresponding tests
2. **Suggests** running the test generator
3. **Runs** tests on both existing and generated test files
4. **Validates** that generated tests are syntactically correct

## Advanced Usage

### Analyzing Existing Code

```bash
# Get detailed module analysis
python3 generate_tests.py data_pipeline.py --analyze-only | jq '.'
```

This outputs structured information about:
- Classes and methods
- Function signatures and return types
- Dependencies requiring mocking
- Import statements

### Custom Test Templates

Modify `test_generator_config.json` to customize:

```json
{
  "test_templates": {
    "class_test": {
      "setup_method": true,
      "teardown_method": false,
      "mock_dependencies": true
    },
    "function_test": {
      "parametrize_inputs": true,
      "test_exceptions": true,
      "test_edge_cases": true
    }
  }
}
```

### FAO-Specific Features

The generator recognizes FAO project patterns:

- **Data Functions**: Generates realistic FAO food price data
- **Date Handling**: Creates appropriate date ranges for time series data
- **Column Names**: Uses actual FAO data column names in mock data
- **Cache Testing**: Includes cache-related test scenarios

## Best Practices

### 1. Review Generated Tests

Generated tests provide scaffolding but should be reviewed and enhanced:

```python
# Generated (basic)
assert isinstance(result, pd.DataFrame)

# Enhanced (specific)
assert len(result) > 0
assert 'date' in result.columns
assert result['date'].is_monotonic_increasing
```

### 2. Add Domain-Specific Assertions

```python
# FAO-specific assertions
assert result['food_price_index'].min() > 0
assert result['date'].dt.freq == 'MS'  # Monthly start frequency
assert not result.duplicated(subset=['date']).any()
```

### 3. Use Generated Tests as Starting Points

- Keep the TODO comments as reminders
- Implement the suggested edge cases
- Add parametrized tests for multiple scenarios

### 4. Maintain Test Coverage

```bash
# Check coverage after adding tests
python3 -m pytest --cov=. --cov-report=html
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are in `requirements.txt`
2. **Type Errors**: Review mock data types in generated tests
3. **Missing Fixtures**: Add project-specific fixtures to config

### Debug Mode

Run with verbose logging:

```bash
python3 generate_tests.py module.py -v
```

### Test Validation

Generated tests should pass basic syntax checks:

```bash
python3 -m py_compile test_generated_file.py
python3 -m pytest test_generated_file.py --collect-only
```

## Contributing

### Extending the Generator

1. **Add New Mock Patterns**: Update `mock_patterns` in config
2. **Improve Type Recognition**: Enhance `_generate_realistic_mock_arg()`
3. **Add Test Templates**: Create new templates in config file

### Feedback Loop

The generator learns from existing tests in the codebase. Well-written tests improve future generation quality.

## Dependencies

```txt
# Core dependencies
pandas>=2.0.0
pytest>=7.0.0

# Optional (for enhanced features)
ast (built-in)
json (built-in)
```

## License

Part of the FAO Dashboard project - see main project LICENSE.