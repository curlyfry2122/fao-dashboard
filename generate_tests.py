#!/usr/bin/env python3
"""
Intelligent Test Generator for Python modules.

Analyzes Python source code using AST parsing and generates comprehensive
pytest-compatible test suites based on existing patterns in the codebase.
"""

import ast
import json
import logging
import re
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestGenerator:
    """
    Analyzes Python modules and generates comprehensive test suites.
    
    Features:
    - AST-based analysis of classes, functions, and methods
    - Pattern recognition from existing tests
    - Smart mock generation for external dependencies
    - Type-aware test generation
    - Customizable test templates
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the test generator.
        
        Args:
            config_path: Path to configuration file (JSON format)
        """
        self.config = self._load_config(config_path)
        self.existing_patterns = {}
        self._analyze_existing_tests()
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from file or use defaults."""
        default_config = {
            "test_file_prefix": "test_",
            "test_class_prefix": "Test",
            "mock_external_deps": True,
            "generate_fixtures": True,
            "include_edge_cases": True,
            "test_timeout": 30,
            "mock_patterns": [
                "requests.",
                "open(",
                "urllib.",
                "Path(",
                "datetime.now",
                "download_",
                "fetch_"
            ],
            "exclude_patterns": [
                "__pycache__",
                ".git",
                "venv",
                ".pyc"
            ]
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
        
        return default_config
    
    def _analyze_existing_tests(self) -> None:
        """Analyze existing test files to learn patterns."""
        test_files = list(Path('.').glob('test_*.py'))
        
        for test_file in test_files:
            try:
                with open(test_file, 'r') as f:
                    content = f.read()
                    tree = ast.parse(content)
                    
                    # Extract common patterns
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef) and node.name.startswith('Test'):
                            self._extract_test_patterns(node, content)
                            
            except Exception as e:
                logger.warning(f"Failed to analyze {test_file}: {e}")
    
    def _extract_test_patterns(self, class_node: ast.ClassDef, content: str) -> None:
        """Extract common patterns from existing test classes."""
        patterns = {
            'setup_methods': [],
            'mock_patterns': [],
            'assertion_patterns': [],
            'fixture_patterns': []
        }
        
        for method in class_node.body:
            if isinstance(method, ast.FunctionDef):
                method_name = method.name
                
                # Extract setup patterns
                if method_name in ['setUp', 'setup_method', 'setup']:
                    patterns['setup_methods'].append(method_name)
                
                # Extract mock usage patterns
                if 'mock' in method_name.lower():
                    patterns['mock_patterns'].append(method_name)
        
        self.existing_patterns[class_node.name] = patterns
    
    def analyze_module(self, module_path: str) -> Dict:
        """
        Analyze a Python module and extract structure for test generation.
        
        Args:
            module_path: Path to the Python module to analyze
            
        Returns:
            Dictionary containing module structure information
        """
        if not Path(module_path).exists():
            raise FileNotFoundError(f"Module not found: {module_path}")
        
        with open(module_path, 'r') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        analysis = {
            'module_name': Path(module_path).stem,
            'imports': self._extract_imports(tree),
            'classes': self._extract_classes(tree),
            'functions': self._extract_functions(tree),
            'constants': self._extract_constants(tree),
            'dependencies': self._identify_dependencies(tree)
        }
        
        return analysis
    
    def _extract_imports(self, tree: ast.AST) -> List[Dict]:
        """Extract import statements from the AST."""
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append({
                        'type': 'import',
                        'name': alias.name,
                        'alias': alias.asname
                    })
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    imports.append({
                        'type': 'from_import',
                        'module': module,
                        'name': alias.name,
                        'alias': alias.asname
                    })
        
        return imports
    
    def _extract_classes(self, tree: ast.AST) -> List[Dict]:
        """Extract class definitions from the AST."""
        classes = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_info = {
                    'name': node.name,
                    'base_classes': [base.id if isinstance(base, ast.Name) else str(base) 
                                   for base in node.bases],
                    'methods': self._extract_methods(node),
                    'docstring': ast.get_docstring(node),
                    'line_number': node.lineno
                }
                classes.append(class_info)
        
        return classes
    
    def _extract_methods(self, class_node: ast.ClassDef) -> List[Dict]:
        """Extract method definitions from a class."""
        methods = []
        
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef):
                method_info = {
                    'name': node.name,
                    'args': self._extract_function_args(node),
                    'returns': self._extract_return_annotation(node),
                    'docstring': ast.get_docstring(node),
                    'is_private': node.name.startswith('_'),
                    'is_property': any(isinstance(d, ast.Name) and d.id == 'property' 
                                     for d in node.decorator_list),
                    'line_number': node.lineno
                }
                methods.append(method_info)
        
        return methods
    
    def _extract_functions(self, tree: ast.AST) -> List[Dict]:
        """Extract standalone function definitions from the AST."""
        functions = []
        
        # Only get top-level functions, not methods inside classes
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                function_info = {
                    'name': node.name,
                    'args': self._extract_function_args(node),
                    'returns': self._extract_return_annotation(node),
                    'docstring': ast.get_docstring(node),
                    'is_private': node.name.startswith('_'),
                    'line_number': node.lineno
                }
                functions.append(function_info)
        
        return functions
    
    def _extract_function_args(self, func_node: ast.FunctionDef) -> List[Dict]:
        """Extract function arguments with type hints."""
        args = []
        
        # Regular arguments
        for i, arg in enumerate(func_node.args.args):
            arg_info = {
                'name': arg.arg,
                'type': self._extract_type_annotation(arg.annotation),
                'default': None,
                'is_self': i == 0 and arg.arg == 'self',
                'is_cls': i == 0 and arg.arg == 'cls'
            }
            args.append(arg_info)
        
        # Default values
        defaults = func_node.args.defaults
        if defaults:
            # Defaults apply to the last len(defaults) arguments
            for i, default in enumerate(defaults):
                arg_index = len(args) - len(defaults) + i
                if arg_index >= 0:
                    args[arg_index]['default'] = ast.unparse(default)
        
        return args
    
    def _extract_return_annotation(self, func_node: ast.FunctionDef) -> Optional[str]:
        """Extract return type annotation."""
        if func_node.returns:
            return self._extract_type_annotation(func_node.returns)
        return None
    
    def _extract_type_annotation(self, annotation) -> Optional[str]:
        """Extract type annotation as string."""
        if annotation:
            try:
                return ast.unparse(annotation)
            except:
                return str(annotation)
        return None
    
    def _extract_constants(self, tree: ast.AST) -> List[Dict]:
        """Extract module-level constants."""
        constants = []
        
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        constants.append({
                            'name': target.id,
                            'value': ast.unparse(node.value),
                            'line_number': node.lineno
                        })
        
        return constants
    
    def _identify_dependencies(self, tree: ast.AST) -> Dict[str, List[str]]:
        """Identify external dependencies that need mocking."""
        dependencies = {
            'external_calls': [],
            'file_operations': [],
            'network_calls': [],
            'datetime_calls': []
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                call_str = ast.unparse(node.func)
                
                # Check against mock patterns
                for pattern in self.config['mock_patterns']:
                    if pattern.rstrip('(').rstrip('.') in call_str:
                        if 'open' in call_str or 'Path' in call_str:
                            dependencies['file_operations'].append(call_str)
                        elif any(net in call_str for net in ['requests', 'urllib', 'download', 'fetch']):
                            dependencies['network_calls'].append(call_str)
                        elif 'datetime' in call_str:
                            dependencies['datetime_calls'].append(call_str)
                        else:
                            dependencies['external_calls'].append(call_str)
        
        return dependencies
    
    def generate_test_file(self, module_path: str, output_path: Optional[str] = None) -> str:
        """
        Generate a complete test file for the given module.
        
        Args:
            module_path: Path to the module to test
            output_path: Optional output path for the test file
            
        Returns:
            Path to the generated test file
        """
        analysis = self.analyze_module(module_path)
        
        if output_path is None:
            module_name = analysis['module_name']
            output_path = f"test_{module_name}.py"
        
        test_content = self._generate_test_content(analysis)
        
        with open(output_path, 'w') as f:
            f.write(test_content)
        
        logger.info(f"Generated test file: {output_path}")
        return output_path
    
    def _generate_test_content(self, analysis: Dict) -> str:
        """Generate the complete test file content."""
        parts = []
        
        # File header and imports
        parts.append(self._generate_header(analysis))
        parts.append(self._generate_imports(analysis))
        
        # Test fixtures
        if self.config['generate_fixtures']:
            parts.append(self._generate_fixtures(analysis))
        
        # Test classes for classes in the module
        for class_info in analysis['classes']:
            parts.append(self._generate_class_tests(class_info, analysis))
        
        # Test functions for standalone functions
        if analysis['functions']:
            parts.append(self._generate_function_tests(analysis['functions'], analysis))
        
        return '\n\n'.join(filter(None, parts))
    
    def _generate_header(self, analysis: Dict) -> str:
        """Generate file header with docstring."""
        module_name = analysis['module_name']
        return f'"""Tests for {module_name} module."""'
    
    def _generate_imports(self, analysis: Dict) -> str:
        """Generate import statements for the test file."""
        imports = [
            "import pytest",
            "from unittest.mock import Mock, patch, MagicMock",
            "from io import BytesIO",
            "from pathlib import Path",
            "import tempfile",
            "import pandas as pd",
            ""
        ]
        
        # Import the module being tested
        module_name = analysis['module_name']
        
        # Import classes
        if analysis['classes']:
            class_names = [cls['name'] for cls in analysis['classes']]
            imports.append(f"from {module_name} import {', '.join(class_names)}")
        
        # Import functions
        if analysis['functions']:
            function_names = [func['name'] for func in analysis['functions'] 
                            if not func['is_private']]
            if function_names:
                imports.append(f"from {module_name} import {', '.join(function_names)}")
        
        return '\n'.join(imports)
    
    def _generate_fixtures(self, analysis: Dict) -> str:
        """Generate pytest fixtures."""
        fixtures = []
        
        # Common fixtures based on dependencies
        if any('pandas' in imp['name'] for imp in analysis['imports']):
            fixtures.append('''@pytest.fixture
def sample_dataframe():
    """Sample DataFrame for testing."""
    return pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=12, freq='MS'),
        'value': range(100, 112)
    })''')
        
        if analysis['dependencies']['file_operations']:
            fixtures.append('''@pytest.fixture
def temp_file():
    """Temporary file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("test content")
        temp_path = f.name
    yield temp_path
    Path(temp_path).unlink(missing_ok=True)''')
        
        if analysis['dependencies']['network_calls']:
            fixtures.append('''@pytest.fixture
def mock_response():
    """Mock HTTP response for testing."""
    mock = Mock()
    mock.status_code = 200
    mock.content = b"test content"
    return mock''')
        
        return '\n\n'.join(fixtures) if fixtures else ""
    
    def _generate_class_tests(self, class_info: Dict, analysis: Dict) -> str:
        """Generate test class for a source class."""
        class_name = class_info['name']
        test_class_name = f"{self.config['test_class_prefix']}{class_name}"
        
        methods = []
        
        # Generate test methods for each method
        for method in class_info['methods']:
            if not method['name'].startswith('__'):  # Skip magic methods
                test_method = self._generate_method_test(method, class_info, analysis)
                methods.append(test_method)
        
        class_content = f'''class {test_class_name}:
    """Test cases for the {class_name} class."""
    
{textwrap.indent(chr(10).join(methods), "    ")}'''
        
        return class_content
    
    def _generate_method_test(self, method: Dict, class_info: Dict, analysis: Dict) -> str:
        """Generate test method for a class method."""
        method_name = method['name']
        test_name = f"test_{method_name}"
        
        # Generate test body based on method characteristics
        test_body = []
        
        # Setup
        class_name = class_info['name']
        if method_name == '__init__':
            test_body.append(f"# Test {class_name} initialization")
            test_body.append(f"instance = {class_name}()")
            test_body.append("assert instance is not None")
        else:
            test_body.append(f"# Test {method_name} method")
            test_body.append(f"instance = {class_name}()")
            
            # Generate method call
            args = [arg['name'] for arg in method['args'] 
                   if not arg['is_self'] and not arg['is_cls']]
            
            if args:
                # Create mock arguments
                mock_args = []
                for arg in method['args']:
                    if not arg['is_self'] and not arg['is_cls']:
                        mock_args.append(self._generate_mock_arg(arg))
                
                call_args = ', '.join(mock_args)
                test_body.append(f"result = instance.{method_name}({call_args})")
            else:
                test_body.append(f"result = instance.{method_name}()")
            
            # Add basic assertions
            if method['returns']:
                test_body.append("assert result is not None")
            
            test_body.append("# TODO: Add specific assertions based on expected behavior")
        
        # Add edge case tests if configured
        if self.config['include_edge_cases']:
            test_body.extend(self._generate_edge_cases(method))
        
        test_content = f'''def {test_name}(self):
    """Test {method_name} method."""
    {chr(10).join(['    ' + line for line in test_body])}'''
        
        return test_content
    
    def _generate_function_tests(self, functions: List[Dict], analysis: Dict) -> str:
        """Generate tests for standalone functions."""
        test_functions = []
        
        for func in functions:
            if not func['is_private']:  # Only test public functions
                test_func = self._generate_function_test(func, analysis)
                test_functions.append(test_func)
        
        return '\n\n'.join(test_functions)
    
    def _generate_function_test(self, func: Dict, analysis: Dict) -> str:
        """Generate test for a standalone function."""
        func_name = func['name']
        test_name = f"test_{func_name}"
        
        test_body = []
        test_body.append(f"\"\"\"Test {func_name} function.\"\"\"")
        test_body.append(f"# Test {func_name} function")
        
        # Generate function call with better mock arguments
        args = [arg for arg in func['args'] if not arg['is_self'] and not arg['is_cls']]
        
        if args:
            # Create realistic mock arguments based on function name and context
            mock_args = [self._generate_realistic_mock_arg(arg, func_name) for arg in args]
            call_args = ', '.join(mock_args)
            test_body.append(f"result = {func_name}({call_args})")
        else:
            test_body.append(f"result = {func_name}()")
        
        # Add assertions based on return type
        if func['returns']:
            return_type = func['returns']
            if 'DataFrame' in return_type:
                test_body.append("assert isinstance(result, pd.DataFrame)")
            elif 'Tuple' in return_type:
                test_body.append("assert isinstance(result, tuple)")
            elif 'List' in return_type:
                test_body.append("assert isinstance(result, list)")
            elif 'bool' in return_type:
                test_body.append("assert isinstance(result, bool)")
            else:
                test_body.append("assert result is not None")
        
        test_body.append("# TODO: Add specific assertions based on expected behavior")
        
        # Add edge case tests
        if self.config['include_edge_cases']:
            test_body.extend(self._generate_edge_cases(func))
        
        test_content = f'''def {test_name}():
{chr(10).join(['    ' + line for line in test_body])}'''
        
        return test_content
    
    def _generate_realistic_mock_arg(self, arg: Dict, func_name: str) -> str:
        """Generate realistic mock arguments based on function context."""
        arg_name = arg['name']
        arg_type = arg['type']
        
        if arg['default'] and arg['default'] != 'None':
            return arg['default']
        
        # Context-aware mock generation
        if 'DataFrame' in str(arg_type):
            if 'date' in func_name.lower() or 'filter' in func_name.lower():
                return '''pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=5, freq='MS'),
        'food_price_index': [100.0, 101.5, 102.0, 103.2, 104.1],
        'meat': [95.0, 96.2, 97.1, 98.0, 99.3]
    })'''
            else:
                return 'pd.DataFrame({"col": [1, 2, 3]})'
        
        # Date-related arguments
        if 'date' in arg_name.lower():
            if 'start' in arg_name.lower():
                return "'2020-01-01'"
            elif 'end' in arg_name.lower():
                return "'2020-12-31'"
            else:
                return "'2020-06-01'"
        
        # Column name arguments
        if 'column' in arg_name.lower() or arg_name in ['date_column']:
            return "'date'"
        
        # List arguments
        if 'List' in str(arg_type):
            if 'indices' in arg_name.lower():
                return "['food_price_index', 'meat']"
            elif 'column' in arg_name.lower():
                return "['date', 'food_price_index', 'meat']"
            else:
                return '[1, 2, 3]'
        
        # Use the original method for other types
        return self._generate_mock_arg(arg)
    
    def _generate_mock_arg(self, arg: Dict) -> str:
        """Generate a mock argument based on type hints."""
        arg_name = arg['name']
        arg_type = arg['type']
        
        if arg['default'] and arg['default'] != 'None':
            return arg['default']
        
        # Generate based on type hint
        if arg_type:
            if 'str' in arg_type:
                return f'"test_{arg_name}"'
            elif 'int' in arg_type:
                return '42'
            elif 'float' in arg_type:
                return '3.14'
            elif 'bool' in arg_type:
                return 'True'
            elif 'List' in arg_type or 'list' in arg_type:
                return '[1, 2, 3]'
            elif 'Dict' in arg_type or 'dict' in arg_type:
                return '{"key": "value"}'
            elif 'DataFrame' in arg_type:
                return 'pd.DataFrame({"col": [1, 2, 3]})'
            elif 'Path' in arg_type:
                return 'Path("test_path")'
        
        # Default mock
        return f'Mock(name="{arg_name}")'
    
    def _generate_edge_cases(self, func_or_method: Dict) -> List[str]:
        """Generate edge case tests."""
        edge_cases = []
        
        # Common edge cases
        edge_cases.append("")
        edge_cases.append("# Edge cases")
        edge_cases.append("# TODO: Test with None inputs")
        edge_cases.append("# TODO: Test with empty inputs")
        edge_cases.append("# TODO: Test exception handling")
        
        return edge_cases


def main():
    """Main CLI interface for the test generator."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate tests for Python modules')
    parser.add_argument('module', help='Path to the Python module to analyze')
    parser.add_argument('-o', '--output', help='Output path for the test file')
    parser.add_argument('-c', '--config', help='Path to configuration file')
    parser.add_argument('-a', '--analyze-only', action='store_true', 
                       help='Only analyze the module, don\'t generate tests')
    
    args = parser.parse_args()
    
    generator = TestGenerator(config_path=args.config)
    
    if args.analyze_only:
        analysis = generator.analyze_module(args.module)
        print(json.dumps(analysis, indent=2))
    else:
        output_path = generator.generate_test_file(args.module, args.output)
        print(f"Generated test file: {output_path}")


if __name__ == '__main__':
    main()