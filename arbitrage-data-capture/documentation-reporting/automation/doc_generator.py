"""
Automated Documentation Generator
Generates comprehensive documentation from code, configs, and runtime data
"""

import ast
import os
import re
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import inspect
import importlib.util
from dataclasses import dataclass
from enum import Enum

import jinja2
from docstring_parser import parse as parse_docstring
import sqlparse
from clickhouse_driver import Client as ClickHouseClient


@dataclass
class CodeEntity:
    """Represents a code entity (function, class, module)"""
    name: str
    type: str  # 'function', 'class', 'module'
    description: str
    parameters: List[Dict[str, Any]]
    returns: Optional[str]
    examples: List[str]
    source_file: str
    line_number: int
    complexity: int
    dependencies: List[str]


class DocumentationGenerator:
    """Automated documentation generation from codebase"""
    
    def __init__(self, project_root: str, output_dir: str):
        self.project_root = Path(project_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.template_loader = jinja2.FileSystemLoader(
            searchpath=str(self.output_dir.parent / 'templates')
        )
        self.template_env = jinja2.Environment(loader=self.template_loader)
        
        self.code_entities = []
        self.database_schema = {}
        self.api_endpoints = []
        self.configuration = {}
        
    def generate_all_documentation(self):
        """Generate complete documentation suite"""
        print("Starting automated documentation generation...")
        
        # Scan and analyze codebase
        self.scan_python_code()
        self.scan_rust_code()
        self.extract_database_schema()
        self.extract_api_documentation()
        self.extract_configuration()
        
        # Generate documentation files
        self.generate_code_documentation()
        self.generate_api_documentation()
        self.generate_database_documentation()
        self.generate_configuration_documentation()
        self.generate_deployment_documentation()
        self.generate_troubleshooting_guide()
        
        # Generate index and navigation
        self.generate_index()
        
        print(f"Documentation generated successfully in {self.output_dir}")
        
    def scan_python_code(self):
        """Scan Python files and extract documentation"""
        for py_file in self.project_root.rglob("*.py"):
            if 'venv' in str(py_file) or '__pycache__' in str(py_file):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    source = f.read()
                    
                tree = ast.parse(source)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        entity = self._extract_function_info(node, py_file, source)
                        self.code_entities.append(entity)
                    elif isinstance(node, ast.ClassDef):
                        entity = self._extract_class_info(node, py_file, source)
                        self.code_entities.append(entity)
                        
            except Exception as e:
                print(f"Error processing {py_file}: {e}")
                
    def _extract_function_info(self, node: ast.FunctionDef, file_path: Path, source: str) -> CodeEntity:
        """Extract function documentation and metadata"""
        docstring = ast.get_docstring(node) or ""
        parsed_doc = parse_docstring(docstring) if docstring else None
        
        parameters = []
        for arg in node.args.args:
            param_info = {
                'name': arg.arg,
                'type': self._get_type_annotation(arg.annotation),
                'description': ''
            }
            
            if parsed_doc:
                for param in parsed_doc.params:
                    if param.arg_name == arg.arg:
                        param_info['description'] = param.description
                        param_info['type'] = param.type_name or param_info['type']
                        
            parameters.append(param_info)
            
        return CodeEntity(
            name=node.name,
            type='function',
            description=parsed_doc.short_description if parsed_doc else "",
            parameters=parameters,
            returns=parsed_doc.returns.description if parsed_doc and parsed_doc.returns else None,
            examples=self._extract_examples(docstring),
            source_file=str(file_path.relative_to(self.project_root)),
            line_number=node.lineno,
            complexity=self._calculate_complexity(node),
            dependencies=self._extract_dependencies(source)
        )
        
    def _extract_class_info(self, node: ast.ClassDef, file_path: Path, source: str) -> CodeEntity:
        """Extract class documentation and metadata"""
        docstring = ast.get_docstring(node) or ""
        parsed_doc = parse_docstring(docstring) if docstring else None
        
        methods = []
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                methods.append({
                    'name': item.name,
                    'description': ast.get_docstring(item) or ""
                })
                
        return CodeEntity(
            name=node.name,
            type='class',
            description=parsed_doc.short_description if parsed_doc else "",
            parameters=methods,  # Store methods as parameters for classes
            returns=None,
            examples=self._extract_examples(docstring),
            source_file=str(file_path.relative_to(self.project_root)),
            line_number=node.lineno,
            complexity=self._calculate_complexity(node),
            dependencies=self._extract_dependencies(source)
        )
        
    def _get_type_annotation(self, annotation) -> str:
        """Extract type annotation as string"""
        if annotation:
            return ast.unparse(annotation)
        return "Any"
        
    def _extract_examples(self, docstring: str) -> List[str]:
        """Extract code examples from docstring"""
        examples = []
        if not docstring:
            return examples
            
        # Look for code blocks in docstring
        pattern = r'```python(.*?)```'
        matches = re.findall(pattern, docstring, re.DOTALL)
        examples.extend(matches)
        
        # Look for >>> style examples
        pattern = r'>>>(.*?)(?=\n(?!\.\.\.|\s)|\Z)'
        matches = re.findall(pattern, docstring, re.MULTILINE)
        examples.extend(matches)
        
        return examples
        
    def _calculate_complexity(self, node) -> int:
        """Calculate cyclomatic complexity of a code node"""
        complexity = 1
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
                
        return complexity
        
    def _extract_dependencies(self, source: str) -> List[str]:
        """Extract import dependencies from source"""
        dependencies = []
        tree = ast.parse(source)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    dependencies.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    dependencies.append(node.module)
                    
        return list(set(dependencies))
        
    def scan_rust_code(self):
        """Scan Rust files and extract documentation"""
        for rs_file in self.project_root.rglob("*.rs"):
            if 'target' in str(rs_file):
                continue
                
            try:
                with open(rs_file, 'r', encoding='utf-8') as f:
                    source = f.read()
                    
                # Extract Rust doc comments
                doc_pattern = r'///\s*(.*?)$'
                docs = re.findall(doc_pattern, source, re.MULTILINE)
                
                # Extract function signatures
                fn_pattern = r'(?:pub\s+)?fn\s+(\w+)\s*\((.*?)\)\s*(?:->\s*(.*?))?\s*\{'
                functions = re.findall(fn_pattern, source, re.DOTALL)
                
                for fn_name, params, return_type in functions:
                    entity = CodeEntity(
                        name=fn_name,
                        type='function',
                        description=' '.join(docs) if docs else "",
                        parameters=self._parse_rust_params(params),
                        returns=return_type.strip() if return_type else "void",
                        examples=[],
                        source_file=str(rs_file.relative_to(self.project_root)),
                        line_number=0,  # Would need more sophisticated parsing
                        complexity=0,
                        dependencies=self._extract_rust_dependencies(source)
                    )
                    self.code_entities.append(entity)
                    
            except Exception as e:
                print(f"Error processing {rs_file}: {e}")
                
    def _parse_rust_params(self, params_str: str) -> List[Dict[str, Any]]:
        """Parse Rust function parameters"""
        params = []
        if not params_str.strip():
            return params
            
        # Simple parameter parsing (would need more sophisticated parsing for complex types)
        param_list = params_str.split(',')
        for param in param_list:
            param = param.strip()
            if ':' in param:
                name, type_str = param.split(':', 1)
                params.append({
                    'name': name.strip(),
                    'type': type_str.strip(),
                    'description': ''
                })
                
        return params
        
    def _extract_rust_dependencies(self, source: str) -> List[str]:
        """Extract Rust dependencies from source"""
        dependencies = []
        
        # Extract use statements
        use_pattern = r'use\s+([\w:]+)'
        uses = re.findall(use_pattern, source)
        dependencies.extend(uses)
        
        # Extract extern crate statements
        crate_pattern = r'extern\s+crate\s+(\w+)'
        crates = re.findall(crate_pattern, source)
        dependencies.extend(crates)
        
        return list(set(dependencies))
        
    def extract_database_schema(self):
        """Extract database schema from SQL files and running database"""
        # Extract from SQL files
        for sql_file in self.project_root.rglob("*.sql"):
            try:
                with open(sql_file, 'r', encoding='utf-8') as f:
                    sql_content = f.read()
                    
                parsed = sqlparse.parse(sql_content)
                for statement in parsed:
                    if statement.get_type() == 'CREATE':
                        self._extract_table_schema(statement)
                        
            except Exception as e:
                print(f"Error processing {sql_file}: {e}")
                
        # Try to connect to ClickHouse and extract live schema
        try:
            client = ClickHouseClient(host='localhost', port=9000)
            tables = client.execute("SHOW TABLES")
            
            for table in tables:
                table_name = table[0]
                columns = client.execute(f"DESCRIBE TABLE {table_name}")
                
                self.database_schema[table_name] = {
                    'columns': [
                        {
                            'name': col[0],
                            'type': col[1],
                            'default': col[2],
                            'comment': col[3] if len(col) > 3 else ''
                        }
                        for col in columns
                    ]
                }
        except Exception as e:
            print(f"Could not connect to ClickHouse: {e}")
            
    def _extract_table_schema(self, statement):
        """Extract table schema from CREATE TABLE statement"""
        # Simplified extraction - would need more sophisticated parsing
        sql_str = str(statement)
        
        # Extract table name
        table_match = re.search(r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)', sql_str, re.IGNORECASE)
        if not table_match:
            return
            
        table_name = table_match.group(1)
        
        # Extract columns (simplified)
        columns_match = re.search(r'\((.*?)\)', sql_str, re.DOTALL)
        if columns_match:
            columns_str = columns_match.group(1)
            columns = []
            
            for line in columns_str.split(','):
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 2:
                        columns.append({
                            'name': parts[0],
                            'type': parts[1],
                            'constraints': ' '.join(parts[2:])
                        })
                        
            self.database_schema[table_name] = {'columns': columns}
            
    def extract_api_documentation(self):
        """Extract API endpoint documentation"""
        # Look for FastAPI/Flask route definitions
        for py_file in self.project_root.rglob("*.py"):
            if 'api' not in str(py_file).lower():
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    source = f.read()
                    
                # FastAPI routes
                fastapi_pattern = r'@\w+\.(get|post|put|delete|patch)\("(.*?)"\)(.*?)def\s+(\w+)'
                matches = re.findall(fastapi_pattern, source, re.DOTALL)
                
                for method, path, decorators, func_name in matches:
                    self.api_endpoints.append({
                        'method': method.upper(),
                        'path': path,
                        'function': func_name,
                        'file': str(py_file.relative_to(self.project_root))
                    })
                    
            except Exception as e:
                print(f"Error processing API file {py_file}: {e}")
                
    def extract_configuration(self):
        """Extract configuration from YAML/JSON files"""
        # YAML files
        for yaml_file in self.project_root.rglob("*.yaml"):
            try:
                with open(yaml_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    if config:
                        self.configuration[yaml_file.stem] = config
            except Exception as e:
                print(f"Error processing {yaml_file}: {e}")
                
        # JSON files
        for json_file in self.project_root.rglob("*.json"):
            if 'node_modules' in str(json_file):
                continue
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    if config:
                        self.configuration[json_file.stem] = config
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
                
    def generate_code_documentation(self):
        """Generate code documentation files"""
        # Group entities by type and module
        modules = {}
        for entity in self.code_entities:
            module = entity.source_file.split('/')[0] if '/' in entity.source_file else 'root'
            if module not in modules:
                modules[module] = []
            modules[module].append(entity)
            
        # Generate documentation for each module
        for module, entities in modules.items():
            output_file = self.output_dir / f"code_{module}.md"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"# {module.title()} Module Documentation\n\n")
                f.write(f"Generated: {datetime.now().isoformat()}\n\n")
                
                # Group by type
                functions = [e for e in entities if e.type == 'function']
                classes = [e for e in entities if e.type == 'class']
                
                if classes:
                    f.write("## Classes\n\n")
                    for cls in classes:
                        f.write(f"### {cls.name}\n\n")
                        f.write(f"**File:** `{cls.source_file}`\n\n")
                        f.write(f"**Description:** {cls.description}\n\n")
                        
                        if cls.parameters:  # Methods
                            f.write("**Methods:**\n\n")
                            for method in cls.parameters:
                                f.write(f"- `{method['name']}`: {method.get('description', '')}\n")
                            f.write("\n")
                            
                        f.write(f"**Complexity:** {cls.complexity}\n\n")
                        f.write("---\n\n")
                        
                if functions:
                    f.write("## Functions\n\n")
                    for func in functions:
                        f.write(f"### {func.name}\n\n")
                        f.write(f"**File:** `{func.source_file}:{func.line_number}`\n\n")
                        f.write(f"**Description:** {func.description}\n\n")
                        
                        if func.parameters:
                            f.write("**Parameters:**\n\n")
                            for param in func.parameters:
                                f.write(f"- `{param['name']}` ({param['type']}): {param.get('description', '')}\n")
                            f.write("\n")
                            
                        if func.returns:
                            f.write(f"**Returns:** {func.returns}\n\n")
                            
                        if func.examples:
                            f.write("**Examples:**\n\n")
                            for example in func.examples:
                                f.write("```python\n")
                                f.write(example.strip())
                                f.write("\n```\n\n")
                                
                        f.write(f"**Complexity:** {func.complexity}\n\n")
                        f.write("---\n\n")
                        
    def generate_api_documentation(self):
        """Generate API documentation"""
        output_file = self.output_dir / "api_documentation.md"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# API Documentation\n\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            
            # Group endpoints by path prefix
            grouped = {}
            for endpoint in self.api_endpoints:
                prefix = endpoint['path'].split('/')[1] if '/' in endpoint['path'] else 'root'
                if prefix not in grouped:
                    grouped[prefix] = []
                grouped[prefix].append(endpoint)
                
            for prefix, endpoints in grouped.items():
                f.write(f"## {prefix.title()} Endpoints\n\n")
                
                for endpoint in endpoints:
                    f.write(f"### {endpoint['method']} {endpoint['path']}\n\n")
                    f.write(f"**Handler:** `{endpoint['function']}` in `{endpoint['file']}`\n\n")
                    
                    # Generate curl example
                    f.write("**Example Request:**\n\n")
                    f.write("```bash\n")
                    f.write(f"curl -X {endpoint['method']} http://localhost:8000{endpoint['path']}")
                    if endpoint['method'] in ['POST', 'PUT', 'PATCH']:
                        f.write(" \\\n  -H 'Content-Type: application/json' \\\n  -d '{}'")
                    f.write("\n```\n\n")
                    f.write("---\n\n")
                    
    def generate_database_documentation(self):
        """Generate database schema documentation"""
        output_file = self.output_dir / "database_schema.md"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# Database Schema Documentation\n\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            
            for table_name, table_info in self.database_schema.items():
                f.write(f"## Table: {table_name}\n\n")
                
                if 'columns' in table_info:
                    f.write("| Column | Type | Constraints | Description |\n")
                    f.write("|--------|------|-------------|-------------|\n")
                    
                    for col in table_info['columns']:
                        constraints = col.get('constraints', col.get('default', ''))
                        description = col.get('comment', '')
                        f.write(f"| {col['name']} | {col['type']} | {constraints} | {description} |\n")
                        
                f.write("\n")
                
                # Add sample queries
                f.write("### Sample Queries\n\n")
                f.write("```sql\n")
                f.write(f"-- Select recent records\n")
                f.write(f"SELECT * FROM {table_name} \n")
                f.write(f"WHERE timestamp > now() - INTERVAL 1 DAY \n")
                f.write(f"ORDER BY timestamp DESC \n")
                f.write(f"LIMIT 100;\n")
                f.write("```\n\n")
                f.write("---\n\n")
                
    def generate_configuration_documentation(self):
        """Generate configuration documentation"""
        output_file = self.output_dir / "configuration.md"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# Configuration Documentation\n\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            
            for config_name, config_data in self.configuration.items():
                f.write(f"## {config_name}\n\n")
                
                # Convert to YAML for readable display
                f.write("```yaml\n")
                f.write(yaml.dump(config_data, default_flow_style=False))
                f.write("```\n\n")
                
                # Document each configuration option
                f.write("### Configuration Options\n\n")
                self._document_config_options(f, config_data)
                f.write("\n---\n\n")
                
    def _document_config_options(self, file_handle, config, prefix=""):
        """Recursively document configuration options"""
        for key, value in config.items():
            full_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                file_handle.write(f"- **{full_key}**: Configuration section\n")
                self._document_config_options(file_handle, value, full_key)
            else:
                value_type = type(value).__name__
                file_handle.write(f"- **{full_key}** ({value_type}): Current value: `{value}`\n")
                
    def generate_deployment_documentation(self):
        """Generate deployment documentation"""
        output_file = self.output_dir / "deployment.md"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# Deployment Guide\n\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            
            # Docker deployment
            f.write("## Docker Deployment\n\n")
            f.write("### Prerequisites\n\n")
            f.write("- Docker Engine 20.10+\n")
            f.write("- Docker Compose 2.0+\n")
            f.write("- 16GB RAM minimum\n")
            f.write("- 100GB SSD storage\n\n")
            
            f.write("### Quick Start\n\n")
            f.write("```bash\n")
            f.write("# Clone repository\n")
            f.write("git clone https://github.com/your-org/arbitrage-detector.git\n")
            f.write("cd arbitrage-detector\n\n")
            f.write("# Configure environment\n")
            f.write("cp .env.example .env\n")
            f.write("# Edit .env with your configuration\n\n")
            f.write("# Start services\n")
            f.write("docker-compose up -d\n\n")
            f.write("# Check status\n")
            f.write("docker-compose ps\n")
            f.write("docker-compose logs -f\n")
            f.write("```\n\n")
            
            # Kubernetes deployment
            f.write("## Kubernetes Deployment\n\n")
            f.write("### Prerequisites\n\n")
            f.write("- Kubernetes 1.20+\n")
            f.write("- Helm 3.0+\n")
            f.write("- kubectl configured\n\n")
            
            f.write("### Deployment Steps\n\n")
            f.write("```bash\n")
            f.write("# Create namespace\n")
            f.write("kubectl create namespace arbitrage\n\n")
            f.write("# Install with Helm\n")
            f.write("helm install arbitrage ./charts/arbitrage \\\n")
            f.write("  --namespace arbitrage \\\n")
            f.write("  --values values.yaml\n\n")
            f.write("# Check deployment\n")
            f.write("kubectl get pods -n arbitrage\n")
            f.write("kubectl get svc -n arbitrage\n")
            f.write("```\n\n")
            
            # Production considerations
            f.write("## Production Considerations\n\n")
            f.write("### Security\n\n")
            f.write("- Use hardware security modules (HSM) for key management\n")
            f.write("- Enable TLS for all communications\n")
            f.write("- Implement rate limiting and DDoS protection\n")
            f.write("- Regular security audits\n\n")
            
            f.write("### Monitoring\n\n")
            f.write("- Configure Prometheus metrics collection\n")
            f.write("- Set up Grafana dashboards\n")
            f.write("- Configure alerting rules\n")
            f.write("- Enable distributed tracing\n\n")
            
            f.write("### Scaling\n\n")
            f.write("- Horizontal pod autoscaling for Kubernetes\n")
            f.write("- Database read replicas for query performance\n")
            f.write("- Redis cluster for caching\n")
            f.write("- CDN for static assets\n\n")
            
    def generate_troubleshooting_guide(self):
        """Generate troubleshooting documentation"""
        output_file = self.output_dir / "troubleshooting.md"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# Troubleshooting Guide\n\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            
            # Common issues
            issues = [
                {
                    'title': 'High Latency in Arbitrage Detection',
                    'symptoms': [
                        'Execution time > 100ms',
                        'Missed opportunities',
                        'Low success rate'
                    ],
                    'causes': [
                        'Network latency to RPC endpoints',
                        'Database query performance',
                        'Insufficient compute resources'
                    ],
                    'solutions': [
                        'Use local/dedicated RPC nodes',
                        'Optimize database indexes',
                        'Scale up compute instances',
                        'Enable caching for frequently accessed data'
                    ]
                },
                {
                    'title': 'Failed Transactions',
                    'symptoms': [
                        'Transactions reverting',
                        'Gas estimation errors',
                        'Slippage too high'
                    ],
                    'causes': [
                        'Insufficient gas limit',
                        'Front-running by competitors',
                        'Price movement during execution'
                    ],
                    'solutions': [
                        'Increase gas multiplier in config',
                        'Use flashbots for private mempool',
                        'Implement dynamic slippage tolerance',
                        'Add MEV protection strategies'
                    ]
                },
                {
                    'title': 'Database Connection Issues',
                    'symptoms': [
                        'Connection timeouts',
                        'Query failures',
                        'Data inconsistency'
                    ],
                    'causes': [
                        'Network issues',
                        'Database overload',
                        'Connection pool exhaustion'
                    ],
                    'solutions': [
                        'Check network connectivity',
                        'Increase connection pool size',
                        'Optimize query performance',
                        'Implement connection retry logic'
                    ]
                }
            ]
            
            for issue in issues:
                f.write(f"## {issue['title']}\n\n")
                
                f.write("### Symptoms\n")
                for symptom in issue['symptoms']:
                    f.write(f"- {symptom}\n")
                f.write("\n")
                
                f.write("### Possible Causes\n")
                for cause in issue['causes']:
                    f.write(f"- {cause}\n")
                f.write("\n")
                
                f.write("### Solutions\n")
                for i, solution in enumerate(issue['solutions'], 1):
                    f.write(f"{i}. {solution}\n")
                f.write("\n")
                
                f.write("### Diagnostic Commands\n\n")
                f.write("```bash\n")
                f.write("# Check system status\n")
                f.write("docker-compose ps\n")
                f.write("docker-compose logs --tail=100 arbitrage-detector\n\n")
                f.write("# Check metrics\n")
                f.write("curl http://localhost:9090/metrics | grep arbitrage\n\n")
                f.write("# Test database connection\n")
                f.write("docker exec -it clickhouse-server clickhouse-client\n")
                f.write("```\n\n")
                f.write("---\n\n")
                
    def generate_index(self):
        """Generate main index file"""
        output_file = self.output_dir / "index.md"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# Arbitrage Detection Infrastructure Documentation\n\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            f.write("## Documentation Overview\n\n")
            
            # Statistics
            f.write("### Project Statistics\n\n")
            f.write(f"- **Total Code Entities:** {len(self.code_entities)}\n")
            f.write(f"- **Database Tables:** {len(self.database_schema)}\n")
            f.write(f"- **API Endpoints:** {len(self.api_endpoints)}\n")
            f.write(f"- **Configuration Files:** {len(self.configuration)}\n\n")
            
            # Navigation
            f.write("### Documentation Sections\n\n")
            
            # List all generated documentation files
            for doc_file in sorted(self.output_dir.glob("*.md")):
                if doc_file.name != "index.md":
                    title = doc_file.stem.replace('_', ' ').title()
                    f.write(f"- [{title}](./{doc_file.name})\n")
                    
            f.write("\n## Quick Links\n\n")
            f.write("- [System Architecture](./architecture/system-overview.md)\n")
            f.write("- [API Reference](./api_documentation.md)\n")
            f.write("- [Database Schema](./database_schema.md)\n")
            f.write("- [Deployment Guide](./deployment.md)\n")
            f.write("- [Troubleshooting](./troubleshooting.md)\n")
            f.write("- [Configuration](./configuration.md)\n")


if __name__ == "__main__":
    # Generate documentation
    generator = DocumentationGenerator(
        project_root="/home/kidgordones/0solana/node/arbitrage-data-capture",
        output_dir="/home/kidgordones/0solana/node/arbitrage-data-capture/documentation-reporting/docs/generated"
    )
    
    generator.generate_all_documentation()