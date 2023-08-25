from typing import List, Optional, Set, Tuple, Dict, Union
from cxxheaderparser.parserstate import ClassBlockState, State
import pcpp
import cxxheaderparser
from cxxheaderparser.visitor import CxxVisitor
from cxxheaderparser import types
from cxxheaderparser.simple import parse_string, ParsedData, NamespaceScope, ClassScope
from pathlib import Path
import json

from header import HeaderFile, sort_headers, filter_includes


class Submodule():
  def __init__(self, name: str, include_path: Path, submodule_file_path: Path):
    self.name = name
    self.include_path = include_path
    self.submodule_file_path = submodule_file_path
    self.config_path = Path('config') /  (name + '.json')
    with open(self.config_path, 'r') as config_file:
      self.config = json.load(config_file)

    assert self.include_path.exists(), f'Submodule path {self.include_path} not found'


  def generate(self) -> None:
    headers = []
    for include_file in self.include_path.iterdir():
      if not include_file.name.endswith('.h') and not include_file.name.endswith('.hpp'):
        continue
      if self.header_should_be_ignored(include_file.name):
        continue
      header = HeaderFile(include_file, self)
      headers.append(header)
    # Sort by dependency level so that generation is in correct order
    headers = sort_headers(headers)

    header_code = []
    includes = []
    for header in headers:
      header_code.append(header.binding_code)
      includes.extend(header.includes)
    includes_set = filter_includes(set(includes))

    body = f'py::module_ submodule = m.def_submodule("{self.name}");\n' + '\n'.join(header_code)
    includes_strs = [f'#include {include}' for include in includes_set]
    includes_str = '\n'.join(includes_strs)
    format_str = f'''
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <map>
{includes_str}

namespace py = pybind11;


void {self.generation_function_name()}(py::module_ &m) {{
  {body}
}}
'''
    with open(self.submodule_file_path, 'w') as submodule_file:
      submodule_file.write(format_str)

  def generation_function_name(self) -> str:
    return f'init_submodule_{self.name}'

  def class_should_be_ignored(self, class_name: str) -> bool:
    if 'ignored_classes' not in self.config:
      return False
    return class_name in self.config['ignored_classes']
  def header_should_be_ignored(self, header_name: str) -> bool:
    if 'ignored_headers' not in self.config:
      return False
    return header_name in self.config['ignored_headers']

  def get_class_config(self, class_name: str) -> Optional[Dict]:
    if 'classes' not in self.config:
      return None
    if class_name not in self.config['classes']:
      return None
    return self.config['classes'][class_name]





def get_submodules(include_path: Path, generate_path: Path) -> List[Submodule]:
  return [
    Submodule('core', Path('/home/sfelton/software/visp-sfelton/modules/core/include/visp3/core'), generate_path / 'core.cpp'),
    # Submodule('visual_features', include_path / 'visual_features', generate_path / 'visual_features.cpp'),
    # Submodule('vs', include_path / 'vs', generate_path / 'vs.cpp')

  ]