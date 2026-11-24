"""Translate the desktop CLI's argument definitions into MCP tools."""
import argparse
from dataclasses import dataclass
import math


READ_ONLY = {'check', 'project_inspect', 'models_list', 'contacts_list', 'contacts_self_list', 'face_inspect'}


def scalar_schema(action):
    kind = {int: 'integer', float: 'number'}.get(action.type, 'string')
    spec = {'type': kind}
    if action.choices is not None:
        spec['enum'] = list(action.choices)
    return spec


def argument_schema(action):
    if action.nargs == 0:
        return {'type': 'boolean'}
    spec = scalar_schema(action)
    if isinstance(action.nargs, int):
        spec = {'type': 'array', 'items': spec, 'minItems': action.nargs, 'maxItems': action.nargs}
    elif action.nargs in ('+', '*'):
        spec = {'type': 'array', 'items': spec, 'minItems': int(action.nargs == '+')}
    if isinstance(action, argparse._AppendAction):
        spec = {'type': 'array', 'items': spec, 'minItems': 1}
    return spec


def validate(value, spec, label):
    kind = spec['type']
    allowed = {'boolean': (bool,), 'integer': (int,), 'number': (int, float),
               'string': (str,), 'array': (list,)}[kind]
    if type(value) not in allowed:
        raise ValueError(f'{label} must be {kind}.')
    if kind == 'array':
        if not spec.get('minItems', 0) <= len(value) <= spec.get('maxItems', 10000):
            raise ValueError(f'{label} has an invalid number of values.')
        for item in value:
            validate(item, spec['items'], label)
    if kind == 'string' and ('\0' in value or value.startswith('--')):
        raise ValueError(f'{label} must be a value, without command flags or NUL characters.')
    if kind == 'number' and type(value) is float and not math.isfinite(value):
        raise ValueError(f'{label} must be finite.')
    if 'enum' in spec and value not in spec['enum']:
        raise ValueError(f'{label} must be one of {spec["enum"]}.')


@dataclass(frozen=True)
class Command:
    path: tuple
    description: str
    actions: tuple

    @property
    def name(self):
        return '_'.join(part.replace('-', '_') for part in self.path)

    def tool(self):
        properties = {}
        required = []
        for action in self.actions:
            spec = argument_schema(action)
            if action.help and action.help != argparse.SUPPRESS:
                spec['description'] = action.help
            if action.required:
                required.append(action.dest)
            properties[action.dest] = spec
        return dict(name=self.name, description=self.description + ' Returns a local job ID.',
                    inputSchema=dict(type='object', properties=properties, required=required,
                                     additionalProperties=False),
                    annotations=dict(readOnlyHint=self.name in READ_ONLY,
                                     destructiveHint=self.name not in READ_ONLY, openWorldHint=True))

    def argv(self, arguments):
        schema = self.tool()['inputSchema']
        if unknown := arguments.keys() - schema['properties'].keys():
            raise ValueError(f'Unknown arguments: {", ".join(sorted(unknown))}.')
        if missing := set(schema['required']) - arguments.keys():
            raise ValueError(f'Missing arguments: {", ".join(sorted(missing))}.')
        positional, options = [], []
        for action in self.actions:
            key = action.dest
            if key not in arguments:
                continue
            value = arguments[key]
            validate(value, schema['properties'][key], key)
            if not action.option_strings:
                positional.extend(str(item) for item in (value if isinstance(value, list) else [value]))
            elif isinstance(action, argparse.BooleanOptionalAction):
                options.append(action.option_strings[0 if value else 1])
            elif action.nargs == 0:
                if value:
                    options.append(action.option_strings[0])
            else:
                groups = value if isinstance(action, argparse._AppendAction) else [value]
                for group in groups:
                    options.append(action.option_strings[0])
                    options.extend(str(item) for item in (group if isinstance(group, list) else [group]))
        return [*self.path, *positional, *options, '--json']


def commands(parser, path=(), description=''):
    """Expose the same command set and options as the managed Windows launcher."""
    sub = next((item for item in parser._actions if isinstance(item, argparse._SubParsersAction)), None)
    if sub is None:
        actions = tuple(item for item in parser._actions if item.dest not in {'help', 'json'})
        yield Command(path, description, actions)
        return
    helps = {item.dest: item.help for item in sub._choices_actions}
    for name, child in sub.choices.items():
        label = helps.get(name) or f'{description.rstrip(".")} ({name}).'
        yield from commands(child, (*path, name), label)
