"""Fail closed on unreviewed native/two-phase contracts in deployed source.

No client imports: controller/serving setup must not require its dependencies.
Method AST fingerprints pin the reviewed transition/budget/masking algorithm;
changing that algorithm requires explicit review rather than a constants check.
"""
import ast
import copy
import hashlib
import sys
from pathlib import Path

CLASSES = {"qwen3.5-27b": "QwenTwoPhaseTokenCompleter",
           "gemma4-31b": "GemmaTwoPhaseTokenCompleter",
           "muse-glimmer-30b": "MuseTwoPhaseTokenCompleter"}
MARKERS = {"qwen3.5-27b": ("<think>", "</think>"),
           "gemma4-31b": ("<|channel>thought\n", "<channel|>"),
           "muse-glimmer-30b": (" to=self<|message|>", " to=user<|message|>")}
# Discover 2444247: 1d662eb4 plus the reviewed native boundary fallback.
# Includes client-side validation of loss masks and behavior logprobs.
# Whitespace/line-number changes do not change these hashes. Update the Discover
# commit, these fingerprints, and the parent gitlink together after reviewing
# changes to these methods. To print replacement hashes from the repo root:
# python -m tpu.swarm.ray_train.thinking_budget.contract --print-method-hashes
METHODS = {
    '_sample': 'cb859a2d12d53fbe1b91c2460e77f4e38bf4e59b1f08536dfcf29ac9d7664505',
    '__call__': '8b5173e56b4fd5e31a5fc291136141d16ff0f4a18e12e0ef5e844bc94e28593c',
    '_two_phase': '5fa022a0fa936abde872d87320136c0b024336465979bbb987ebb43ddc275f9c',
    'sample_group': 'f3975e8e7cad291b77615b6091ac52e454342f0f476b43c3f2a5553626188d34',
    '_native_group': 'd4a1024f95059a1d9f3bbbf3db53bc8cf48e9b0ee1d956304640f368d24319f6',
}
CONSTANTS = {"THINK_CLOSE", "THINK_CLOSE_MARKER", "ANSWER_CUE"}


def method_fingerprint(node):
    node = copy.deepcopy(node)
    # Python 3.12 adds empty type_params to function ASTs. Capture semantics,
    # not the Python minor version used by the builder/serving environment.
    for child in ast.walk(node):
        if 'type_params' in child._fields and not child.type_params:
            child._fields = tuple(field for field in child._fields if field != 'type_params')
    # 3.13 omits empty fields unless explicitly requested; earlier versions
    # always include them and do not accept show_empty.
    options = {"show_empty": True} if sys.version_info >= (3, 13) else {}
    return hashlib.sha256(ast.dump(node, include_attributes=False, **options).encode()).hexdigest()


def _constant(node, names):
    if isinstance(node, ast.Name):
        return names[node.id]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _constant(node.left, names) + _constant(node.right, names)
    if isinstance(node, (ast.Tuple, ast.List)):
        return tuple(_constant(value, names) for value in node.elts)
    if isinstance(node, ast.Dict):
        return {_constant(k, names): _constant(v, names) for k, v in zip(node.keys, node.values)}
    return ast.literal_eval(node)


def _assign(target, value, values):
    if isinstance(target, ast.Name):
        values[target.id] = value
    elif isinstance(target, (ast.Tuple, ast.List)) and len(target.elts) == len(value):
        for child, item in zip(target.elts, value):
            _assign(child, item, values)
    else:
        raise ValueError("unsupported contract assignment")


def _names(target):
    return {n.id for n in ast.walk(target) if isinstance(n, ast.Name)}


def _constants(nodes, keys, inherited=None):
    values = dict(inherited or {})
    for node in nodes:
        targets = node.targets if isinstance(node, ast.Assign) else (
            [node.target] if isinstance(node, ast.AnnAssign) and node.value is not None else [])
        if any(_names(t) & keys for t in targets):
            value = _constant(node.value, values)
            for target in targets:
                _assign(target, value, values)
    return values


def check(source, model=None, require_client=True):
    selected = [model] if model is not None else list(CLASSES)
    try:
        if any(name not in CLASSES for name in selected):
            raise ValueError(f"unsupported model: {model}")
        api = ast.parse(Path(__file__).with_name('thinking_budget_api.py').read_text())
        constants = _constants(api.body, {'ANSWER_CUE', 'FORMATS', 'CLOSE_MARKERS'})
        formats, markers = constants['FORMATS'], constants['CLOSE_MARKERS']
        for name in selected:
            if formats[name][:2] != MARKERS[name]:
                raise ValueError(f"unreviewed native open/end markers for {name}")
        if not require_client:
            return True  # inference-only source need not contain Discover
        path = Path(source) / 'third_party/discover/ttt_discover/tinker_utils/completers.py'
        tree = ast.parse(path.read_text())
        classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
        relevant = {CLASSES[n] for n in selected} | {'QwenTwoPhaseTokenCompleter'}
        # Reject post-class monkeypatches, including annotated/tuple targets.
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.ctx, (ast.Store, ast.Del)):
                if isinstance(node.value, ast.Name) and node.value.id in relevant:
                    raise ValueError(f"post-class mutation of {node.value.id}")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {'setattr', 'delattr'}:
                if node.args and isinstance(node.args[0], ast.Name) and node.args[0].id in relevant:
                    raise ValueError('dynamic completer mutation')
        base_name = 'QwenTwoPhaseTokenCompleter'
        base = classes[base_name]
        if [ast.unparse(n) for n in base.bases] != ['TokenCompleter']:
            raise ValueError('unreviewed Qwen completer inheritance')
        base_values = _constants(base.body, CONSTANTS)
        base_methods = {n.name: n for n in base.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        for name in selected:
            cls = classes[CLASSES[name]]
            if cls is not base and [ast.unparse(n) for n in cls.bases] != [base_name]:
                raise ValueError(f'unreviewed inheritance for {cls.name}')
            values = _constants(cls.body, CONSTANTS, base_values)
            if (values['THINK_CLOSE'] + values['ANSWER_CUE'] != formats[name][2]
                    or values['THINK_CLOSE_MARKER'] != markers[name]):
                raise ValueError(f'native completion contract differs from packaged two-phase {name}')
            methods = dict(base_methods)
            methods.update({n.name: n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))})
            for method, expected in METHODS.items():
                digest = method_fingerprint(methods[method])
                if digest != expected:
                    raise ValueError(f'unreviewed two-phase method {cls.name}.{method}')
        return True
    except (KeyError, TypeError, ValueError, SyntaxError, OSError) as error:
        raise ValueError(f'native completion contract error: {error}') from error


if __name__ == '__main__':
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--print-method-hashes', action='store_true')
    parser.add_argument('--source', type=Path, default=Path.cwd())
    args = parser.parse_args()
    if args.print_method_hashes:
        source = args.source / 'third_party/discover/ttt_discover/tinker_utils/completers.py'
        tree = ast.parse(source.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                   and n.name == 'QwenTwoPhaseTokenCompleter')
        print(json.dumps({n.name: method_fingerprint(n) for n in cls.body
                          if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}, indent=2))
    else:
        check(args.source)
