"""Check the Python sources and callback references for each HDA runtime."""

import argparse
import ast
import builtins
from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import re
import sys

if __package__:
    from .section_build import manifest_entries
else:
    from section_build import manifest_entries


SRC_ROOT = Path(__file__).resolve().parents[1]
HDAS = ("msh_reader", "polyfem", "readpvd", "write_msh")
_AMBIENT = set(vars(builtins)) | {
    "__annotations__", "__builtins__", "__cached__", "__doc__", "__file__",
    "__loader__", "__name__", "__package__", "__spec__", "kwargs",
}
_CALLBACK_PREFIX = r"(?:\bhou\s*\.\s*phm|\bhdaModule)\s*\(\s*\)\s*\.\s*"
_CALLBACK = re.compile(_CALLBACK_PREFIX + r"(\w+)")
# PolyFEM's legacy importer constructs a table, then converts its values to
# frozensets under the same name. Match only those two unchanged statements.
_LEGACY_IMPORT_KEYS = (
    "467ab59db3bd042b724bc2d0079a6205890bdb7fd6df91ad5623dae32dc3d0b7",
    "313dd6d1f5370d30e302ec6970cfd8c19a0241183b373abd27205e7d0f396a89",
)


@dataclass(frozen=True)
class CheckError:
    asset: str
    rule: str
    path: str
    line: int
    name: str
    detail: str

    def __str__(self):
        return f"{self.asset}: {self.path}:{self.line}: {self.rule}: {self.detail}"


@dataclass(frozen=True)
class SourcePart:
    path: str
    text: str


@dataclass
class _Scope:
    kind: str
    parent: "_Scope | None" = None
    eager: bool = True
    locals: set = field(default_factory=set)
    bound: set = field(default_factory=set)
    globals: set = field(default_factory=set)
    nonlocals: set = field(default_factory=set)

    def ancestors(self):
        scope = self.parent
        while scope is not None:
            yield scope
            scope = scope.parent


class _Directives(ast.NodeVisitor):
    """Find declarations for a block without entering its nested scopes."""

    def __init__(self, scope, body):
        self.scope = scope
        for stmt in body:
            self.visit(stmt)

    def visit_Global(self, node):
        self.scope.globals.update(node.names)

    def visit_Nonlocal(self, node):
        self.scope.nonlocals.update(node.names)

    def visit_FunctionDef(self, node):
        pass

    visit_AsyncFunctionDef = visit_ClassDef = visit_FunctionDef


@dataclass(frozen=True)
class _Read:
    part: SourcePart
    node: ast.Name
    scope: _Scope
    bound_here: bool
    bound_in_enclosing: frozenset
    bound_in_module: bool
    eager: bool


class _Names(ast.NodeVisitor):
    """Collect lexical bindings and snapshot bindings at each eager read."""

    def __init__(self):
        self.module = _Scope("module")
        self.scope = self.module
        self.reads = []
        self.future_annotations = False
        self.deferred_annotation = False

    def scan(self, part, tree):
        self.part = part
        self.future_annotations = any(
            isinstance(stmt, ast.ImportFrom) and stmt.module == "__future__"
            and any(alias.name == "annotations" for alias in stmt.names)
            for stmt in tree.body
        )
        self.visit(tree)

    def _bind(self, name, scope=None, initialized=True):
        scope = self.scope if scope is None else scope
        scope.locals.add(name)
        if initialized:
            scope.bound.add(name)
        if scope.kind == "class" and scope.eager and name in scope.globals:
            self.module.locals.add(name)
            if initialized:
                self.module.bound.add(name)

    def _within(self, scope, nodes):
        previous = self.scope
        self.scope = scope
        for node in nodes:
            self.visit(node)
        self.scope = previous

    def visit_Name(self, node):
        name = node.id
        if isinstance(node.ctx, ast.Load):
            scope = self.scope
            enclosing = frozenset(id(outer) for outer in scope.ancestors()
                                  if name in outer.bound)
            self.reads.append(_Read(self.part, node, scope, name in scope.bound,
                                    enclosing, name in self.module.bound,
                                    scope.eager and not self.deferred_annotation))
        elif isinstance(node.ctx, ast.Store):
            self._bind(name)
        elif self.scope.kind != "module":
            # del is also a local declaration in a function.
            self._bind(name, initialized=False)
            self.scope.bound.discard(name)
        else:
            self.module.bound.discard(name)

    def visit_Import(self, node):
        for alias in node.names:
            if alias.name != "*":
                self._bind(alias.asname or alias.name.split(".")[0])

    visit_ImportFrom = visit_Import

    def visit_Assign(self, node):
        self.visit(node.value)
        for target in node.targets:
            self.visit(target)

    def _annotation(self, node, eager):
        if node is None:
            return
        previous = self.deferred_annotation
        self.deferred_annotation |= not eager or self.future_annotations
        self.visit(node)
        self.deferred_annotation = previous

    def visit_AnnAssign(self, node):
        if node.value is not None:
            self.visit(node.value)
            self.visit(node.target)
        elif isinstance(node.target, ast.Name):
            self._bind(node.target.id, initialized=False)
        else:
            self.visit(node.target)
        self._annotation(node.annotation, self.scope.kind in ("module", "class"))

    def visit_AugAssign(self, node):
        if isinstance(node.target, ast.Name):
            self.visit_Name(ast.copy_location(ast.Name(node.target.id, ast.Load()), node.target))
        else:
            self.visit(node.target)
        self.visit(node.value)
        if isinstance(node.target, ast.Name):
            self._bind(node.target.id)

    def visit_NamedExpr(self, node):
        self.visit(node.value)
        scope = self.scope
        while scope.kind == "comprehension":
            scope = scope.parent
        self._bind(node.target.id, scope)

    def _type_parameters(self, node):
        parameters = getattr(node, "type_params", ())
        scope = _Scope("type_parameters", self.scope, self.scope.eager)
        for parameter in parameters:
            self._bind(parameter.name, scope)
        previous = self.deferred_annotation
        self.deferred_annotation = True
        self._within(scope, parameters)
        self.deferred_annotation = previous
        return scope

    def _function(self, node, body):
        args = node.args
        params = [*args.posonlyargs, *args.args, *args.kwonlyargs]
        if args.vararg:
            params.append(args.vararg)
        if args.kwarg:
            params.append(args.kwarg)
        for value in (*args.defaults, *args.kw_defaults):
            if value is not None:
                self.visit(value)
        outer = self.scope
        if getattr(node, "type_params", ()):
            self.scope = self._type_parameters(node)
        for param in params:
            self._annotation(param.annotation, eager=True)
        self._annotation(getattr(node, "returns", None), eager=True)
        scope = _Scope("function", self.scope, eager=False)
        scope.locals.update(param.arg for param in params)
        _Directives(scope, body)
        self.scope = outer
        self._within(scope, body)

    def visit_FunctionDef(self, node):
        for decorator in node.decorator_list:
            self.visit(decorator)
        self._function(node, node.body)
        self._bind(node.name)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node):
        self._function(node, [node.body])

    def visit_ClassDef(self, node):
        for expr in node.decorator_list:
            self.visit(expr)
        enclosing = self._type_parameters(node) if getattr(node, "type_params", ()) else self.scope
        self._within(enclosing, (*node.bases, *node.keywords))
        scope = _Scope("class", enclosing, enclosing.eager)
        scope.bound.update(("__module__", "__qualname__"))
        scope.locals.update(scope.bound)
        _Directives(scope, node.body)
        self._within(scope, node.body)
        self._bind(node.name)

    def visit_TypeAlias(self, node):
        scope = self._type_parameters(node)
        scope.eager = False
        self._within(scope, [node.value])
        self.visit(node.name)

    def _comprehension(self, node, values, eager):
        outer = self.scope
        first, *rest = node.generators
        self.visit(first.iter)
        # A walrus in a comprehension binds in the surrounding scope, but the
        # body may run zero times, or (in a generator) only after module load.
        before = []
        scope = outer
        while scope is not None:
            before.append((scope, scope.bound.copy()))
            scope = scope.parent
        scope = _Scope("comprehension", outer, outer.eager and eager)
        body = [first.target, *first.ifs, *rest, *values]
        self._within(scope, body)
        for surrounding, bound in before:
            surrounding.bound = bound

    def visit_comprehension(self, node):
        self.visit(node.iter)
        self.visit(node.target)
        for condition in node.ifs:
            self.visit(condition)

    def visit_ListComp(self, node):
        self._comprehension(node, [node.elt], eager=True)

    visit_SetComp = visit_ListComp

    def visit_GeneratorExp(self, node):
        self._comprehension(node, [node.elt], eager=False)

    def visit_DictComp(self, node):
        self._comprehension(node, [node.key, node.value], eager=True)

    def visit_Dict(self, node):
        for key, value in zip(node.keys, node.values):
            if key is not None:
                self.visit(key)
            self.visit(value)

    def _is_main_guard(self, test):
        if not isinstance(test, ast.Compare) or len(test.ops) != 1 or len(test.comparators) != 1:
            return False
        if not isinstance(test.ops[0], ast.Eq):
            return False
        left, right = test.left, test.comparators[0]
        return (
            (isinstance(left, ast.Name) and left.id == "__name__"
             and isinstance(right, ast.Constant) and right.value == "__main__")
            or (isinstance(right, ast.Name) and right.id == "__name__"
                and isinstance(left, ast.Constant) and left.value == "__main__")
        )

    def visit_If(self, node):
        self.visit(node.test)
        if self.scope is self.module and self._is_main_guard(node.test):
            # Inside an HDA, __name__ is never "__main__"; lint the guard body
            # in an isolated child scope so its bindings do not leak into
            # module.locals for other parts.
            guard_scope = _Scope("main_guard", self.module, eager=True)
            self._within(guard_scope, node.body)
            if node.orelse:
                self._within(self.scope, node.orelse)
            return
        self._branches(node.body, node.orelse)

    def visit_IfExp(self, node):
        self.visit(node.test)
        self._branches([node.body], [node.orelse])

    def _branches(self, *branches):
        scope = self.scope
        before = scope.bound.copy()
        after = []
        for branch in branches:
            scope.bound = before.copy()
            self._within(scope, branch)
            after.append(scope.bound)
        scope.bound = set.intersection(*after)

    def visit_For(self, node):
        self.visit(node.iter)
        self._branches([node.target, *node.body, *node.orelse], node.orelse)

    visit_AsyncFor = visit_For

    def visit_While(self, node):
        self.visit(node.test)
        self._branches([*node.body, *node.orelse], node.orelse)

    def visit_Try(self, node):
        self._branches([*node.body, *node.orelse], *([handler] for handler in node.handlers))
        self._within(self.scope, node.finalbody)

    visit_TryStar = visit_Try

    def visit_ExceptHandler(self, node):
        if node.type is not None:
            self.visit(node.type)
        if node.name:
            self._bind(node.name)
        self._within(self.scope, node.body)
        if node.name:
            self.scope.bound.discard(node.name)

    def visit_Match(self, node):
        self.visit(node.subject)
        self._branches(*([case] for case in node.cases), [])

    def visit_MatchAs(self, node):
        if node.pattern is not None:
            self.visit(node.pattern)
        if node.name:
            self._bind(node.name)

    def visit_MatchStar(self, node):
        if node.name:
            self._bind(node.name)

    def visit_MatchMapping(self, node):
        self.generic_visit(node)
        if node.rest:
            self._bind(node.rest)

    def resolve(self, read):
        """Return (lexically visible, bound when this expression runs)."""
        scope, name = read.scope, read.node.id
        bound_locally = name in scope.locals and name not in scope.nonlocals
        if scope.kind == "class":
            if read.bound_here:
                return True, True
            # Uninitialized class locals fall back to the module globals.
            if bound_locally or name in scope.globals:
                scope = self.module
        elif scope.kind not in ("module", "class"):
            if bound_locally and name not in scope.globals:
                return True, read.bound_here
            if name in scope.globals:
                if bound_locally:
                    return True, read.bound_here
                scope = self.module

        child = scope
        for parent in scope.ancestors():
            if parent is self.module:
                break
            if parent.kind == "class":
                if child.kind == "type_parameters" and id(parent) in read.bound_in_enclosing:
                    return True, True
                if name == "__class__" and scope.kind != "class":
                    return True, True
            elif name in parent.globals:
                break
            elif name in parent.locals and name not in parent.nonlocals:
                initialized = parent.kind == "function" or id(parent) in read.bound_in_enclosing
                return True, initialized
            child = parent

        return (name in self.module.locals or name in _AMBIENT,
                read.bound_in_module or name in _AMBIENT)


def _targets(target):
    if isinstance(target, ast.Name):
        yield target.id
    elif isinstance(target, (ast.Tuple, ast.List)):
        for item in target.elts:
            yield from _targets(item)
    elif isinstance(target, ast.Starred):
        yield from _targets(target.value)


def _definitions(stmt):
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        yield stmt.name
    elif isinstance(stmt, ast.Assign):
        for target in stmt.targets:
            yield from _targets(target)
    elif isinstance(stmt, ast.AnnAssign):
        yield from _targets(stmt.target)


def _legacy_conversion(part, previous, current):
    old_part, old_stmt = previous
    if part.path != "polyfem/runtime/96_import_apply.py" or old_part is not part:
        return False
    return tuple(
        hashlib.sha256(ast.get_source_segment(part.text, stmt).encode()).hexdigest()
        for stmt in (old_stmt, current)
    ) == _LEGACY_IMPORT_KEYS


def _literal_strings(expr):
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        yield expr.value
    elif isinstance(expr, ast.IfExp):
        yield from _literal_strings(expr.body)
        yield from _literal_strings(expr.orelse)


def _callback_references(part, factories, calls):
    for match in _CALLBACK.finditer(part.text):
        yield match[1], part.text.count("\n", 0, match.start(1)) + 1
    if not part.path.endswith(".py"):
        return

    # Collect f-string factories and their calls across the builder sources.
    # Callback names can also come from assignments inside a factory.
    tree = ast.parse(part.text, filename=part.path)
    # Match sibling imports such as `from .helpers import _callback` to their
    # defining source so unrelated functions with the same name stay distinct.
    path = Path(part.path)
    imports = {
        alias.asname or alias.name: (path.with_name(f"{stmt.module}.py").as_posix(), alias.name)
        for stmt in tree.body
        if isinstance(stmt, ast.ImportFrom) and stmt.level == 1
        and stmt.module and "." not in stmt.module
        for alias in stmt.names
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            factory = imports.get(node.func.id, (part.path, node.func.id))
            calls.append((part, node, factory))
        if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                and node.end_lineno == node.lineno):
            for match in _CALLBACK.finditer(node.value):
                yield match[1], node.lineno
    for function in tree.body:
        if not isinstance(function, ast.FunctionDef):
            continue
        arguments = [arg.arg for arg in (*function.args.posonlyargs, *function.args.args)]
        for node in ast.walk(function):
            if not isinstance(node, ast.JoinedStr):
                continue
            for prefix, value in zip(node.values, node.values[1:]):
                if (not isinstance(prefix, ast.Constant) or
                        not re.search(_CALLBACK_PREFIX + "$", prefix.value) or
                        not isinstance(value, ast.FormattedValue) or
                        not isinstance(value.value, ast.Name)):
                    continue
                variable = value.value.id
                for assignment in ast.walk(function):
                    if isinstance(assignment, ast.Assign) and any(
                        variable in _targets(target) for target in assignment.targets
                    ):
                        for name in _literal_strings(assignment.value):
                            yield name, assignment.lineno
                if variable in arguments:
                    factories.setdefault((part.path, function.name), set()).add(
                        (arguments.index(variable), variable))


def _callback_call_references(factories, calls):
    for part, call, factory in calls:
        for index, argument in sorted(factories.get(factory, ())):
            values = [arg.value for arg in call.keywords if arg.arg == argument]
            values.extend(call.args[index:index + 1])
            for expr in values:
                for name in _literal_strings(expr):
                    yield part, name, call.lineno


def check_sources(asset, parts, callback_sources=()):
    """Check ordered SourceParts and callback-bearing sources, without Houdini."""
    errors = []
    names = _Names()
    definitions = {}

    def report(rule, part, line, name, detail):
        errors.append(CheckError(asset, rule, part.path, line, name, detail))

    for part in parts:
        try:
            tree = ast.parse(part.text, filename=part.path)
            compile(tree, part.path, "exec", dont_inherit=True)
        except SyntaxError as error:
            report("syntax", part, error.lineno or 1, "", error.msg)
            continue
        for stmt in tree.body:
            for name in dict.fromkeys(_definitions(stmt)):
                previous = definitions.get(name)
                if previous is None:
                    definitions[name] = (part, stmt)
                elif not (asset == "polyfem" and name == "_IMPORT_KEYS"
                          and _legacy_conversion(part, previous, stmt)):
                    first_part, first_stmt = previous
                    report("duplicate-definition", part, stmt.lineno, name,
                           f"{name!r} was already defined at "
                           f"{first_part.path}:{first_stmt.lineno}")
        names.scan(part, tree)

    for read in names.reads:
        visible, ready = names.resolve(read)
        name = read.node.id
        if not visible:
            report("undefined-name", read.part, read.node.lineno, name,
                   f"{name!r} has no binding in this runtime")
        elif read.eager and not ready:
            report("load-order", read.part, read.node.lineno, name,
                   f"{name!r} is unbound when this expression runs")

    factories, calls, references = {}, [], {}
    for part in callback_sources:
        try:
            for name, line in _callback_references(part, factories, calls):
                references[part.path, name, line] = part
        except SyntaxError as error:
            report("syntax", part, error.lineno or 1, "", error.msg)
    for part, name, line in _callback_call_references(factories, calls):
        references[part.path, name, line] = part
    for (_, name, line), part in references.items():
        if name not in names.module.bound:
            report("missing-callback", part, line, name,
                   f"{name!r} is absent from the runtime namespace")
    return errors


def _manifest_entries(path):
    """Read the manifest's literal PARTS without importing asset code."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for stmt in tree.body:
        if isinstance(stmt, (ast.Assign, ast.AnnAssign)) and "PARTS" in _definitions(stmt):
            return manifest_entries(ast.literal_eval(stmt.value))
    raise ValueError(f"{path.name} must assign a literal PARTS list")


def check_hda(src_root, asset):
    root = Path(src_root)
    directory = root / asset
    try:
        entries = _manifest_entries(directory / "runtime_manifest.py")
        parts = [SourcePart("/".join(path), root.joinpath(*path).read_text(encoding="utf-8"))
                 for _, path, _ in entries]

        callbacks = []
        for path in sorted(directory.rglob("*")):
            relative = path.relative_to(directory)
            if path.is_file() and (
                path.name in {"build.py", "DialogScript", "network.py", "scripts.py",
                              "Tools.shelf", "ViewerStateModule.py"}
                or path.stem in {"OnCreated", "OnLoaded", "OnUpdated"}
                or (relative.parts[0] == "parms" and path.suffix == ".py")
                or (relative.parts[:2] == ("sections", "dialog")
                    and path.suffix in {".ds", ".in"})
            ):
                callbacks.append(SourcePart(path.relative_to(root).as_posix(),
                                            path.read_text(encoding="utf-8")))
    except (OSError, KeyError, TypeError, ValueError, SyntaxError) as error:
        return [CheckError(asset, "source", f"{asset}/", 1, "", str(error))]

    return check_sources(asset, parts, callbacks)


def check_all(src_root=SRC_ROOT):
    return [error for asset in HDAS for error in check_hda(src_root, asset)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-root", type=Path, default=SRC_ROOT)
    args = parser.parse_args()
    errors = check_all(args.src_root)
    for error in errors:
        print(error, file=sys.stderr)
    print(f"Checked {', '.join(HDAS)}: {len(errors)} errors")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
