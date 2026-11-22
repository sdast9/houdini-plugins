"""Expand the source fragments and visibility macros in a Houdini DialogScript."""

from pathlib import Path
import re


_SRC_ROOT = Path(__file__).resolve().parents[1]
_INCLUDE = re.compile(r"^([ \t]*)@INCLUDE\s+([^\s@]+)@\r?\n?$")


def expand_includes(entry_relpath):
    """Read a source-root-relative entry and recursively insert its fragments.

    Includes are relative to the file that contains them. Fragment text is
    copied verbatim, so indentation in the directive serves only readability.
    """
    entry = (entry_relpath,) if isinstance(entry_relpath, (str, Path)) else entry_relpath

    def expand(directory, relpath, stack):
        if relpath.is_absolute() or ".." in relpath.parts:
            raise ValueError(f"DialogScript include must stay under src/: {relpath}")
        path = (directory / relpath).resolve()
        if not path.is_relative_to(_SRC_ROOT):
            raise ValueError(f"DialogScript include escapes src/: {path}")
        if path in stack:
            cycle = " -> ".join(str(p.relative_to(_SRC_ROOT)) for p in (*stack, path))
            raise ValueError(f"DialogScript include cycle: {cycle}")

        source = path.read_bytes().decode("utf-8").replace("\r\n", "\n")
        if stack and source and not source.endswith("\n"):
            raise ValueError(
                f"Included DialogScript fragment must end with a newline: "
                f"{path.relative_to(_SRC_ROOT)}")
        chunks = []
        for number, line in enumerate(source.splitlines(keepends=True), 1):
            match = _INCLUDE.fullmatch(line)
            if match:
                chunks.append(expand(path.parent, Path(match[2]), (*stack, path)))
            elif "@INCLUDE" in line:
                raise ValueError(
                    f"{path.relative_to(_SRC_ROOT)}:{number}: malformed @INCLUDE directive")
            else:
                chunks.append(line)
        return "".join(chunks)

    text = expand(_SRC_ROOT, Path(*entry), ())
    if "@INCLUDE" in text:
        raise ValueError("unexpanded @INCLUDE directive in assembled DialogScript")
    return text


def expand_visibility_macros(text):
    """Expand @SHOW_FOR(...)@ / @SHOW_FOR_DIRECT(...)@ in the DialogScript.

    Houdini conditionals are written as the negative ("hide when ..."), groups
    AND inside braces and OR between them, so "show this parameter for models
    A or B" already needs one !=-per-model. Composite materials double that:
    an isotropic parameter must also show when the volume is a MaterialSum
    whose MATRIX is one of those models, i.e. hide = A and (B or C), which
    expands to two groups. Writing that by hand for every parameter is where
    conditional bugs come from, so the source carries the model list and this
    generates the condition.

        @SHOW_FOR(NeoHookean LinearElasticity)@
            -> visible for those models directly, or as a composite matrix
        @SHOW_FOR_DIRECT(HGOFiber HGODispersion)@
            -> visible only when the volume itself is one of those models
               (fiber families inside a composite have their own parameters)
    """
    def group(parm, tokens):
        return " ".join(f"{parm} != {token}" for token in tokens)

    def direct(match):
        return "{ " + group("materials#_#", match.group(1).split()) + " }"

    def composite(match):
        tokens = match.group(1).split()
        not_listed = group("materials#_#", tokens)
        return (f"{{ {not_listed} materials#_# != MaterialSum }} "
                f"{{ {not_listed} {group('matrix_model#_#', tokens)} }}")

    text = re.sub(r"@SHOW_FOR_DIRECT\(([^)]*)\)@", direct, text)
    text = re.sub(r"@SHOW_FOR\(([^)]*)\)@", composite, text)
    leftover = re.search(r"@SHOW_FOR[A-Z_]*\(", text)
    if leftover:
        raise RuntimeError(
            f"unexpanded visibility macro at offset {leftover.start()}")
    return text
