"""Run 47 (reproducibility, pointer review point 2): list the local notes/wp3 modules a set of scripts imports, transitively.
Static (ast) only; nothing is executed. Usage: python3 import_closure.py inject_loc.py raw_rescore45.py ..."""
import ast, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))


def local_imports(path):
    mods = set()
    for node in ast.walk(ast.parse(open(path).read())):
        names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module] if isinstance(node, ast.ImportFrom) and node.module else []
        mods |= {n.split(".")[0] for n in names if os.path.exists(os.path.join(HERE, n.split(".")[0] + ".py"))}
    return mods


seen, todo = set(), [os.path.basename(p)[:-3] for p in sys.argv[1:]]
while todo:
    m = todo.pop()
    if m in seen: continue
    seen.add(m); todo += sorted(local_imports(os.path.join(HERE, m + ".py")) - seen)
print(" ".join(sorted(f"{m}.py" for m in seen)))
