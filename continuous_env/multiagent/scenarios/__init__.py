import importlib.util
import os

def load(name):
    pathname = os.path.join(os.path.dirname(__file__), name + ".py")
    spec = importlib.util.spec_from_file_location(name, pathname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
