# coding: utf-8

# pip install pdbpp
# import pdb; pdb.set_trace()

# copy paste below line and run in pdb, then get a new pi function to dir(obj) attributes
# example usage:
# import builtins
# # find all attributes of builtins, whose name starts with 'm'
# pi(builtins, "^m") # output
    # >>> pi(builtins, "^m")
    # [('callable:True', 'map', <class 'type'>),
    #  ('callable:True', 'max', <class 'builtin_function_or_method'>),
    #  ('callable:True', 'memoryview', <class 'type'>),
    #  ('callable:True', 'min', <class 'builtin_function_or_method'>)]
def pi(obj, pattern=""):  import pprint as pp; import re; pp.pprint([(f"callable:{callable(getattr(obj,name))}", name, type(getattr(obj,name))) for name in dir(obj) if not name.startswith("_") and len(re.findall(pattern, name)) > 0]);

