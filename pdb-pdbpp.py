# coding: utf-8

# pip install pdbpp
# import pdb; pdb.set_trace()

# copy paste below line and run in pdb, then get a new pi function to dir(obj) attributes
def pi(obj, pattern=""):  import pprint as pp; import re; pp.pprint([(f"callable:{callable(getattr(obj,name))}", name, type(getattr(obj,name))) for name in dir(obj) if len(re.findall(pattern, name)) > 0]);