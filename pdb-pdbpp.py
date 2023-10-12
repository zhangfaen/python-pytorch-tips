# coding: utf-8

# pip install pdbpp
# import pdb; pdb.set_trace()

# copy paste below line and run in pdb, then get a new pi function to dir(obj) attributes
def pi(obj): import pprint as pp;  pp.pprint([(name,type(getattr(obj,name))) for name in dir(obj)])
def pii(obj, pattern):  import pprint as pp; import re; pp.pprint([(name,type(getattr(obj,name))) for name in dir(obj) if len(re.findall(pattern, name)) > 0])