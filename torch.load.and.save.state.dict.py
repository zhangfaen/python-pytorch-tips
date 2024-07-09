
torch.save(model.half().state_dict(), "pytorch_model.temp.half.oldzip.bin", _use_new_zipfile_serialization=False)

# it is zip file. you can rename it to *.zip and unzip it to check.


[k1 for (k1, v1), (k2, v2) in zip(mydict1.items(), mydict2.items()) if not torch.all(v1 == v2)]

set([v.dtype for k,v in mydict2.items()])

mydict = torch.load("model/pytorch_model.bin")

