import torch
import argparse
from pathlib import Path
from pprint import pprint

parser = argparse.ArgumentParser()
parser.add_argument('pthfile',help='path to pth file')
args = parser.parse_args()

pthdict = torch.load(args.pthfile)

# pprint(pthdict)
pprint(pthdict.keys())
# print(pthdict['scheduler'])
# print(len(pthdict['scheduler']['base_lrs']))
# print(len(pthdict['scheduler']['_last_lr']))

del pthdict['scheduler']
del pthdict['iteration']
del pthdict['optimizer']
pprint(pthdict.keys())
# pthdict['scheduler']['milestones'] = ('60100','120000')
# print(pthdict['scheduler'])

og_path = Path(args.pthfile)
new_path = og_path.parent / '{}{}{}'.format(og_path.stem,'_wts-only',og_path.suffix) 
print('Writing new state dict to {}'.format(new_path))
torch.save( pthdict, str(new_path) )