import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('pthfile',help='path to pth file')
args = parser.parse_args()

if args.pthfile.endswith('.pkl'):
    import pickle
    with open(args.pthfile, "rb") as f:
        data = pickle.load(f, encoding="latin1")
elif args.pthfile.endswith('.pth'):
    import torch
    data = torch.load(args.pthfile)

anchorkeys = []
for k in data['model'].keys():
    if 'anchor' in k:
        anchorkeys.append(k)

for k in anchorkeys:
    del data['model'][k]
    print('deleted {}!'.format(k))

if len(anchorkeys) > 0:
    og_path = Path(args.pthfile)
    new_path = og_path.parent / '{}{}{}'.format(og_path.stem,'_anchor-removed',og_path.suffix) 

    print('Writing new state dict to {}'.format(new_path))
    if args.pthfile.endswith('.pkl'):
        with new_path.open('wb') as f:
            pickle.dump(data, f)
    else:
        torch.save( data, str(new_path) )
else:
    print('No anchor values stored in here!')