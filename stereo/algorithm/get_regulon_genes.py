import json
import os
import sys

"""
python get_regulon_genes.py hotspot scro
or
python get_regulon_genes.py hotspot scro L2
"""


def get_genes(method, target_reg):
    times=['E14-16h', 'E16-18h', 'L1', 'L2','L3']
    base = '/dellfsqd2/ST_OCEAN/USER/liyao1/spatialGRN/exp/fly3d/regulons'
    
    reg = target_reg+'(+)'

    total_genes = []
    for time in times:
        folder=os.path.join(base, time)
        fn = os.path.join(folder, f'{method}_regulons.json')
        regulons = json.load(open(fn))
        if reg in regulons.keys():
            target_genes = regulons[reg]
            total_genes = total_genes + target_genes

    total_genes = list(set(total_genes))

    with open(f'{target_reg}_{method}.txt', 'w') as f:
        f.writelines('\n'.join(total_genes))


def get_cytoscape(time, method, target_reg):
    base = '/dellfsqd2/ST_OCEAN/USER/liyao1/spatialGRN/exp/fly3d/regulons'
    
    reg = target_reg+'(+)'

    folder=os.path.join(base, time)
    fn = os.path.join(folder, f'{method}_regulons.json')
    regulons = json.load(open(fn))
    if reg in regulons.keys():
        target_genes = regulons[reg]

    total_genes = list(set(target_genes))

    with open(f'{time}_{target_reg}_{method}.txt', 'w') as f:
        f.writelines('\n'.join(total_genes))


def get_genes_time(time, method, target_reg):
    base = '/dellfsqd2/ST_OCEAN/USER/liyao1/spatialGRN/exp/fly3d/regulons'
    
    reg = target_reg+'(+)'

    folder=os.path.join(base, time)
    fn = os.path.join(folder, f'{method}_regulons.json')
    regulons = json.load(open(fn))
    if reg in regulons.keys():
        target_genes = regulons[reg]

    total_genes = list(set(target_genes))

    with open(f'{time}_{target_reg}_{method}.txt', 'w') as f:
        f.writelines('\n'.join(total_genes))


if __name__ == '__main__':
    method=sys.argv[1]
    target_reg=sys.argv[2]
    if len(sys.argv) > 2:
        time=sys.argv[3]
        get_genes_time(time, method, target_reg)
    else:
        get_genes(method, target_reg)

