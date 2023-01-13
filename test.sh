/dellfsqd2/ST_OCEAN/USER/hankai/software/miniconda/envs/st-pipe/bin/pyscenic grn resource/WT_smes_cell_norm.csv resource/TFs.txt -o adj.csv --num_workers 6


#/dellfsqd2/ST_OCEAN/USER/hankai/software/miniconda/envs/st-pipe/bin/pyscenic ctx adj.csv \
#    /dellfsqd2/ST_OCEAN/USER/hankai/software/SpatialTranscript/scenic/cistarget_databases.plarian/planrian.regions_vs_motifs.rankings.feather \
#    --annotations_fname /dellfsqd2/ST_OCEAN/USER/hankai/software/SpatialTranscript/scenic/cistarget_databases.plarian/motifs-v9-nr.planarian-m0.001-o0.0.tbl \
#    --expression_mtx_fname ./resource/WT_smes_cell_norm.csv \
#    --output reg.prune.csv \
#    --mask_dropouts \
#    --num_workers 24
#
#
#/dellfsqd2/ST_OCEAN/USER/hankai/software/miniconda/envs/st-pipe/bin/pyscenic aucell ./resource/WT_smes_cell_norm.csv reg.prune.csv --output test.loom --num_workers 24
