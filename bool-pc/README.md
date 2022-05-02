## Getting started

### Environment
Bool-PC requires the following packages:
`pip install cdt`
`pip install pandas`
`pip install matplotlib`
`pip install networkx`
`pip install pandas`


### Arguments
Run Bool-PC on experimental data as formatted by the included helper tool.
The proper commmand structure is as follows:

`python bool_pc.py --dual [dual] [data] [gold] [pert]`

For reference, all experimental data need to reproduce the results found in
Rast and Lopez, 2022 have been included in the "data" directory. AUPRC can be calculated with the following command:

`python bool_pc.py --dual 1 data/data.tsv data/gold.tsv data/pert.tsv`


### Generation of new dataset
New experimental data can be simulated using the [GeneNetWeaver tool](http://gnw.sourceforge.net/)

1. Add the network structure to GNW.
2. Generate stable gene expression data. Label this data "stable.tsv"
3. Run `python multi_knockout.py` to generate perturbations file.
4. Rename this file with the appropriate name, as required by GNW. 
5. Run GNW with stable and dualkockout experiments.
6. Perturbation file, dual knockout file, and goldstandard network file are
   required for Bool-PC
