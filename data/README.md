### txt data format

* 1st line: `N` number of graphs; then the following `N` blocks describe the graphs  
* for each block of text:
  - a line contains `n l`, where `n` is number of nodes in the current graph, and `l` is the graph label
  - following `n` lines: 
    - the `i`th line describes the information of `i`th node (0 based), which starts with `t m`, where `t` is the tag of current node, and `m` is the number of neighbors of current node;
    - following `m` numbers indicate the neighbor indices (starting from 0). 
    - following `d` numbers (if any) indicate the continuous node features (attributes)
