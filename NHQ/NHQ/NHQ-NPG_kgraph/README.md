# NHQ-NPG_kgraph

## Compile on Linux

Go to the root directory of NHQ-NPG_kgraph and run the following scripts.    

```shell
cd NHQ-NPG_kgraph/
mkdir build && cd build
cmake ..
make
```

## Build NHQ-NPG_kgraph index
First: 

```shell
cd NHQ-NPG_kgraph/build/tests/
```

Then: 

```shell
./test_dng_index data_file att_file save_graph save_attributetable K L iter S R Range PL B M
```

```shell
./test_dng_index \
../../../../data/arxiv/arxiv_base.fvecs \
../../../../equal_length_experiment/arxiv/label_NHQ_base.txt \ 
../../../../equal_length_experiment/index_files/NHQ_kgraph/arxiv/test_graph \
../../../../equal_length_experiment/index_files/NHQ_kgraph/arxiv/test_attr.txt \
100 \
200 \
10 \
10 \
200 \
20 \
100 \
0.4 \
1.0
```


```bash
./run_dng.sh ../../../../data/arxiv/arxiv_base.fvecs   ../../../../equal_length_experiment/arxiv/label_NHQ_base_header.txt   ../../../../equal_length_experiment/index_files/NHQ_kgraph/arxiv/test_graph.bin   ../../../../equal_length_experiment/index_files/NHQ_kgraph/arxiv/test_attr.txt 100 200 12 10 200 20 100 0.4 1.0
```

```bash
./build/tests/run_dng.sh ../../data/arxiv/arxiv_base.fvecs ../../equal_length_experiment/arxiv/label_NHQ_base.txt ../../equal_length_experiment/index_files/NHQ_kgraph/arxiv/K=100_L=200_iter=12_S=10_R=200_RANGE=20_PL=100_B=0.4_M=1/test_graph.bin ../../equal_length_experiment/index_files/NHQ_kgraph/arxiv/K=100_L=200_iter=12_S=10_R=200_RANGE=20_PL=100_B=0.4_M=1/test_attr.txt 100 200 12 10 200 20 100 0.4 1
```

 Meaning of the parameters:    

```
<data_file> is the path to the original object set.
<att_file> is the path to the structured attributes of the original objects.
<save_graph> is the path of the NHQ-NPG_kgraph to be saved.
<save_attributetable> is the path of the attributes codes to be saved.
<K> is the 'K' of kNN graph.
<L> is the parameter controlling the graph quality, larger is more accurate but slower, no smaller than K.
<iter> is the parameter controlling the maximum iteration times, iter usually < 30.
<S> is the parameter contollling the graph quality, larger is more accurate but slower.
<R> is the parameter controlling the graph quality, larger is more accurate but slower.
<RANGE> controls the index size of the graph, the best R is related to the intrinsic dimension of the dataset.
<PL> controls the quality of the NHQ-NPG_kgraph, the larger the better.
<B> controls the quality of the NHQ-NPG_kgraph.
<M> controls the edge selection of NHQ-NPG_kgraph.
```

```shell
./test_dng_index data_file att_file save_graph save_attributetable K L iter S R Range PL B M
```
./test_dng_index ../../sift/sift_base.fvecs ../../sift/label_sift_base.txt ../../sift/graph_ ../../sift_label_label_file 32 64 15 10 10 100 10 1.2 1.2

## Search on NHQ-NPG_kgraph
```shell
./test_dng_optimized_search graph_path attributetable_path data_path query_path query_att_path groundtruth_path
```
./test_dng_optimized_search ../../sift/graph_ ../../sift_label_label_file ../../sift/sift_base.fvecs ../../sift/sift_query.fvecs ../../sift/label_sift_query.txt ../../sift/sift_ground_truth.txt

```shell
./test_dng_optimized_search ../../../../equal_length_experiment/index_files/NHQ_kgraph/arxiv/test_graph.bin ../../../../equal_length_experiment/index_files/NHQ_kgraph/arxiv/test_attr.txt ../../../../data/arxiv/arxiv_base.fvecs ../../../../equal_length_experiment/arxiv/arxiv_query_NHQ.fvecs ../../../../equal_length_experiment/arxiv/label_NHQ_query_header.txt ../../../../equal_length_experiment/arxiv/arxiv_gt_NHQ.txt 
```
 Meaning of the parameters:    

```
<graph_path> is the path of the pre-built NHQ-NPG_kgraph.
<attributetable_path> is the path of the attributes codes.
<data_path> is the path of the original object set.
<query_path> is the path of the query object.
<query_att_path> is the path of the corresponding structured attributes of the query object.
<groundtruth_path> is the path of the groundtruth data.
<output_path> is the path of the output file, which contains the search results.
```
```bash
./test_dng_optimized_search \
../../../../equal_length_experiment/index_files/NHQ_kgraph/arxiv/K=100_L=200_iter=12_S=10_R=100_RANGE=20_PL=100_B=0.4_M=1/graph.bin \
../../../../equal_length_experiment/index_files/NHQ_kgraph/arxiv/K=100_L=200_iter=12_S=10_R=100_RANGE=20_PL=100_B=0.4_M=1/attr.txt \
../../../../data/arxiv/arxiv_base.fvecs \
../../../../equal_length_experiment/arxiv/arxiv_query_NHQ.fvecs \
../../../../equal_length_experiment/arxiv/label_NHQ_query_header.txt \
../../../../equal_length_experiment/arxiv/arxiv_gt_NHQ.txt \
../../../../equal_length_experiment/result/NHQ_kgraph/arxiv/K=100_L=200_iter=12_S=10_R=100_RANGE=20_PL=100_B=0.4_M=1_result.csv
```

```bash
./test_dng_optimized_search \
../../../../equal_length_experiment/index_files/NHQ_kgraph/arxiv/K=100_L=200_iter=12_S=10_R=200_RANGE=20_PL=100_B=0.4_M=1/graph.bin \
../../../../equal_length_experiment/index_files/NHQ_kgraph/arxiv/K=100_L=200_iter=12_S=10_R=200_RANGE=20_PL=100_B=0.4_M=1/attr.txt \
../../../../data/arxiv/arxiv_base.fvecs \
../../../../equal_length_experiment/arxiv/arxiv_query_NHQ.fvecs \
../../../../equal_length_experiment/arxiv/label_NHQ_query_header.txt \
../../../../equal_length_experiment/arxiv/arxiv_gt_NHQ.txt \
../../../../equal_length_experiment/result/NHQ_kgraph/arxiv/K=100_L=200_iter=12_S=10_R=200_RANGE=20_PL=100_B=0.4_M=1_result.csv
```