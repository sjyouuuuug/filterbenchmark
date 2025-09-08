# [CAPS: A Practical Partition Index for Filtered Similarity Search](https://LAION1M.org/abs/2308.15014v1)
**Authors**: Gaurav Gupta, Jonah Yi, Benjamin Coleman, Chen Luo, Vihan Lakshman, Anshumali Shrivastava

## Abstract
With the surging popularity of approximate near-neighbor search (ANNS), driven by advances in neural representation learning, the ability to serve queries accompanied by a set of constraints has become an area of intense interest. While the community has recently proposed several algorithms for constrained ANNS, almost all of these methods focus on integration with graph-based indexes, the predominant class of algorithms achieving state-of-the-art performance in latency-recall tradeoffs. In this work, we take a different approach and focus on developing a constrained ANNS algorithm via space partitioning as opposed to graphs. To that end, we introduce Constrained Approximate Partitioned Search (CAPS), an index for ANNS with filters via space partitions that not only retains the benefits of a partition-based algorithm but also outperforms state-of-the-art graph-based constrained search techniques in recall-latency tradeoffs, with only 10% of the index size.


Install (only for Faiss Kmeans clustering)
- Faiss
   ```
   cd ..
   git clone https://github.com/facebookresearch/faiss.git
   cd faiss
   cmake -B build .
   make -C build -j faiss
   make -C build install
   ```
   This will generate the libfaiss.a

- OpenBLAS
  ```
  git clone https://github.com/xianyi/OpenBLAS.git
  make
  ```
  This will generate the libopenblas.a
  

Provide path at INC and LFLAGS in Makefile

- INC=-I faiss -I include/
- LFLAGS=faiss/build/faiss/libfaiss.a OpenBLAS/libopenblas.a -lpthread -lm -ldl -lgfortran -fopenmp


Get Data:
- Download from https://github.com/AshenOn3/NHQ
- To genearate synthetic data: Run generateRandomTokens.py then getGT-filterSearch.py, each time changing the attribute length (default =3) of attributes to generate the synthetic attributes and groundtruth.
 
For your own data
- base vectors and query vectors are stored in .fvecs format
- base and query attributes are stored in .txt files. 
- Example -
```
<num points> <num attributes>
2 outdoor night
1 indoor daytime
3 outdoor night
2 indoor daytime
3 outdoor daytime
```
 
Where "2 outdoor night" is an example of space seperated 3 attributes.

Make sure to have these files in the data folder
```
data/sift/base.fvecs 
data/sift/label_base_3.txt
data/sift/query.fvecs 
data/sift/label_query_3.txt 
data/sift/label_3_hard_groundtruth.ivecs
python getGT-filterSearch.py --data data/sift
```

If using bliss run
```
python3 include/bliss/dataPrepare_constrained.py --data="data/sift"
python3 include/bliss/construct.py --index='sift_epc40_K10_B1024_R1' --hdim=256 --mode=1 --kn=10
make index
./index data/sift/base.fvecs data/sift/label_base_3.txt indices/sift1024blissMode1 1024 bliss 1
make query
./query data/sift/base.fvecs data/sift/label_base_3.txt data/sift/query.fvecs data/sift/label_query_3.txt indices/sift1024blissMode1 data/sift/label_3_hard_groundtruth.ivecs 1024 bliss 1 500
```

If using faiss kmeans run
```
make index
./index data/audio/base.fvecs data/audio/label_base_3.txt indices/audio1024blissMode1 1024 kmeans 1
make query
./query data/audio/base.fvecs data/audio/label_base_3.txt data/audio/query.fvecs data/audio/label_query_3.txt indices/audio1024blissMode1 data/audio/ground_truth_3.txt 1024 kmeans 1 1000
```

```
./index ../equal_length_experiment/arxiv/arxiv_base.fvecs ../equal_length_experiment/arxiv/label_NHQ_base.txt ../equal_length_experiment/index_files/caps/arxiv/nb_1024 1024 kmeans 1

./query ../equal_length_experiment/arxiv/arxiv_base.fvecs ../equal_length_experiment/arxiv/label_NHQ_base.txt ../equal_length_experiment/arxiv/arxiv_query_NHQ.fvecs ../equal_length_experiment/arxiv/label_NHQ_query.txt ../equal_length_experiment/index_files/caps/arxiv/nb_1024 ../equal_length_experiment/arxiv/arxiv_gt_NHQ.txt ../equal_length_experiment/result/caps/arxiv/nb_1024_results.csv 1024 kmeans 1 1000
```

Functionalities: 
- Variable number of attributes
- AND among attributes
- Large number of attributes
