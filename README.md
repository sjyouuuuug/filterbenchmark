# Filtered Approximate Nearest Neighbor Search: A Unified Benchmark and Systematic Experimental Study [Experiment, Analysis & Benchmark]


## UNG-dev & Brute Force

### build

```bash
cd UNG-dev/

mkdir build

cd build

cmake -DCMAKE_BUILD_TYPE=Release ../codes/

make -j
```

### Usage

Each dataset is available in two raw data formats: `dataset_name.fvecs` and `dataset_name.bin`. We offer these options to ensure compatibility with a variety of tools and workflows. If you need to convert the data, please consult the detailed instructions in `UNG-dev/README.md`. Also, refer to https://github.com/YZ-Cai/Unified-Navigating-Graph for detailed environment setup.

We provide a small dataset on github to facilitate testing and experimentation.

***Brute Force***

For containment scenario:

```bash
./build/tools/compute_groundtruth \
    --data_type float --dist_fn L2 \
    --base_bin_file ../data/words/words_base.bin --query_bin_file ../data/words/words_query_and.bin \
    --base_label_file ../data/words/label_base.txt --query_label_file ../data/t/words_query_and.txt \
    --gt_file ../data/words/words_gt_and.bin --scenario containment --K 10 --num_threads 16
```

For overlap scenario:

```bash
./build/tools/compute_groundtruth \
    --data_type float --dist_fn L2 \
    --base_bin_file ../data/words/words_base.bin --query_bin_file ../data/words/words_query_or.bin \
    --base_label_file ../data/words/label_base.txt --query_label_file ../data/words/words_query_or.txt \
    --gt_file ../data/words/words_gt_or.bin --scenario overlap --K 10 --num_threads 16
```

The result will be saved in the specified ground truth files (`words_gt_and.bin` and `words_gt_or.bin`) and the statistics will be printed to the console.

***UNG Parameter Tuning Workflow***

To facilitate parameter tuning for the UNG model, we provide a suite of scripts located in the `bash/` directory. This workflow is designed to be straightforward and can be adapted for other models with minor modifications.

1. ***Step 1: Explore the Parameter Space***

This script systematically traverses the defined parameter space for UNG, building a small index for each parameter combination to enable initial analysis.

```bash
cd bash
python3 ./traverse_param_space.py
```

2. ***Step 2: Perform Subspace Search***

This script executes a hybrid search across the parameter subspace generated in the previous step. You can customize the search scenarios by modifying the `search_in_subspace.py` file.

```bash
cd bash
python3 ./search_in_subspace.py
```

3. ***Step 3: Aggregate Search Results***

This script combines the results from the various scenario searches for each parameter set and calculates the overall performance metrics.

```bash
cd bash
python3 ./combine_search_result.py
```

4. ***Step 4: Select Representative Samples***

From the aggregated results, this script identifies and selects a set of representative samples from each subspace. The final selection is saved to a CSV file for review.

```bash
cd bash
python3 ./select_representative.py
```

5. ***Step 5: Build Representative Graphs***

Using the samples selected in the previous step, this script constructs the final representative graphs.

```bash
cd bash
python3 ./build_representative_graphs.py
```

6. ***Step 6: Run Customized Evaluations***

```bash
python search_basic_exp.py
python search_base_exp.py
python search_NHQ_exp.py
python search_selectivity_exp.py
# ... and other evaluation scripts
```

## Post-filter HNSW/IVFPQ

The post-filter HNSW and post-filter IVFPQ

### build