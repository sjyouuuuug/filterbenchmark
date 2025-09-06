# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

find_package(OpenMP REQUIRED)

add_executable(1-Flat EXCLUDE_FROM_ALL 1-Flat.cpp)
target_link_libraries(1-Flat PRIVATE faiss)

add_executable(2-IVFFlat EXCLUDE_FROM_ALL 2-IVFFlat.cpp)
target_link_libraries(2-IVFFlat PRIVATE faiss)

add_executable(3-IVFPQ EXCLUDE_FROM_ALL 3-IVFPQ.cpp)
target_link_libraries(3-IVFPQ PRIVATE faiss)

add_executable(4-GPU EXCLUDE_FROM_ALL 4-GPU.cpp)
target_link_libraries(4-GPU PRIVATE faiss)

add_executable(5-Multiple-GPUs EXCLUDE_FROM_ALL 5-Multiple-GPUs.cpp)
target_link_libraries(5-Multiple-GPUs PRIVATE faiss)

add_executable(6-HNSW EXCLUDE_FROM_ALL 6-HNSW.cpp)
target_link_libraries(6-HNSW PRIVATE faiss)

add_executable(7-PQFastScan EXCLUDE_FROM_ALL 7-PQFastScan.cpp)
target_link_libraries(7-PQFastScan PRIVATE faiss)

add_executable(8-PQFastScanRefine EXCLUDE_FROM_ALL 8-PQFastScanRefine.cpp)
target_link_libraries(8-PQFastScanRefine PRIVATE faiss)

add_executable(9-RefineComparison EXCLUDE_FROM_ALL 9-RefineComparison.cpp)
target_link_libraries(9-RefineComparison PRIVATE faiss)

add_executable(build_HNSW_index EXCLUDE_FROM_ALL build_HNSW_index.cpp)
target_link_libraries(build_HNSW_index PRIVATE faiss)

add_executable(search_HNSW_index EXCLUDE_FROM_ALL search_HNSW_index.cpp)
target_link_libraries(search_HNSW_index PRIVATE faiss OpenMP::OpenMP_CXX)

add_executable(search_prefilter_HNSW_index EXCLUDE_FROM_ALL search_prefilter_HNSW_index.cpp)
target_link_libraries(search_prefilter_HNSW_index PRIVATE faiss OpenMP::OpenMP_CXX)

add_executable(build_IVFPQ_index EXCLUDE_FROM_ALL build_IVFPQ_index.cpp)
target_link_libraries(build_IVFPQ_index PRIVATE faiss)

add_executable(search_IVFPQ_index EXCLUDE_FROM_ALL search_IVFPQ_index.cpp)
target_link_libraries(search_IVFPQ_index PRIVATE faiss OpenMP::OpenMP_CXX)