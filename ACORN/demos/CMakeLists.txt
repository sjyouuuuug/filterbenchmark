# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

find_package(OpenMP REQUIRED)

add_executable(utils EXCLUDE_FROM_ALL utils.cpp)
target_link_libraries(utils PRIVATE faiss)

add_executable(test_acorn EXCLUDE_FROM_ALL test_acorn.cpp)
target_link_libraries(test_acorn PRIVATE faiss OpenMP::OpenMP_CXX)

add_executable(test_containment EXCLUDE_FROM_ALL test_containment.cpp)
target_link_libraries(test_containment PRIVATE faiss OpenMP::OpenMP_CXX)

add_executable(test_overlap EXCLUDE_FROM_ALL test_overlap.cpp)
target_link_libraries(test_overlap PRIVATE faiss OpenMP::OpenMP_CXX)

add_executable(test_multilabel EXCLUDE_FROM_ALL test_multilabel.cpp)
target_link_libraries(test_multilabel PRIVATE faiss OpenMP::OpenMP_CXX)

add_executable(test_distribution EXCLUDE_FROM_ALL test_distribution.cpp)
target_link_libraries(test_distribution PRIVATE faiss OpenMP::OpenMP_CXX)

add_executable(test_sel EXCLUDE_FROM_ALL test_sel.cpp)
target_link_libraries(test_sel PRIVATE faiss OpenMP::OpenMP_CXX)

add_executable(test_parse EXCLUDE_FROM_ALL test_parse.cpp)
target_link_libraries(test_parse PRIVATE faiss OpenMP::OpenMP_CXX)

add_executable(test_parse_containment EXCLUDE_FROM_ALL test_parse_containment.cpp)
target_link_libraries(test_parse_containment PRIVATE faiss OpenMP::OpenMP_CXX)

add_executable(build_acorn_index EXCLUDE_FROM_ALL build_acorn_index.cpp)
target_link_libraries(build_acorn_index PRIVATE faiss OpenMP::OpenMP_CXX)

add_executable(search_acorn_index EXCLUDE_FROM_ALL search_acorn_index.cpp)
target_link_libraries(search_acorn_index PRIVATE faiss OpenMP::OpenMP_CXX)

add_executable(search_acorn_index_parse EXCLUDE_FROM_ALL search_acorn_index_parse.cpp)
target_link_libraries(search_acorn_index_parse PRIVATE faiss OpenMP::OpenMP_CXX UNG)

add_executable(build_gamma_index EXCLUDE_FROM_ALL build_gamma_index.cpp)
target_link_libraries(build_gamma_index PRIVATE faiss OpenMP::OpenMP_CXX)

add_executable(build_1_index EXCLUDE_FROM_ALL build_1_index.cpp)
target_link_libraries(build_1_index PRIVATE faiss OpenMP::OpenMP_CXX)

# add_executable(parse_file EXCLUDE_FROM_ALL parse_file.cpp)
# target_link_libraries(parse_file PRIVATE faiss OpenMP_CXX)