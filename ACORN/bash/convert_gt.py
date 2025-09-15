import numpy as np

def read_groundtruth_file(filename):
    with open(filename, 'rb') as f:
        npts = int.from_bytes(f.read(4), byteorder='little')
        ndims = int.from_bytes(f.read(4), byteorder='little')
        data = np.frombuffer(f.read(npts * ndims * 4), dtype=np.uint32).reshape((npts, ndims))
        distances = np.frombuffer(f.read(npts * ndims * 4), dtype=np.float32).reshape((npts, ndims))
    
    print("npts:", npts)
    print("ndims:", ndims)
    print("data:", data)
    print("distances:", distances)
    
    return npts, ndims, data, distances

'''
read groundtruth file saved by UNG index, namely, saved as:
void write_gt_file(const std::string& filename, const std::pair<IdxType, float>* gt, uint32_t num_queries, uint32_t K) {
    std::ofstream fout(filename, std::ios::binary);
    fout.write(reinterpret_cast<const char*>(gt), num_queries * K * sizeof(std::pair<IdxType, float>));
    std::cout << "Ground truth written to " << filename << std::endl;
}
'''

def read_ung_gt(filename, num_queries, K):
    with open(filename, 'rb') as f:
        gt_data = np.frombuffer(f.read(num_queries * K * 8), dtype=np.dtype([('idx', np.uint32), ('dist', np.float32)]))
        gt_data = gt_data.reshape((num_queries, K))
    
    # Extract indices and distances
    indices = gt_data['idx']
    distances = gt_data['dist']
    
    print("indices:", indices)
    print("distances:", distances)
    
    return indices, distances

num_queries = 200
# num_queries = 354

# dataset = 'sift'
dataset = 'words'
scenarios = ['and', 'or']
# scenario = 'NHQ'
# for K in [1, 25, 50, 100]:
K = 10
# for sel in [1, 25, 50, 75]:
for scenario in scenarios:
    gt_path = "../data/" + dataset + "/" + dataset + "_gt_" + scenario + ".bin"
    output_path = "../data/" + dataset + "/" + dataset + "_gt_" + scenario + ".txt"

    gt, _ = read_ung_gt(gt_path, num_queries, K)

    with open(output_path, 'w') as f:
        for i in range(num_queries):
            f.write(' '.join([str(x) for x in gt[i]]) + '\n')
