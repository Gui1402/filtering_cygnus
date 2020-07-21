

class FilterSettings:
    # constructor
    def __init__(self):
        self.data_folder = '../data/simulations.json'
        self.noise_file = '../data/noise/noise_data.h5'
        self.run_number = 817
        self.sup = 16
        self.inf = -26
        self.roc_grid = 120
        self.output_file_path = '../data/'
        self.output_file_name = 'tv'
        self.filters = {

                        #'bilateral': [[i, j, k] for i in range(3, 19, 2) for j in range(1, 5, 2) for k in range(1, 19, 2)],
                        #'nlmeans': [[i, j] for i in range(1, 25, 3) for j in range(1, 25, 3)],
                        #'nlmeans':[[11, 7]],
                        #'mean': [[3], [7], [9], [11], [13], [17], [21], [23]],
                        #'gaussian': [[3], [7], [9], [11], [13], [17], [21], [23]],
                        #'gaussian': [[11], [13]],
                        #'mean': [[7]],
                        #'median': [[3], [5], [7], [9], [11], [13], [15], [17], [19], [21], [23], [25], [27]],
                        #'wiener': [[1], [2]],
                        #'bm3D': [[1], [2], [3], [4], [5], [6], [7], [8]],
                        #'FCAIDE': [[1], [2], [3], [4], [5]],
                        #'DnCNN': [[self.output_file_path + 'snapshot/DnCNN_Net/filtering_cygnus_snapshot_save_DnCNN_run817_2020-03-19-19-19-14_model_40.h5'],
                        #          [self.output_file_path + 'snapshot/DnCNN_Net/model_50.hdf5'],
                        #          [self.output_file_path + 'snapshot/BRD_Net/15/model_50.h5'],
                        #          [self.output_file_path + 'snapshot/BRD_Net/25/model_50.h5'],
                        #          [self.output_file_path + 'snapshot/BRD_Net/50/model_50.h5']]
                        #'cygno': [],
                        'tv': [[.1], [.2], [.3], [.4], [.5], [.6], [.7], [.8], [.9]]
                        }

