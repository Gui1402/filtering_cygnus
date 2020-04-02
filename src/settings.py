

class FilterSettings:
    # constructor
    def __init__(self):
        self.data_folder = '../data/Runs001'
        self.noise_file = '../data/noise/noise_data.h5'
        self.run_number = 817
        self.sup = 16
        self.inf = -26
        self.roc_grid = 120
        self.output_file_path = '../data/'
        self.output_file_name = 'cluster_eval'
        self.filters = {'bilateral': [[i, j, k] for i in range(3, 11, 2) for j in range(1, 10, 2) for k in range(1, 10, 2)],
                        'nlmeans': [[i, j] for i in range(1, 13, 2) for j in range(1, 13, 2)],
                        'gaussian': [[3], [5], [7]],
                        'mean': [[1], [3], [5], [7]],
                        'median': [[3], [5], [7]],
                        'wiener': [[1], [3], [5]],
                        'bm3D': [[1], [2], [3], [4], [5]],
                        'FCAIDE': [[1], [2], [3], [4], [5]],
                        'DnCNN': [[self.output_file_path + 'snapshot/DnCNN_Net/filtering_cygnus_snapshot_save_DnCNN_run817_2020-03-19-19-19-14_model_40.h5'],
                                  [self.output_file_path + 'snapshot/DnCNN_Net/model_50.hdf5'],
                                  [self.output_file_path + 'snapshot/BRD_Net/15/model_50.h5'],
                                  [self.output_file_path + 'snapshot/BRD_Net/25/model_50.h5'],
                                  [self.output_file_path + 'snapshot/BRD_Net/50/model_50.h5']]
                        }

        self.best_filters = {'nlmeans': [[11, 7]],
                             'bm3D': [[4]],
                             'gaussian': [[7]],
                             'mean': [[1], [7]],
                             'median': [[5]],
                             'wiener': [[1]],
                             'bilateral': [[9, 1, 9]],
                             'FCAIDE': ['lut']
                            }

        self.threshold_parameters = {'nlmeans': 0.85,
                                     'bm3D': 0.48,
                                     'gaussian': 1.02,
                                     'none': 0.66,
                                     'mean': 1.04,
                                     'median': 1.09,
                                     'wiener': 0.96,
                                     'bilateral': 0.45,
                                     'FCAIDE': 0.78
                                    }