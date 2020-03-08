

class FilterSettings:
    # constructor
    def __init__(self):
        self.data_folder = '../data/Runs001'
        self.noise_file = '../data/noise/noise_data.h5'
        self.run_number = 817
        self.sup = 16
        self.inf = -26
        self.roc_grid = 20
        self.filters = {'bilateral': [[i, j, k] for i in range(3, 11, 2) for j in range(1, 10, 2) for k in range(1, 10, 2)],
                        'nlmeans': [[i, j] for i in range(1, 13, 2) for j in range(1, 13, 2)],
                        'gaussian': [[3], [5], [7]],
                        'mean': [[1], [3], [5], [7]],
                        'median': [[3], [5], [7]],
                        'wiener': [[1], [3], [5]],
                        'bm3D': [[1], [2], [3], [4], [5]],
                        'FCAIDE': ['lut']
                        }
        self.output_file_path = '../data/'
        self.output_file_name = 'all_filters'
