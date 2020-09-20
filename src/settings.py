

class FilterSettings:
    # constructor
    def __init__(self):
        self.data_folder = '../data/simulations.json'
        self.noise_file = '../data/noise/noise_data.h5'
        self.preprocessing_params = ('../data/clustering_params_he_30kev.json', ['Digi_He_30_kev'])
        self.run_number = 817
        self.sup = 16
        self.inf = -26
        self.roc_grid = 120
        self.nsamples = 100
        self.output_file_path = '../data/'
        self.output_file_name = 'cygno_adjusted'
        self.filters = {

                        #'bilateral': [[i, j, k] for i in range(3, 19, 2) for j in range(1, 5, 2) for k in range(1, 19, 2)],
                        #'nlmeans': [[i, j] for i in range(1, 25, 3) for j in range(1, 25, 3)],
                        #'nlmeans':[[11, 7]],
                        #'mean': [[3], [7], [9], [11], [13], [17], [21], [23]],
                        #'gaussian': [[3], [7], [9], [11], [13], [17], [21], [23]],
                        #'gaussian': [[11]],
                        #'mean': [[7]],
                        #'median': [[3], [5], [7], [9], [11], [13], [15], [17], [19], [21], [23], [25], [27]],
                        #'wiener': [[1], [2]],
                        #'bm3D': [[1], [2], [3], [4], [5], [6], [7], [8]],
                        'cygno': [],
                        #'tv': [[.1], [.2], [.3], [.4], [.5], [.6], [.7], [.8], [.9]],
                        #'wavelets' : [[None]]
                        }


