

class FilterSettings:
    # constructor
    def __init__(self):
        self.data_folder = '../data/simulations.json'
        self.noise_file = '../data/noise/noise_data.h5'
        self.packs = [['Digi_He_30_kev']]
        self.preprocessing_params = ('../data/clustering_params.json', ['Digi_He_30_kev'], ['median', 'cygno', 'gaussian', 'mean'])
        self.run_number = 817
        self.sup = 16
        self.inf = -26
        self.roc_grid = 120
        self.nsamples = 100
        self.output_file_path = '../data/'
        self.output_file_name = 'new_filter_analysis'
        self.filters = {

                        #'bilateral': [[i, j, k] for i in range(3, 19, 2) for j in range(1, 5, 2) for k in range(1, 19, 2)],
                        #'nlmeans': [[i, j] for i in range(1, 25, 3) for j in range(1, 25, 3)],
                        #'nlmeans':[[11, 7]],
                        'mean': [[9], [11], [13], [15], [17], [19],[21]],
                        'gaussian': [[9], [11], [13], [15], [17], [19], [21]],
                        #'gaussian': [[11]],
                        #'mean': [[7]],
                        'median': [[9], [11], [13], [15], [17], [19], [21]],
                        #'wiener': [[1], [2]],
                        #'bm3D': [[1], [2], [3], [4], [5], [6], [7], [8]],
                        'cygno': [],
                        #'tv': [[.1], [.2], [.3], [.4], [.5], [.6], [.7], [.8], [.9]],
                        #'wavelets' : [[None]]
                        }


