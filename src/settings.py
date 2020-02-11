class FilterSettings:
    # constructor
    def __init__(self):
        self.data_folder = '../data/Runs001'
        self.noise_file = '../data/noise/noise_data.h5'
        self.run_number = 817
        self.sup = 16
        self.inf = -26
        self.roc_grid = 5
        self.filters = {'gaussian': [1, 3],
                        'mean': [1],
                        'median': [1],
                        'wiener': [3, 5]}
        self.output_file_path = '../data/'
        self.output_file_name = 'result_test_wierner'
