class FilterSettings:
    # constructor
    def __init__(self):
        self.data_folder = '../data/Runs001'
        self.noise_file = '../data/noise/noise_data.h5'
        self.run_number = 817
        self.batch_size = 32
        self.sup = 16
        self.inf = -26
        self.roc_grid = 120
        self.w_range = [1, 3, 5, 7, 9]
        self.filters = ['unsupervised_wiener','gauss','mean','median']
        self.output_file_path = '../data/'
        self.output_file_name = 'result_all_filters'
