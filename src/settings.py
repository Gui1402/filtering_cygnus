class FilterSettings:
    # constructor
    def __init__(self):
        self.data_folder = '../data/Runs001'
        self.noise_file = '../data/noise/noise_data.h5'
        self.run_number = 817
        self.batch_size = 32
        self.sup = 4
        self.inf = -4
        self.roc_grid = 100
        self.w_range = [1, 3, 5, 7]
        self.filters = ['mean', 'gauss', 'median']
