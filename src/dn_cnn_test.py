from keras.models import load_model


class NnTest:

    def __init__(self, model_path, image_input):
        self.model_path = model_path
        self.image_input = image_input

    def test(self):
        """ Input: A noised image MxN standardized between 0 to 1
            Output: A MxN filtered image
        """
        x_test = self.image_input.reshape(1, self.image_input.shape[0], self.image_input.shape[1], 1)
        model = load_model(self.model_path, compile=False)
        y_predict = model.predict(x_test)
        return y_predict.reshape(self.image_input.shape[0], self.image_input.shape[1])





