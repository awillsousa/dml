from compare_mlp import calculate_distance_pairs, load_models,get_filenames, plot_distance_pairs, plot_distances_from_target
import unittest

class DistanceTestCase(unittest.TestCase):

    def setUp(self):
        self.afiles = load_models(get_filenames("best_model_mlp", "zero_blur_a.pkl"))
        self.bfiles = load_models(get_filenames("best_model_mlp", "rand.pkl"))

    def testDistanceBetweenZeroAndRandModels(self):

        distances = calculate_distance_pairs(self.afiles, self.bfiles)
        plot_distance_pairs(distances)

    def testDistanceBetweenZeroModelsAndZeroTarget(self):
        plot_distances_from_target(self.afiles[-1], self.afiles)

    def testDistanceBetweenRandModelsAndRandTarget(self):
        plot_distances_from_target(self.bfiles[-1], self.bfiles)

    # def testDistanceBetweenRandModelsAndItself(self):
    #     distances = calculate_distance_pairs(self.bfiles, self.bfiles)
    #     plot_distance_pairs(distances)


if __name__ == '__main__':
    unittest.main()