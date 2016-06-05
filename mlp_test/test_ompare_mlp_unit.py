from compare_mlp import calculate_distance_pairs, load_models,get_filenames, plot_distance_pairs, plot_distances_from_target
import unittest

class DistanceTestCase(unittest.TestCase):

    def setUp(self):
        thefiles = get_filenames("best_model_mlp")
        self.zerofiles = load_models(thefiles[0])
        self.randfiles = load_models(thefiles[1])

    def testDistanceBetweenZeroAndRandModels(self):

        distances = calculate_distance_pairs(self.zerofiles, self.randfiles)
        plot_distance_pairs(distances)

    def testDistanceBetweenZeroModelsAndZeroTarget(self):
        plot_distances_from_target(self.zerofiles[-1], self.zerofiles)

    def testDistanceBetweenRandModelsAndRandTarget(self):
        plot_distances_from_target(self.randfiles[-1], self.randfiles)


if __name__ == '__main__':
    unittest.main()