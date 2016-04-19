import network3_2
training_data, validation_data, test_data = network3_2.load_data_shared()
import cPickle

gg = open('savedSelf.saved', 'rb')
savedNet= cPickle.load(gg)
gg.close()

savedNet.outputAccuracy(test_data)
savedNet.singleOutputPrediction(test_data, 9716)#OK
savedNet.singleOutputPrediction(test_data, 9904)#KO