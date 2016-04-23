import network3_2
test_data = network3_2.load_data_shared()[2]
import cPickle

gg = open('savedSelf.p', 'rb')
savedNet= cPickle.load(gg)
gg.close()
index_1 = 9716
index_2 = 9904
out_1 = savedNet.singleOutputPrediction(test_data, index_1)
out_2 = savedNet.singleOutputPrediction(test_data, index_2)

savedNet.outputAccuracy(test_data)
print "%s gives %s" % (index_1, out_1)
print "%s gives %s" % (index_2, out_2)