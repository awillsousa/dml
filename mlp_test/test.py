import mlp_modified as mlp
index = 9

filename = 'best_model_mlp_100.pkl'

pred = mlp.predict_mlp(filename, index)
correct = (pred[0] == pred[1])
print('The prediction for index '+str(index) + ' is ' + str(pred[0])+'.')
print ('The correct value is ' + str(pred[1])+'.')
print ('The prediction is ' + str(correct)+'.')

mlp.predict_mlp_all_fast(filename)