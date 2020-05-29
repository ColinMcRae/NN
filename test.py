import net1
import numpy as np

#test data to see that network works correct

data = np.array([
  [-0.2, -0.1],  # Alice
  [0.5, 0.6],   # Bob
  [0.7, 0.4],   # Charlie
  [-0.5, -0.6], # Diana
])
results = np.array([
  [1, 0], # Alice
  [0, 1], # Bob
  [0, 1], # Charlie
  [1, 0]# Diana
])

net = net1.Network([2,5,3,2])
net.load('test',1000)

for ep in range(1000):  
  for i in range(len(data)):
    net.train(data[i], results[i], ep)    
  if ep % 10 == 0:            
    print(net1.mse_loss([1, 0], net.feedforward([-7, -4])))    
#net.train_2(data[0], results[0], 0)
print('biases ', net.biases)
print("weights: ", net.weights)
print("**********************************")
print("**********************************")
print("**********************************")

emily = np.array([-0.7, -0.3]) # 128 pounds, 63 inches
frank = np.array([2, 0.2])  # 155 pounds, 68 inches
print("Emily: ", net.feedforward(emily))
print("Frank: ", net.feedforward(frank))

net.save('test',1000)