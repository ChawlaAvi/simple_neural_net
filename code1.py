import numpy as np
import pandas as pd

def sigmoid(x, deriv=False):
	if deriv:
		return(sigmoid(x)*(1-sigmoid(x)))

	a=[]
	for j in x:
		b=[]
		for i in j:
			if i>10:
				b.append(1)
			elif i<-20:
				b.append(0)	
			else:
				b.append(1/(1+np.exp(-i)))

		a.append(b)

	a=np.array(a)
	
	return a		
			
def relu(x,deriv = False):
	if deriv:
		a= []
		for i in x:
			if i>0:
				a.append(1)
			else:
				a.append(0)
		a=np.array(a)		
		return a.reshape((-1,1))
	else:
		a=[]
		for i in x:
			if i>0:
				a.append(i)
			else:
				a.append(0)
		a=np.array(a)
				
		return a.reshape((-1,1))

# df = pd.read_csv('data.csv')

# X_train = np.array(df.iloc[:,0:11]).reshape((-1,11))
# Y_train = np.array(df.iloc[:,11]).reshape((-1,1))

X_train = np.array([[1,1,0],[1,0,1],[0,1,0],[0,0,1]])
Y_train = np.array([[1],[1],[0],[0]])

# print("Y_train -> ",Y_train.shape)

assert X_train.shape[0] == Y_train.shape[0]

np.random.seed(1)

weight1 = np.random.random((X_train.shape[1],2)) 
weight2 = np.random.random((2,1)) 

lr = 0.01
for i in range(10000):

	a1 = X_train
	# print("a1 -> ", a1.shape)
	z2 = np.dot(a1,weight1)
	# print("z2 -> ", z2.shape)
	a2 = sigmoid(z2)
	# print("a2 -> ", a2.shape)
	z3 = np.dot(a2,weight2)
	# print("z3 -> ", z3.shape)
	a3 = sigmoid(z3)*10
	# print("a3 -> ", a3.shape)
	error_3 = Y_train - a3

	# print("error_3 -> ", error_3.shape)
	print("ERROR: "+str(np.mean(np.abs(error_3))))


	delta_3 = -(error_3)*sigmoid(a3,deriv=True)*10
	grad_2 = np.dot(a2.T,delta_3)

	delta_2 = np.dot(delta_3,weight2.T)*sigmoid(z2,deriv=True)
	grad_1 = np.dot(X_train.T,delta_2)

	weight1 -= lr*grad_1
	weight2 -= lr*grad_2


print("Output after training")
print(a3,Y_train)


	




