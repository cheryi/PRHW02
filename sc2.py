import numpy as np
import math
import matplotlib.pyplot as plt


def generate_gaussian(dim,N):
	sample_matrix=[]
	for node in range(N):
		current_vector = []
		for j in range(dim):
			r = np.random.normal(loc=0, scale=1, size=None)#get 1 gaussion
			current_vector.append(r)
		sample_matrix.append(current_vector)
	return sample_matrix
	#generate a node a time
	#ret=[]
	#for i in range(dimensions):
	#	v=np.random.normal(loc=0, scale=1, size=100)#get gaussion with 100 node
	#	ret.append(v)
	#return ret


def generate_plot(dim,p,N,f1,f2):#p refers to sigma
	known = generate_gaussian(dim,N)
	mean = np.zeros((dim,1))
	#mean = np.matrix([[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]])
	#print(mean)
	#produce covariance matrix
	cov=[]
	for i in range(dim):
		row=[]
		for j in range(dim):
			row.append(math.pow(p,abs(i-j)))
		cov.append(row)
	#transfer to ndarray
	cov_m=np.asarray(cov)
	#print(cov_m)

	[eigenvalues, eigenvectors] = np.linalg.eig(cov_m)
	lamda = np.matrix(np.diag(np.sqrt(eigenvalues)))
	Q = np.matrix(eigenvectors) * lamda
	x1_tweaked = []
	x2_tweaked = []
	tweaked_all = []
	for each in known:
		original = np.matrix(each).copy().transpose()
		#print(original)#M20*1
		#print(Q)#M20*20
		tweaked = (Q * original) + mean
		#print(tweaked)#M20*1
		x1_tweaked.append(float(tweaked[f1-1]))
		x2_tweaked.append(float(tweaked[f2-1]))
		tweaked_all.append( tweaked )
	return x1_tweaked, x2_tweaked

if __name__ == "__main__":
	x1,y1=generate_plot(20,0.9,100,3,5)
	x2,y2=generate_plot(20,0.5,100,1,2)

	
	plt.scatter(x2,y2,c='r',marker='s',label='2nd class p=0.5')
	plt.axis([-4, 4, -4, 4])
	plt.hlines(0, -4, 4)
	plt.vlines(0, -4, 4)
	plt.legend(loc='upper left')
	plt.savefig('class_2.png')
	plt.clf()
	
	plt.scatter(x1,y1,c='b',marker='x',label='1st class p=0.9')
	plt.axis([-4, 4, -4, 4])
	plt.hlines(0, -4, 4)
	plt.vlines(0, -4, 4)
	plt.legend(loc='upper left')
	plt.savefig('class_1.png')
	
	plt.scatter(x2,y2,c='r',marker='s',label='2nd class p=0.5')
	plt.savefig('class_1&2.png')
	plt.show()
