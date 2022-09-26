
import numpy as np
#this is useful for machine learning

print(np.__version__)

#multiplication
a = np.array([[1,2,3],[4,5,6]])
b = a
c = a*b
print(c)
print(a+b) #real addition
d = np.array((1,2,3)) #we can also pass tuple
#d = np.array("hello")
print(d)
#in lists we can't
""" matrix multiplication is not getting with * operator"""
a = np.array([[1,2,3],[4,5,6]])
b = np.array([[4,5,6],[1,2,3]])
print("multiplication",a*b)
#dimension of an array
print(a.ndim)
#shape of an array
print(a.shape)
#datatype of an array
print(a.dtype) #int32
arr = np.array([[2,3],[8,5]],dtype='int16')
print(arr)

#size and no of elements
a = np.array([[1,2,3],[4,5,6]])
print(a.size) #6 no of elements
print(a.itemsize) #4 itemsize int32 if int16 -> 2
print(a.size*a.itemsize) # total size of array in bytes

#accesing,changing and slicing arrays [start:stop:step]
a = np.array([[1,2,3,4,5,6,7],[4,5,6,4,5,6,7]])
print(a[0,0]) #1
print(a[0,:2]) #[1,2]
print(a[:,2]) #[3,6]
print(a[0,:-1:2]) # [1,3,5]
a[0,0] = 3
print(a)
a[:,2] = 5
print(a)
a[0] = 3
print(a) #works

#working with 3-d arrays also same

#creating some default arrays (shape,dtype)
arr_0 = np.zeros((2,3))#default float 32
print(arr_0)
arr_1 = np.ones((4,3),dtype="int8")
print(arr_1)
arr_ = np.full((3,2),34) #(shape,number)
print(arr_)
a = np.array([[1,2,3],[4,5,6]])
arr_1 = np.full_like(a,66)#(sample_array,number)
print(arr_1)

#for random decimal numbers np.random has methods related to random
arr_r = np.random.rand(2,3)#(directly shape)
print(arr_r)
#with some element shape
arr_r2 = np.random.random_sample(a.shape)
print(arr_r2)

#random integers randint()
arr_int = np.random.randint(3,7,size=(2,3))#(starts,endvalue,size) starts with zeero
print("random")
print(arr_int)
#unity array
print(np.identity(3)) #no tuple just a matrix

#repeating an array
a = np.array([[1,2,3],[4,5,6]])
r1 = np.repeat(a,3,axis=1)#(array,times,axis) #sorts atomatically
print("repeat")
print(r1)
#creating big arrays in simple application of array
"""
[[1. 1. 1. 1. 1.]
 [1. 0. 0. 0. 1.]
 [1. 0. 9. 0. 1.]
 [1. 0. 0. 0. 1.]
 [1. 1. 1. 1. 1.]]
                  """
output = np.ones((5,5))
z = np.zeros((3,3))
z[1,1] = 9
output[1:-1,1:-1] = z
print(output)

#copying array
a = np.array([[1,2,3],[4,5,6]])
b = a #the problem is b is pointing to the same location where a pointing
b[0,0] = 0
print(b)
print(a)
b = a.copy() # to avoid this we use copy method
b[0,0] = 100
print(a)
print(b)
print(np.pi) #gives pi value
#sine function
x = np.sin(np.pi/2)#sin(arr) cos(arr)
print(x)#1
a = np.array([[1,2,3],[4,5,6]])
x = np.sin(a)
print(x)
#radians to degrees
a = np.array([np.pi,np.pi/2])
d = np.rad2deg(a) #similarly deg2rad()
print(a)
print(np.sin(np.pi))

#matrix multiplication in arrays ##matmul()
mat1 = np.ones((2,3))
mat2 = np.full((3,2),2)
c = np.matmul(mat1,mat2)
print(c)
#the determinant
mat1 = np.ones((3,3))
det = np.linalg.det(mat1)
print(det)
#min and max and sum
a = np.array([[1,2,3],[4,5,6]])
print(np.min(a)) #default takes numbers
print(np.max(a))
print(np.sum(a))

print(np.min(a,axis=0))#[1,2,3]
print(np.max(a,axis=1)) # [3,6]

#reshaping array
a = np.array([[1,2,3],[4,5,6]])
b = a.reshape((3,2))
#b = a.reshape((4,2)) cannot reshape array in this case
print(b)

#vertical stack
#horizantal stack

#getting data from a file
data = np.genfromtxt("data.txt",delimiter=",")
print(data)
#conversion datatype
data2 = data.astype("int32") #or use dtype attribute
print(data2)

#boolean masking and advanced indexing

dat = data < 50
print(dat)
dat = data[data < 400]
print(dat)

a = np.array([1,2,3,4,5,6,7,8])
b = a[[1,3,5]] #specify what index values you want
a = np.array([[1,2,3],[4,5,6]])
#you can secify index as well slicing in []
b = a[[0,1],[1,2]]
print(b)
c = np.any(data > 100, axis = 0)
c = np.all(data > 100,axis = 0)
print(data)
print(c)
"""a = [1,2,3]
b = [4,5,6]
#print(a*b)

print(a+b)#concats
#print(a-b) cant possible"""
