#!/usr/bin/env python
# coding: utf-8

# # **Assignment For Numpy**

# Difficulty Level **Beginner**

# 1. Import the numpy package under the name np

# In[1]:


import numpy as np


# 2. Create a null vector of size 10 

# In[1]:


import numpy as np
y=np.zeros(10)
print(y)


# 3. Create a vector with values ranging from 10 to 49

# In[8]:


import numpy as np
vectors=np.arange(10,49)
print(vectors)


# 4. Find the shape of previous array in question 3

# In[113]:


import numpy as np
vectors=np.arange(10,49)
vectors.shape


# 5. Print the type of the previous array in question 3

# In[112]:


import numpy as np
vectors=np.arange(10,49)
vectors.dtype


# 6. Print the numpy version and the configuration
# 

# In[7]:


import numpy as np
print(np.__version__)
print(np.show_config())


# 7. Print the dimension of the array in question 3
# 

# In[111]:


import numpy as np
vectors=np.arange(10,49)
vectors.ndim


# 8. Create a boolean array with all the True values

# In[114]:


import numpy as np
vectors=np.arange(10,49)
vectors<49


# 9. Create a two dimensional array
# 
# 
# 

# In[14]:


x=np.random.randn(3,2)
x


# 10. Create a three dimensional array
# 
# 

# In[17]:


x=np.arange(3,99,3)
y=x.reshape((4,4,2))
y


# Difficulty Level **Easy**

# 11. Reverse a vector (first element becomes last)

# In[5]:


import numpy as np
vectors=np.arange(10,49)
reversed_arr = vectors[::-1]
print(reversed_arr)


# 12. Create a null vector of size 10 but the fifth value which is 1 

# In[115]:


import numpy as np
x=np.zeros(10)
x[4]=1
x


# 13. Create a 3x3 identity matrix

# In[22]:


import numpy as np
y=np.identity(3)
y


# 14. arr = np.array([1, 2, 3, 4, 5]) 
# 
# ---
# 
#  Convert the data type of the given array from int to float 

# In[116]:


import numpy as np
arr=np.array([1,2,3,4,5])
arr=arr.astype('float64')
print(arr)
arr.dtype


# 15. arr1 =          np.array([[1., 2., 3.],
# 
#                     [4., 5., 6.]])  
#                       
#     arr2 = np.array([[0., 4., 1.],
#      
#                    [7., 2., 12.]])
# 
# ---
# 
# 
# Multiply arr1 with arr2
# 

# In[25]:


import numpy as np
arr1 = np.array([[1., 2., 3.],[4., 5., 6.]]) 
arr2 = np.array([[0., 4., 1.],[7., 2., 12.]])
arr1*arr2


# 16. arr1 = np.array([[1., 2., 3.],
#                     [4., 5., 6.]]) 
#                     
#     arr2 = np.array([[0., 4., 1.], 
#                     [7., 2., 12.]])
# 
# 
# ---
# 
# Make an array by comparing both the arrays provided above

# In[26]:


import numpy as np
arr1 = np.array([[1., 2., 3.],[4., 5., 6.]]) 
arr2 = np.array([[0., 4., 1.],[7., 2., 12.]])
comparison = arr1 == arr2
equal_arrays = comparison.all()   
print(equal_arrays)


# 17. Extract all odd numbers from arr with values(0-9)

# In[9]:


import numpy as np
x=np.arange(10)
x[x%2==1]


# 18. Replace all odd numbers to -1 from previous array

# In[10]:


import numpy as np
x=np.arange(10)
np.where(x%2==1,-1,x)


# 19. arr = np.arange(10)
# 
# 
# ---
# 
# Replace the values of indexes 5,6,7 and 8 to **12**

# In[23]:


import numpy as np
arr=np.arange(10)
arr[5:9]=12
arr


# 20. Create a 2d array with 1 on the border and 0 inside

# In[24]:


import numpy as np
x=np.ones((4,4))
x[1:-1,1:-1]=0
x


# Difficulty Level **Medium**

# 21. arr2d = np.array([[1, 2, 3],
# 
#                     [4, 5, 6], 
# 
#                     [7, 8, 9]])
# 
# ---
# 
# Replace the value 5 to 12

# In[7]:


import numpy as np
arr2d = np.array([[1, 2, 3],[4, 5, 6],[7, 8, 9]])
arr2d[1,1]=12     ##arr2d[1:-1,1:-1]=12
arr2d


# 22. arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
# 
# ---
# Convert all the values of 1st array to 64
# 

# In[9]:


import numpy as np
arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
arr3d[0]=64     ####arr3d[0:3:2,::]=64
arr3d


# 23. Make a 2-Dimensional array with values 0-9 and slice out the first 1st 1-D array from it

# In[43]:


import numpy as np
arr1 = np.array([[0,1,2,3,4],[5,6,7,8,9]])
arr1[0]


# 24. Make a 2-Dimensional array with values 0-9 and slice out the 2nd value from 2nd 1-D array from it

# In[44]:


import numpy as np
arr1 = np.array([[0,1,2,3,4],[5,6,7,8,9]])
arr1[1][1]


# 25. Make a 2-Dimensional array with values 0-9 and slice out the third column but only the first two rows

# In[14]:


import numpy as np
arr1 = np.array([[0,1,2],[3,4,5],[6,7,8]])
print(arr1)
arr1[:2,2]


# 26. Create a 10x10 array with random values and find the minimum and maximum values

# In[50]:


import numpy as np
x=np.random.randn(10,10)
print(x)
print(x.min())
print(x.max())


# 27. a = np.array([1,2,3,2,3,4,3,4,5,6]) b = np.array([7,2,10,2,7,4,9,4,9,8])
# ---
# Find the common items between a and b
# 

# In[54]:


import numpy as np
a = np.array([1,2,3,2,3,4,3,4,5,6])
b = np.array([7,2,10,2,7,4,9,4,9,8])
print(np.intersect1d(a,b))


# 28. a = np.array([1,2,3,2,3,4,3,4,5,6])
# b = np.array([7,2,10,2,7,4,9,4,9,8])
# 
# ---
# Find the positions where elements of a and b match
# 
# 

# In[16]:


import numpy as np
a = np.array([1,2,3,2,3,4,3,4,5,6])
b = np.array([7,2,10,2,7,4,9,4,9,8])
np.where(a==b)


# 29.  names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])  data = np.random.randn(7, 4)
# 
# ---
# Find all the values from array **data** where the values from array **names** are not equal to **Will**
# 

# In[66]:


import numpy as np
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) 
data = np.random.randn(7, 4)
print(data)
data[names!='Will']


# 30. names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) data = np.random.randn(7, 4)
# 
# ---
# Find all the values from array **data** where the values from array **names** are not equal to **Will** and **Joe**
# 
# 

# In[78]:


import numpy as np
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) 
data = np.random.randn(7, 4)
print(data)
mask= (names!='Will') & (names!='Joe')
data[mask]


# Difficulty Level **Hard**

# 31. Create a 2D array of shape 5x3 to contain decimal numbers between 1 and 15.

# In[4]:


import numpy as np
arr = np.arange(1,16)
x = arr.reshape((5,3))
print(x)


# 32. Create an array of shape (2, 2, 4) with decimal numbers between 1 to 16.

# In[2]:


import numpy as np
arr = np.arange(1,17)
x = arr.reshape((2,2,4))
print(x)


# 33. Swap axes of the array you created in Question 32

# In[3]:


import numpy as np
arr = np.arange(1,17)
x = arr.reshape((2,2,4))
x.swapaxes(1,2)


# 34. Create an array of size 10, and find the square root of every element in the array, if the values less than 0.5, replace them with 0

# In[106]:


import numpy as np
x = np.array([0.1,1.8,5,0.2,1,1.5,0.5,0,9,0.122])
y=np.sqrt(x)
print(y)
np.where(y<0.5,0,y)


# 35. Create two random arrays of range 12 and make an array with the maximum values between each element of the two arrays

# In[107]:


import numpy as np
arr1 = np.random.randn(12)
arr2 = np.random.randn(12)
print(arr1)
print(arr2)
np.maximum(arr1,arr2)


# 36. names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
# 
# ---
# Find the unique names and sort them out!
# 

# In[108]:


import numpy as np
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
np.unique(names)


# 37. a = np.array([1,2,3,4,5])
# b = np.array([5,6,7,8,9])
# 
# ---
# From array a remove all items present in array b
# 
# 

# In[109]:


import numpy as np
a = np.array([1,2,3,4,5])
b = np.array([5,6,7,8,9])
unique = np.setdiff1d(a, b)
print(unique)


# 38.  Following is the input NumPy array delete column two and insert following new column in its place.
# 
# ---
# sampleArray = numpy.array([[34,43,73],[82,22,12],[53,94,66]]) 
# 
# 
# ---
# 
# newColumn = numpy.array([[10,10,10]])
# 

# In[19]:


import numpy as np
sampleArray = np.array([[34,43,73],[82,22,12],[53,94,66]])
newColumn = np.array([[10,10,10]])
sampleArray[:,1]=newColumn
sampleArray


# 39. x = np.array([[1., 2., 3.], [4., 5., 6.]]) y = np.array([[6., 23.], [-1, 7], [8, 9]])
# 
# 
# ---
# Find the dot product of the above two matrix
# 

# In[41]:


import numpy as np
x = np.array([[1., 2., 3.], [4., 5., 6.]])
y = np.array([[6., 23.], [-1, 7], [8, 9]])
x.dot(y)


# 40. Generate a matrix of 20 random values and find its cumulative sum

# In[40]:


import numpy as np
x=np.random.randn(20)
print(x)
print(x.cumsum())

