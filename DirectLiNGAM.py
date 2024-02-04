import numpy as np
import pandas as pd
import graphviz
import lingam
from lingam.utils import make_dot

print([np.__version__, pd.__version__, graphviz.__version__, lingam.__version__])

np.set_printoptions(precision=3, suppress=True)
np.random.seed(100)

#We create test data consisting of 6 variables.
# m = np.array([[0.0, 0.0, 0.0, 3.0, 0.0, 0.0],
#               [3.0, 0.0, 2.0, 0.0, 0.0, 0.0],
#               [0.0, 0.0, 0.0, 6.0, 0.0, 0.0],
#               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#               [8.0, 0.0,-1.0, 0.0, 0.0, 0.0],
#               [4.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
#
# dot = make_dot(m)
#
# # Save png
# dot.format = 'png'
# dot.render('dag')


x3 = np.random.uniform(size=1000)
x0 = 3.0*x3 + np.random.uniform(size=1000)
x2 = 6.0*x3 + np.random.uniform(size=1000)
x1 = 3.0*x0 + 2.0*x2 + np.random.uniform(size=1000)
x5 = 4.0*x0 + np.random.uniform(size=1000)
x4 = 8.0*x0 - 1.0*x2 + np.random.uniform(size=1000)
X = pd.DataFrame(np.array([x0, x1, x2, x3, x4, x5]).T ,columns=['x0', 'x1', 'x2', 'x3', 'x4', 'x5'])
print(X.head())

model = lingam.DirectLiNGAM()
model.fit(X)

print(f'{model.causal_order_=}')
print(f'{model.adjacency_matrix_=}')

dot = make_dot(model.adjacency_matrix_)
dot.format = 'png'
dot.render('fit_dag')

