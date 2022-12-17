$$
\begin{bmatrix}
1-a-b & a & 0\\
b & 1-a-b & a\\
0 &b &1-a-b
\end{bmatrix}
\begin{bmatrix}
x_{11} & x_{12} & x_{13}\\
x_{21} & x_{22} & x_{23}\\
x_{31} & x_{32} & x_{33}
\end{bmatrix}
\begin{bmatrix}
1-a-b & a & 0\\
b & 1-a-b & a\\
0 &b &1-a-b\\
\end{bmatrix}
$$

$$
\begin{bmatrix}
(1-a-b)x_{11}+ax_{21} & (1-a-b)x_{12}+ax_{22} & (1-a-b)x_{13}+ax_{23}\\
bx_{11}+(1-a-b)x_{21}+ax_{31} & bx_{12}+(1-a-b)x_{22}+ax_{32} & bx_{13}+(1-a-b)x_{23}+ax_{33}\\
bx_{21}+(1-a-b)x_{31} &bx_{22}+(1-a-b)x_{32} &bx_{23}+(1-a-b)x_{33}
\end{bmatrix}
\begin{bmatrix}
1-a-b & a & 0\\
b & 1-a-b & a\\
0 &b &1-a-b\\
\end{bmatrix}
$$

$abx_{11}+a(1-a-b)x_{21}+a^2x_{31}+(1-a-b)bx_{12}+(1-a-b)^2x_{22}+(1-a-b)ax_{32}+b^2x_{13}+(1-a-b)bx_{23}+abx_{33}$

Box Blur:

```
np.diag([1/3]*n, 0)+np.diag([1/3]*(n-1), 1)+np.diag([1/3]*(n-1), -1)
```

Gaussian Blur 3x3:

```
np.diag([1/2]*n, 0)+np.diag([1/4]*(n-1), 1)+np.diag([1/4]*(n-1), -1)
```

Gaussian Blur 5x5:

```
np.diag([3/8]*n, 0)+np.diag([1/4]*(n-1), 1)+np.diag([1/4]*(n-1), -1)+np.diag([1/16]*(n-2), 2)+np.diag([1/16]*(n-2), -2)
```

