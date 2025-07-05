
### Working on this...

$$
Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]}
$$

$$
A^{[l]} =
\begin{cases}
\tanh(Z^{[l]}), & \text{if } l < L \\
\dfrac{1}{1 + e^{-Z^{[L]}}}, & \text{if } l = L
\end{cases}
$$

$$
J(A^{[L]}, Y) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log A^{[L](i)} + (1 - y^{(i)}) \log(1 - A^{[L](i)}) \right]
$$

$$
dW^{[l]} = \frac{1}{m} dZ^{[l]} (A^{[l-1]})^T
\quad
db^{[l]} = \frac{1}{m} \sum dZ^{[l]}
\quad
dA^{[l-1]} = (W^{[l]})^T dZ^{[l]}
$$


