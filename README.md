# circular_correlation_CNN
circular correlation with CNN

single Layer CNN

## Definition
circular correlation (@) can be defined as following:

suppose a, b are vectors with same length, than  a@b = ans

    ans = np.zeros(dim)
    for i in range(dim):
        for j in range(dim):
            ans[i] += a[j] * b[(j+i) % dim]
            

## Operation

rotate left of vector a to construct matix:
```
  a_0, a_1, .. a_n-1
  a_1, a_2, .. a_0
  .
  .
  a_n-1, ..
```

set b as conv_kernel , and than just the same as CNN

(Attention: please normalize vectors when use **L2/MSEloss** or others)

(We use **SmoothL1Loss** to test)

## Test
written by PyTorch , python3

train & test:

```   python cir_conv_grad.py ```
 
