---
layout: page
title: "Basic Python"
---
# Fundamental Types, Lists, and Tuples


```python
x = 1
print(type(x))
x = 1.0
print(type(x))
x = True
print(type(x))
x = 1.0 +1.j
print(type(x))
x = 'my string'
print(type(x), len(x))
```

    <class 'int'>
    <class 'float'>
    <class 'bool'>
    <class 'complex'>
    <class 'str'> 9


**List are mutable** but **Tuples are immutable**, i.e. lists can add or remove elements dynamically but tuples are static and fixed. Tuples uses less memory than lists and are faster. 


```python
x = [1, 2, 3.0, 2.+3j] # List
y = (1, 2, 3.0, 2.+3j) # Tuple
print(type(x))
print(type(y))
```

    <class 'list'>
    <class 'tuple'>



```python
x[2] = 'ss' # The third element is now str
print([type(i) for i in x], [type(i) for i in y])
```

    [<class 'int'>, <class 'int'>, <class 'str'>, <class 'complex'>] [<class 'int'>, <class 'int'>, <class 'float'>, <class 'complex'>]


The **append()** method appends an element to the end of the list  
* `list.append(elmnt)`


```python
r = [] # List
for i in x:
    r.append(type(i))
print(r)
```

    [<class 'int'>, <class 'int'>, <class 'str'>, <class 'complex'>]


**Generator** are a type of iterable that compute the values on demand and one at a time. 
* Generator expressions: 
    ```py
    gen = (x**2 for x in range(5))
    ```
* Generator functions: use `yield` instead of `return`
    ```py 
    def my_gen():
        for i in range(3):
            yield i
    ```


```python
r = (type(i) for i in x)
print(r)
print(type(r))

for i in r:
    print(i)

[type(i) for i in x]
```

    <generator object <genexpr> at 0x10fed9700>
    <class 'generator'>
    <class 'int'>
    <class 'int'>
    <class 'str'>
    <class 'complex'>





    [int, int, str, complex]



# Formating strings and print statement


```python
help(print)
```

    Help on built-in function print in module builtins:
    
    print(*args, sep=' ', end='\n', file=None, flush=False)
        Prints the values to a stream, or to sys.stdout by default.
    
        sep
          string inserted between values, default a space.
        end
          string appended after the last value, default a newline.
        file
          a file-like object (stream); defaults to the current sys.stdout.
        flush
          whether to forcibly flush the stream.
    



```python
print("My name is {} and I am a {} years old.".format("John", 25))
```

    My name is John and I am a 25 years old.


An older formatting method is the **String Modulo Operator** `%`.
```py
    %[flags][width][.precision]type
```
* `s, r, a` for string
* `d, i, u` for integer
* `f, F` for floating point
* `e, E` for floating point with e notation


```python
name = "John"
age = 25
height = 175
print("My name is %s and I am a %d years old."%(name, age))
print("I am %.2f cm tall."%(height)) 
# %.2f means float with 2 decimal places
```

    My name is John and I am a 25 years old.
    I am 175.00 cm tall.



```python
from math import *
print("The value of pi is %10.3f" % (pi), ".")
print("Or approximately %-10.d" % (pi), ".")
```

    The value of pi is      3.142 .
    Or approximately 3          .


**f-string** stands for **formatted string literal in Python 3.6 which allows directly embed variables into a string.
  
Escape Charaters:
* `\n` → Newline
* `\t` → Tab
* `\\` → Backslash
* `\"` → Double quote
* `\'` → Single quote


```python
print(f"Pi is approximately {pi:.2f}")
print(f"My name is {name} and I am {age} years old.\n")

print("Padding Example:")
print(f"|{'left':<10}|")
print(f"|{'center':^10}|")
print(f"|{'right':>10}|")


```

    Pi is approximately 3.14
    My name is John and I am 25 years old.
    
    Padding Example:
    |left      |
    |  center  |
    |     right|


# Lists `[...]`


```python
l1 = [1, 'a', 1.0, -1.1]
print(l1[0], l1[-1])
print(l1[1::2],'\n')

l2 = [1, [1,[2,3]]] # Nested list
print(l2,'\n')

print(list(range(10)))
print(list(range(1, 10, 2))) # Start, End, Step

```

    1 -1.1
    ['a', -1.1] 
    
    [1, [1, [2, 3]]] 
    
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    [1, 3, 5, 7, 9]



```python
w = range(1, 20, 2)
v = list(w)
print(w)
print(v)
print(list(v))
```

    range(1, 20, 2)
    [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]



```python
print(v[-1])
print(v[1:-1:2])
```

    19
    [3, 7, 11, 15]


The **numpy** package (module) is used in almost all numerical computation using Python. It is a package that provide high-performance vector, matrix and higher-dimensional data structures for Python. 


```python
from numpy import *
array([[1j, 2], [3, 1]])
```




    array([[0.+1.j, 2.+0.j],
           [3.+0.j, 1.+0.j]])




```python
l = [] # Empty list
l.append('A')
l.append('d')
l.append('f')
print(l)

# Modify an elenment
l[1] = 'e'
print(l)

# Insert at specific location
l.insert(1, 'k') 
print(l)

# Remove specific element
l.remove('f')

# Remove element at specific location
del l[1]
print(l)
```

    ['A', 'd', 'f']
    ['A', 'e', 'f']
    ['A', 'k', 'e', 'f']
    ['A', 'e']



```python
# Delete from back instead of front
l = list(range(20))
print(l)

to_delete = sorted([3, 5, 7, 19]) # in acending order
print(to_delete)
print(to_delete[::-1]) # in reverse steps, decending

for i in to_delete[::-1]:
    print("Remove ", l[i])
    del l[i]
print(l)
```

    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    [3, 5, 7, 19]
    [19, 7, 5, 3]
    Remove  19
    Remove  7
    Remove  5
    Remove  3
    [0, 1, 2, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]



```python
a = [3,4,7,8,9,10]
if (20 in a):
    a.remove(20)
print(a)

what = 3 in a
print(what)

if (10 in a):
    print(a)
```

    [3, 4, 7, 8, 9, 10]
    True
    [3, 4, 7, 8, 9, 10]


# Tuples `(...)`
Tuples are like lists, except that they cannot be modified once created, that is they are immutable. Their most important use is in returning from a function and in print statement, and as keys to dictionary.


```python
point = (10,20,30)
print(point[2])

# point[0] = 1
# TypeError: 'tuple' object does not support item assignment
# Trying to modify will throw a TypeError.

# Modifying the entire tuple is allowed.
point = (1,20,30)
x, y, z = point
print("(x,y) is ", x,y)

# A function that returns a tuple.
def Simple():
    return ('a', 'b', 'c')

x, y, z = Simple()
print(x,y,z)

```

    30
    (x,y) is  1 20
    a b c


# Dictionaries `{...}`
Dictionaries are like lists, but each element is a key-value pair. Keys are not integers, they can be any scalar type or tuple of scalar types.



```python
# Just like list
di = {}
di[0] = 'a'
di[1] = 'b'
di[2] = 'c'
print(di)
for i in di:
    print(di[i])
```

    {0: 'a', 1: 'b', 2: 'c'}
    a
    b
    c



```python
# Generic keys, any immutable type (including tuples)
dj = {}
dj[300] = 'c'
dj[600] = 'd'
dj['cde'] = 1
dj[(1,2)] = 5
print(dj)
print(dj[(1,2)])
```

    {300: 'c', 600: 'd', 'cde': 1, (1, 2): 5}
    5



```python
if (3,4) in dj:
    print(dj[(3,4)])
else:
    dj[(3,4)] = 'something'

print(dj)
```

    {300: 'c', 600: 'd', 'cde': 1, (1, 2): 5, (3, 4): 'something'}



```python
dk = {
    300: 'c',
    600: 'd',
    'cde': 1,
    (1,2): 5
}

# Iterate over keys
for k in dk.keys():
    # The plus operate only concantanate strings
    print('p1[' + str(k) + '] is ' , dk[k])
```

    p1[300] is  c
    p1[600] is  d
    p1[cde] is  1
    p1[(1, 2)] is  5



```python
# Similarly
for k in dk:
    print('p2[' + str(k) + ']', dk[k])

print()

for k, v in dk.items():
    print('p3[' + str(k) + '] is ', v)
```

    p2[300] c
    p2[600] d
    p2[cde] 1
    p2[(1, 2)] 5
    
    p3[300] is  c
    p3[600] is  d
    p3[cde] is  1
    p3[(1, 2)] is  5



```python
print(list(dk.keys()))
print(list(dk.items()))
print(list(dk.values()))
```

    [300, 600, 'cde', (1, 2)]
    [(300, 'c'), (600, 'd'), ('cde', 1), ((1, 2), 5)]
    ['c', 'd', 1, 5]



```python
from math import *

# Spare matrix
di = {(100,100): pi,
      (100,300): 2*pi
      }
print(di)

# Index to list
r = [10, 100, 30, 5]
d = {}
# assigning each key to a value
for i in range(len(r)):
    d[r[i]] = i
print(d)

# Equivalently but nicer
for i, v in enumerate(r):
    d[v] = i
print(d)

list(enumerate(r))
```

    {(100, 100): 3.141592653589793, (100, 300): 6.283185307179586}
    {10: 0, 100: 1, 30: 2, 5: 3}
    {10: 0, 100: 1, 30: 2, 5: 3}





    [(0, 10), (1, 100), (2, 30), (3, 5)]



# Control flow
* `if, else, elif`
* loops using `for` & `while`
* list comprehsions: `inline for`
```py 
    [< do something with i> for i in data]
```    


```python
statement1 = statement2 = statement3 = True
statement2 = False

if (statement1):
    if (statement2):
        print('Both statements are true.')
    elif (statement3):
        print('statement2 is false but statement3 is true.')
```

    statement2 is false but statement3 is true.



```python
# Scalar versus non-scalar values
# Scalar behaviour in Python
print('Scalar')
a = b = g = 3
g = 4
print('a= ', a, ' b=', b, ' g=', g)
b = 1
print('a= ', a, ' b=', b, ' g=', g)
print()

print('Non-scalar')
c = d = f = [3] # A list with a single element
d[0] = 4 # applies to both c and f
print('c=', c, ' d=', d, 'f=', f)
c.append(4) # applies to both d and f
print('c=', c, ' d=', d, 'f=', f)
d = [3] # d is decoupled from c and f
print('c=', c, ' d=', d, 'f=', f)

```

    Scalar
    a=  3  b= 3  g= 4
    a=  3  b= 1  g= 4
    
    Non-scalar
    c= [4]  d= [4] f= [4]
    c= [4, 4]  d= [4, 4] f= [4, 4]
    c= [4, 4]  d= [3] f= [4, 4]



```python
bool(100)
```




    True




```python
for x in range(-3,3,1):
    print(x)

ls = ['scientific', 'computing', 'in', 'Python']
for word in ls:
    print(word)

for i, word in enumerate(ls):
    print('At index', i, 'we have string', word)

print()
print('Length of ls is', len(ls))
# Equivalently
for i in range(len(ls)):
    word = ls[i]
    print('At index', i, 'we have string', word)
```

    -3
    -2
    -1
    0
    1
    2
    scientific
    computing
    in
    Python
    At index 0 we have string scientific
    At index 1 we have string computing
    At index 2 we have string in
    At index 3 we have string Python
    
    Length of ls is 4
    At index 0 we have string scientific
    At index 1 we have string computing
    At index 2 we have string in
    At index 3 we have string Python



```python
list(enumerate(ls))
```




    [(0, 'scientific'), (1, 'computing'), (2, 'in'), (3, 'Python')]




```python
i = 0
while i < 5:
    i += 1
    print(i)
```

    1
    2
    3
    4
    5



```python
for i in range(100):
    if i >= 5: break
    print(i)
```

    0
    1
    2
    3
    4



```python
l1 = []
for x in range(5):
    l1.append(x**2)
print(l1)
```

    [0, 1, 4, 9, 16]



```python
l1 = [x**2 for x in range(5)]
print(l1)
```

    [0, 1, 4, 9, 16]



```python
# A nested list of tuples
[(j, i) for j in range(5) for i in range(5)]
```




    [(0, 0),
     (0, 1),
     (0, 2),
     (0, 3),
     (0, 4),
     (1, 0),
     (1, 1),
     (1, 2),
     (1, 3),
     (1, 4),
     (2, 0),
     (2, 1),
     (2, 2),
     (2, 3),
     (2, 4),
     (3, 0),
     (3, 1),
     (3, 2),
     (3, 3),
     (3, 4),
     (4, 0),
     (4, 1),
     (4, 2),
     (4, 3),
     (4, 4)]




```python
for i in range(5):
    for j in range(5):
        print(i,j)
```

    0 0
    0 1
    0 2
    0 3
    0 4
    1 0
    1 1
    1 2
    1 3
    1 4
    2 0
    2 1
    2 2
    2 3
    2 4
    3 0
    3 1
    3 2
    3 3
    3 4
    4 0
    4 1
    4 2
    4 3
    4 4


# Functions
* A function starts with keyword `def`.
* Use `docstring` to explains what the function does.
* Multiple things (classes or values) can be returned by tuples.
* Functions can have default and keyword arguments.


```python
def funca(s):
    """ Print a string and tell how many character it has.
        Return a tuple with the length and the string.
    """
    print('s has', len(s), 'characters.')
    return (len(s), s)

# Example
(l, s) = funca('something')
print(l, s)

# First. Call the function
# Second. Print the tuple
print(funca('This string'))
```

    s has 9 characters.
    9 something
    s has 11 characters.
    (11, 'This string')



```python
# Look at the docstring
help(funca)
```

    Help on function funca in module __main__:
    
    funca(s)
        Print a string and tell how many character it has.
        Return a tuple with the length and the string.
    



```python
def funca(s, prnt=False, extra='p'):
    """ Print a string with extra attached.
        Returns new string.
    """
    sn = s + extra
    if prnt:
        print(sn, 'has', len(sn), 'character.')
    return sn

# Examples
funca('This str', prnt=True)
funca('This str', extra='q')
```

    This strp has 9 character.





    'This strq'



# `lambda` functions (Unnamed functions)


```python
# Normal function
def f2(x):
    return x**2

# Lambda function
f1 = lambda x: x**2
```


```python
f1(3), f2(3)
```




    (9, 9)




```python
from numpy import *
from scipy import integrate
help(integrate)
```

    Help on package scipy.integrate in scipy:
    
    NAME
        scipy.integrate
    
    DESCRIPTION
        =============================================
        Integration and ODEs (:mod:`scipy.integrate`)
        =============================================
    
        .. currentmodule:: scipy.integrate
    
        Integrating functions, given function object
        ============================================
    
        .. autosummary::
           :toctree: generated/
    
           quad          -- General purpose integration
           quad_vec      -- General purpose integration of vector-valued functions
           dblquad       -- General purpose double integration
           tplquad       -- General purpose triple integration
           nquad         -- General purpose N-D integration
           fixed_quad    -- Integrate func(x) using Gaussian quadrature of order n
           quadrature    -- Integrate with given tolerance using Gaussian quadrature
           romberg       -- Integrate func using Romberg integration
           newton_cotes  -- Weights and error coefficient for Newton-Cotes integration
           qmc_quad      -- N-D integration using Quasi-Monte Carlo quadrature
           IntegrationWarning -- Warning on issues during integration
           AccuracyWarning  -- Warning on issues during quadrature integration
    
        Integrating functions, given fixed samples
        ==========================================
    
        .. autosummary::
           :toctree: generated/
    
           trapezoid            -- Use trapezoidal rule to compute integral.
           cumulative_trapezoid -- Use trapezoidal rule to cumulatively compute integral.
           simpson              -- Use Simpson's rule to compute integral from samples.
           cumulative_simpson   -- Use Simpson's rule to cumulatively compute integral from samples.
           romb                 -- Use Romberg Integration to compute integral from
                                -- (2**k + 1) evenly-spaced samples.
    
        .. seealso::
    
           :mod:`scipy.special` for orthogonal polynomials (special) for Gaussian
           quadrature roots and weights for other weighting factors and regions.
    
        Solving initial value problems for ODE systems
        ==============================================
    
        The solvers are implemented as individual classes, which can be used directly
        (low-level usage) or through a convenience function.
    
        .. autosummary::
           :toctree: generated/
    
           solve_ivp     -- Convenient function for ODE integration.
           RK23          -- Explicit Runge-Kutta solver of order 3(2).
           RK45          -- Explicit Runge-Kutta solver of order 5(4).
           DOP853        -- Explicit Runge-Kutta solver of order 8.
           Radau         -- Implicit Runge-Kutta solver of order 5.
           BDF           -- Implicit multi-step variable order (1 to 5) solver.
           LSODA         -- LSODA solver from ODEPACK Fortran package.
           OdeSolver     -- Base class for ODE solvers.
           DenseOutput   -- Local interpolant for computing a dense output.
           OdeSolution   -- Class which represents a continuous ODE solution.
    
    
        Old API
        -------
    
        These are the routines developed earlier for SciPy. They wrap older solvers
        implemented in Fortran (mostly ODEPACK). While the interface to them is not
        particularly convenient and certain features are missing compared to the new
        API, the solvers themselves are of good quality and work fast as compiled
        Fortran code. In some cases, it might be worth using this old API.
    
        .. autosummary::
           :toctree: generated/
    
           odeint        -- General integration of ordinary differential equations.
           ode           -- Integrate ODE using VODE and ZVODE routines.
           complex_ode   -- Convert a complex-valued ODE to real-valued and integrate.
           ODEintWarning -- Warning raised during the execution of `odeint`.
    
    
        Solving boundary value problems for ODE systems
        ===============================================
    
        .. autosummary::
           :toctree: generated/
    
           solve_bvp     -- Solve a boundary value problem for a system of ODEs.
    
    PACKAGE CONTENTS
        _bvp
        _dop
        _ivp (package)
        _lsoda
        _ode
        _odepack
        _odepack_py
        _quad_vec
        _quadpack
        _quadpack_py
        _quadrature
        _tanhsinh
        _test_multivariate
        _test_odeint_banded
        _vode
        dop
        lsoda
        odepack
        quadpack
        tests (package)
        vode
    
    CLASSES
        builtins.UserWarning(builtins.Warning)
            scipy.integrate._quadpack_py.IntegrationWarning
        builtins.Warning(builtins.Exception)
            scipy.integrate._odepack_py.ODEintWarning
            scipy.integrate._quadrature.AccuracyWarning
        builtins.object
            scipy.integrate._ivp.base.DenseOutput
            scipy.integrate._ivp.base.OdeSolver
                scipy.integrate._ivp.bdf.BDF
                scipy.integrate._ivp.lsoda.LSODA
                scipy.integrate._ivp.radau.Radau
            scipy.integrate._ivp.common.OdeSolution
            scipy.integrate._ode.ode
                scipy.integrate._ode.complex_ode
        scipy.integrate._ivp.rk.RungeKutta(scipy.integrate._ivp.base.OdeSolver)
            scipy.integrate._ivp.rk.DOP853
            scipy.integrate._ivp.rk.RK23
            scipy.integrate._ivp.rk.RK45
    
        class AccuracyWarning(builtins.Warning)
         |  Method resolution order:
         |      AccuracyWarning
         |      builtins.Warning
         |      builtins.Exception
         |      builtins.BaseException
         |      builtins.object
         |
         |  Data descriptors defined here:
         |
         |  __weakref__
         |      list of weak references to the object
         |
         |  ----------------------------------------------------------------------
         |  Methods inherited from builtins.Warning:
         |
         |  __init__(self, /, *args, **kwargs)
         |      Initialize self.  See help(type(self)) for accurate signature.
         |
         |  ----------------------------------------------------------------------
         |  Static methods inherited from builtins.Warning:
         |
         |  __new__(*args, **kwargs) class method of builtins.Warning
         |      Create and return a new object.  See help(type) for accurate signature.
         |
         |  ----------------------------------------------------------------------
         |  Methods inherited from builtins.BaseException:
         |
         |  __getattribute__(self, name, /)
         |      Return getattr(self, name).
         |
         |  __reduce__(...)
         |      Helper for pickle.
         |
         |  __repr__(self, /)
         |      Return repr(self).
         |
         |  __setstate__(...)
         |
         |  __str__(self, /)
         |      Return str(self).
         |
         |  add_note(...)
         |      Exception.add_note(note) --
         |      add a note to the exception
         |
         |  with_traceback(...)
         |      Exception.with_traceback(tb) --
         |      set self.__traceback__ to tb and return self.
         |
         |  ----------------------------------------------------------------------
         |  Data descriptors inherited from builtins.BaseException:
         |
         |  __cause__
         |      exception cause
         |
         |  __context__
         |      exception context
         |
         |  __dict__
         |
         |  __suppress_context__
         |
         |  __traceback__
         |
         |  args
    
        class BDF(scipy.integrate._ivp.base.OdeSolver)
         |  BDF(fun, t0, y0, t_bound, max_step=inf, rtol=0.001, atol=1e-06, jac=None, jac_sparsity=None, vectorized=False, first_step=None, **extraneous)
         |
         |  Implicit method based on backward-differentiation formulas.
         |
         |  This is a variable order method with the order varying automatically from
         |  1 to 5. The general framework of the BDF algorithm is described in [1]_.
         |  This class implements a quasi-constant step size as explained in [2]_.
         |  The error estimation strategy for the constant-step BDF is derived in [3]_.
         |  An accuracy enhancement using modified formulas (NDF) [2]_ is also implemented.
         |
         |  Can be applied in the complex domain.
         |
         |  Parameters
         |  ----------
         |  fun : callable
         |      Right-hand side of the system: the time derivative of the state ``y``
         |      at time ``t``. The calling signature is ``fun(t, y)``, where ``t`` is a
         |      scalar and ``y`` is an ndarray with ``len(y) = len(y0)``. ``fun`` must
         |      return an array of the same shape as ``y``. See `vectorized` for more
         |      information.
         |  t0 : float
         |      Initial time.
         |  y0 : array_like, shape (n,)
         |      Initial state.
         |  t_bound : float
         |      Boundary time - the integration won't continue beyond it. It also
         |      determines the direction of the integration.
         |  first_step : float or None, optional
         |      Initial step size. Default is ``None`` which means that the algorithm
         |      should choose.
         |  max_step : float, optional
         |      Maximum allowed step size. Default is np.inf, i.e., the step size is not
         |      bounded and determined solely by the solver.
         |  rtol, atol : float and array_like, optional
         |      Relative and absolute tolerances. The solver keeps the local error
         |      estimates less than ``atol + rtol * abs(y)``. Here `rtol` controls a
         |      relative accuracy (number of correct digits), while `atol` controls
         |      absolute accuracy (number of correct decimal places). To achieve the
         |      desired `rtol`, set `atol` to be smaller than the smallest value that
         |      can be expected from ``rtol * abs(y)`` so that `rtol` dominates the
         |      allowable error. If `atol` is larger than ``rtol * abs(y)`` the
         |      number of correct digits is not guaranteed. Conversely, to achieve the
         |      desired `atol` set `rtol` such that ``rtol * abs(y)`` is always smaller
         |      than `atol`. If components of y have different scales, it might be
         |      beneficial to set different `atol` values for different components by
         |      passing array_like with shape (n,) for `atol`. Default values are
         |      1e-3 for `rtol` and 1e-6 for `atol`.
         |  jac : {None, array_like, sparse_matrix, callable}, optional
         |      Jacobian matrix of the right-hand side of the system with respect to y,
         |      required by this method. The Jacobian matrix has shape (n, n) and its
         |      element (i, j) is equal to ``d f_i / d y_j``.
         |      There are three ways to define the Jacobian:
         |
         |          * If array_like or sparse_matrix, the Jacobian is assumed to
         |            be constant.
         |          * If callable, the Jacobian is assumed to depend on both
         |            t and y; it will be called as ``jac(t, y)`` as necessary.
         |            For the 'Radau' and 'BDF' methods, the return value might be a
         |            sparse matrix.
         |          * If None (default), the Jacobian will be approximated by
         |            finite differences.
         |
         |      It is generally recommended to provide the Jacobian rather than
         |      relying on a finite-difference approximation.
         |  jac_sparsity : {None, array_like, sparse matrix}, optional
         |      Defines a sparsity structure of the Jacobian matrix for a
         |      finite-difference approximation. Its shape must be (n, n). This argument
         |      is ignored if `jac` is not `None`. If the Jacobian has only few non-zero
         |      elements in *each* row, providing the sparsity structure will greatly
         |      speed up the computations [4]_. A zero entry means that a corresponding
         |      element in the Jacobian is always zero. If None (default), the Jacobian
         |      is assumed to be dense.
         |  vectorized : bool, optional
         |      Whether `fun` can be called in a vectorized fashion. Default is False.
         |
         |      If ``vectorized`` is False, `fun` will always be called with ``y`` of
         |      shape ``(n,)``, where ``n = len(y0)``.
         |
         |      If ``vectorized`` is True, `fun` may be called with ``y`` of shape
         |      ``(n, k)``, where ``k`` is an integer. In this case, `fun` must behave
         |      such that ``fun(t, y)[:, i] == fun(t, y[:, i])`` (i.e. each column of
         |      the returned array is the time derivative of the state corresponding
         |      with a column of ``y``).
         |
         |      Setting ``vectorized=True`` allows for faster finite difference
         |      approximation of the Jacobian by this method, but may result in slower
         |      execution overall in some circumstances (e.g. small ``len(y0)``).
         |
         |  Attributes
         |  ----------
         |  n : int
         |      Number of equations.
         |  status : string
         |      Current status of the solver: 'running', 'finished' or 'failed'.
         |  t_bound : float
         |      Boundary time.
         |  direction : float
         |      Integration direction: +1 or -1.
         |  t : float
         |      Current time.
         |  y : ndarray
         |      Current state.
         |  t_old : float
         |      Previous time. None if no steps were made yet.
         |  step_size : float
         |      Size of the last successful step. None if no steps were made yet.
         |  nfev : int
         |      Number of evaluations of the right-hand side.
         |  njev : int
         |      Number of evaluations of the Jacobian.
         |  nlu : int
         |      Number of LU decompositions.
         |
         |  References
         |  ----------
         |  .. [1] G. D. Byrne, A. C. Hindmarsh, "A Polyalgorithm for the Numerical
         |         Solution of Ordinary Differential Equations", ACM Transactions on
         |         Mathematical Software, Vol. 1, No. 1, pp. 71-96, March 1975.
         |  .. [2] L. F. Shampine, M. W. Reichelt, "THE MATLAB ODE SUITE", SIAM J. SCI.
         |         COMPUTE., Vol. 18, No. 1, pp. 1-22, January 1997.
         |  .. [3] E. Hairer, G. Wanner, "Solving Ordinary Differential Equations I:
         |         Nonstiff Problems", Sec. III.2.
         |  .. [4] A. Curtis, M. J. D. Powell, and J. Reid, "On the estimation of
         |         sparse Jacobian matrices", Journal of the Institute of Mathematics
         |         and its Applications, 13, pp. 117-120, 1974.
         |
         |  Method resolution order:
         |      BDF
         |      scipy.integrate._ivp.base.OdeSolver
         |      builtins.object
         |
         |  Methods defined here:
         |
         |  __init__(self, fun, t0, y0, t_bound, max_step=inf, rtol=0.001, atol=1e-06, jac=None, jac_sparsity=None, vectorized=False, first_step=None, **extraneous)
         |      Initialize self.  See help(type(self)) for accurate signature.
         |
         |  ----------------------------------------------------------------------
         |  Methods inherited from scipy.integrate._ivp.base.OdeSolver:
         |
         |  dense_output(self)
         |      Compute a local interpolant over the last successful step.
         |
         |      Returns
         |      -------
         |      sol : `DenseOutput`
         |          Local interpolant over the last successful step.
         |
         |  step(self)
         |      Perform one integration step.
         |
         |      Returns
         |      -------
         |      message : string or None
         |          Report from the solver. Typically a reason for a failure if
         |          `self.status` is 'failed' after the step was taken or None
         |          otherwise.
         |
         |  ----------------------------------------------------------------------
         |  Readonly properties inherited from scipy.integrate._ivp.base.OdeSolver:
         |
         |  step_size
         |
         |  ----------------------------------------------------------------------
         |  Data descriptors inherited from scipy.integrate._ivp.base.OdeSolver:
         |
         |  __dict__
         |      dictionary for instance variables
         |
         |  __weakref__
         |      list of weak references to the object
         |
         |  ----------------------------------------------------------------------
         |  Data and other attributes inherited from scipy.integrate._ivp.base.OdeSolver:
         |
         |  TOO_SMALL_STEP = 'Required step size is less than spacing between numb...
    
        class DOP853(RungeKutta)
         |  DOP853(fun, t0, y0, t_bound, max_step=inf, rtol=0.001, atol=1e-06, vectorized=False, first_step=None, **extraneous)
         |
         |  Explicit Runge-Kutta method of order 8.
         |
         |  This is a Python implementation of "DOP853" algorithm originally written
         |  in Fortran [1]_, [2]_. Note that this is not a literal translation, but
         |  the algorithmic core and coefficients are the same.
         |
         |  Can be applied in the complex domain.
         |
         |  Parameters
         |  ----------
         |  fun : callable
         |      Right-hand side of the system. The calling signature is ``fun(t, y)``.
         |      Here, ``t`` is a scalar, and there are two options for the ndarray ``y``:
         |      It can either have shape (n,); then ``fun`` must return array_like with
         |      shape (n,). Alternatively it can have shape (n, k); then ``fun``
         |      must return an array_like with shape (n, k), i.e. each column
         |      corresponds to a single column in ``y``. The choice between the two
         |      options is determined by `vectorized` argument (see below).
         |  t0 : float
         |      Initial time.
         |  y0 : array_like, shape (n,)
         |      Initial state.
         |  t_bound : float
         |      Boundary time - the integration won't continue beyond it. It also
         |      determines the direction of the integration.
         |  first_step : float or None, optional
         |      Initial step size. Default is ``None`` which means that the algorithm
         |      should choose.
         |  max_step : float, optional
         |      Maximum allowed step size. Default is np.inf, i.e. the step size is not
         |      bounded and determined solely by the solver.
         |  rtol, atol : float and array_like, optional
         |      Relative and absolute tolerances. The solver keeps the local error
         |      estimates less than ``atol + rtol * abs(y)``. Here `rtol` controls a
         |      relative accuracy (number of correct digits), while `atol` controls
         |      absolute accuracy (number of correct decimal places). To achieve the
         |      desired `rtol`, set `atol` to be smaller than the smallest value that
         |      can be expected from ``rtol * abs(y)`` so that `rtol` dominates the
         |      allowable error. If `atol` is larger than ``rtol * abs(y)`` the
         |      number of correct digits is not guaranteed. Conversely, to achieve the
         |      desired `atol` set `rtol` such that ``rtol * abs(y)`` is always smaller
         |      than `atol`. If components of y have different scales, it might be
         |      beneficial to set different `atol` values for different components by
         |      passing array_like with shape (n,) for `atol`. Default values are
         |      1e-3 for `rtol` and 1e-6 for `atol`.
         |  vectorized : bool, optional
         |      Whether `fun` is implemented in a vectorized fashion. Default is False.
         |
         |  Attributes
         |  ----------
         |  n : int
         |      Number of equations.
         |  status : string
         |      Current status of the solver: 'running', 'finished' or 'failed'.
         |  t_bound : float
         |      Boundary time.
         |  direction : float
         |      Integration direction: +1 or -1.
         |  t : float
         |      Current time.
         |  y : ndarray
         |      Current state.
         |  t_old : float
         |      Previous time. None if no steps were made yet.
         |  step_size : float
         |      Size of the last successful step. None if no steps were made yet.
         |  nfev : int
         |      Number evaluations of the system's right-hand side.
         |  njev : int
         |      Number of evaluations of the Jacobian. Is always 0 for this solver
         |      as it does not use the Jacobian.
         |  nlu : int
         |      Number of LU decompositions. Is always 0 for this solver.
         |
         |  References
         |  ----------
         |  .. [1] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
         |         Equations I: Nonstiff Problems", Sec. II.
         |  .. [2] `Page with original Fortran code of DOP853
         |          <http://www.unige.ch/~hairer/software.html>`_.
         |
         |  Method resolution order:
         |      DOP853
         |      RungeKutta
         |      scipy.integrate._ivp.base.OdeSolver
         |      builtins.object
         |
         |  Methods defined here:
         |
         |  __init__(self, fun, t0, y0, t_bound, max_step=inf, rtol=0.001, atol=1e-06, vectorized=False, first_step=None, **extraneous)
         |      Initialize self.  See help(type(self)) for accurate signature.
         |
         |  ----------------------------------------------------------------------
         |  Data and other attributes defined here:
         |
         |  A = array([[ 0.00000000e+00,  0.00000000e+00,  0.000...23605672e+01,  ...
         |
         |  A_EXTRA = array([[ 5.61675023e-02,  0.00000000e+00,  0.000...e+00, -9....
         |
         |  B = array([ 0.05429373,  0.        ,  0.        ,  0...7, -0.15216095,...
         |
         |  C = array([0.        , 0.05260015, 0.07890023, 0.118...8205, 0.6      ...
         |
         |  C_EXTRA = array([0.1       , 0.2       , 0.77777778])
         |
         |  D = array([[-8.42893828e+00,  0.00000000e+00,  0.000...e+01, -3.917726...
         |
         |  E3 = array([-0.18980075,  0.        ,  0.        ,  0...5,
         |          0.2...
         |
         |  E5 = array([ 0.01312004,  0.        ,  0.        ,  0...2,
         |          0.0...
         |
         |  __annotations__ = {}
         |
         |  error_estimator_order = 7
         |
         |  n_stages = 12
         |
         |  order = 8
         |
         |  ----------------------------------------------------------------------
         |  Data and other attributes inherited from RungeKutta:
         |
         |  E = NotImplemented
         |
         |  P = NotImplemented
         |
         |  ----------------------------------------------------------------------
         |  Methods inherited from scipy.integrate._ivp.base.OdeSolver:
         |
         |  dense_output(self)
         |      Compute a local interpolant over the last successful step.
         |
         |      Returns
         |      -------
         |      sol : `DenseOutput`
         |          Local interpolant over the last successful step.
         |
         |  step(self)
         |      Perform one integration step.
         |
         |      Returns
         |      -------
         |      message : string or None
         |          Report from the solver. Typically a reason for a failure if
         |          `self.status` is 'failed' after the step was taken or None
         |          otherwise.
         |
         |  ----------------------------------------------------------------------
         |  Readonly properties inherited from scipy.integrate._ivp.base.OdeSolver:
         |
         |  step_size
         |
         |  ----------------------------------------------------------------------
         |  Data descriptors inherited from scipy.integrate._ivp.base.OdeSolver:
         |
         |  __dict__
         |      dictionary for instance variables
         |
         |  __weakref__
         |      list of weak references to the object
         |
         |  ----------------------------------------------------------------------
         |  Data and other attributes inherited from scipy.integrate._ivp.base.OdeSolver:
         |
         |  TOO_SMALL_STEP = 'Required step size is less than spacing between numb...
    
        class DenseOutput(builtins.object)
         |  DenseOutput(t_old, t)
         |
         |  Base class for local interpolant over step made by an ODE solver.
         |
         |  It interpolates between `t_min` and `t_max` (see Attributes below).
         |  Evaluation outside this interval is not forbidden, but the accuracy is not
         |  guaranteed.
         |
         |  Attributes
         |  ----------
         |  t_min, t_max : float
         |      Time range of the interpolation.
         |
         |  Methods defined here:
         |
         |  __call__(self, t)
         |      Evaluate the interpolant.
         |
         |      Parameters
         |      ----------
         |      t : float or array_like with shape (n_points,)
         |          Points to evaluate the solution at.
         |
         |      Returns
         |      -------
         |      y : ndarray, shape (n,) or (n, n_points)
         |          Computed values. Shape depends on whether `t` was a scalar or a
         |          1-D array.
         |
         |  __init__(self, t_old, t)
         |      Initialize self.  See help(type(self)) for accurate signature.
         |
         |  ----------------------------------------------------------------------
         |  Data descriptors defined here:
         |
         |  __dict__
         |      dictionary for instance variables
         |
         |  __weakref__
         |      list of weak references to the object
    
        class IntegrationWarning(builtins.UserWarning)
         |  Warning on issues during integration.
         |
         |  Method resolution order:
         |      IntegrationWarning
         |      builtins.UserWarning
         |      builtins.Warning
         |      builtins.Exception
         |      builtins.BaseException
         |      builtins.object
         |
         |  Data descriptors defined here:
         |
         |  __weakref__
         |      list of weak references to the object
         |
         |  ----------------------------------------------------------------------
         |  Methods inherited from builtins.UserWarning:
         |
         |  __init__(self, /, *args, **kwargs)
         |      Initialize self.  See help(type(self)) for accurate signature.
         |
         |  ----------------------------------------------------------------------
         |  Static methods inherited from builtins.UserWarning:
         |
         |  __new__(*args, **kwargs) class method of builtins.UserWarning
         |      Create and return a new object.  See help(type) for accurate signature.
         |
         |  ----------------------------------------------------------------------
         |  Methods inherited from builtins.BaseException:
         |
         |  __getattribute__(self, name, /)
         |      Return getattr(self, name).
         |
         |  __reduce__(...)
         |      Helper for pickle.
         |
         |  __repr__(self, /)
         |      Return repr(self).
         |
         |  __setstate__(...)
         |
         |  __str__(self, /)
         |      Return str(self).
         |
         |  add_note(...)
         |      Exception.add_note(note) --
         |      add a note to the exception
         |
         |  with_traceback(...)
         |      Exception.with_traceback(tb) --
         |      set self.__traceback__ to tb and return self.
         |
         |  ----------------------------------------------------------------------
         |  Data descriptors inherited from builtins.BaseException:
         |
         |  __cause__
         |      exception cause
         |
         |  __context__
         |      exception context
         |
         |  __dict__
         |
         |  __suppress_context__
         |
         |  __traceback__
         |
         |  args
    
        class LSODA(scipy.integrate._ivp.base.OdeSolver)
         |  LSODA(fun, t0, y0, t_bound, first_step=None, min_step=0.0, max_step=inf, rtol=0.001, atol=1e-06, jac=None, lband=None, uband=None, vectorized=False, **extraneous)
         |
         |  Adams/BDF method with automatic stiffness detection and switching.
         |
         |  This is a wrapper to the Fortran solver from ODEPACK [1]_. It switches
         |  automatically between the nonstiff Adams method and the stiff BDF method.
         |  The method was originally detailed in [2]_.
         |
         |  Parameters
         |  ----------
         |  fun : callable
         |      Right-hand side of the system: the time derivative of the state ``y``
         |      at time ``t``. The calling signature is ``fun(t, y)``, where ``t`` is a
         |      scalar and ``y`` is an ndarray with ``len(y) = len(y0)``. ``fun`` must
         |      return an array of the same shape as ``y``. See `vectorized` for more
         |      information.
         |  t0 : float
         |      Initial time.
         |  y0 : array_like, shape (n,)
         |      Initial state.
         |  t_bound : float
         |      Boundary time - the integration won't continue beyond it. It also
         |      determines the direction of the integration.
         |  first_step : float or None, optional
         |      Initial step size. Default is ``None`` which means that the algorithm
         |      should choose.
         |  min_step : float, optional
         |      Minimum allowed step size. Default is 0.0, i.e., the step size is not
         |      bounded and determined solely by the solver.
         |  max_step : float, optional
         |      Maximum allowed step size. Default is np.inf, i.e., the step size is not
         |      bounded and determined solely by the solver.
         |  rtol, atol : float and array_like, optional
         |      Relative and absolute tolerances. The solver keeps the local error
         |      estimates less than ``atol + rtol * abs(y)``. Here `rtol` controls a
         |      relative accuracy (number of correct digits), while `atol` controls
         |      absolute accuracy (number of correct decimal places). To achieve the
         |      desired `rtol`, set `atol` to be smaller than the smallest value that
         |      can be expected from ``rtol * abs(y)`` so that `rtol` dominates the
         |      allowable error. If `atol` is larger than ``rtol * abs(y)`` the
         |      number of correct digits is not guaranteed. Conversely, to achieve the
         |      desired `atol` set `rtol` such that ``rtol * abs(y)`` is always smaller
         |      than `atol`. If components of y have different scales, it might be
         |      beneficial to set different `atol` values for different components by
         |      passing array_like with shape (n,) for `atol`. Default values are
         |      1e-3 for `rtol` and 1e-6 for `atol`.
         |  jac : None or callable, optional
         |      Jacobian matrix of the right-hand side of the system with respect to
         |      ``y``. The Jacobian matrix has shape (n, n) and its element (i, j) is
         |      equal to ``d f_i / d y_j``. The function will be called as
         |      ``jac(t, y)``. If None (default), the Jacobian will be
         |      approximated by finite differences. It is generally recommended to
         |      provide the Jacobian rather than relying on a finite-difference
         |      approximation.
         |  lband, uband : int or None
         |      Parameters defining the bandwidth of the Jacobian,
         |      i.e., ``jac[i, j] != 0 only for i - lband <= j <= i + uband``. Setting
         |      these requires your jac routine to return the Jacobian in the packed format:
         |      the returned array must have ``n`` columns and ``uband + lband + 1``
         |      rows in which Jacobian diagonals are written. Specifically
         |      ``jac_packed[uband + i - j , j] = jac[i, j]``. The same format is used
         |      in `scipy.linalg.solve_banded` (check for an illustration).
         |      These parameters can be also used with ``jac=None`` to reduce the
         |      number of Jacobian elements estimated by finite differences.
         |  vectorized : bool, optional
         |      Whether `fun` may be called in a vectorized fashion. False (default)
         |      is recommended for this solver.
         |
         |      If ``vectorized`` is False, `fun` will always be called with ``y`` of
         |      shape ``(n,)``, where ``n = len(y0)``.
         |
         |      If ``vectorized`` is True, `fun` may be called with ``y`` of shape
         |      ``(n, k)``, where ``k`` is an integer. In this case, `fun` must behave
         |      such that ``fun(t, y)[:, i] == fun(t, y[:, i])`` (i.e. each column of
         |      the returned array is the time derivative of the state corresponding
         |      with a column of ``y``).
         |
         |      Setting ``vectorized=True`` allows for faster finite difference
         |      approximation of the Jacobian by methods 'Radau' and 'BDF', but
         |      will result in slower execution for this solver.
         |
         |  Attributes
         |  ----------
         |  n : int
         |      Number of equations.
         |  status : string
         |      Current status of the solver: 'running', 'finished' or 'failed'.
         |  t_bound : float
         |      Boundary time.
         |  direction : float
         |      Integration direction: +1 or -1.
         |  t : float
         |      Current time.
         |  y : ndarray
         |      Current state.
         |  t_old : float
         |      Previous time. None if no steps were made yet.
         |  nfev : int
         |      Number of evaluations of the right-hand side.
         |  njev : int
         |      Number of evaluations of the Jacobian.
         |
         |  References
         |  ----------
         |  .. [1] A. C. Hindmarsh, "ODEPACK, A Systematized Collection of ODE
         |         Solvers," IMACS Transactions on Scientific Computation, Vol 1.,
         |         pp. 55-64, 1983.
         |  .. [2] L. Petzold, "Automatic selection of methods for solving stiff and
         |         nonstiff systems of ordinary differential equations", SIAM Journal
         |         on Scientific and Statistical Computing, Vol. 4, No. 1, pp. 136-148,
         |         1983.
         |
         |  Method resolution order:
         |      LSODA
         |      scipy.integrate._ivp.base.OdeSolver
         |      builtins.object
         |
         |  Methods defined here:
         |
         |  __init__(self, fun, t0, y0, t_bound, first_step=None, min_step=0.0, max_step=inf, rtol=0.001, atol=1e-06, jac=None, lband=None, uband=None, vectorized=False, **extraneous)
         |      Initialize self.  See help(type(self)) for accurate signature.
         |
         |  ----------------------------------------------------------------------
         |  Data and other attributes defined here:
         |
         |  __annotations__ = {}
         |
         |  ----------------------------------------------------------------------
         |  Methods inherited from scipy.integrate._ivp.base.OdeSolver:
         |
         |  dense_output(self)
         |      Compute a local interpolant over the last successful step.
         |
         |      Returns
         |      -------
         |      sol : `DenseOutput`
         |          Local interpolant over the last successful step.
         |
         |  step(self)
         |      Perform one integration step.
         |
         |      Returns
         |      -------
         |      message : string or None
         |          Report from the solver. Typically a reason for a failure if
         |          `self.status` is 'failed' after the step was taken or None
         |          otherwise.
         |
         |  ----------------------------------------------------------------------
         |  Readonly properties inherited from scipy.integrate._ivp.base.OdeSolver:
         |
         |  step_size
         |
         |  ----------------------------------------------------------------------
         |  Data descriptors inherited from scipy.integrate._ivp.base.OdeSolver:
         |
         |  __dict__
         |      dictionary for instance variables
         |
         |  __weakref__
         |      list of weak references to the object
         |
         |  ----------------------------------------------------------------------
         |  Data and other attributes inherited from scipy.integrate._ivp.base.OdeSolver:
         |
         |  TOO_SMALL_STEP = 'Required step size is less than spacing between numb...
    
        class ODEintWarning(builtins.Warning)
         |  Warning raised during the execution of `odeint`.
         |
         |  Method resolution order:
         |      ODEintWarning
         |      builtins.Warning
         |      builtins.Exception
         |      builtins.BaseException
         |      builtins.object
         |
         |  Data descriptors defined here:
         |
         |  __weakref__
         |      list of weak references to the object
         |
         |  ----------------------------------------------------------------------
         |  Methods inherited from builtins.Warning:
         |
         |  __init__(self, /, *args, **kwargs)
         |      Initialize self.  See help(type(self)) for accurate signature.
         |
         |  ----------------------------------------------------------------------
         |  Static methods inherited from builtins.Warning:
         |
         |  __new__(*args, **kwargs) class method of builtins.Warning
         |      Create and return a new object.  See help(type) for accurate signature.
         |
         |  ----------------------------------------------------------------------
         |  Methods inherited from builtins.BaseException:
         |
         |  __getattribute__(self, name, /)
         |      Return getattr(self, name).
         |
         |  __reduce__(...)
         |      Helper for pickle.
         |
         |  __repr__(self, /)
         |      Return repr(self).
         |
         |  __setstate__(...)
         |
         |  __str__(self, /)
         |      Return str(self).
         |
         |  add_note(...)
         |      Exception.add_note(note) --
         |      add a note to the exception
         |
         |  with_traceback(...)
         |      Exception.with_traceback(tb) --
         |      set self.__traceback__ to tb and return self.
         |
         |  ----------------------------------------------------------------------
         |  Data descriptors inherited from builtins.BaseException:
         |
         |  __cause__
         |      exception cause
         |
         |  __context__
         |      exception context
         |
         |  __dict__
         |
         |  __suppress_context__
         |
         |  __traceback__
         |
         |  args
    
        class OdeSolution(builtins.object)
         |  OdeSolution(ts, interpolants, alt_segment=False)
         |
         |  Continuous ODE solution.
         |
         |  It is organized as a collection of `DenseOutput` objects which represent
         |  local interpolants. It provides an algorithm to select a right interpolant
         |  for each given point.
         |
         |  The interpolants cover the range between `t_min` and `t_max` (see
         |  Attributes below). Evaluation outside this interval is not forbidden, but
         |  the accuracy is not guaranteed.
         |
         |  When evaluating at a breakpoint (one of the values in `ts`) a segment with
         |  the lower index is selected.
         |
         |  Parameters
         |  ----------
         |  ts : array_like, shape (n_segments + 1,)
         |      Time instants between which local interpolants are defined. Must
         |      be strictly increasing or decreasing (zero segment with two points is
         |      also allowed).
         |  interpolants : list of DenseOutput with n_segments elements
         |      Local interpolants. An i-th interpolant is assumed to be defined
         |      between ``ts[i]`` and ``ts[i + 1]``.
         |  alt_segment : boolean
         |      Requests the alternative interpolant segment selection scheme. At each
         |      solver integration point, two interpolant segments are available. The
         |      default (False) and alternative (True) behaviours select the segment
         |      for which the requested time corresponded to ``t`` and ``t_old``,
         |      respectively. This functionality is only relevant for testing the
         |      interpolants' accuracy: different integrators use different
         |      construction strategies.
         |
         |  Attributes
         |  ----------
         |  t_min, t_max : float
         |      Time range of the interpolation.
         |
         |  Methods defined here:
         |
         |  __call__(self, t)
         |      Evaluate the solution.
         |
         |      Parameters
         |      ----------
         |      t : float or array_like with shape (n_points,)
         |          Points to evaluate at.
         |
         |      Returns
         |      -------
         |      y : ndarray, shape (n_states,) or (n_states, n_points)
         |          Computed values. Shape depends on whether `t` is a scalar or a
         |          1-D array.
         |
         |  __init__(self, ts, interpolants, alt_segment=False)
         |      Initialize self.  See help(type(self)) for accurate signature.
         |
         |  ----------------------------------------------------------------------
         |  Data descriptors defined here:
         |
         |  __dict__
         |      dictionary for instance variables
         |
         |  __weakref__
         |      list of weak references to the object
    
        class OdeSolver(builtins.object)
         |  OdeSolver(fun, t0, y0, t_bound, vectorized, support_complex=False)
         |
         |  Base class for ODE solvers.
         |
         |  In order to implement a new solver you need to follow the guidelines:
         |
         |      1. A constructor must accept parameters presented in the base class
         |         (listed below) along with any other parameters specific to a solver.
         |      2. A constructor must accept arbitrary extraneous arguments
         |         ``**extraneous``, but warn that these arguments are irrelevant
         |         using `common.warn_extraneous` function. Do not pass these
         |         arguments to the base class.
         |      3. A solver must implement a private method `_step_impl(self)` which
         |         propagates a solver one step further. It must return tuple
         |         ``(success, message)``, where ``success`` is a boolean indicating
         |         whether a step was successful, and ``message`` is a string
         |         containing description of a failure if a step failed or None
         |         otherwise.
         |      4. A solver must implement a private method `_dense_output_impl(self)`,
         |         which returns a `DenseOutput` object covering the last successful
         |         step.
         |      5. A solver must have attributes listed below in Attributes section.
         |         Note that ``t_old`` and ``step_size`` are updated automatically.
         |      6. Use `fun(self, t, y)` method for the system rhs evaluation, this
         |         way the number of function evaluations (`nfev`) will be tracked
         |         automatically.
         |      7. For convenience, a base class provides `fun_single(self, t, y)` and
         |         `fun_vectorized(self, t, y)` for evaluating the rhs in
         |         non-vectorized and vectorized fashions respectively (regardless of
         |         how `fun` from the constructor is implemented). These calls don't
         |         increment `nfev`.
         |      8. If a solver uses a Jacobian matrix and LU decompositions, it should
         |         track the number of Jacobian evaluations (`njev`) and the number of
         |         LU decompositions (`nlu`).
         |      9. By convention, the function evaluations used to compute a finite
         |         difference approximation of the Jacobian should not be counted in
         |         `nfev`, thus use `fun_single(self, t, y)` or
         |         `fun_vectorized(self, t, y)` when computing a finite difference
         |         approximation of the Jacobian.
         |
         |  Parameters
         |  ----------
         |  fun : callable
         |      Right-hand side of the system: the time derivative of the state ``y``
         |      at time ``t``. The calling signature is ``fun(t, y)``, where ``t`` is a
         |      scalar and ``y`` is an ndarray with ``len(y) = len(y0)``. ``fun`` must
         |      return an array of the same shape as ``y``. See `vectorized` for more
         |      information.
         |  t0 : float
         |      Initial time.
         |  y0 : array_like, shape (n,)
         |      Initial state.
         |  t_bound : float
         |      Boundary time --- the integration won't continue beyond it. It also
         |      determines the direction of the integration.
         |  vectorized : bool
         |      Whether `fun` can be called in a vectorized fashion. Default is False.
         |
         |      If ``vectorized`` is False, `fun` will always be called with ``y`` of
         |      shape ``(n,)``, where ``n = len(y0)``.
         |
         |      If ``vectorized`` is True, `fun` may be called with ``y`` of shape
         |      ``(n, k)``, where ``k`` is an integer. In this case, `fun` must behave
         |      such that ``fun(t, y)[:, i] == fun(t, y[:, i])`` (i.e. each column of
         |      the returned array is the time derivative of the state corresponding
         |      with a column of ``y``).
         |
         |      Setting ``vectorized=True`` allows for faster finite difference
         |      approximation of the Jacobian by methods 'Radau' and 'BDF', but
         |      will result in slower execution for other methods. It can also
         |      result in slower overall execution for 'Radau' and 'BDF' in some
         |      circumstances (e.g. small ``len(y0)``).
         |  support_complex : bool, optional
         |      Whether integration in a complex domain should be supported.
         |      Generally determined by a derived solver class capabilities.
         |      Default is False.
         |
         |  Attributes
         |  ----------
         |  n : int
         |      Number of equations.
         |  status : string
         |      Current status of the solver: 'running', 'finished' or 'failed'.
         |  t_bound : float
         |      Boundary time.
         |  direction : float
         |      Integration direction: +1 or -1.
         |  t : float
         |      Current time.
         |  y : ndarray
         |      Current state.
         |  t_old : float
         |      Previous time. None if no steps were made yet.
         |  step_size : float
         |      Size of the last successful step. None if no steps were made yet.
         |  nfev : int
         |      Number of the system's rhs evaluations.
         |  njev : int
         |      Number of the Jacobian evaluations.
         |  nlu : int
         |      Number of LU decompositions.
         |
         |  Methods defined here:
         |
         |  __init__(self, fun, t0, y0, t_bound, vectorized, support_complex=False)
         |      Initialize self.  See help(type(self)) for accurate signature.
         |
         |  dense_output(self)
         |      Compute a local interpolant over the last successful step.
         |
         |      Returns
         |      -------
         |      sol : `DenseOutput`
         |          Local interpolant over the last successful step.
         |
         |  step(self)
         |      Perform one integration step.
         |
         |      Returns
         |      -------
         |      message : string or None
         |          Report from the solver. Typically a reason for a failure if
         |          `self.status` is 'failed' after the step was taken or None
         |          otherwise.
         |
         |  ----------------------------------------------------------------------
         |  Readonly properties defined here:
         |
         |  step_size
         |
         |  ----------------------------------------------------------------------
         |  Data descriptors defined here:
         |
         |  __dict__
         |      dictionary for instance variables
         |
         |  __weakref__
         |      list of weak references to the object
         |
         |  ----------------------------------------------------------------------
         |  Data and other attributes defined here:
         |
         |  TOO_SMALL_STEP = 'Required step size is less than spacing between numb...
         |
         |  __annotations__ = {}
    
        class RK23(RungeKutta)
         |  RK23(fun, t0, y0, t_bound, max_step=inf, rtol=0.001, atol=1e-06, vectorized=False, first_step=None, **extraneous)
         |
         |  Explicit Runge-Kutta method of order 3(2).
         |
         |  This uses the Bogacki-Shampine pair of formulas [1]_. The error is controlled
         |  assuming accuracy of the second-order method, but steps are taken using the
         |  third-order accurate formula (local extrapolation is done). A cubic Hermite
         |  polynomial is used for the dense output.
         |
         |  Can be applied in the complex domain.
         |
         |  Parameters
         |  ----------
         |  fun : callable
         |      Right-hand side of the system: the time derivative of the state ``y``
         |      at time ``t``. The calling signature is ``fun(t, y)``, where ``t`` is a
         |      scalar and ``y`` is an ndarray with ``len(y) = len(y0)``. ``fun`` must
         |      return an array of the same shape as ``y``. See `vectorized` for more
         |      information.
         |  t0 : float
         |      Initial time.
         |  y0 : array_like, shape (n,)
         |      Initial state.
         |  t_bound : float
         |      Boundary time - the integration won't continue beyond it. It also
         |      determines the direction of the integration.
         |  first_step : float or None, optional
         |      Initial step size. Default is ``None`` which means that the algorithm
         |      should choose.
         |  max_step : float, optional
         |      Maximum allowed step size. Default is np.inf, i.e., the step size is not
         |      bounded and determined solely by the solver.
         |  rtol, atol : float and array_like, optional
         |      Relative and absolute tolerances. The solver keeps the local error
         |      estimates less than ``atol + rtol * abs(y)``. Here `rtol` controls a
         |      relative accuracy (number of correct digits), while `atol` controls
         |      absolute accuracy (number of correct decimal places). To achieve the
         |      desired `rtol`, set `atol` to be smaller than the smallest value that
         |      can be expected from ``rtol * abs(y)`` so that `rtol` dominates the
         |      allowable error. If `atol` is larger than ``rtol * abs(y)`` the
         |      number of correct digits is not guaranteed. Conversely, to achieve the
         |      desired `atol` set `rtol` such that ``rtol * abs(y)`` is always smaller
         |      than `atol`. If components of y have different scales, it might be
         |      beneficial to set different `atol` values for different components by
         |      passing array_like with shape (n,) for `atol`. Default values are
         |      1e-3 for `rtol` and 1e-6 for `atol`.
         |  vectorized : bool, optional
         |      Whether `fun` may be called in a vectorized fashion. False (default)
         |      is recommended for this solver.
         |
         |      If ``vectorized`` is False, `fun` will always be called with ``y`` of
         |      shape ``(n,)``, where ``n = len(y0)``.
         |
         |      If ``vectorized`` is True, `fun` may be called with ``y`` of shape
         |      ``(n, k)``, where ``k`` is an integer. In this case, `fun` must behave
         |      such that ``fun(t, y)[:, i] == fun(t, y[:, i])`` (i.e. each column of
         |      the returned array is the time derivative of the state corresponding
         |      with a column of ``y``).
         |
         |      Setting ``vectorized=True`` allows for faster finite difference
         |      approximation of the Jacobian by methods 'Radau' and 'BDF', but
         |      will result in slower execution for this solver.
         |
         |  Attributes
         |  ----------
         |  n : int
         |      Number of equations.
         |  status : string
         |      Current status of the solver: 'running', 'finished' or 'failed'.
         |  t_bound : float
         |      Boundary time.
         |  direction : float
         |      Integration direction: +1 or -1.
         |  t : float
         |      Current time.
         |  y : ndarray
         |      Current state.
         |  t_old : float
         |      Previous time. None if no steps were made yet.
         |  step_size : float
         |      Size of the last successful step. None if no steps were made yet.
         |  nfev : int
         |      Number evaluations of the system's right-hand side.
         |  njev : int
         |      Number of evaluations of the Jacobian.
         |      Is always 0 for this solver as it does not use the Jacobian.
         |  nlu : int
         |      Number of LU decompositions. Is always 0 for this solver.
         |
         |  References
         |  ----------
         |  .. [1] P. Bogacki, L.F. Shampine, "A 3(2) Pair of Runge-Kutta Formulas",
         |         Appl. Math. Lett. Vol. 2, No. 4. pp. 321-325, 1989.
         |
         |  Method resolution order:
         |      RK23
         |      RungeKutta
         |      scipy.integrate._ivp.base.OdeSolver
         |      builtins.object
         |
         |  Data and other attributes defined here:
         |
         |  A = array([[0.  , 0.  , 0.  ],
         |         [0.5 , 0.  , 0.  ],
         |         [0.  ...
         |
         |  B = array([0.22222222, 0.33333333, 0.44444444])
         |
         |  C = array([0.  , 0.5 , 0.75])
         |
         |  E = array([ 0.06944444, -0.08333333, -0.11111111,  0.125     ])
         |
         |  P = array([[ 1.        , -1.33333333,  0.55555556],
         |  ...
         |         [ 0.   ...
         |
         |  __annotations__ = {}
         |
         |  error_estimator_order = 2
         |
         |  n_stages = 3
         |
         |  order = 3
         |
         |  ----------------------------------------------------------------------
         |  Methods inherited from RungeKutta:
         |
         |  __init__(self, fun, t0, y0, t_bound, max_step=inf, rtol=0.001, atol=1e-06, vectorized=False, first_step=None, **extraneous)
         |      Initialize self.  See help(type(self)) for accurate signature.
         |
         |  ----------------------------------------------------------------------
         |  Methods inherited from scipy.integrate._ivp.base.OdeSolver:
         |
         |  dense_output(self)
         |      Compute a local interpolant over the last successful step.
         |
         |      Returns
         |      -------
         |      sol : `DenseOutput`
         |          Local interpolant over the last successful step.
         |
         |  step(self)
         |      Perform one integration step.
         |
         |      Returns
         |      -------
         |      message : string or None
         |          Report from the solver. Typically a reason for a failure if
         |          `self.status` is 'failed' after the step was taken or None
         |          otherwise.
         |
         |  ----------------------------------------------------------------------
         |  Readonly properties inherited from scipy.integrate._ivp.base.OdeSolver:
         |
         |  step_size
         |
         |  ----------------------------------------------------------------------
         |  Data descriptors inherited from scipy.integrate._ivp.base.OdeSolver:
         |
         |  __dict__
         |      dictionary for instance variables
         |
         |  __weakref__
         |      list of weak references to the object
         |
         |  ----------------------------------------------------------------------
         |  Data and other attributes inherited from scipy.integrate._ivp.base.OdeSolver:
         |
         |  TOO_SMALL_STEP = 'Required step size is less than spacing between numb...
    
        class RK45(RungeKutta)
         |  RK45(fun, t0, y0, t_bound, max_step=inf, rtol=0.001, atol=1e-06, vectorized=False, first_step=None, **extraneous)
         |
         |  Explicit Runge-Kutta method of order 5(4).
         |
         |  This uses the Dormand-Prince pair of formulas [1]_. The error is controlled
         |  assuming accuracy of the fourth-order method accuracy, but steps are taken
         |  using the fifth-order accurate formula (local extrapolation is done).
         |  A quartic interpolation polynomial is used for the dense output [2]_.
         |
         |  Can be applied in the complex domain.
         |
         |  Parameters
         |  ----------
         |  fun : callable
         |      Right-hand side of the system. The calling signature is ``fun(t, y)``.
         |      Here ``t`` is a scalar, and there are two options for the ndarray ``y``:
         |      It can either have shape (n,); then ``fun`` must return array_like with
         |      shape (n,). Alternatively it can have shape (n, k); then ``fun``
         |      must return an array_like with shape (n, k), i.e., each column
         |      corresponds to a single column in ``y``. The choice between the two
         |      options is determined by `vectorized` argument (see below).
         |  t0 : float
         |      Initial time.
         |  y0 : array_like, shape (n,)
         |      Initial state.
         |  t_bound : float
         |      Boundary time - the integration won't continue beyond it. It also
         |      determines the direction of the integration.
         |  first_step : float or None, optional
         |      Initial step size. Default is ``None`` which means that the algorithm
         |      should choose.
         |  max_step : float, optional
         |      Maximum allowed step size. Default is np.inf, i.e., the step size is not
         |      bounded and determined solely by the solver.
         |  rtol, atol : float and array_like, optional
         |      Relative and absolute tolerances. The solver keeps the local error
         |      estimates less than ``atol + rtol * abs(y)``. Here `rtol` controls a
         |      relative accuracy (number of correct digits), while `atol` controls
         |      absolute accuracy (number of correct decimal places). To achieve the
         |      desired `rtol`, set `atol` to be smaller than the smallest value that
         |      can be expected from ``rtol * abs(y)`` so that `rtol` dominates the
         |      allowable error. If `atol` is larger than ``rtol * abs(y)`` the
         |      number of correct digits is not guaranteed. Conversely, to achieve the
         |      desired `atol` set `rtol` such that ``rtol * abs(y)`` is always smaller
         |      than `atol`. If components of y have different scales, it might be
         |      beneficial to set different `atol` values for different components by
         |      passing array_like with shape (n,) for `atol`. Default values are
         |      1e-3 for `rtol` and 1e-6 for `atol`.
         |  vectorized : bool, optional
         |      Whether `fun` is implemented in a vectorized fashion. Default is False.
         |
         |  Attributes
         |  ----------
         |  n : int
         |      Number of equations.
         |  status : string
         |      Current status of the solver: 'running', 'finished' or 'failed'.
         |  t_bound : float
         |      Boundary time.
         |  direction : float
         |      Integration direction: +1 or -1.
         |  t : float
         |      Current time.
         |  y : ndarray
         |      Current state.
         |  t_old : float
         |      Previous time. None if no steps were made yet.
         |  step_size : float
         |      Size of the last successful step. None if no steps were made yet.
         |  nfev : int
         |      Number evaluations of the system's right-hand side.
         |  njev : int
         |      Number of evaluations of the Jacobian.
         |      Is always 0 for this solver as it does not use the Jacobian.
         |  nlu : int
         |      Number of LU decompositions. Is always 0 for this solver.
         |
         |  References
         |  ----------
         |  .. [1] J. R. Dormand, P. J. Prince, "A family of embedded Runge-Kutta
         |         formulae", Journal of Computational and Applied Mathematics, Vol. 6,
         |         No. 1, pp. 19-26, 1980.
         |  .. [2] L. W. Shampine, "Some Practical Runge-Kutta Formulas", Mathematics
         |         of Computation,, Vol. 46, No. 173, pp. 135-150, 1986.
         |
         |  Method resolution order:
         |      RK45
         |      RungeKutta
         |      scipy.integrate._ivp.base.OdeSolver
         |      builtins.object
         |
         |  Data and other attributes defined here:
         |
         |  A = array([[  0.        ,   0.        ,   0.        ...8.90642272,   0...
         |
         |  B = array([ 0.09114583,  0.        ,  0.4492363 ,  0.65104167, -0.3223...
         |
         |  C = array([0.        , 0.2       , 0.3       , 0.8       , 0.88888889,...
         |
         |  E = array([-0.00123264,  0.        ,  0.00425277, -0...7,  0.0508638 ,...
         |
         |  P = array([[ 1.        , -2.85358007,  3.07174346, -...       ,  1.382...
         |
         |  __annotations__ = {}
         |
         |  error_estimator_order = 4
         |
         |  n_stages = 6
         |
         |  order = 5
         |
         |  ----------------------------------------------------------------------
         |  Methods inherited from RungeKutta:
         |
         |  __init__(self, fun, t0, y0, t_bound, max_step=inf, rtol=0.001, atol=1e-06, vectorized=False, first_step=None, **extraneous)
         |      Initialize self.  See help(type(self)) for accurate signature.
         |
         |  ----------------------------------------------------------------------
         |  Methods inherited from scipy.integrate._ivp.base.OdeSolver:
         |
         |  dense_output(self)
         |      Compute a local interpolant over the last successful step.
         |
         |      Returns
         |      -------
         |      sol : `DenseOutput`
         |          Local interpolant over the last successful step.
         |
         |  step(self)
         |      Perform one integration step.
         |
         |      Returns
         |      -------
         |      message : string or None
         |          Report from the solver. Typically a reason for a failure if
         |          `self.status` is 'failed' after the step was taken or None
         |          otherwise.
         |
         |  ----------------------------------------------------------------------
         |  Readonly properties inherited from scipy.integrate._ivp.base.OdeSolver:
         |
         |  step_size
         |
         |  ----------------------------------------------------------------------
         |  Data descriptors inherited from scipy.integrate._ivp.base.OdeSolver:
         |
         |  __dict__
         |      dictionary for instance variables
         |
         |  __weakref__
         |      list of weak references to the object
         |
         |  ----------------------------------------------------------------------
         |  Data and other attributes inherited from scipy.integrate._ivp.base.OdeSolver:
         |
         |  TOO_SMALL_STEP = 'Required step size is less than spacing between numb...
    
        class Radau(scipy.integrate._ivp.base.OdeSolver)
         |  Radau(fun, t0, y0, t_bound, max_step=inf, rtol=0.001, atol=1e-06, jac=None, jac_sparsity=None, vectorized=False, first_step=None, **extraneous)
         |
         |  Implicit Runge-Kutta method of Radau IIA family of order 5.
         |
         |  The implementation follows [1]_. The error is controlled with a
         |  third-order accurate embedded formula. A cubic polynomial which satisfies
         |  the collocation conditions is used for the dense output.
         |
         |  Parameters
         |  ----------
         |  fun : callable
         |      Right-hand side of the system: the time derivative of the state ``y``
         |      at time ``t``. The calling signature is ``fun(t, y)``, where ``t`` is a
         |      scalar and ``y`` is an ndarray with ``len(y) = len(y0)``. ``fun`` must
         |      return an array of the same shape as ``y``. See `vectorized` for more
         |      information.
         |  t0 : float
         |      Initial time.
         |  y0 : array_like, shape (n,)
         |      Initial state.
         |  t_bound : float
         |      Boundary time - the integration won't continue beyond it. It also
         |      determines the direction of the integration.
         |  first_step : float or None, optional
         |      Initial step size. Default is ``None`` which means that the algorithm
         |      should choose.
         |  max_step : float, optional
         |      Maximum allowed step size. Default is np.inf, i.e., the step size is not
         |      bounded and determined solely by the solver.
         |  rtol, atol : float and array_like, optional
         |      Relative and absolute tolerances. The solver keeps the local error
         |      estimates less than ``atol + rtol * abs(y)``. HHere `rtol` controls a
         |      relative accuracy (number of correct digits), while `atol` controls
         |      absolute accuracy (number of correct decimal places). To achieve the
         |      desired `rtol`, set `atol` to be smaller than the smallest value that
         |      can be expected from ``rtol * abs(y)`` so that `rtol` dominates the
         |      allowable error. If `atol` is larger than ``rtol * abs(y)`` the
         |      number of correct digits is not guaranteed. Conversely, to achieve the
         |      desired `atol` set `rtol` such that ``rtol * abs(y)`` is always smaller
         |      than `atol`. If components of y have different scales, it might be
         |      beneficial to set different `atol` values for different components by
         |      passing array_like with shape (n,) for `atol`. Default values are
         |      1e-3 for `rtol` and 1e-6 for `atol`.
         |  jac : {None, array_like, sparse_matrix, callable}, optional
         |      Jacobian matrix of the right-hand side of the system with respect to
         |      y, required by this method. The Jacobian matrix has shape (n, n) and
         |      its element (i, j) is equal to ``d f_i / d y_j``.
         |      There are three ways to define the Jacobian:
         |
         |          * If array_like or sparse_matrix, the Jacobian is assumed to
         |            be constant.
         |          * If callable, the Jacobian is assumed to depend on both
         |            t and y; it will be called as ``jac(t, y)`` as necessary.
         |            For the 'Radau' and 'BDF' methods, the return value might be a
         |            sparse matrix.
         |          * If None (default), the Jacobian will be approximated by
         |            finite differences.
         |
         |      It is generally recommended to provide the Jacobian rather than
         |      relying on a finite-difference approximation.
         |  jac_sparsity : {None, array_like, sparse matrix}, optional
         |      Defines a sparsity structure of the Jacobian matrix for a
         |      finite-difference approximation. Its shape must be (n, n). This argument
         |      is ignored if `jac` is not `None`. If the Jacobian has only few non-zero
         |      elements in *each* row, providing the sparsity structure will greatly
         |      speed up the computations [2]_. A zero entry means that a corresponding
         |      element in the Jacobian is always zero. If None (default), the Jacobian
         |      is assumed to be dense.
         |  vectorized : bool, optional
         |      Whether `fun` can be called in a vectorized fashion. Default is False.
         |
         |      If ``vectorized`` is False, `fun` will always be called with ``y`` of
         |      shape ``(n,)``, where ``n = len(y0)``.
         |
         |      If ``vectorized`` is True, `fun` may be called with ``y`` of shape
         |      ``(n, k)``, where ``k`` is an integer. In this case, `fun` must behave
         |      such that ``fun(t, y)[:, i] == fun(t, y[:, i])`` (i.e. each column of
         |      the returned array is the time derivative of the state corresponding
         |      with a column of ``y``).
         |
         |      Setting ``vectorized=True`` allows for faster finite difference
         |      approximation of the Jacobian by this method, but may result in slower
         |      execution overall in some circumstances (e.g. small ``len(y0)``).
         |
         |  Attributes
         |  ----------
         |  n : int
         |      Number of equations.
         |  status : string
         |      Current status of the solver: 'running', 'finished' or 'failed'.
         |  t_bound : float
         |      Boundary time.
         |  direction : float
         |      Integration direction: +1 or -1.
         |  t : float
         |      Current time.
         |  y : ndarray
         |      Current state.
         |  t_old : float
         |      Previous time. None if no steps were made yet.
         |  step_size : float
         |      Size of the last successful step. None if no steps were made yet.
         |  nfev : int
         |      Number of evaluations of the right-hand side.
         |  njev : int
         |      Number of evaluations of the Jacobian.
         |  nlu : int
         |      Number of LU decompositions.
         |
         |  References
         |  ----------
         |  .. [1] E. Hairer, G. Wanner, "Solving Ordinary Differential Equations II:
         |         Stiff and Differential-Algebraic Problems", Sec. IV.8.
         |  .. [2] A. Curtis, M. J. D. Powell, and J. Reid, "On the estimation of
         |         sparse Jacobian matrices", Journal of the Institute of Mathematics
         |         and its Applications, 13, pp. 117-120, 1974.
         |
         |  Method resolution order:
         |      Radau
         |      scipy.integrate._ivp.base.OdeSolver
         |      builtins.object
         |
         |  Methods defined here:
         |
         |  __init__(self, fun, t0, y0, t_bound, max_step=inf, rtol=0.001, atol=1e-06, jac=None, jac_sparsity=None, vectorized=False, first_step=None, **extraneous)
         |      Initialize self.  See help(type(self)) for accurate signature.
         |
         |  ----------------------------------------------------------------------
         |  Data and other attributes defined here:
         |
         |  __annotations__ = {}
         |
         |  ----------------------------------------------------------------------
         |  Methods inherited from scipy.integrate._ivp.base.OdeSolver:
         |
         |  dense_output(self)
         |      Compute a local interpolant over the last successful step.
         |
         |      Returns
         |      -------
         |      sol : `DenseOutput`
         |          Local interpolant over the last successful step.
         |
         |  step(self)
         |      Perform one integration step.
         |
         |      Returns
         |      -------
         |      message : string or None
         |          Report from the solver. Typically a reason for a failure if
         |          `self.status` is 'failed' after the step was taken or None
         |          otherwise.
         |
         |  ----------------------------------------------------------------------
         |  Readonly properties inherited from scipy.integrate._ivp.base.OdeSolver:
         |
         |  step_size
         |
         |  ----------------------------------------------------------------------
         |  Data descriptors inherited from scipy.integrate._ivp.base.OdeSolver:
         |
         |  __dict__
         |      dictionary for instance variables
         |
         |  __weakref__
         |      list of weak references to the object
         |
         |  ----------------------------------------------------------------------
         |  Data and other attributes inherited from scipy.integrate._ivp.base.OdeSolver:
         |
         |  TOO_SMALL_STEP = 'Required step size is less than spacing between numb...
    
        class complex_ode(ode)
         |  complex_ode(f, jac=None)
         |
         |  A wrapper of ode for complex systems.
         |
         |  This functions similarly as `ode`, but re-maps a complex-valued
         |  equation system to a real-valued one before using the integrators.
         |
         |  Parameters
         |  ----------
         |  f : callable ``f(t, y, *f_args)``
         |      Rhs of the equation. t is a scalar, ``y.shape == (n,)``.
         |      ``f_args`` is set by calling ``set_f_params(*args)``.
         |  jac : callable ``jac(t, y, *jac_args)``
         |      Jacobian of the rhs, ``jac[i,j] = d f[i] / d y[j]``.
         |      ``jac_args`` is set by calling ``set_f_params(*args)``.
         |
         |  Attributes
         |  ----------
         |  t : float
         |      Current time.
         |  y : ndarray
         |      Current variable values.
         |
         |  Examples
         |  --------
         |  For usage examples, see `ode`.
         |
         |  Method resolution order:
         |      complex_ode
         |      ode
         |      builtins.object
         |
         |  Methods defined here:
         |
         |  __init__(self, f, jac=None)
         |      Initialize self.  See help(type(self)) for accurate signature.
         |
         |  integrate(self, t, step=False, relax=False)
         |      Find y=y(t), set y as an initial condition, and return y.
         |
         |      Parameters
         |      ----------
         |      t : float
         |          The endpoint of the integration step.
         |      step : bool
         |          If True, and if the integrator supports the step method,
         |          then perform a single integration step and return.
         |          This parameter is provided in order to expose internals of
         |          the implementation, and should not be changed from its default
         |          value in most cases.
         |      relax : bool
         |          If True and if the integrator supports the run_relax method,
         |          then integrate until t_1 >= t and return. ``relax`` is not
         |          referenced if ``step=True``.
         |          This parameter is provided in order to expose internals of
         |          the implementation, and should not be changed from its default
         |          value in most cases.
         |
         |      Returns
         |      -------
         |      y : float
         |          The integrated value at t
         |
         |  set_initial_value(self, y, t=0.0)
         |      Set initial conditions y(t) = y.
         |
         |  set_integrator(self, name, **integrator_params)
         |      Set integrator by name.
         |
         |      Parameters
         |      ----------
         |      name : str
         |          Name of the integrator
         |      **integrator_params
         |          Additional parameters for the integrator.
         |
         |  set_solout(self, solout)
         |      Set callable to be called at every successful integration step.
         |
         |      Parameters
         |      ----------
         |      solout : callable
         |          ``solout(t, y)`` is called at each internal integrator step,
         |          t is a scalar providing the current independent position
         |          y is the current solution ``y.shape == (n,)``
         |          solout should return -1 to stop integration
         |          otherwise it should return None or 0
         |
         |  ----------------------------------------------------------------------
         |  Readonly properties defined here:
         |
         |  y
         |
         |  ----------------------------------------------------------------------
         |  Methods inherited from ode:
         |
         |  get_return_code(self)
         |      Extracts the return code for the integration to enable better control
         |      if the integration fails.
         |
         |      In general, a return code > 0 implies success, while a return code < 0
         |      implies failure.
         |
         |      Notes
         |      -----
         |      This section describes possible return codes and their meaning, for available
         |      integrators that can be selected by `set_integrator` method.
         |
         |      "vode"
         |
         |      ===========  =======
         |      Return Code  Message
         |      ===========  =======
         |      2            Integration successful.
         |      -1           Excess work done on this call. (Perhaps wrong MF.)
         |      -2           Excess accuracy requested. (Tolerances too small.)
         |      -3           Illegal input detected. (See printed message.)
         |      -4           Repeated error test failures. (Check all input.)
         |      -5           Repeated convergence failures. (Perhaps bad Jacobian
         |                   supplied or wrong choice of MF or tolerances.)
         |      -6           Error weight became zero during problem. (Solution
         |                   component i vanished, and ATOL or ATOL(i) = 0.)
         |      ===========  =======
         |
         |      "zvode"
         |
         |      ===========  =======
         |      Return Code  Message
         |      ===========  =======
         |      2            Integration successful.
         |      -1           Excess work done on this call. (Perhaps wrong MF.)
         |      -2           Excess accuracy requested. (Tolerances too small.)
         |      -3           Illegal input detected. (See printed message.)
         |      -4           Repeated error test failures. (Check all input.)
         |      -5           Repeated convergence failures. (Perhaps bad Jacobian
         |                   supplied or wrong choice of MF or tolerances.)
         |      -6           Error weight became zero during problem. (Solution
         |                   component i vanished, and ATOL or ATOL(i) = 0.)
         |      ===========  =======
         |
         |      "dopri5"
         |
         |      ===========  =======
         |      Return Code  Message
         |      ===========  =======
         |      1            Integration successful.
         |      2            Integration successful (interrupted by solout).
         |      -1           Input is not consistent.
         |      -2           Larger nsteps is needed.
         |      -3           Step size becomes too small.
         |      -4           Problem is probably stiff (interrupted).
         |      ===========  =======
         |
         |      "dop853"
         |
         |      ===========  =======
         |      Return Code  Message
         |      ===========  =======
         |      1            Integration successful.
         |      2            Integration successful (interrupted by solout).
         |      -1           Input is not consistent.
         |      -2           Larger nsteps is needed.
         |      -3           Step size becomes too small.
         |      -4           Problem is probably stiff (interrupted).
         |      ===========  =======
         |
         |      "lsoda"
         |
         |      ===========  =======
         |      Return Code  Message
         |      ===========  =======
         |      2            Integration successful.
         |      -1           Excess work done on this call (perhaps wrong Dfun type).
         |      -2           Excess accuracy requested (tolerances too small).
         |      -3           Illegal input detected (internal error).
         |      -4           Repeated error test failures (internal error).
         |      -5           Repeated convergence failures (perhaps bad Jacobian or tolerances).
         |      -6           Error weight became zero during problem.
         |      -7           Internal workspace insufficient to finish (internal error).
         |      ===========  =======
         |
         |  set_f_params(self, *args)
         |      Set extra parameters for user-supplied function f.
         |
         |  set_jac_params(self, *args)
         |      Set extra parameters for user-supplied function jac.
         |
         |  successful(self)
         |      Check if integration was successful.
         |
         |  ----------------------------------------------------------------------
         |  Data descriptors inherited from ode:
         |
         |  __dict__
         |      dictionary for instance variables
         |
         |  __weakref__
         |      list of weak references to the object
    
        class ode(builtins.object)
         |  ode(f, jac=None)
         |
         |  A generic interface class to numeric integrators.
         |
         |  Solve an equation system :math:`y'(t) = f(t,y)` with (optional) ``jac = df/dy``.
         |
         |  *Note*: The first two arguments of ``f(t, y, ...)`` are in the
         |  opposite order of the arguments in the system definition function used
         |  by `scipy.integrate.odeint`.
         |
         |  Parameters
         |  ----------
         |  f : callable ``f(t, y, *f_args)``
         |      Right-hand side of the differential equation. t is a scalar,
         |      ``y.shape == (n,)``.
         |      ``f_args`` is set by calling ``set_f_params(*args)``.
         |      `f` should return a scalar, array or list (not a tuple).
         |  jac : callable ``jac(t, y, *jac_args)``, optional
         |      Jacobian of the right-hand side, ``jac[i,j] = d f[i] / d y[j]``.
         |      ``jac_args`` is set by calling ``set_jac_params(*args)``.
         |
         |  Attributes
         |  ----------
         |  t : float
         |      Current time.
         |  y : ndarray
         |      Current variable values.
         |
         |  See also
         |  --------
         |  odeint : an integrator with a simpler interface based on lsoda from ODEPACK
         |  quad : for finding the area under a curve
         |
         |  Notes
         |  -----
         |  Available integrators are listed below. They can be selected using
         |  the `set_integrator` method.
         |
         |  "vode"
         |
         |      Real-valued Variable-coefficient Ordinary Differential Equation
         |      solver, with fixed-leading-coefficient implementation. It provides
         |      implicit Adams method (for non-stiff problems) and a method based on
         |      backward differentiation formulas (BDF) (for stiff problems).
         |
         |      Source: http://www.netlib.org/ode/vode.f
         |
         |      .. warning::
         |
         |         This integrator is not re-entrant. You cannot have two `ode`
         |         instances using the "vode" integrator at the same time.
         |
         |      This integrator accepts the following parameters in `set_integrator`
         |      method of the `ode` class:
         |
         |      - atol : float or sequence
         |        absolute tolerance for solution
         |      - rtol : float or sequence
         |        relative tolerance for solution
         |      - lband : None or int
         |      - uband : None or int
         |        Jacobian band width, jac[i,j] != 0 for i-lband <= j <= i+uband.
         |        Setting these requires your jac routine to return the jacobian
         |        in packed format, jac_packed[i-j+uband, j] = jac[i,j]. The
         |        dimension of the matrix must be (lband+uband+1, len(y)).
         |      - method: 'adams' or 'bdf'
         |        Which solver to use, Adams (non-stiff) or BDF (stiff)
         |      - with_jacobian : bool
         |        This option is only considered when the user has not supplied a
         |        Jacobian function and has not indicated (by setting either band)
         |        that the Jacobian is banded. In this case, `with_jacobian` specifies
         |        whether the iteration method of the ODE solver's correction step is
         |        chord iteration with an internally generated full Jacobian or
         |        functional iteration with no Jacobian.
         |      - nsteps : int
         |        Maximum number of (internally defined) steps allowed during one
         |        call to the solver.
         |      - first_step : float
         |      - min_step : float
         |      - max_step : float
         |        Limits for the step sizes used by the integrator.
         |      - order : int
         |        Maximum order used by the integrator,
         |        order <= 12 for Adams, <= 5 for BDF.
         |
         |  "zvode"
         |
         |      Complex-valued Variable-coefficient Ordinary Differential Equation
         |      solver, with fixed-leading-coefficient implementation. It provides
         |      implicit Adams method (for non-stiff problems) and a method based on
         |      backward differentiation formulas (BDF) (for stiff problems).
         |
         |      Source: http://www.netlib.org/ode/zvode.f
         |
         |      .. warning::
         |
         |         This integrator is not re-entrant. You cannot have two `ode`
         |         instances using the "zvode" integrator at the same time.
         |
         |      This integrator accepts the same parameters in `set_integrator`
         |      as the "vode" solver.
         |
         |      .. note::
         |
         |          When using ZVODE for a stiff system, it should only be used for
         |          the case in which the function f is analytic, that is, when each f(i)
         |          is an analytic function of each y(j). Analyticity means that the
         |          partial derivative df(i)/dy(j) is a unique complex number, and this
         |          fact is critical in the way ZVODE solves the dense or banded linear
         |          systems that arise in the stiff case. For a complex stiff ODE system
         |          in which f is not analytic, ZVODE is likely to have convergence
         |          failures, and for this problem one should instead use DVODE on the
         |          equivalent real system (in the real and imaginary parts of y).
         |
         |  "lsoda"
         |
         |      Real-valued Variable-coefficient Ordinary Differential Equation
         |      solver, with fixed-leading-coefficient implementation. It provides
         |      automatic method switching between implicit Adams method (for non-stiff
         |      problems) and a method based on backward differentiation formulas (BDF)
         |      (for stiff problems).
         |
         |      Source: http://www.netlib.org/odepack
         |
         |      .. warning::
         |
         |         This integrator is not re-entrant. You cannot have two `ode`
         |         instances using the "lsoda" integrator at the same time.
         |
         |      This integrator accepts the following parameters in `set_integrator`
         |      method of the `ode` class:
         |
         |      - atol : float or sequence
         |        absolute tolerance for solution
         |      - rtol : float or sequence
         |        relative tolerance for solution
         |      - lband : None or int
         |      - uband : None or int
         |        Jacobian band width, jac[i,j] != 0 for i-lband <= j <= i+uband.
         |        Setting these requires your jac routine to return the jacobian
         |        in packed format, jac_packed[i-j+uband, j] = jac[i,j].
         |      - with_jacobian : bool
         |        *Not used.*
         |      - nsteps : int
         |        Maximum number of (internally defined) steps allowed during one
         |        call to the solver.
         |      - first_step : float
         |      - min_step : float
         |      - max_step : float
         |        Limits for the step sizes used by the integrator.
         |      - max_order_ns : int
         |        Maximum order used in the nonstiff case (default 12).
         |      - max_order_s : int
         |        Maximum order used in the stiff case (default 5).
         |      - max_hnil : int
         |        Maximum number of messages reporting too small step size (t + h = t)
         |        (default 0)
         |      - ixpr : int
         |        Whether to generate extra printing at method switches (default False).
         |
         |  "dopri5"
         |
         |      This is an explicit runge-kutta method of order (4)5 due to Dormand &
         |      Prince (with stepsize control and dense output).
         |
         |      Authors:
         |
         |          E. Hairer and G. Wanner
         |          Universite de Geneve, Dept. de Mathematiques
         |          CH-1211 Geneve 24, Switzerland
         |          e-mail:  ernst.hairer@math.unige.ch, gerhard.wanner@math.unige.ch
         |
         |      This code is described in [HNW93]_.
         |
         |      This integrator accepts the following parameters in set_integrator()
         |      method of the ode class:
         |
         |      - atol : float or sequence
         |        absolute tolerance for solution
         |      - rtol : float or sequence
         |        relative tolerance for solution
         |      - nsteps : int
         |        Maximum number of (internally defined) steps allowed during one
         |        call to the solver.
         |      - first_step : float
         |      - max_step : float
         |      - safety : float
         |        Safety factor on new step selection (default 0.9)
         |      - ifactor : float
         |      - dfactor : float
         |        Maximum factor to increase/decrease step size by in one step
         |      - beta : float
         |        Beta parameter for stabilised step size control.
         |      - verbosity : int
         |        Switch for printing messages (< 0 for no messages).
         |
         |  "dop853"
         |
         |      This is an explicit runge-kutta method of order 8(5,3) due to Dormand
         |      & Prince (with stepsize control and dense output).
         |
         |      Options and references the same as "dopri5".
         |
         |  Examples
         |  --------
         |
         |  A problem to integrate and the corresponding jacobian:
         |
         |  >>> from scipy.integrate import ode
         |  >>>
         |  >>> y0, t0 = [1.0j, 2.0], 0
         |  >>>
         |  >>> def f(t, y, arg1):
         |  ...     return [1j*arg1*y[0] + y[1], -arg1*y[1]**2]
         |  >>> def jac(t, y, arg1):
         |  ...     return [[1j*arg1, 1], [0, -arg1*2*y[1]]]
         |
         |  The integration:
         |
         |  >>> r = ode(f, jac).set_integrator('zvode', method='bdf')
         |  >>> r.set_initial_value(y0, t0).set_f_params(2.0).set_jac_params(2.0)
         |  >>> t1 = 10
         |  >>> dt = 1
         |  >>> while r.successful() and r.t < t1:
         |  ...     print(r.t+dt, r.integrate(r.t+dt))
         |  1 [-0.71038232+0.23749653j  0.40000271+0.j        ]
         |  2.0 [0.19098503-0.52359246j 0.22222356+0.j        ]
         |  3.0 [0.47153208+0.52701229j 0.15384681+0.j        ]
         |  4.0 [-0.61905937+0.30726255j  0.11764744+0.j        ]
         |  5.0 [0.02340997-0.61418799j 0.09523835+0.j        ]
         |  6.0 [0.58643071+0.339819j 0.08000018+0.j      ]
         |  7.0 [-0.52070105+0.44525141j  0.06896565+0.j        ]
         |  8.0 [-0.15986733-0.61234476j  0.06060616+0.j        ]
         |  9.0 [0.64850462+0.15048982j 0.05405414+0.j        ]
         |  10.0 [-0.38404699+0.56382299j  0.04878055+0.j        ]
         |
         |  References
         |  ----------
         |  .. [HNW93] E. Hairer, S.P. Norsett and G. Wanner, Solving Ordinary
         |      Differential Equations i. Nonstiff Problems. 2nd edition.
         |      Springer Series in Computational Mathematics,
         |      Springer-Verlag (1993)
         |
         |  Methods defined here:
         |
         |  __init__(self, f, jac=None)
         |      Initialize self.  See help(type(self)) for accurate signature.
         |
         |  get_return_code(self)
         |      Extracts the return code for the integration to enable better control
         |      if the integration fails.
         |
         |      In general, a return code > 0 implies success, while a return code < 0
         |      implies failure.
         |
         |      Notes
         |      -----
         |      This section describes possible return codes and their meaning, for available
         |      integrators that can be selected by `set_integrator` method.
         |
         |      "vode"
         |
         |      ===========  =======
         |      Return Code  Message
         |      ===========  =======
         |      2            Integration successful.
         |      -1           Excess work done on this call. (Perhaps wrong MF.)
         |      -2           Excess accuracy requested. (Tolerances too small.)
         |      -3           Illegal input detected. (See printed message.)
         |      -4           Repeated error test failures. (Check all input.)
         |      -5           Repeated convergence failures. (Perhaps bad Jacobian
         |                   supplied or wrong choice of MF or tolerances.)
         |      -6           Error weight became zero during problem. (Solution
         |                   component i vanished, and ATOL or ATOL(i) = 0.)
         |      ===========  =======
         |
         |      "zvode"
         |
         |      ===========  =======
         |      Return Code  Message
         |      ===========  =======
         |      2            Integration successful.
         |      -1           Excess work done on this call. (Perhaps wrong MF.)
         |      -2           Excess accuracy requested. (Tolerances too small.)
         |      -3           Illegal input detected. (See printed message.)
         |      -4           Repeated error test failures. (Check all input.)
         |      -5           Repeated convergence failures. (Perhaps bad Jacobian
         |                   supplied or wrong choice of MF or tolerances.)
         |      -6           Error weight became zero during problem. (Solution
         |                   component i vanished, and ATOL or ATOL(i) = 0.)
         |      ===========  =======
         |
         |      "dopri5"
         |
         |      ===========  =======
         |      Return Code  Message
         |      ===========  =======
         |      1            Integration successful.
         |      2            Integration successful (interrupted by solout).
         |      -1           Input is not consistent.
         |      -2           Larger nsteps is needed.
         |      -3           Step size becomes too small.
         |      -4           Problem is probably stiff (interrupted).
         |      ===========  =======
         |
         |      "dop853"
         |
         |      ===========  =======
         |      Return Code  Message
         |      ===========  =======
         |      1            Integration successful.
         |      2            Integration successful (interrupted by solout).
         |      -1           Input is not consistent.
         |      -2           Larger nsteps is needed.
         |      -3           Step size becomes too small.
         |      -4           Problem is probably stiff (interrupted).
         |      ===========  =======
         |
         |      "lsoda"
         |
         |      ===========  =======
         |      Return Code  Message
         |      ===========  =======
         |      2            Integration successful.
         |      -1           Excess work done on this call (perhaps wrong Dfun type).
         |      -2           Excess accuracy requested (tolerances too small).
         |      -3           Illegal input detected (internal error).
         |      -4           Repeated error test failures (internal error).
         |      -5           Repeated convergence failures (perhaps bad Jacobian or tolerances).
         |      -6           Error weight became zero during problem.
         |      -7           Internal workspace insufficient to finish (internal error).
         |      ===========  =======
         |
         |  integrate(self, t, step=False, relax=False)
         |      Find y=y(t), set y as an initial condition, and return y.
         |
         |      Parameters
         |      ----------
         |      t : float
         |          The endpoint of the integration step.
         |      step : bool
         |          If True, and if the integrator supports the step method,
         |          then perform a single integration step and return.
         |          This parameter is provided in order to expose internals of
         |          the implementation, and should not be changed from its default
         |          value in most cases.
         |      relax : bool
         |          If True and if the integrator supports the run_relax method,
         |          then integrate until t_1 >= t and return. ``relax`` is not
         |          referenced if ``step=True``.
         |          This parameter is provided in order to expose internals of
         |          the implementation, and should not be changed from its default
         |          value in most cases.
         |
         |      Returns
         |      -------
         |      y : float
         |          The integrated value at t
         |
         |  set_f_params(self, *args)
         |      Set extra parameters for user-supplied function f.
         |
         |  set_initial_value(self, y, t=0.0)
         |      Set initial conditions y(t) = y.
         |
         |  set_integrator(self, name, **integrator_params)
         |      Set integrator by name.
         |
         |      Parameters
         |      ----------
         |      name : str
         |          Name of the integrator.
         |      **integrator_params
         |          Additional parameters for the integrator.
         |
         |  set_jac_params(self, *args)
         |      Set extra parameters for user-supplied function jac.
         |
         |  set_solout(self, solout)
         |      Set callable to be called at every successful integration step.
         |
         |      Parameters
         |      ----------
         |      solout : callable
         |          ``solout(t, y)`` is called at each internal integrator step,
         |          t is a scalar providing the current independent position
         |          y is the current solution ``y.shape == (n,)``
         |          solout should return -1 to stop integration
         |          otherwise it should return None or 0
         |
         |  successful(self)
         |      Check if integration was successful.
         |
         |  ----------------------------------------------------------------------
         |  Readonly properties defined here:
         |
         |  y
         |
         |  ----------------------------------------------------------------------
         |  Data descriptors defined here:
         |
         |  __dict__
         |      dictionary for instance variables
         |
         |  __weakref__
         |      list of weak references to the object
    
    FUNCTIONS
        cumulative_simpson(y, *, x=None, dx=1.0, axis=-1, initial=None)
            Cumulatively integrate y(x) using the composite Simpson's 1/3 rule.
            The integral of the samples at every point is calculated by assuming a
            quadratic relationship between each point and the two adjacent points.
    
            Parameters
            ----------
            y : array_like
                Values to integrate. Requires at least one point along `axis`. If two or fewer
                points are provided along `axis`, Simpson's integration is not possible and the
                result is calculated with `cumulative_trapezoid`.
            x : array_like, optional
                The coordinate to integrate along. Must have the same shape as `y` or
                must be 1D with the same length as `y` along `axis`. `x` must also be
                strictly increasing along `axis`.
                If `x` is None (default), integration is performed using spacing `dx`
                between consecutive elements in `y`.
            dx : scalar or array_like, optional
                Spacing between elements of `y`. Only used if `x` is None. Can either
                be a float, or an array with the same shape as `y`, but of length one along
                `axis`. Default is 1.0.
            axis : int, optional
                Specifies the axis to integrate along. Default is -1 (last axis).
            initial : scalar or array_like, optional
                If given, insert this value at the beginning of the returned result,
                and add it to the rest of the result. Default is None, which means no
                value at ``x[0]`` is returned and `res` has one element less than `y`
                along the axis of integration. Can either be a float, or an array with
                the same shape as `y`, but of length one along `axis`.
    
            Returns
            -------
            res : ndarray
                The result of cumulative integration of `y` along `axis`.
                If `initial` is None, the shape is such that the axis of integration
                has one less value than `y`. If `initial` is given, the shape is equal
                to that of `y`.
    
            See Also
            --------
            numpy.cumsum
            cumulative_trapezoid : cumulative integration using the composite
                trapezoidal rule
            simpson : integrator for sampled data using the Composite Simpson's Rule
    
            Notes
            -----
    
            .. versionadded:: 1.12.0
    
            The composite Simpson's 1/3 method can be used to approximate the definite
            integral of a sampled input function :math:`y(x)` [1]_. The method assumes
            a quadratic relationship over the interval containing any three consecutive
            sampled points.
    
            Consider three consecutive points:
            :math:`(x_1, y_1), (x_2, y_2), (x_3, y_3)`.
    
            Assuming a quadratic relationship over the three points, the integral over
            the subinterval between :math:`x_1` and :math:`x_2` is given by formula
            (8) of [2]_:
    
            .. math::
                \int_{x_1}^{x_2} y(x) dx\ &= \frac{x_2-x_1}{6}\left[\
                \left\{3-\frac{x_2-x_1}{x_3-x_1}\right\} y_1 + \
                \left\{3 + \frac{(x_2-x_1)^2}{(x_3-x_2)(x_3-x_1)} + \
                \frac{x_2-x_1}{x_3-x_1}\right\} y_2\\
                - \frac{(x_2-x_1)^2}{(x_3-x_2)(x_3-x_1)} y_3\right]
    
            The integral between :math:`x_2` and :math:`x_3` is given by swapping
            appearances of :math:`x_1` and :math:`x_3`. The integral is estimated
            separately for each subinterval and then cumulatively summed to obtain
            the final result.
    
            For samples that are equally spaced, the result is exact if the function
            is a polynomial of order three or less [1]_ and the number of subintervals
            is even. Otherwise, the integral is exact for polynomials of order two or
            less.
    
            References
            ----------
            .. [1] Wikipedia page: https://en.wikipedia.org/wiki/Simpson's_rule
            .. [2] Cartwright, Kenneth V. Simpson's Rule Cumulative Integration with
                    MS Excel and Irregularly-spaced Data. Journal of Mathematical
                    Sciences and Mathematics Education. 12 (2): 1-9
    
            Examples
            --------
            >>> from scipy import integrate
            >>> import numpy as np
            >>> import matplotlib.pyplot as plt
            >>> x = np.linspace(-2, 2, num=20)
            >>> y = x**2
            >>> y_int = integrate.cumulative_simpson(y, x=x, initial=0)
            >>> fig, ax = plt.subplots()
            >>> ax.plot(x, y_int, 'ro', x, x**3/3 - (x[0])**3/3, 'b-')
            >>> ax.grid()
            >>> plt.show()
    
            The output of `cumulative_simpson` is similar to that of iteratively
            calling `simpson` with successively higher upper limits of integration, but
            not identical.
    
            >>> def cumulative_simpson_reference(y, x):
            ...     return np.asarray([integrate.simpson(y[:i], x=x[:i])
            ...                        for i in range(2, len(y) + 1)])
            >>>
            >>> rng = np.random.default_rng(354673834679465)
            >>> x, y = rng.random(size=(2, 10))
            >>> x.sort()
            >>>
            >>> res = integrate.cumulative_simpson(y, x=x)
            >>> ref = cumulative_simpson_reference(y, x)
            >>> equal = np.abs(res - ref) < 1e-15
            >>> equal  # not equal when `simpson` has even number of subintervals
            array([False,  True, False,  True, False,  True, False,  True,  True])
    
            This is expected: because `cumulative_simpson` has access to more
            information than `simpson`, it can typically produce more accurate
            estimates of the underlying integral over subintervals.
    
        cumulative_trapezoid(y, x=None, dx=1.0, axis=-1, initial=None)
            Cumulatively integrate y(x) using the composite trapezoidal rule.
    
            Parameters
            ----------
            y : array_like
                Values to integrate.
            x : array_like, optional
                The coordinate to integrate along. If None (default), use spacing `dx`
                between consecutive elements in `y`.
            dx : float, optional
                Spacing between elements of `y`. Only used if `x` is None.
            axis : int, optional
                Specifies the axis to cumulate. Default is -1 (last axis).
            initial : scalar, optional
                If given, insert this value at the beginning of the returned result.
                0 or None are the only values accepted. Default is None, which means
                `res` has one element less than `y` along the axis of integration.
    
                .. deprecated:: 1.12.0
                    The option for non-zero inputs for `initial` will be deprecated in
                    SciPy 1.15.0. After this time, a ValueError will be raised if
                    `initial` is not None or 0.
    
            Returns
            -------
            res : ndarray
                The result of cumulative integration of `y` along `axis`.
                If `initial` is None, the shape is such that the axis of integration
                has one less value than `y`. If `initial` is given, the shape is equal
                to that of `y`.
    
            See Also
            --------
            numpy.cumsum, numpy.cumprod
            cumulative_simpson : cumulative integration using Simpson's 1/3 rule
            quad : adaptive quadrature using QUADPACK
            fixed_quad : fixed-order Gaussian quadrature
            dblquad : double integrals
            tplquad : triple integrals
            romb : integrators for sampled data
    
            Examples
            --------
            >>> from scipy import integrate
            >>> import numpy as np
            >>> import matplotlib.pyplot as plt
    
            >>> x = np.linspace(-2, 2, num=20)
            >>> y = x
            >>> y_int = integrate.cumulative_trapezoid(y, x, initial=0)
            >>> plt.plot(x, y_int, 'ro', x, y[0] + 0.5 * x**2, 'b-')
            >>> plt.show()
    
        dblquad(func, a, b, gfun, hfun, args=(), epsabs=1.49e-08, epsrel=1.49e-08)
            Compute a double integral.
    
            Return the double (definite) integral of ``func(y, x)`` from ``x = a..b``
            and ``y = gfun(x)..hfun(x)``.
    
            Parameters
            ----------
            func : callable
                A Python function or method of at least two variables: y must be the
                first argument and x the second argument.
            a, b : float
                The limits of integration in x: `a` < `b`
            gfun : callable or float
                The lower boundary curve in y which is a function taking a single
                floating point argument (x) and returning a floating point result
                or a float indicating a constant boundary curve.
            hfun : callable or float
                The upper boundary curve in y (same requirements as `gfun`).
            args : sequence, optional
                Extra arguments to pass to `func`.
            epsabs : float, optional
                Absolute tolerance passed directly to the inner 1-D quadrature
                integration. Default is 1.49e-8. ``dblquad`` tries to obtain
                an accuracy of ``abs(i-result) <= max(epsabs, epsrel*abs(i))``
                where ``i`` = inner integral of ``func(y, x)`` from ``gfun(x)``
                to ``hfun(x)``, and ``result`` is the numerical approximation.
                See `epsrel` below.
            epsrel : float, optional
                Relative tolerance of the inner 1-D integrals. Default is 1.49e-8.
                If ``epsabs <= 0``, `epsrel` must be greater than both 5e-29
                and ``50 * (machine epsilon)``. See `epsabs` above.
    
            Returns
            -------
            y : float
                The resultant integral.
            abserr : float
                An estimate of the error.
    
            See Also
            --------
            quad : single integral
            tplquad : triple integral
            nquad : N-dimensional integrals
            fixed_quad : fixed-order Gaussian quadrature
            simpson : integrator for sampled data
            romb : integrator for sampled data
            scipy.special : for coefficients and roots of orthogonal polynomials
    
    
            Notes
            -----
            For valid results, the integral must converge; behavior for divergent
            integrals is not guaranteed.
    
            **Details of QUADPACK level routines**
    
            `quad` calls routines from the FORTRAN library QUADPACK. This section
            provides details on the conditions for each routine to be called and a
            short description of each routine. For each level of integration, ``qagse``
            is used for finite limits or ``qagie`` is used if either limit (or both!)
            are infinite. The following provides a short description from [1]_ for each
            routine.
    
            qagse
                is an integrator based on globally adaptive interval
                subdivision in connection with extrapolation, which will
                eliminate the effects of integrand singularities of
                several types.
            qagie
                handles integration over infinite intervals. The infinite range is
                mapped onto a finite interval and subsequently the same strategy as
                in ``QAGS`` is applied.
    
            References
            ----------
    
            .. [1] Piessens, Robert; de Doncker-Kapenga, Elise;
                   Überhuber, Christoph W.; Kahaner, David (1983).
                   QUADPACK: A subroutine package for automatic integration.
                   Springer-Verlag.
                   ISBN 978-3-540-12553-2.
    
            Examples
            --------
            Compute the double integral of ``x * y**2`` over the box
            ``x`` ranging from 0 to 2 and ``y`` ranging from 0 to 1.
            That is, :math:`\int^{x=2}_{x=0} \int^{y=1}_{y=0} x y^2 \,dy \,dx`.
    
            >>> import numpy as np
            >>> from scipy import integrate
            >>> f = lambda y, x: x*y**2
            >>> integrate.dblquad(f, 0, 2, 0, 1)
                (0.6666666666666667, 7.401486830834377e-15)
    
            Calculate :math:`\int^{x=\pi/4}_{x=0} \int^{y=\cos(x)}_{y=\sin(x)} 1
            \,dy \,dx`.
    
            >>> f = lambda y, x: 1
            >>> integrate.dblquad(f, 0, np.pi/4, np.sin, np.cos)
                (0.41421356237309503, 1.1083280054755938e-14)
    
            Calculate :math:`\int^{x=1}_{x=0} \int^{y=2-x}_{y=x} a x y \,dy \,dx`
            for :math:`a=1, 3`.
    
            >>> f = lambda y, x, a: a*x*y
            >>> integrate.dblquad(f, 0, 1, lambda x: x, lambda x: 2-x, args=(1,))
                (0.33333333333333337, 5.551115123125783e-15)
            >>> integrate.dblquad(f, 0, 1, lambda x: x, lambda x: 2-x, args=(3,))
                (0.9999999999999999, 1.6653345369377348e-14)
    
            Compute the two-dimensional Gaussian Integral, which is the integral of the
            Gaussian function :math:`f(x,y) = e^{-(x^{2} + y^{2})}`, over
            :math:`(-\infty,+\infty)`. That is, compute the integral
            :math:`\iint^{+\infty}_{-\infty} e^{-(x^{2} + y^{2})} \,dy\,dx`.
    
            >>> f = lambda x, y: np.exp(-(x ** 2 + y ** 2))
            >>> integrate.dblquad(f, -np.inf, np.inf, -np.inf, np.inf)
                (3.141592653589777, 2.5173086737433208e-08)
    
        fixed_quad(func, a, b, args=(), n=5)
            Compute a definite integral using fixed-order Gaussian quadrature.
    
            Integrate `func` from `a` to `b` using Gaussian quadrature of
            order `n`.
    
            Parameters
            ----------
            func : callable
                A Python function or method to integrate (must accept vector inputs).
                If integrating a vector-valued function, the returned array must have
                shape ``(..., len(x))``.
            a : float
                Lower limit of integration.
            b : float
                Upper limit of integration.
            args : tuple, optional
                Extra arguments to pass to function, if any.
            n : int, optional
                Order of quadrature integration. Default is 5.
    
            Returns
            -------
            val : float
                Gaussian quadrature approximation to the integral
            none : None
                Statically returned value of None
    
            See Also
            --------
            quad : adaptive quadrature using QUADPACK
            dblquad : double integrals
            tplquad : triple integrals
            romb : integrators for sampled data
            simpson : integrators for sampled data
            cumulative_trapezoid : cumulative integration for sampled data
    
            Examples
            --------
            >>> from scipy import integrate
            >>> import numpy as np
            >>> f = lambda x: x**8
            >>> integrate.fixed_quad(f, 0.0, 1.0, n=4)
            (0.1110884353741496, None)
            >>> integrate.fixed_quad(f, 0.0, 1.0, n=5)
            (0.11111111111111102, None)
            >>> print(1/9.0)  # analytical result
            0.1111111111111111
    
            >>> integrate.fixed_quad(np.cos, 0.0, np.pi/2, n=4)
            (0.9999999771971152, None)
            >>> integrate.fixed_quad(np.cos, 0.0, np.pi/2, n=5)
            (1.000000000039565, None)
            >>> np.sin(np.pi/2)-np.sin(0)  # analytical result
            1.0
    
        newton_cotes(rn, equal=0)
            Return weights and error coefficient for Newton-Cotes integration.
    
            Suppose we have (N+1) samples of f at the positions
            x_0, x_1, ..., x_N. Then an N-point Newton-Cotes formula for the
            integral between x_0 and x_N is:
    
            :math:`\int_{x_0}^{x_N} f(x)dx = \Delta x \sum_{i=0}^{N} a_i f(x_i)
            + B_N (\Delta x)^{N+2} f^{N+1} (\xi)`
    
            where :math:`\xi \in [x_0,x_N]`
            and :math:`\Delta x = \frac{x_N-x_0}{N}` is the average samples spacing.
    
            If the samples are equally-spaced and N is even, then the error
            term is :math:`B_N (\Delta x)^{N+3} f^{N+2}(\xi)`.
    
            Parameters
            ----------
            rn : int
                The integer order for equally-spaced data or the relative positions of
                the samples with the first sample at 0 and the last at N, where N+1 is
                the length of `rn`. N is the order of the Newton-Cotes integration.
            equal : int, optional
                Set to 1 to enforce equally spaced data.
    
            Returns
            -------
            an : ndarray
                1-D array of weights to apply to the function at the provided sample
                positions.
            B : float
                Error coefficient.
    
            Notes
            -----
            Normally, the Newton-Cotes rules are used on smaller integration
            regions and a composite rule is used to return the total integral.
    
            Examples
            --------
            Compute the integral of sin(x) in [0, :math:`\pi`]:
    
            >>> from scipy.integrate import newton_cotes
            >>> import numpy as np
            >>> def f(x):
            ...     return np.sin(x)
            >>> a = 0
            >>> b = np.pi
            >>> exact = 2
            >>> for N in [2, 4, 6, 8, 10]:
            ...     x = np.linspace(a, b, N + 1)
            ...     an, B = newton_cotes(N, 1)
            ...     dx = (b - a) / N
            ...     quad = dx * np.sum(an * f(x))
            ...     error = abs(quad - exact)
            ...     print('{:2d}  {:10.9f}  {:.5e}'.format(N, quad, error))
            ...
             2   2.094395102   9.43951e-02
             4   1.998570732   1.42927e-03
             6   2.000017814   1.78136e-05
             8   1.999999835   1.64725e-07
            10   2.000000001   1.14677e-09
    
        nquad(func, ranges, args=None, opts=None, full_output=False)
            Integration over multiple variables.
    
            Wraps `quad` to enable integration over multiple variables.
            Various options allow improved integration of discontinuous functions, as
            well as the use of weighted integration, and generally finer control of the
            integration process.
    
            Parameters
            ----------
            func : {callable, scipy.LowLevelCallable}
                The function to be integrated. Has arguments of ``x0, ... xn``,
                ``t0, ... tm``, where integration is carried out over ``x0, ... xn``,
                which must be floats.  Where ``t0, ... tm`` are extra arguments
                passed in args.
                Function signature should be ``func(x0, x1, ..., xn, t0, t1, ..., tm)``.
                Integration is carried out in order.  That is, integration over ``x0``
                is the innermost integral, and ``xn`` is the outermost.
    
                If the user desires improved integration performance, then `f` may
                be a `scipy.LowLevelCallable` with one of the signatures::
    
                    double func(int n, double *xx)
                    double func(int n, double *xx, void *user_data)
    
                where ``n`` is the number of variables and args.  The ``xx`` array
                contains the coordinates and extra arguments. ``user_data`` is the data
                contained in the `scipy.LowLevelCallable`.
            ranges : iterable object
                Each element of ranges may be either a sequence  of 2 numbers, or else
                a callable that returns such a sequence. ``ranges[0]`` corresponds to
                integration over x0, and so on. If an element of ranges is a callable,
                then it will be called with all of the integration arguments available,
                as well as any parametric arguments. e.g., if
                ``func = f(x0, x1, x2, t0, t1)``, then ``ranges[0]`` may be defined as
                either ``(a, b)`` or else as ``(a, b) = range0(x1, x2, t0, t1)``.
            args : iterable object, optional
                Additional arguments ``t0, ... tn``, required by ``func``, ``ranges``,
                and ``opts``.
            opts : iterable object or dict, optional
                Options to be passed to `quad`. May be empty, a dict, or
                a sequence of dicts or functions that return a dict. If empty, the
                default options from scipy.integrate.quad are used. If a dict, the same
                options are used for all levels of integraion. If a sequence, then each
                element of the sequence corresponds to a particular integration. e.g.,
                ``opts[0]`` corresponds to integration over ``x0``, and so on. If a
                callable, the signature must be the same as for ``ranges``. The
                available options together with their default values are:
    
                  - epsabs = 1.49e-08
                  - epsrel = 1.49e-08
                  - limit  = 50
                  - points = None
                  - weight = None
                  - wvar   = None
                  - wopts  = None
    
                For more information on these options, see `quad`.
    
            full_output : bool, optional
                Partial implementation of ``full_output`` from scipy.integrate.quad.
                The number of integrand function evaluations ``neval`` can be obtained
                by setting ``full_output=True`` when calling nquad.
    
            Returns
            -------
            result : float
                The result of the integration.
            abserr : float
                The maximum of the estimates of the absolute error in the various
                integration results.
            out_dict : dict, optional
                A dict containing additional information on the integration.
    
            See Also
            --------
            quad : 1-D numerical integration
            dblquad, tplquad : double and triple integrals
            fixed_quad : fixed-order Gaussian quadrature
    
            Notes
            -----
            For valid results, the integral must converge; behavior for divergent
            integrals is not guaranteed.
    
            **Details of QUADPACK level routines**
    
            `nquad` calls routines from the FORTRAN library QUADPACK. This section
            provides details on the conditions for each routine to be called and a
            short description of each routine. The routine called depends on
            `weight`, `points` and the integration limits `a` and `b`.
    
            ================  ==============  ==========  =====================
            QUADPACK routine  `weight`        `points`    infinite bounds
            ================  ==============  ==========  =====================
            qagse             None            No          No
            qagie             None            No          Yes
            qagpe             None            Yes         No
            qawoe             'sin', 'cos'    No          No
            qawfe             'sin', 'cos'    No          either `a` or `b`
            qawse             'alg*'          No          No
            qawce             'cauchy'        No          No
            ================  ==============  ==========  =====================
    
            The following provides a short description from [1]_ for each
            routine.
    
            qagse
                is an integrator based on globally adaptive interval
                subdivision in connection with extrapolation, which will
                eliminate the effects of integrand singularities of
                several types.
            qagie
                handles integration over infinite intervals. The infinite range is
                mapped onto a finite interval and subsequently the same strategy as
                in ``QAGS`` is applied.
            qagpe
                serves the same purposes as QAGS, but also allows the
                user to provide explicit information about the location
                and type of trouble-spots i.e. the abscissae of internal
                singularities, discontinuities and other difficulties of
                the integrand function.
            qawoe
                is an integrator for the evaluation of
                :math:`\int^b_a \cos(\omega x)f(x)dx` or
                :math:`\int^b_a \sin(\omega x)f(x)dx`
                over a finite interval [a,b], where :math:`\omega` and :math:`f`
                are specified by the user. The rule evaluation component is based
                on the modified Clenshaw-Curtis technique
    
                An adaptive subdivision scheme is used in connection
                with an extrapolation procedure, which is a modification
                of that in ``QAGS`` and allows the algorithm to deal with
                singularities in :math:`f(x)`.
            qawfe
                calculates the Fourier transform
                :math:`\int^\infty_a \cos(\omega x)f(x)dx` or
                :math:`\int^\infty_a \sin(\omega x)f(x)dx`
                for user-provided :math:`\omega` and :math:`f`. The procedure of
                ``QAWO`` is applied on successive finite intervals, and convergence
                acceleration by means of the :math:`\varepsilon`-algorithm is applied
                to the series of integral approximations.
            qawse
                approximate :math:`\int^b_a w(x)f(x)dx`, with :math:`a < b` where
                :math:`w(x) = (x-a)^{\alpha}(b-x)^{\beta}v(x)` with
                :math:`\alpha,\beta > -1`, where :math:`v(x)` may be one of the
                following functions: :math:`1`, :math:`\log(x-a)`, :math:`\log(b-x)`,
                :math:`\log(x-a)\log(b-x)`.
    
                The user specifies :math:`\alpha`, :math:`\beta` and the type of the
                function :math:`v`. A globally adaptive subdivision strategy is
                applied, with modified Clenshaw-Curtis integration on those
                subintervals which contain `a` or `b`.
            qawce
                compute :math:`\int^b_a f(x) / (x-c)dx` where the integral must be
                interpreted as a Cauchy principal value integral, for user specified
                :math:`c` and :math:`f`. The strategy is globally adaptive. Modified
                Clenshaw-Curtis integration is used on those intervals containing the
                point :math:`x = c`.
    
            References
            ----------
    
            .. [1] Piessens, Robert; de Doncker-Kapenga, Elise;
                   Überhuber, Christoph W.; Kahaner, David (1983).
                   QUADPACK: A subroutine package for automatic integration.
                   Springer-Verlag.
                   ISBN 978-3-540-12553-2.
    
            Examples
            --------
            Calculate
    
            .. math::
    
                \int^{1}_{-0.15} \int^{0.8}_{0.13} \int^{1}_{-1} \int^{1}_{0}
                f(x_0, x_1, x_2, x_3) \,dx_0 \,dx_1 \,dx_2 \,dx_3 ,
    
            where
    
            .. math::
    
                f(x_0, x_1, x_2, x_3) = \begin{cases}
                  x_0^2+x_1 x_2-x_3^3+ \sin{x_0}+1 & (x_0-0.2 x_3-0.5-0.25 x_1 > 0) \\
                  x_0^2+x_1 x_2-x_3^3+ \sin{x_0}+0 & (x_0-0.2 x_3-0.5-0.25 x_1 \leq 0)
                \end{cases} .
    
            >>> import numpy as np
            >>> from scipy import integrate
            >>> func = lambda x0,x1,x2,x3 : x0**2 + x1*x2 - x3**3 + np.sin(x0) + (
            ...                                 1 if (x0-.2*x3-.5-.25*x1>0) else 0)
            >>> def opts0(*args, **kwargs):
            ...     return {'points':[0.2*args[2] + 0.5 + 0.25*args[0]]}
            >>> integrate.nquad(func, [[0,1], [-1,1], [.13,.8], [-.15,1]],
            ...                 opts=[opts0,{},{},{}], full_output=True)
            (1.5267454070738633, 2.9437360001402324e-14, {'neval': 388962})
    
            Calculate
    
            .. math::
    
                \int^{t_0+t_1+1}_{t_0+t_1-1}
                \int^{x_2+t_0^2 t_1^3+1}_{x_2+t_0^2 t_1^3-1}
                \int^{t_0 x_1+t_1 x_2+1}_{t_0 x_1+t_1 x_2-1}
                f(x_0,x_1, x_2,t_0,t_1)
                \,dx_0 \,dx_1 \,dx_2,
    
            where
    
            .. math::
    
                f(x_0, x_1, x_2, t_0, t_1) = \begin{cases}
                  x_0 x_2^2 + \sin{x_1}+2 & (x_0+t_1 x_1-t_0 > 0) \\
                  x_0 x_2^2 +\sin{x_1}+1 & (x_0+t_1 x_1-t_0 \leq 0)
                \end{cases}
    
            and :math:`(t_0, t_1) = (0, 1)` .
    
            >>> def func2(x0, x1, x2, t0, t1):
            ...     return x0*x2**2 + np.sin(x1) + 1 + (1 if x0+t1*x1-t0>0 else 0)
            >>> def lim0(x1, x2, t0, t1):
            ...     return [t0*x1 + t1*x2 - 1, t0*x1 + t1*x2 + 1]
            >>> def lim1(x2, t0, t1):
            ...     return [x2 + t0**2*t1**3 - 1, x2 + t0**2*t1**3 + 1]
            >>> def lim2(t0, t1):
            ...     return [t0 + t1 - 1, t0 + t1 + 1]
            >>> def opts0(x1, x2, t0, t1):
            ...     return {'points' : [t0 - t1*x1]}
            >>> def opts1(x2, t0, t1):
            ...     return {}
            >>> def opts2(t0, t1):
            ...     return {}
            >>> integrate.nquad(func2, [lim0, lim1, lim2], args=(0,1),
            ...                 opts=[opts0, opts1, opts2])
            (36.099919226771625, 1.8546948553373528e-07)
    
        odeint(func, y0, t, args=(), Dfun=None, col_deriv=0, full_output=0, ml=None, mu=None, rtol=None, atol=None, tcrit=None, h0=0.0, hmax=0.0, hmin=0.0, ixpr=0, mxstep=0, mxhnil=0, mxordn=12, mxords=5, printmessg=0, tfirst=False)
            Integrate a system of ordinary differential equations.
    
            .. note:: For new code, use `scipy.integrate.solve_ivp` to solve a
                      differential equation.
    
            Solve a system of ordinary differential equations using lsoda from the
            FORTRAN library odepack.
    
            Solves the initial value problem for stiff or non-stiff systems
            of first order ode-s::
    
                dy/dt = func(y, t, ...)  [or func(t, y, ...)]
    
            where y can be a vector.
    
            .. note:: By default, the required order of the first two arguments of
                      `func` are in the opposite order of the arguments in the system
                      definition function used by the `scipy.integrate.ode` class and
                      the function `scipy.integrate.solve_ivp`. To use a function with
                      the signature ``func(t, y, ...)``, the argument `tfirst` must be
                      set to ``True``.
    
            Parameters
            ----------
            func : callable(y, t, ...) or callable(t, y, ...)
                Computes the derivative of y at t.
                If the signature is ``callable(t, y, ...)``, then the argument
                `tfirst` must be set ``True``.
                `func` must not modify the data in `y`, as it is a
                view of the data used internally by the ODE solver.
            y0 : array
                Initial condition on y (can be a vector).
            t : array
                A sequence of time points for which to solve for y. The initial
                value point should be the first element of this sequence.
                This sequence must be monotonically increasing or monotonically
                decreasing; repeated values are allowed.
            args : tuple, optional
                Extra arguments to pass to function.
            Dfun : callable(y, t, ...) or callable(t, y, ...)
                Gradient (Jacobian) of `func`.
                If the signature is ``callable(t, y, ...)``, then the argument
                `tfirst` must be set ``True``.
                `Dfun` must not modify the data in `y`, as it is a
                view of the data used internally by the ODE solver.
            col_deriv : bool, optional
                True if `Dfun` defines derivatives down columns (faster),
                otherwise `Dfun` should define derivatives across rows.
            full_output : bool, optional
                True if to return a dictionary of optional outputs as the second output
            printmessg : bool, optional
                Whether to print the convergence message
            tfirst : bool, optional
                If True, the first two arguments of `func` (and `Dfun`, if given)
                must ``t, y`` instead of the default ``y, t``.
    
                .. versionadded:: 1.1.0
    
            Returns
            -------
            y : array, shape (len(t), len(y0))
                Array containing the value of y for each desired time in t,
                with the initial value `y0` in the first row.
            infodict : dict, only returned if full_output == True
                Dictionary containing additional output information
    
                =======  ============================================================
                key      meaning
                =======  ============================================================
                'hu'     vector of step sizes successfully used for each time step
                'tcur'   vector with the value of t reached for each time step
                         (will always be at least as large as the input times)
                'tolsf'  vector of tolerance scale factors, greater than 1.0,
                         computed when a request for too much accuracy was detected
                'tsw'    value of t at the time of the last method switch
                         (given for each time step)
                'nst'    cumulative number of time steps
                'nfe'    cumulative number of function evaluations for each time step
                'nje'    cumulative number of jacobian evaluations for each time step
                'nqu'    a vector of method orders for each successful step
                'imxer'  index of the component of largest magnitude in the
                         weighted local error vector (e / ewt) on an error return, -1
                         otherwise
                'lenrw'  the length of the double work array required
                'leniw'  the length of integer work array required
                'mused'  a vector of method indicators for each successful time step:
                         1: adams (nonstiff), 2: bdf (stiff)
                =======  ============================================================
    
            Other Parameters
            ----------------
            ml, mu : int, optional
                If either of these are not None or non-negative, then the
                Jacobian is assumed to be banded. These give the number of
                lower and upper non-zero diagonals in this banded matrix.
                For the banded case, `Dfun` should return a matrix whose
                rows contain the non-zero bands (starting with the lowest diagonal).
                Thus, the return matrix `jac` from `Dfun` should have shape
                ``(ml + mu + 1, len(y0))`` when ``ml >=0`` or ``mu >=0``.
                The data in `jac` must be stored such that ``jac[i - j + mu, j]``
                holds the derivative of the ``i``\ th equation with respect to the
                ``j``\ th state variable.  If `col_deriv` is True, the transpose of
                this `jac` must be returned.
            rtol, atol : float, optional
                The input parameters `rtol` and `atol` determine the error
                control performed by the solver.  The solver will control the
                vector, e, of estimated local errors in y, according to an
                inequality of the form ``max-norm of (e / ewt) <= 1``,
                where ewt is a vector of positive error weights computed as
                ``ewt = rtol * abs(y) + atol``.
                rtol and atol can be either vectors the same length as y or scalars.
                Defaults to 1.49012e-8.
            tcrit : ndarray, optional
                Vector of critical points (e.g., singularities) where integration
                care should be taken.
            h0 : float, (0: solver-determined), optional
                The step size to be attempted on the first step.
            hmax : float, (0: solver-determined), optional
                The maximum absolute step size allowed.
            hmin : float, (0: solver-determined), optional
                The minimum absolute step size allowed.
            ixpr : bool, optional
                Whether to generate extra printing at method switches.
            mxstep : int, (0: solver-determined), optional
                Maximum number of (internally defined) steps allowed for each
                integration point in t.
            mxhnil : int, (0: solver-determined), optional
                Maximum number of messages printed.
            mxordn : int, (0: solver-determined), optional
                Maximum order to be allowed for the non-stiff (Adams) method.
            mxords : int, (0: solver-determined), optional
                Maximum order to be allowed for the stiff (BDF) method.
    
            See Also
            --------
            solve_ivp : solve an initial value problem for a system of ODEs
            ode : a more object-oriented integrator based on VODE
            quad : for finding the area under a curve
    
            Examples
            --------
            The second order differential equation for the angle `theta` of a
            pendulum acted on by gravity with friction can be written::
    
                theta''(t) + b*theta'(t) + c*sin(theta(t)) = 0
    
            where `b` and `c` are positive constants, and a prime (') denotes a
            derivative. To solve this equation with `odeint`, we must first convert
            it to a system of first order equations. By defining the angular
            velocity ``omega(t) = theta'(t)``, we obtain the system::
    
                theta'(t) = omega(t)
                omega'(t) = -b*omega(t) - c*sin(theta(t))
    
            Let `y` be the vector [`theta`, `omega`]. We implement this system
            in Python as:
    
            >>> import numpy as np
            >>> def pend(y, t, b, c):
            ...     theta, omega = y
            ...     dydt = [omega, -b*omega - c*np.sin(theta)]
            ...     return dydt
            ...
    
            We assume the constants are `b` = 0.25 and `c` = 5.0:
    
            >>> b = 0.25
            >>> c = 5.0
    
            For initial conditions, we assume the pendulum is nearly vertical
            with `theta(0)` = `pi` - 0.1, and is initially at rest, so
            `omega(0)` = 0.  Then the vector of initial conditions is
    
            >>> y0 = [np.pi - 0.1, 0.0]
    
            We will generate a solution at 101 evenly spaced samples in the interval
            0 <= `t` <= 10.  So our array of times is:
    
            >>> t = np.linspace(0, 10, 101)
    
            Call `odeint` to generate the solution. To pass the parameters
            `b` and `c` to `pend`, we give them to `odeint` using the `args`
            argument.
    
            >>> from scipy.integrate import odeint
            >>> sol = odeint(pend, y0, t, args=(b, c))
    
            The solution is an array with shape (101, 2). The first column
            is `theta(t)`, and the second is `omega(t)`. The following code
            plots both components.
    
            >>> import matplotlib.pyplot as plt
            >>> plt.plot(t, sol[:, 0], 'b', label='theta(t)')
            >>> plt.plot(t, sol[:, 1], 'g', label='omega(t)')
            >>> plt.legend(loc='best')
            >>> plt.xlabel('t')
            >>> plt.grid()
            >>> plt.show()
    
        qmc_quad(func, a, b, *, n_estimates=8, n_points=1024, qrng=None, log=False)
            Compute an integral in N-dimensions using Quasi-Monte Carlo quadrature.
    
            Parameters
            ----------
            func : callable
                The integrand. Must accept a single argument ``x``, an array which
                specifies the point(s) at which to evaluate the scalar-valued
                integrand, and return the value(s) of the integrand.
                For efficiency, the function should be vectorized to accept an array of
                shape ``(d, n_points)``, where ``d`` is the number of variables (i.e.
                the dimensionality of the function domain) and `n_points` is the number
                of quadrature points, and return an array of shape ``(n_points,)``,
                the integrand at each quadrature point.
            a, b : array-like
                One-dimensional arrays specifying the lower and upper integration
                limits, respectively, of each of the ``d`` variables.
            n_estimates, n_points : int, optional
                `n_estimates` (default: 8) statistically independent QMC samples, each
                of `n_points` (default: 1024) points, will be generated by `qrng`.
                The total number of points at which the integrand `func` will be
                evaluated is ``n_points * n_estimates``. See Notes for details.
            qrng : `~scipy.stats.qmc.QMCEngine`, optional
                An instance of the QMCEngine from which to sample QMC points.
                The QMCEngine must be initialized to a number of dimensions ``d``
                corresponding with the number of variables ``x1, ..., xd`` passed to
                `func`.
                The provided QMCEngine is used to produce the first integral estimate.
                If `n_estimates` is greater than one, additional QMCEngines are
                spawned from the first (with scrambling enabled, if it is an option.)
                If a QMCEngine is not provided, the default `scipy.stats.qmc.Halton`
                will be initialized with the number of dimensions determine from
                the length of `a`.
            log : boolean, default: False
                When set to True, `func` returns the log of the integrand, and
                the result object contains the log of the integral.
    
            Returns
            -------
            result : object
                A result object with attributes:
    
                integral : float
                    The estimate of the integral.
                standard_error :
                    The error estimate. See Notes for interpretation.
    
            Notes
            -----
            Values of the integrand at each of the `n_points` points of a QMC sample
            are used to produce an estimate of the integral. This estimate is drawn
            from a population of possible estimates of the integral, the value of
            which we obtain depends on the particular points at which the integral
            was evaluated. We perform this process `n_estimates` times, each time
            evaluating the integrand at different scrambled QMC points, effectively
            drawing i.i.d. random samples from the population of integral estimates.
            The sample mean :math:`m` of these integral estimates is an
            unbiased estimator of the true value of the integral, and the standard
            error of the mean :math:`s` of these estimates may be used to generate
            confidence intervals using the t distribution with ``n_estimates - 1``
            degrees of freedom. Perhaps counter-intuitively, increasing `n_points`
            while keeping the total number of function evaluation points
            ``n_points * n_estimates`` fixed tends to reduce the actual error, whereas
            increasing `n_estimates` tends to decrease the error estimate.
    
            Examples
            --------
            QMC quadrature is particularly useful for computing integrals in higher
            dimensions. An example integrand is the probability density function
            of a multivariate normal distribution.
    
            >>> import numpy as np
            >>> from scipy import stats
            >>> dim = 8
            >>> mean = np.zeros(dim)
            >>> cov = np.eye(dim)
            >>> def func(x):
            ...     # `multivariate_normal` expects the _last_ axis to correspond with
            ...     # the dimensionality of the space, so `x` must be transposed
            ...     return stats.multivariate_normal.pdf(x.T, mean, cov)
    
            To compute the integral over the unit hypercube:
    
            >>> from scipy.integrate import qmc_quad
            >>> a = np.zeros(dim)
            >>> b = np.ones(dim)
            >>> rng = np.random.default_rng()
            >>> qrng = stats.qmc.Halton(d=dim, seed=rng)
            >>> n_estimates = 8
            >>> res = qmc_quad(func, a, b, n_estimates=n_estimates, qrng=qrng)
            >>> res.integral, res.standard_error
            (0.00018429555666024108, 1.0389431116001344e-07)
    
            A two-sided, 99% confidence interval for the integral may be estimated
            as:
    
            >>> t = stats.t(df=n_estimates-1, loc=res.integral,
            ...             scale=res.standard_error)
            >>> t.interval(0.99)
            (0.0001839319802536469, 0.00018465913306683527)
    
            Indeed, the value reported by `scipy.stats.multivariate_normal` is
            within this range.
    
            >>> stats.multivariate_normal.cdf(b, mean, cov, lower_limit=a)
            0.00018430867675187443
    
        quad(func, a, b, args=(), full_output=0, epsabs=1.49e-08, epsrel=1.49e-08, limit=50, points=None, weight=None, wvar=None, wopts=None, maxp1=50, limlst=50, complex_func=False)
            Compute a definite integral.
    
            Integrate func from `a` to `b` (possibly infinite interval) using a
            technique from the Fortran library QUADPACK.
    
            Parameters
            ----------
            func : {function, scipy.LowLevelCallable}
                A Python function or method to integrate. If `func` takes many
                arguments, it is integrated along the axis corresponding to the
                first argument.
    
                If the user desires improved integration performance, then `f` may
                be a `scipy.LowLevelCallable` with one of the signatures::
    
                    double func(double x)
                    double func(double x, void *user_data)
                    double func(int n, double *xx)
                    double func(int n, double *xx, void *user_data)
    
                The ``user_data`` is the data contained in the `scipy.LowLevelCallable`.
                In the call forms with ``xx``,  ``n`` is the length of the ``xx``
                array which contains ``xx[0] == x`` and the rest of the items are
                numbers contained in the ``args`` argument of quad.
    
                In addition, certain ctypes call signatures are supported for
                backward compatibility, but those should not be used in new code.
            a : float
                Lower limit of integration (use -numpy.inf for -infinity).
            b : float
                Upper limit of integration (use numpy.inf for +infinity).
            args : tuple, optional
                Extra arguments to pass to `func`.
            full_output : int, optional
                Non-zero to return a dictionary of integration information.
                If non-zero, warning messages are also suppressed and the
                message is appended to the output tuple.
            complex_func : bool, optional
                Indicate if the function's (`func`) return type is real
                (``complex_func=False``: default) or complex (``complex_func=True``).
                In both cases, the function's argument is real.
                If full_output is also non-zero, the `infodict`, `message`, and
                `explain` for the real and complex components are returned in
                a dictionary with keys "real output" and "imag output".
    
            Returns
            -------
            y : float
                The integral of func from `a` to `b`.
            abserr : float
                An estimate of the absolute error in the result.
            infodict : dict
                A dictionary containing additional information.
            message
                A convergence message.
            explain
                Appended only with 'cos' or 'sin' weighting and infinite
                integration limits, it contains an explanation of the codes in
                infodict['ierlst']
    
            Other Parameters
            ----------------
            epsabs : float or int, optional
                Absolute error tolerance. Default is 1.49e-8. `quad` tries to obtain
                an accuracy of ``abs(i-result) <= max(epsabs, epsrel*abs(i))``
                where ``i`` = integral of `func` from `a` to `b`, and ``result`` is the
                numerical approximation. See `epsrel` below.
            epsrel : float or int, optional
                Relative error tolerance. Default is 1.49e-8.
                If ``epsabs <= 0``, `epsrel` must be greater than both 5e-29
                and ``50 * (machine epsilon)``. See `epsabs` above.
            limit : float or int, optional
                An upper bound on the number of subintervals used in the adaptive
                algorithm.
            points : (sequence of floats,ints), optional
                A sequence of break points in the bounded integration interval
                where local difficulties of the integrand may occur (e.g.,
                singularities, discontinuities). The sequence does not have
                to be sorted. Note that this option cannot be used in conjunction
                with ``weight``.
            weight : float or int, optional
                String indicating weighting function. Full explanation for this
                and the remaining arguments can be found below.
            wvar : optional
                Variables for use with weighting functions.
            wopts : optional
                Optional input for reusing Chebyshev moments.
            maxp1 : float or int, optional
                An upper bound on the number of Chebyshev moments.
            limlst : int, optional
                Upper bound on the number of cycles (>=3) for use with a sinusoidal
                weighting and an infinite end-point.
    
            See Also
            --------
            dblquad : double integral
            tplquad : triple integral
            nquad : n-dimensional integrals (uses `quad` recursively)
            fixed_quad : fixed-order Gaussian quadrature
            simpson : integrator for sampled data
            romb : integrator for sampled data
            scipy.special : for coefficients and roots of orthogonal polynomials
    
            Notes
            -----
            For valid results, the integral must converge; behavior for divergent
            integrals is not guaranteed.
    
            **Extra information for quad() inputs and outputs**
    
            If full_output is non-zero, then the third output argument
            (infodict) is a dictionary with entries as tabulated below. For
            infinite limits, the range is transformed to (0,1) and the
            optional outputs are given with respect to this transformed range.
            Let M be the input argument limit and let K be infodict['last'].
            The entries are:
    
            'neval'
                The number of function evaluations.
            'last'
                The number, K, of subintervals produced in the subdivision process.
            'alist'
                A rank-1 array of length M, the first K elements of which are the
                left end points of the subintervals in the partition of the
                integration range.
            'blist'
                A rank-1 array of length M, the first K elements of which are the
                right end points of the subintervals.
            'rlist'
                A rank-1 array of length M, the first K elements of which are the
                integral approximations on the subintervals.
            'elist'
                A rank-1 array of length M, the first K elements of which are the
                moduli of the absolute error estimates on the subintervals.
            'iord'
                A rank-1 integer array of length M, the first L elements of
                which are pointers to the error estimates over the subintervals
                with ``L=K`` if ``K<=M/2+2`` or ``L=M+1-K`` otherwise. Let I be the
                sequence ``infodict['iord']`` and let E be the sequence
                ``infodict['elist']``.  Then ``E[I[1]], ..., E[I[L]]`` forms a
                decreasing sequence.
    
            If the input argument points is provided (i.e., it is not None),
            the following additional outputs are placed in the output
            dictionary. Assume the points sequence is of length P.
    
            'pts'
                A rank-1 array of length P+2 containing the integration limits
                and the break points of the intervals in ascending order.
                This is an array giving the subintervals over which integration
                will occur.
            'level'
                A rank-1 integer array of length M (=limit), containing the
                subdivision levels of the subintervals, i.e., if (aa,bb) is a
                subinterval of ``(pts[1], pts[2])`` where ``pts[0]`` and ``pts[2]``
                are adjacent elements of ``infodict['pts']``, then (aa,bb) has level l
                if ``|bb-aa| = |pts[2]-pts[1]| * 2**(-l)``.
            'ndin'
                A rank-1 integer array of length P+2. After the first integration
                over the intervals (pts[1], pts[2]), the error estimates over some
                of the intervals may have been increased artificially in order to
                put their subdivision forward. This array has ones in slots
                corresponding to the subintervals for which this happens.
    
            **Weighting the integrand**
    
            The input variables, *weight* and *wvar*, are used to weight the
            integrand by a select list of functions. Different integration
            methods are used to compute the integral with these weighting
            functions, and these do not support specifying break points. The
            possible values of weight and the corresponding weighting functions are.
    
            ==========  ===================================   =====================
            ``weight``  Weight function used                  ``wvar``
            ==========  ===================================   =====================
            'cos'       cos(w*x)                              wvar = w
            'sin'       sin(w*x)                              wvar = w
            'alg'       g(x) = ((x-a)**alpha)*((b-x)**beta)   wvar = (alpha, beta)
            'alg-loga'  g(x)*log(x-a)                         wvar = (alpha, beta)
            'alg-logb'  g(x)*log(b-x)                         wvar = (alpha, beta)
            'alg-log'   g(x)*log(x-a)*log(b-x)                wvar = (alpha, beta)
            'cauchy'    1/(x-c)                               wvar = c
            ==========  ===================================   =====================
    
            wvar holds the parameter w, (alpha, beta), or c depending on the weight
            selected. In these expressions, a and b are the integration limits.
    
            For the 'cos' and 'sin' weighting, additional inputs and outputs are
            available.
    
            For finite integration limits, the integration is performed using a
            Clenshaw-Curtis method which uses Chebyshev moments. For repeated
            calculations, these moments are saved in the output dictionary:
    
            'momcom'
                The maximum level of Chebyshev moments that have been computed,
                i.e., if ``M_c`` is ``infodict['momcom']`` then the moments have been
                computed for intervals of length ``|b-a| * 2**(-l)``,
                ``l=0,1,...,M_c``.
            'nnlog'
                A rank-1 integer array of length M(=limit), containing the
                subdivision levels of the subintervals, i.e., an element of this
                array is equal to l if the corresponding subinterval is
                ``|b-a|* 2**(-l)``.
            'chebmo'
                A rank-2 array of shape (25, maxp1) containing the computed
                Chebyshev moments. These can be passed on to an integration
                over the same interval by passing this array as the second
                element of the sequence wopts and passing infodict['momcom'] as
                the first element.
    
            If one of the integration limits is infinite, then a Fourier integral is
            computed (assuming w neq 0). If full_output is 1 and a numerical error
            is encountered, besides the error message attached to the output tuple,
            a dictionary is also appended to the output tuple which translates the
            error codes in the array ``info['ierlst']`` to English messages. The
            output information dictionary contains the following entries instead of
            'last', 'alist', 'blist', 'rlist', and 'elist':
    
            'lst'
                The number of subintervals needed for the integration (call it ``K_f``).
            'rslst'
                A rank-1 array of length M_f=limlst, whose first ``K_f`` elements
                contain the integral contribution over the interval
                ``(a+(k-1)c, a+kc)`` where ``c = (2*floor(|w|) + 1) * pi / |w|``
                and ``k=1,2,...,K_f``.
            'erlst'
                A rank-1 array of length ``M_f`` containing the error estimate
                corresponding to the interval in the same position in
                ``infodict['rslist']``.
            'ierlst'
                A rank-1 integer array of length ``M_f`` containing an error flag
                corresponding to the interval in the same position in
                ``infodict['rslist']``.  See the explanation dictionary (last entry
                in the output tuple) for the meaning of the codes.
    
    
            **Details of QUADPACK level routines**
    
            `quad` calls routines from the FORTRAN library QUADPACK. This section
            provides details on the conditions for each routine to be called and a
            short description of each routine. The routine called depends on
            `weight`, `points` and the integration limits `a` and `b`.
    
            ================  ==============  ==========  =====================
            QUADPACK routine  `weight`        `points`    infinite bounds
            ================  ==============  ==========  =====================
            qagse             None            No          No
            qagie             None            No          Yes
            qagpe             None            Yes         No
            qawoe             'sin', 'cos'    No          No
            qawfe             'sin', 'cos'    No          either `a` or `b`
            qawse             'alg*'          No          No
            qawce             'cauchy'        No          No
            ================  ==============  ==========  =====================
    
            The following provides a short description from [1]_ for each
            routine.
    
            qagse
                is an integrator based on globally adaptive interval
                subdivision in connection with extrapolation, which will
                eliminate the effects of integrand singularities of
                several types.
            qagie
                handles integration over infinite intervals. The infinite range is
                mapped onto a finite interval and subsequently the same strategy as
                in ``QAGS`` is applied.
            qagpe
                serves the same purposes as QAGS, but also allows the
                user to provide explicit information about the location
                and type of trouble-spots i.e. the abscissae of internal
                singularities, discontinuities and other difficulties of
                the integrand function.
            qawoe
                is an integrator for the evaluation of
                :math:`\int^b_a \cos(\omega x)f(x)dx` or
                :math:`\int^b_a \sin(\omega x)f(x)dx`
                over a finite interval [a,b], where :math:`\omega` and :math:`f`
                are specified by the user. The rule evaluation component is based
                on the modified Clenshaw-Curtis technique
    
                An adaptive subdivision scheme is used in connection
                with an extrapolation procedure, which is a modification
                of that in ``QAGS`` and allows the algorithm to deal with
                singularities in :math:`f(x)`.
            qawfe
                calculates the Fourier transform
                :math:`\int^\infty_a \cos(\omega x)f(x)dx` or
                :math:`\int^\infty_a \sin(\omega x)f(x)dx`
                for user-provided :math:`\omega` and :math:`f`. The procedure of
                ``QAWO`` is applied on successive finite intervals, and convergence
                acceleration by means of the :math:`\varepsilon`-algorithm is applied
                to the series of integral approximations.
            qawse
                approximate :math:`\int^b_a w(x)f(x)dx`, with :math:`a < b` where
                :math:`w(x) = (x-a)^{\alpha}(b-x)^{\beta}v(x)` with
                :math:`\alpha,\beta > -1`, where :math:`v(x)` may be one of the
                following functions: :math:`1`, :math:`\log(x-a)`, :math:`\log(b-x)`,
                :math:`\log(x-a)\log(b-x)`.
    
                The user specifies :math:`\alpha`, :math:`\beta` and the type of the
                function :math:`v`. A globally adaptive subdivision strategy is
                applied, with modified Clenshaw-Curtis integration on those
                subintervals which contain `a` or `b`.
            qawce
                compute :math:`\int^b_a f(x) / (x-c)dx` where the integral must be
                interpreted as a Cauchy principal value integral, for user specified
                :math:`c` and :math:`f`. The strategy is globally adaptive. Modified
                Clenshaw-Curtis integration is used on those intervals containing the
                point :math:`x = c`.
    
            **Integration of Complex Function of a Real Variable**
    
            A complex valued function, :math:`f`, of a real variable can be written as
            :math:`f = g + ih`.  Similarly, the integral of :math:`f` can be
            written as
    
            .. math::
                \int_a^b f(x) dx = \int_a^b g(x) dx + i\int_a^b h(x) dx
    
            assuming that the integrals of :math:`g` and :math:`h` exist
            over the interval :math:`[a,b]` [2]_. Therefore, ``quad`` integrates
            complex-valued functions by integrating the real and imaginary components
            separately.
    
    
            References
            ----------
    
            .. [1] Piessens, Robert; de Doncker-Kapenga, Elise;
                   Überhuber, Christoph W.; Kahaner, David (1983).
                   QUADPACK: A subroutine package for automatic integration.
                   Springer-Verlag.
                   ISBN 978-3-540-12553-2.
    
            .. [2] McCullough, Thomas; Phillips, Keith (1973).
                   Foundations of Analysis in the Complex Plane.
                   Holt Rinehart Winston.
                   ISBN 0-03-086370-8
    
            Examples
            --------
            Calculate :math:`\int^4_0 x^2 dx` and compare with an analytic result
    
            >>> from scipy import integrate
            >>> import numpy as np
            >>> x2 = lambda x: x**2
            >>> integrate.quad(x2, 0, 4)
            (21.333333333333332, 2.3684757858670003e-13)
            >>> print(4**3 / 3.)  # analytical result
            21.3333333333
    
            Calculate :math:`\int^\infty_0 e^{-x} dx`
    
            >>> invexp = lambda x: np.exp(-x)
            >>> integrate.quad(invexp, 0, np.inf)
            (1.0, 5.842605999138044e-11)
    
            Calculate :math:`\int^1_0 a x \,dx` for :math:`a = 1, 3`
    
            >>> f = lambda x, a: a*x
            >>> y, err = integrate.quad(f, 0, 1, args=(1,))
            >>> y
            0.5
            >>> y, err = integrate.quad(f, 0, 1, args=(3,))
            >>> y
            1.5
    
            Calculate :math:`\int^1_0 x^2 + y^2 dx` with ctypes, holding
            y parameter as 1::
    
                testlib.c =>
                    double func(int n, double args[n]){
                        return args[0]*args[0] + args[1]*args[1];}
                compile to library testlib.*
    
            ::
    
               from scipy import integrate
               import ctypes
               lib = ctypes.CDLL('/home/.../testlib.*') #use absolute path
               lib.func.restype = ctypes.c_double
               lib.func.argtypes = (ctypes.c_int,ctypes.c_double)
               integrate.quad(lib.func,0,1,(1))
               #(1.3333333333333333, 1.4802973661668752e-14)
               print((1.0**3/3.0 + 1.0) - (0.0**3/3.0 + 0.0)) #Analytic result
               # 1.3333333333333333
    
            Be aware that pulse shapes and other sharp features as compared to the
            size of the integration interval may not be integrated correctly using
            this method. A simplified example of this limitation is integrating a
            y-axis reflected step function with many zero values within the integrals
            bounds.
    
            >>> y = lambda x: 1 if x<=0 else 0
            >>> integrate.quad(y, -1, 1)
            (1.0, 1.1102230246251565e-14)
            >>> integrate.quad(y, -1, 100)
            (1.0000000002199108, 1.0189464580163188e-08)
            >>> integrate.quad(y, -1, 10000)
            (0.0, 0.0)
    
        quad_vec(f, a, b, epsabs=1e-200, epsrel=1e-08, norm='2', cache_size=100000000.0, limit=10000, workers=1, points=None, quadrature=None, full_output=False, *, args=())
            Adaptive integration of a vector-valued function.
    
            Parameters
            ----------
            f : callable
                Vector-valued function f(x) to integrate.
            a : float
                Initial point.
            b : float
                Final point.
            epsabs : float, optional
                Absolute tolerance.
            epsrel : float, optional
                Relative tolerance.
            norm : {'max', '2'}, optional
                Vector norm to use for error estimation.
            cache_size : int, optional
                Number of bytes to use for memoization.
            limit : float or int, optional
                An upper bound on the number of subintervals used in the adaptive
                algorithm.
            workers : int or map-like callable, optional
                If `workers` is an integer, part of the computation is done in
                parallel subdivided to this many tasks (using
                :class:`python:multiprocessing.pool.Pool`).
                Supply `-1` to use all cores available to the Process.
                Alternatively, supply a map-like callable, such as
                :meth:`python:multiprocessing.pool.Pool.map` for evaluating the
                population in parallel.
                This evaluation is carried out as ``workers(func, iterable)``.
            points : list, optional
                List of additional breakpoints.
            quadrature : {'gk21', 'gk15', 'trapezoid'}, optional
                Quadrature rule to use on subintervals.
                Options: 'gk21' (Gauss-Kronrod 21-point rule),
                'gk15' (Gauss-Kronrod 15-point rule),
                'trapezoid' (composite trapezoid rule).
                Default: 'gk21' for finite intervals and 'gk15' for (semi-)infinite
            full_output : bool, optional
                Return an additional ``info`` dictionary.
            args : tuple, optional
                Extra arguments to pass to function, if any.
    
                .. versionadded:: 1.8.0
    
            Returns
            -------
            res : {float, array-like}
                Estimate for the result
            err : float
                Error estimate for the result in the given norm
            info : dict
                Returned only when ``full_output=True``.
                Info dictionary. Is an object with the attributes:
    
                    success : bool
                        Whether integration reached target precision.
                    status : int
                        Indicator for convergence, success (0),
                        failure (1), and failure due to rounding error (2).
                    neval : int
                        Number of function evaluations.
                    intervals : ndarray, shape (num_intervals, 2)
                        Start and end points of subdivision intervals.
                    integrals : ndarray, shape (num_intervals, ...)
                        Integral for each interval.
                        Note that at most ``cache_size`` values are recorded,
                        and the array may contains *nan* for missing items.
                    errors : ndarray, shape (num_intervals,)
                        Estimated integration error for each interval.
    
            Notes
            -----
            The algorithm mainly follows the implementation of QUADPACK's
            DQAG* algorithms, implementing global error control and adaptive
            subdivision.
    
            The algorithm here has some differences to the QUADPACK approach:
    
            Instead of subdividing one interval at a time, the algorithm
            subdivides N intervals with largest errors at once. This enables
            (partial) parallelization of the integration.
    
            The logic of subdividing "next largest" intervals first is then
            not implemented, and we rely on the above extension to avoid
            concentrating on "small" intervals only.
    
            The Wynn epsilon table extrapolation is not used (QUADPACK uses it
            for infinite intervals). This is because the algorithm here is
            supposed to work on vector-valued functions, in an user-specified
            norm, and the extension of the epsilon algorithm to this case does
            not appear to be widely agreed. For max-norm, using elementwise
            Wynn epsilon could be possible, but we do not do this here with
            the hope that the epsilon extrapolation is mainly useful in
            special cases.
    
            References
            ----------
            [1] R. Piessens, E. de Doncker, QUADPACK (1983).
    
            Examples
            --------
            We can compute integrations of a vector-valued function:
    
            >>> from scipy.integrate import quad_vec
            >>> import numpy as np
            >>> import matplotlib.pyplot as plt
            >>> alpha = np.linspace(0.0, 2.0, num=30)
            >>> f = lambda x: x**alpha
            >>> x0, x1 = 0, 2
            >>> y, err = quad_vec(f, x0, x1)
            >>> plt.plot(alpha, y)
            >>> plt.xlabel(r"$\alpha$")
            >>> plt.ylabel(r"$\int_{0}^{2} x^\alpha dx$")
            >>> plt.show()
    
        quadrature(func, a, b, args=(), tol=1.49e-08, rtol=1.49e-08, maxiter=50, vec_func=True, miniter=1)
            Compute a definite integral using fixed-tolerance Gaussian quadrature.
    
            .. deprecated:: 1.12.0
    
                  This function is deprecated as of SciPy 1.12.0 and will be removed
                  in SciPy 1.15.0. Please use `scipy.integrate.quad` instead.
    
            Integrate `func` from `a` to `b` using Gaussian quadrature
            with absolute tolerance `tol`.
    
            Parameters
            ----------
            func : function
                A Python function or method to integrate.
            a : float
                Lower limit of integration.
            b : float
                Upper limit of integration.
            args : tuple, optional
                Extra arguments to pass to function.
            tol, rtol : float, optional
                Iteration stops when error between last two iterates is less than
                `tol` OR the relative change is less than `rtol`.
            maxiter : int, optional
                Maximum order of Gaussian quadrature.
            vec_func : bool, optional
                True or False if func handles arrays as arguments (is
                a "vector" function). Default is True.
            miniter : int, optional
                Minimum order of Gaussian quadrature.
    
            Returns
            -------
            val : float
                Gaussian quadrature approximation (within tolerance) to integral.
            err : float
                Difference between last two estimates of the integral.
    
            See Also
            --------
            fixed_quad : fixed-order Gaussian quadrature
            quad : adaptive quadrature using QUADPACK
            dblquad : double integrals
            tplquad : triple integrals
            romb : integrator for sampled data
            simpson : integrator for sampled data
            cumulative_trapezoid : cumulative integration for sampled data
    
            Examples
            --------
            >>> from scipy import integrate
            >>> import numpy as np
            >>> f = lambda x: x**8
            >>> integrate.quadrature(f, 0.0, 1.0)
            (0.11111111111111106, 4.163336342344337e-17)
            >>> print(1/9.0)  # analytical result
            0.1111111111111111
    
            >>> integrate.quadrature(np.cos, 0.0, np.pi/2)
            (0.9999999999999536, 3.9611425250996035e-11)
            >>> np.sin(np.pi/2)-np.sin(0)  # analytical result
            1.0
    
        romb(y, dx=1.0, axis=-1, show=False)
            Romberg integration using samples of a function.
    
            Parameters
            ----------
            y : array_like
                A vector of ``2**k + 1`` equally-spaced samples of a function.
            dx : float, optional
                The sample spacing. Default is 1.
            axis : int, optional
                The axis along which to integrate. Default is -1 (last axis).
            show : bool, optional
                When `y` is a single 1-D array, then if this argument is True
                print the table showing Richardson extrapolation from the
                samples. Default is False.
    
            Returns
            -------
            romb : ndarray
                The integrated result for `axis`.
    
            See Also
            --------
            quad : adaptive quadrature using QUADPACK
            fixed_quad : fixed-order Gaussian quadrature
            dblquad : double integrals
            tplquad : triple integrals
            simpson : integrators for sampled data
            cumulative_trapezoid : cumulative integration for sampled data
    
            Examples
            --------
            >>> from scipy import integrate
            >>> import numpy as np
            >>> x = np.arange(10, 14.25, 0.25)
            >>> y = np.arange(3, 12)
    
            >>> integrate.romb(y)
            56.0
    
            >>> y = np.sin(np.power(x, 2.5))
            >>> integrate.romb(y)
            -0.742561336672229
    
            >>> integrate.romb(y, show=True)
            Richardson Extrapolation Table for Romberg Integration
            ======================================================
            -0.81576
             4.63862  6.45674
            -1.10581 -3.02062 -3.65245
            -2.57379 -3.06311 -3.06595 -3.05664
            -1.34093 -0.92997 -0.78776 -0.75160 -0.74256
            ======================================================
            -0.742561336672229  # may vary
    
        romberg(function, a, b, args=(), tol=1.48e-08, rtol=1.48e-08, show=False, divmax=10, vec_func=False)
            Romberg integration of a callable function or method.
    
            .. deprecated:: 1.12.0
    
                  This function is deprecated as of SciPy 1.12.0 and will be removed
                  in SciPy 1.15.0. Please use `scipy.integrate.quad` instead.
    
            Returns the integral of `function` (a function of one variable)
            over the interval (`a`, `b`).
    
            If `show` is 1, the triangular array of the intermediate results
            will be printed. If `vec_func` is True (default is False), then
            `function` is assumed to support vector arguments.
    
            Parameters
            ----------
            function : callable
                Function to be integrated.
            a : float
                Lower limit of integration.
            b : float
                Upper limit of integration.
    
            Returns
            -------
            results : float
                Result of the integration.
    
            Other Parameters
            ----------------
            args : tuple, optional
                Extra arguments to pass to function. Each element of `args` will
                be passed as a single argument to `func`. Default is to pass no
                extra arguments.
            tol, rtol : float, optional
                The desired absolute and relative tolerances. Defaults are 1.48e-8.
            show : bool, optional
                Whether to print the results. Default is False.
            divmax : int, optional
                Maximum order of extrapolation. Default is 10.
            vec_func : bool, optional
                Whether `func` handles arrays as arguments (i.e., whether it is a
                "vector" function). Default is False.
    
            See Also
            --------
            fixed_quad : Fixed-order Gaussian quadrature.
            quad : Adaptive quadrature using QUADPACK.
            dblquad : Double integrals.
            tplquad : Triple integrals.
            romb : Integrators for sampled data.
            simpson : Integrators for sampled data.
            cumulative_trapezoid : Cumulative integration for sampled data.
    
            References
            ----------
            .. [1] 'Romberg's method' https://en.wikipedia.org/wiki/Romberg%27s_method
    
            Examples
            --------
            Integrate a gaussian from 0 to 1 and compare to the error function.
    
            >>> from scipy import integrate
            >>> from scipy.special import erf
            >>> import numpy as np
            >>> gaussian = lambda x: 1/np.sqrt(np.pi) * np.exp(-x**2)
            >>> result = integrate.romberg(gaussian, 0, 1, show=True)
            Romberg integration of <function vfunc at ...> from [0, 1]
    
            ::
    
               Steps  StepSize  Results
                   1  1.000000  0.385872
                   2  0.500000  0.412631  0.421551
                   4  0.250000  0.419184  0.421368  0.421356
                   8  0.125000  0.420810  0.421352  0.421350  0.421350
                  16  0.062500  0.421215  0.421350  0.421350  0.421350  0.421350
                  32  0.031250  0.421317  0.421350  0.421350  0.421350  0.421350  0.421350
    
            The final result is 0.421350396475 after 33 function evaluations.
    
            >>> print("%g %g" % (2*result, erf(1)))
            0.842701 0.842701
    
        simpson(y, *, x=None, dx=1.0, axis=-1)
            Integrate y(x) using samples along the given axis and the composite
            Simpson's rule. If x is None, spacing of dx is assumed.
    
            If there are an even number of samples, N, then there are an odd
            number of intervals (N-1), but Simpson's rule requires an even number
            of intervals. The parameter 'even' controls how this is handled.
    
            Parameters
            ----------
            y : array_like
                Array to be integrated.
            x : array_like, optional
                If given, the points at which `y` is sampled.
            dx : float, optional
                Spacing of integration points along axis of `x`. Only used when
                `x` is None. Default is 1.
            axis : int, optional
                Axis along which to integrate. Default is the last axis.
    
            Returns
            -------
            float
                The estimated integral computed with the composite Simpson's rule.
    
            See Also
            --------
            quad : adaptive quadrature using QUADPACK
            fixed_quad : fixed-order Gaussian quadrature
            dblquad : double integrals
            tplquad : triple integrals
            romb : integrators for sampled data
            cumulative_trapezoid : cumulative integration for sampled data
            cumulative_simpson : cumulative integration using Simpson's 1/3 rule
    
            Notes
            -----
            For an odd number of samples that are equally spaced the result is
            exact if the function is a polynomial of order 3 or less. If
            the samples are not equally spaced, then the result is exact only
            if the function is a polynomial of order 2 or less.
    
            References
            ----------
            .. [1] Cartwright, Kenneth V. Simpson's Rule Cumulative Integration with
                   MS Excel and Irregularly-spaced Data. Journal of Mathematical
                   Sciences and Mathematics Education. 12 (2): 1-9
    
            Examples
            --------
            >>> from scipy import integrate
            >>> import numpy as np
            >>> x = np.arange(0, 10)
            >>> y = np.arange(0, 10)
    
            >>> integrate.simpson(y, x=x)
            40.5
    
            >>> y = np.power(x, 3)
            >>> integrate.simpson(y, x=x)
            1640.5
            >>> integrate.quad(lambda x: x**3, 0, 9)[0]
            1640.25
    
        solve_bvp(fun, bc, x, y, p=None, S=None, fun_jac=None, bc_jac=None, tol=0.001, max_nodes=1000, verbose=0, bc_tol=None)
            Solve a boundary value problem for a system of ODEs.
    
            This function numerically solves a first order system of ODEs subject to
            two-point boundary conditions::
    
                dy / dx = f(x, y, p) + S * y / (x - a), a <= x <= b
                bc(y(a), y(b), p) = 0
    
            Here x is a 1-D independent variable, y(x) is an N-D
            vector-valued function and p is a k-D vector of unknown
            parameters which is to be found along with y(x). For the problem to be
            determined, there must be n + k boundary conditions, i.e., bc must be an
            (n + k)-D function.
    
            The last singular term on the right-hand side of the system is optional.
            It is defined by an n-by-n matrix S, such that the solution must satisfy
            S y(a) = 0. This condition will be forced during iterations, so it must not
            contradict boundary conditions. See [2]_ for the explanation how this term
            is handled when solving BVPs numerically.
    
            Problems in a complex domain can be solved as well. In this case, y and p
            are considered to be complex, and f and bc are assumed to be complex-valued
            functions, but x stays real. Note that f and bc must be complex
            differentiable (satisfy Cauchy-Riemann equations [4]_), otherwise you
            should rewrite your problem for real and imaginary parts separately. To
            solve a problem in a complex domain, pass an initial guess for y with a
            complex data type (see below).
    
            Parameters
            ----------
            fun : callable
                Right-hand side of the system. The calling signature is ``fun(x, y)``,
                or ``fun(x, y, p)`` if parameters are present. All arguments are
                ndarray: ``x`` with shape (m,), ``y`` with shape (n, m), meaning that
                ``y[:, i]`` corresponds to ``x[i]``, and ``p`` with shape (k,). The
                return value must be an array with shape (n, m) and with the same
                layout as ``y``.
            bc : callable
                Function evaluating residuals of the boundary conditions. The calling
                signature is ``bc(ya, yb)``, or ``bc(ya, yb, p)`` if parameters are
                present. All arguments are ndarray: ``ya`` and ``yb`` with shape (n,),
                and ``p`` with shape (k,). The return value must be an array with
                shape (n + k,).
            x : array_like, shape (m,)
                Initial mesh. Must be a strictly increasing sequence of real numbers
                with ``x[0]=a`` and ``x[-1]=b``.
            y : array_like, shape (n, m)
                Initial guess for the function values at the mesh nodes, ith column
                corresponds to ``x[i]``. For problems in a complex domain pass `y`
                with a complex data type (even if the initial guess is purely real).
            p : array_like with shape (k,) or None, optional
                Initial guess for the unknown parameters. If None (default), it is
                assumed that the problem doesn't depend on any parameters.
            S : array_like with shape (n, n) or None
                Matrix defining the singular term. If None (default), the problem is
                solved without the singular term.
            fun_jac : callable or None, optional
                Function computing derivatives of f with respect to y and p. The
                calling signature is ``fun_jac(x, y)``, or ``fun_jac(x, y, p)`` if
                parameters are present. The return must contain 1 or 2 elements in the
                following order:
    
                    * df_dy : array_like with shape (n, n, m), where an element
                      (i, j, q) equals to d f_i(x_q, y_q, p) / d (y_q)_j.
                    * df_dp : array_like with shape (n, k, m), where an element
                      (i, j, q) equals to d f_i(x_q, y_q, p) / d p_j.
    
                Here q numbers nodes at which x and y are defined, whereas i and j
                number vector components. If the problem is solved without unknown
                parameters, df_dp should not be returned.
    
                If `fun_jac` is None (default), the derivatives will be estimated
                by the forward finite differences.
            bc_jac : callable or None, optional
                Function computing derivatives of bc with respect to ya, yb, and p.
                The calling signature is ``bc_jac(ya, yb)``, or ``bc_jac(ya, yb, p)``
                if parameters are present. The return must contain 2 or 3 elements in
                the following order:
    
                    * dbc_dya : array_like with shape (n, n), where an element (i, j)
                      equals to d bc_i(ya, yb, p) / d ya_j.
                    * dbc_dyb : array_like with shape (n, n), where an element (i, j)
                      equals to d bc_i(ya, yb, p) / d yb_j.
                    * dbc_dp : array_like with shape (n, k), where an element (i, j)
                      equals to d bc_i(ya, yb, p) / d p_j.
    
                If the problem is solved without unknown parameters, dbc_dp should not
                be returned.
    
                If `bc_jac` is None (default), the derivatives will be estimated by
                the forward finite differences.
            tol : float, optional
                Desired tolerance of the solution. If we define ``r = y' - f(x, y)``,
                where y is the found solution, then the solver tries to achieve on each
                mesh interval ``norm(r / (1 + abs(f)) < tol``, where ``norm`` is
                estimated in a root mean squared sense (using a numerical quadrature
                formula). Default is 1e-3.
            max_nodes : int, optional
                Maximum allowed number of the mesh nodes. If exceeded, the algorithm
                terminates. Default is 1000.
            verbose : {0, 1, 2}, optional
                Level of algorithm's verbosity:
    
                    * 0 (default) : work silently.
                    * 1 : display a termination report.
                    * 2 : display progress during iterations.
            bc_tol : float, optional
                Desired absolute tolerance for the boundary condition residuals: `bc`
                value should satisfy ``abs(bc) < bc_tol`` component-wise.
                Equals to `tol` by default. Up to 10 iterations are allowed to achieve this
                tolerance.
    
            Returns
            -------
            Bunch object with the following fields defined:
            sol : PPoly
                Found solution for y as `scipy.interpolate.PPoly` instance, a C1
                continuous cubic spline.
            p : ndarray or None, shape (k,)
                Found parameters. None, if the parameters were not present in the
                problem.
            x : ndarray, shape (m,)
                Nodes of the final mesh.
            y : ndarray, shape (n, m)
                Solution values at the mesh nodes.
            yp : ndarray, shape (n, m)
                Solution derivatives at the mesh nodes.
            rms_residuals : ndarray, shape (m - 1,)
                RMS values of the relative residuals over each mesh interval (see the
                description of `tol` parameter).
            niter : int
                Number of completed iterations.
            status : int
                Reason for algorithm termination:
    
                    * 0: The algorithm converged to the desired accuracy.
                    * 1: The maximum number of mesh nodes is exceeded.
                    * 2: A singular Jacobian encountered when solving the collocation
                      system.
    
            message : string
                Verbal description of the termination reason.
            success : bool
                True if the algorithm converged to the desired accuracy (``status=0``).
    
            Notes
            -----
            This function implements a 4th order collocation algorithm with the
            control of residuals similar to [1]_. A collocation system is solved
            by a damped Newton method with an affine-invariant criterion function as
            described in [3]_.
    
            Note that in [1]_  integral residuals are defined without normalization
            by interval lengths. So, their definition is different by a multiplier of
            h**0.5 (h is an interval length) from the definition used here.
    
            .. versionadded:: 0.18.0
    
            References
            ----------
            .. [1] J. Kierzenka, L. F. Shampine, "A BVP Solver Based on Residual
                   Control and the Maltab PSE", ACM Trans. Math. Softw., Vol. 27,
                   Number 3, pp. 299-316, 2001.
            .. [2] L.F. Shampine, P. H. Muir and H. Xu, "A User-Friendly Fortran BVP
                   Solver".
            .. [3] U. Ascher, R. Mattheij and R. Russell "Numerical Solution of
                   Boundary Value Problems for Ordinary Differential Equations".
            .. [4] `Cauchy-Riemann equations
                    <https://en.wikipedia.org/wiki/Cauchy-Riemann_equations>`_ on
                    Wikipedia.
    
            Examples
            --------
            In the first example, we solve Bratu's problem::
    
                y'' + k * exp(y) = 0
                y(0) = y(1) = 0
    
            for k = 1.
    
            We rewrite the equation as a first-order system and implement its
            right-hand side evaluation::
    
                y1' = y2
                y2' = -exp(y1)
    
            >>> import numpy as np
            >>> def fun(x, y):
            ...     return np.vstack((y[1], -np.exp(y[0])))
    
            Implement evaluation of the boundary condition residuals:
    
            >>> def bc(ya, yb):
            ...     return np.array([ya[0], yb[0]])
    
            Define the initial mesh with 5 nodes:
    
            >>> x = np.linspace(0, 1, 5)
    
            This problem is known to have two solutions. To obtain both of them, we
            use two different initial guesses for y. We denote them by subscripts
            a and b.
    
            >>> y_a = np.zeros((2, x.size))
            >>> y_b = np.zeros((2, x.size))
            >>> y_b[0] = 3
    
            Now we are ready to run the solver.
    
            >>> from scipy.integrate import solve_bvp
            >>> res_a = solve_bvp(fun, bc, x, y_a)
            >>> res_b = solve_bvp(fun, bc, x, y_b)
    
            Let's plot the two found solutions. We take an advantage of having the
            solution in a spline form to produce a smooth plot.
    
            >>> x_plot = np.linspace(0, 1, 100)
            >>> y_plot_a = res_a.sol(x_plot)[0]
            >>> y_plot_b = res_b.sol(x_plot)[0]
            >>> import matplotlib.pyplot as plt
            >>> plt.plot(x_plot, y_plot_a, label='y_a')
            >>> plt.plot(x_plot, y_plot_b, label='y_b')
            >>> plt.legend()
            >>> plt.xlabel("x")
            >>> plt.ylabel("y")
            >>> plt.show()
    
            We see that the two solutions have similar shape, but differ in scale
            significantly.
    
            In the second example, we solve a simple Sturm-Liouville problem::
    
                y'' + k**2 * y = 0
                y(0) = y(1) = 0
    
            It is known that a non-trivial solution y = A * sin(k * x) is possible for
            k = pi * n, where n is an integer. To establish the normalization constant
            A = 1 we add a boundary condition::
    
                y'(0) = k
    
            Again, we rewrite our equation as a first-order system and implement its
            right-hand side evaluation::
    
                y1' = y2
                y2' = -k**2 * y1
    
            >>> def fun(x, y, p):
            ...     k = p[0]
            ...     return np.vstack((y[1], -k**2 * y[0]))
    
            Note that parameters p are passed as a vector (with one element in our
            case).
    
            Implement the boundary conditions:
    
            >>> def bc(ya, yb, p):
            ...     k = p[0]
            ...     return np.array([ya[0], yb[0], ya[1] - k])
    
            Set up the initial mesh and guess for y. We aim to find the solution for
            k = 2 * pi, to achieve that we set values of y to approximately follow
            sin(2 * pi * x):
    
            >>> x = np.linspace(0, 1, 5)
            >>> y = np.zeros((2, x.size))
            >>> y[0, 1] = 1
            >>> y[0, 3] = -1
    
            Run the solver with 6 as an initial guess for k.
    
            >>> sol = solve_bvp(fun, bc, x, y, p=[6])
    
            We see that the found k is approximately correct:
    
            >>> sol.p[0]
            6.28329460046
    
            And, finally, plot the solution to see the anticipated sinusoid:
    
            >>> x_plot = np.linspace(0, 1, 100)
            >>> y_plot = sol.sol(x_plot)[0]
            >>> plt.plot(x_plot, y_plot)
            >>> plt.xlabel("x")
            >>> plt.ylabel("y")
            >>> plt.show()
    
        solve_ivp(fun, t_span, y0, method='RK45', t_eval=None, dense_output=False, events=None, vectorized=False, args=None, **options)
            Solve an initial value problem for a system of ODEs.
    
            This function numerically integrates a system of ordinary differential
            equations given an initial value::
    
                dy / dt = f(t, y)
                y(t0) = y0
    
            Here t is a 1-D independent variable (time), y(t) is an
            N-D vector-valued function (state), and an N-D
            vector-valued function f(t, y) determines the differential equations.
            The goal is to find y(t) approximately satisfying the differential
            equations, given an initial value y(t0)=y0.
    
            Some of the solvers support integration in the complex domain, but note
            that for stiff ODE solvers, the right-hand side must be
            complex-differentiable (satisfy Cauchy-Riemann equations [11]_).
            To solve a problem in the complex domain, pass y0 with a complex data type.
            Another option always available is to rewrite your problem for real and
            imaginary parts separately.
    
            Parameters
            ----------
            fun : callable
                Right-hand side of the system: the time derivative of the state ``y``
                at time ``t``. The calling signature is ``fun(t, y)``, where ``t`` is a
                scalar and ``y`` is an ndarray with ``len(y) = len(y0)``. Additional
                arguments need to be passed if ``args`` is used (see documentation of
                ``args`` argument). ``fun`` must return an array of the same shape as
                ``y``. See `vectorized` for more information.
            t_span : 2-member sequence
                Interval of integration (t0, tf). The solver starts with t=t0 and
                integrates until it reaches t=tf. Both t0 and tf must be floats
                or values interpretable by the float conversion function.
            y0 : array_like, shape (n,)
                Initial state. For problems in the complex domain, pass `y0` with a
                complex data type (even if the initial value is purely real).
            method : string or `OdeSolver`, optional
                Integration method to use:
    
                    * 'RK45' (default): Explicit Runge-Kutta method of order 5(4) [1]_.
                      The error is controlled assuming accuracy of the fourth-order
                      method, but steps are taken using the fifth-order accurate
                      formula (local extrapolation is done). A quartic interpolation
                      polynomial is used for the dense output [2]_. Can be applied in
                      the complex domain.
                    * 'RK23': Explicit Runge-Kutta method of order 3(2) [3]_. The error
                      is controlled assuming accuracy of the second-order method, but
                      steps are taken using the third-order accurate formula (local
                      extrapolation is done). A cubic Hermite polynomial is used for the
                      dense output. Can be applied in the complex domain.
                    * 'DOP853': Explicit Runge-Kutta method of order 8 [13]_.
                      Python implementation of the "DOP853" algorithm originally
                      written in Fortran [14]_. A 7-th order interpolation polynomial
                      accurate to 7-th order is used for the dense output.
                      Can be applied in the complex domain.
                    * 'Radau': Implicit Runge-Kutta method of the Radau IIA family of
                      order 5 [4]_. The error is controlled with a third-order accurate
                      embedded formula. A cubic polynomial which satisfies the
                      collocation conditions is used for the dense output.
                    * 'BDF': Implicit multi-step variable-order (1 to 5) method based
                      on a backward differentiation formula for the derivative
                      approximation [5]_. The implementation follows the one described
                      in [6]_. A quasi-constant step scheme is used and accuracy is
                      enhanced using the NDF modification. Can be applied in the
                      complex domain.
                    * 'LSODA': Adams/BDF method with automatic stiffness detection and
                      switching [7]_, [8]_. This is a wrapper of the Fortran solver
                      from ODEPACK.
    
                Explicit Runge-Kutta methods ('RK23', 'RK45', 'DOP853') should be used
                for non-stiff problems and implicit methods ('Radau', 'BDF') for
                stiff problems [9]_. Among Runge-Kutta methods, 'DOP853' is recommended
                for solving with high precision (low values of `rtol` and `atol`).
    
                If not sure, first try to run 'RK45'. If it makes unusually many
                iterations, diverges, or fails, your problem is likely to be stiff and
                you should use 'Radau' or 'BDF'. 'LSODA' can also be a good universal
                choice, but it might be somewhat less convenient to work with as it
                wraps old Fortran code.
    
                You can also pass an arbitrary class derived from `OdeSolver` which
                implements the solver.
            t_eval : array_like or None, optional
                Times at which to store the computed solution, must be sorted and lie
                within `t_span`. If None (default), use points selected by the solver.
            dense_output : bool, optional
                Whether to compute a continuous solution. Default is False.
            events : callable, or list of callables, optional
                Events to track. If None (default), no events will be tracked.
                Each event occurs at the zeros of a continuous function of time and
                state. Each function must have the signature ``event(t, y)`` where
                additional argument have to be passed if ``args`` is used (see
                documentation of ``args`` argument). Each function must return a
                float. The solver will find an accurate value of `t` at which
                ``event(t, y(t)) = 0`` using a root-finding algorithm. By default,
                all zeros will be found. The solver looks for a sign change over
                each step, so if multiple zero crossings occur within one step,
                events may be missed. Additionally each `event` function might
                have the following attributes:
    
                    terminal: bool or int, optional
                        When boolean, whether to terminate integration if this event occurs.
                        When integral, termination occurs after the specified the number of
                        occurences of this event.
                        Implicitly False if not assigned.
                    direction: float, optional
                        Direction of a zero crossing. If `direction` is positive,
                        `event` will only trigger when going from negative to positive,
                        and vice versa if `direction` is negative. If 0, then either
                        direction will trigger event. Implicitly 0 if not assigned.
    
                You can assign attributes like ``event.terminal = True`` to any
                function in Python.
            vectorized : bool, optional
                Whether `fun` can be called in a vectorized fashion. Default is False.
    
                If ``vectorized`` is False, `fun` will always be called with ``y`` of
                shape ``(n,)``, where ``n = len(y0)``.
    
                If ``vectorized`` is True, `fun` may be called with ``y`` of shape
                ``(n, k)``, where ``k`` is an integer. In this case, `fun` must behave
                such that ``fun(t, y)[:, i] == fun(t, y[:, i])`` (i.e. each column of
                the returned array is the time derivative of the state corresponding
                with a column of ``y``).
    
                Setting ``vectorized=True`` allows for faster finite difference
                approximation of the Jacobian by methods 'Radau' and 'BDF', but
                will result in slower execution for other methods and for 'Radau' and
                'BDF' in some circumstances (e.g. small ``len(y0)``).
            args : tuple, optional
                Additional arguments to pass to the user-defined functions.  If given,
                the additional arguments are passed to all user-defined functions.
                So if, for example, `fun` has the signature ``fun(t, y, a, b, c)``,
                then `jac` (if given) and any event functions must have the same
                signature, and `args` must be a tuple of length 3.
            **options
                Options passed to a chosen solver. All options available for already
                implemented solvers are listed below.
            first_step : float or None, optional
                Initial step size. Default is `None` which means that the algorithm
                should choose.
            max_step : float, optional
                Maximum allowed step size. Default is np.inf, i.e., the step size is not
                bounded and determined solely by the solver.
            rtol, atol : float or array_like, optional
                Relative and absolute tolerances. The solver keeps the local error
                estimates less than ``atol + rtol * abs(y)``. Here `rtol` controls a
                relative accuracy (number of correct digits), while `atol` controls
                absolute accuracy (number of correct decimal places). To achieve the
                desired `rtol`, set `atol` to be smaller than the smallest value that
                can be expected from ``rtol * abs(y)`` so that `rtol` dominates the
                allowable error. If `atol` is larger than ``rtol * abs(y)`` the
                number of correct digits is not guaranteed. Conversely, to achieve the
                desired `atol` set `rtol` such that ``rtol * abs(y)`` is always smaller
                than `atol`. If components of y have different scales, it might be
                beneficial to set different `atol` values for different components by
                passing array_like with shape (n,) for `atol`. Default values are
                1e-3 for `rtol` and 1e-6 for `atol`.
            jac : array_like, sparse_matrix, callable or None, optional
                Jacobian matrix of the right-hand side of the system with respect
                to y, required by the 'Radau', 'BDF' and 'LSODA' method. The
                Jacobian matrix has shape (n, n) and its element (i, j) is equal to
                ``d f_i / d y_j``.  There are three ways to define the Jacobian:
    
                    * If array_like or sparse_matrix, the Jacobian is assumed to
                      be constant. Not supported by 'LSODA'.
                    * If callable, the Jacobian is assumed to depend on both
                      t and y; it will be called as ``jac(t, y)``, as necessary.
                      Additional arguments have to be passed if ``args`` is
                      used (see documentation of ``args`` argument).
                      For 'Radau' and 'BDF' methods, the return value might be a
                      sparse matrix.
                    * If None (default), the Jacobian will be approximated by
                      finite differences.
    
                It is generally recommended to provide the Jacobian rather than
                relying on a finite-difference approximation.
            jac_sparsity : array_like, sparse matrix or None, optional
                Defines a sparsity structure of the Jacobian matrix for a finite-
                difference approximation. Its shape must be (n, n). This argument
                is ignored if `jac` is not `None`. If the Jacobian has only few
                non-zero elements in *each* row, providing the sparsity structure
                will greatly speed up the computations [10]_. A zero entry means that
                a corresponding element in the Jacobian is always zero. If None
                (default), the Jacobian is assumed to be dense.
                Not supported by 'LSODA', see `lband` and `uband` instead.
            lband, uband : int or None, optional
                Parameters defining the bandwidth of the Jacobian for the 'LSODA'
                method, i.e., ``jac[i, j] != 0 only for i - lband <= j <= i + uband``.
                Default is None. Setting these requires your jac routine to return the
                Jacobian in the packed format: the returned array must have ``n``
                columns and ``uband + lband + 1`` rows in which Jacobian diagonals are
                written. Specifically ``jac_packed[uband + i - j , j] = jac[i, j]``.
                The same format is used in `scipy.linalg.solve_banded` (check for an
                illustration).  These parameters can be also used with ``jac=None`` to
                reduce the number of Jacobian elements estimated by finite differences.
            min_step : float, optional
                The minimum allowed step size for 'LSODA' method.
                By default `min_step` is zero.
    
            Returns
            -------
            Bunch object with the following fields defined:
            t : ndarray, shape (n_points,)
                Time points.
            y : ndarray, shape (n, n_points)
                Values of the solution at `t`.
            sol : `OdeSolution` or None
                Found solution as `OdeSolution` instance; None if `dense_output` was
                set to False.
            t_events : list of ndarray or None
                Contains for each event type a list of arrays at which an event of
                that type event was detected. None if `events` was None.
            y_events : list of ndarray or None
                For each value of `t_events`, the corresponding value of the solution.
                None if `events` was None.
            nfev : int
                Number of evaluations of the right-hand side.
            njev : int
                Number of evaluations of the Jacobian.
            nlu : int
                Number of LU decompositions.
            status : int
                Reason for algorithm termination:
    
                    * -1: Integration step failed.
                    *  0: The solver successfully reached the end of `tspan`.
                    *  1: A termination event occurred.
    
            message : string
                Human-readable description of the termination reason.
            success : bool
                True if the solver reached the interval end or a termination event
                occurred (``status >= 0``).
    
            References
            ----------
            .. [1] J. R. Dormand, P. J. Prince, "A family of embedded Runge-Kutta
                   formulae", Journal of Computational and Applied Mathematics, Vol. 6,
                   No. 1, pp. 19-26, 1980.
            .. [2] L. W. Shampine, "Some Practical Runge-Kutta Formulas", Mathematics
                   of Computation,, Vol. 46, No. 173, pp. 135-150, 1986.
            .. [3] P. Bogacki, L.F. Shampine, "A 3(2) Pair of Runge-Kutta Formulas",
                   Appl. Math. Lett. Vol. 2, No. 4. pp. 321-325, 1989.
            .. [4] E. Hairer, G. Wanner, "Solving Ordinary Differential Equations II:
                   Stiff and Differential-Algebraic Problems", Sec. IV.8.
            .. [5] `Backward Differentiation Formula
                    <https://en.wikipedia.org/wiki/Backward_differentiation_formula>`_
                    on Wikipedia.
            .. [6] L. F. Shampine, M. W. Reichelt, "THE MATLAB ODE SUITE", SIAM J. SCI.
                   COMPUTE., Vol. 18, No. 1, pp. 1-22, January 1997.
            .. [7] A. C. Hindmarsh, "ODEPACK, A Systematized Collection of ODE
                   Solvers," IMACS Transactions on Scientific Computation, Vol 1.,
                   pp. 55-64, 1983.
            .. [8] L. Petzold, "Automatic selection of methods for solving stiff and
                   nonstiff systems of ordinary differential equations", SIAM Journal
                   on Scientific and Statistical Computing, Vol. 4, No. 1, pp. 136-148,
                   1983.
            .. [9] `Stiff equation <https://en.wikipedia.org/wiki/Stiff_equation>`_ on
                   Wikipedia.
            .. [10] A. Curtis, M. J. D. Powell, and J. Reid, "On the estimation of
                    sparse Jacobian matrices", Journal of the Institute of Mathematics
                    and its Applications, 13, pp. 117-120, 1974.
            .. [11] `Cauchy-Riemann equations
                     <https://en.wikipedia.org/wiki/Cauchy-Riemann_equations>`_ on
                     Wikipedia.
            .. [12] `Lotka-Volterra equations
                    <https://en.wikipedia.org/wiki/Lotka%E2%80%93Volterra_equations>`_
                    on Wikipedia.
            .. [13] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
                    Equations I: Nonstiff Problems", Sec. II.
            .. [14] `Page with original Fortran code of DOP853
                    <http://www.unige.ch/~hairer/software.html>`_.
    
            Examples
            --------
            Basic exponential decay showing automatically chosen time points.
    
            >>> import numpy as np
            >>> from scipy.integrate import solve_ivp
            >>> def exponential_decay(t, y): return -0.5 * y
            >>> sol = solve_ivp(exponential_decay, [0, 10], [2, 4, 8])
            >>> print(sol.t)
            [ 0.          0.11487653  1.26364188  3.06061781  4.81611105  6.57445806
              8.33328988 10.        ]
            >>> print(sol.y)
            [[2.         1.88836035 1.06327177 0.43319312 0.18017253 0.07483045
              0.03107158 0.01350781]
             [4.         3.7767207  2.12654355 0.86638624 0.36034507 0.14966091
              0.06214316 0.02701561]
             [8.         7.5534414  4.25308709 1.73277247 0.72069014 0.29932181
              0.12428631 0.05403123]]
    
            Specifying points where the solution is desired.
    
            >>> sol = solve_ivp(exponential_decay, [0, 10], [2, 4, 8],
            ...                 t_eval=[0, 1, 2, 4, 10])
            >>> print(sol.t)
            [ 0  1  2  4 10]
            >>> print(sol.y)
            [[2.         1.21305369 0.73534021 0.27066736 0.01350938]
             [4.         2.42610739 1.47068043 0.54133472 0.02701876]
             [8.         4.85221478 2.94136085 1.08266944 0.05403753]]
    
            Cannon fired upward with terminal event upon impact. The ``terminal`` and
            ``direction`` fields of an event are applied by monkey patching a function.
            Here ``y[0]`` is position and ``y[1]`` is velocity. The projectile starts
            at position 0 with velocity +10. Note that the integration never reaches
            t=100 because the event is terminal.
    
            >>> def upward_cannon(t, y): return [y[1], -0.5]
            >>> def hit_ground(t, y): return y[0]
            >>> hit_ground.terminal = True
            >>> hit_ground.direction = -1
            >>> sol = solve_ivp(upward_cannon, [0, 100], [0, 10], events=hit_ground)
            >>> print(sol.t_events)
            [array([40.])]
            >>> print(sol.t)
            [0.00000000e+00 9.99900010e-05 1.09989001e-03 1.10988901e-02
             1.11088891e-01 1.11098890e+00 1.11099890e+01 4.00000000e+01]
    
            Use `dense_output` and `events` to find position, which is 100, at the apex
            of the cannonball's trajectory. Apex is not defined as terminal, so both
            apex and hit_ground are found. There is no information at t=20, so the sol
            attribute is used to evaluate the solution. The sol attribute is returned
            by setting ``dense_output=True``. Alternatively, the `y_events` attribute
            can be used to access the solution at the time of the event.
    
            >>> def apex(t, y): return y[1]
            >>> sol = solve_ivp(upward_cannon, [0, 100], [0, 10],
            ...                 events=(hit_ground, apex), dense_output=True)
            >>> print(sol.t_events)
            [array([40.]), array([20.])]
            >>> print(sol.t)
            [0.00000000e+00 9.99900010e-05 1.09989001e-03 1.10988901e-02
             1.11088891e-01 1.11098890e+00 1.11099890e+01 4.00000000e+01]
            >>> print(sol.sol(sol.t_events[1][0]))
            [100.   0.]
            >>> print(sol.y_events)
            [array([[-5.68434189e-14, -1.00000000e+01]]),
             array([[1.00000000e+02, 1.77635684e-15]])]
    
            As an example of a system with additional parameters, we'll implement
            the Lotka-Volterra equations [12]_.
    
            >>> def lotkavolterra(t, z, a, b, c, d):
            ...     x, y = z
            ...     return [a*x - b*x*y, -c*y + d*x*y]
            ...
    
            We pass in the parameter values a=1.5, b=1, c=3 and d=1 with the `args`
            argument.
    
            >>> sol = solve_ivp(lotkavolterra, [0, 15], [10, 5], args=(1.5, 1, 3, 1),
            ...                 dense_output=True)
    
            Compute a dense solution and plot it.
    
            >>> t = np.linspace(0, 15, 300)
            >>> z = sol.sol(t)
            >>> import matplotlib.pyplot as plt
            >>> plt.plot(t, z.T)
            >>> plt.xlabel('t')
            >>> plt.legend(['x', 'y'], shadow=True)
            >>> plt.title('Lotka-Volterra System')
            >>> plt.show()
    
            A couple examples of using solve_ivp to solve the differential
            equation ``y' = Ay`` with complex matrix ``A``.
    
            >>> A = np.array([[-0.25 + 0.14j, 0, 0.33 + 0.44j],
            ...               [0.25 + 0.58j, -0.2 + 0.14j, 0],
            ...               [0, 0.2 + 0.4j, -0.1 + 0.97j]])
    
            Solving an IVP with ``A`` from above and ``y`` as 3x1 vector:
    
            >>> def deriv_vec(t, y):
            ...     return A @ y
            >>> result = solve_ivp(deriv_vec, [0, 25],
            ...                    np.array([10 + 0j, 20 + 0j, 30 + 0j]),
            ...                    t_eval=np.linspace(0, 25, 101))
            >>> print(result.y[:, 0])
            [10.+0.j 20.+0.j 30.+0.j]
            >>> print(result.y[:, -1])
            [18.46291039+45.25653651j 10.01569306+36.23293216j
             -4.98662741+80.07360388j]
    
            Solving an IVP with ``A`` from above with ``y`` as 3x3 matrix :
    
            >>> def deriv_mat(t, y):
            ...     return (A @ y.reshape(3, 3)).flatten()
            >>> y0 = np.array([[2 + 0j, 3 + 0j, 4 + 0j],
            ...                [5 + 0j, 6 + 0j, 7 + 0j],
            ...                [9 + 0j, 34 + 0j, 78 + 0j]])
    
            >>> result = solve_ivp(deriv_mat, [0, 25], y0.flatten(),
            ...                    t_eval=np.linspace(0, 25, 101))
            >>> print(result.y[:, 0].reshape(3, 3))
            [[ 2.+0.j  3.+0.j  4.+0.j]
             [ 5.+0.j  6.+0.j  7.+0.j]
             [ 9.+0.j 34.+0.j 78.+0.j]]
            >>> print(result.y[:, -1].reshape(3, 3))
            [[  5.67451179 +12.07938445j  17.2888073  +31.03278837j
                37.83405768 +63.25138759j]
             [  3.39949503 +11.82123994j  21.32530996 +44.88668871j
                53.17531184+103.80400411j]
             [ -2.26105874 +22.19277664j -15.1255713  +70.19616341j
               -38.34616845+153.29039931j]]
    
        tplquad(func, a, b, gfun, hfun, qfun, rfun, args=(), epsabs=1.49e-08, epsrel=1.49e-08)
            Compute a triple (definite) integral.
    
            Return the triple integral of ``func(z, y, x)`` from ``x = a..b``,
            ``y = gfun(x)..hfun(x)``, and ``z = qfun(x,y)..rfun(x,y)``.
    
            Parameters
            ----------
            func : function
                A Python function or method of at least three variables in the
                order (z, y, x).
            a, b : float
                The limits of integration in x: `a` < `b`
            gfun : function or float
                The lower boundary curve in y which is a function taking a single
                floating point argument (x) and returning a floating point result
                or a float indicating a constant boundary curve.
            hfun : function or float
                The upper boundary curve in y (same requirements as `gfun`).
            qfun : function or float
                The lower boundary surface in z.  It must be a function that takes
                two floats in the order (x, y) and returns a float or a float
                indicating a constant boundary surface.
            rfun : function or float
                The upper boundary surface in z. (Same requirements as `qfun`.)
            args : tuple, optional
                Extra arguments to pass to `func`.
            epsabs : float, optional
                Absolute tolerance passed directly to the innermost 1-D quadrature
                integration. Default is 1.49e-8.
            epsrel : float, optional
                Relative tolerance of the innermost 1-D integrals. Default is 1.49e-8.
    
            Returns
            -------
            y : float
                The resultant integral.
            abserr : float
                An estimate of the error.
    
            See Also
            --------
            quad : Adaptive quadrature using QUADPACK
            fixed_quad : Fixed-order Gaussian quadrature
            dblquad : Double integrals
            nquad : N-dimensional integrals
            romb : Integrators for sampled data
            simpson : Integrators for sampled data
            scipy.special : For coefficients and roots of orthogonal polynomials
    
            Notes
            -----
            For valid results, the integral must converge; behavior for divergent
            integrals is not guaranteed.
    
            **Details of QUADPACK level routines**
    
            `quad` calls routines from the FORTRAN library QUADPACK. This section
            provides details on the conditions for each routine to be called and a
            short description of each routine. For each level of integration, ``qagse``
            is used for finite limits or ``qagie`` is used, if either limit (or both!)
            are infinite. The following provides a short description from [1]_ for each
            routine.
    
            qagse
                is an integrator based on globally adaptive interval
                subdivision in connection with extrapolation, which will
                eliminate the effects of integrand singularities of
                several types.
            qagie
                handles integration over infinite intervals. The infinite range is
                mapped onto a finite interval and subsequently the same strategy as
                in ``QAGS`` is applied.
    
            References
            ----------
    
            .. [1] Piessens, Robert; de Doncker-Kapenga, Elise;
                   Überhuber, Christoph W.; Kahaner, David (1983).
                   QUADPACK: A subroutine package for automatic integration.
                   Springer-Verlag.
                   ISBN 978-3-540-12553-2.
    
            Examples
            --------
            Compute the triple integral of ``x * y * z``, over ``x`` ranging
            from 1 to 2, ``y`` ranging from 2 to 3, ``z`` ranging from 0 to 1.
            That is, :math:`\int^{x=2}_{x=1} \int^{y=3}_{y=2} \int^{z=1}_{z=0} x y z
            \,dz \,dy \,dx`.
    
            >>> import numpy as np
            >>> from scipy import integrate
            >>> f = lambda z, y, x: x*y*z
            >>> integrate.tplquad(f, 1, 2, 2, 3, 0, 1)
            (1.8749999999999998, 3.3246447942574074e-14)
    
            Calculate :math:`\int^{x=1}_{x=0} \int^{y=1-2x}_{y=0}
            \int^{z=1-x-2y}_{z=0} x y z \,dz \,dy \,dx`.
            Note: `qfun`/`rfun` takes arguments in the order (x, y), even though ``f``
            takes arguments in the order (z, y, x).
    
            >>> f = lambda z, y, x: x*y*z
            >>> integrate.tplquad(f, 0, 1, 0, lambda x: 1-2*x, 0, lambda x, y: 1-x-2*y)
            (0.05416666666666668, 2.1774196738157757e-14)
    
            Calculate :math:`\int^{x=1}_{x=0} \int^{y=1}_{y=0} \int^{z=1}_{z=0}
            a x y z \,dz \,dy \,dx` for :math:`a=1, 3`.
    
            >>> f = lambda z, y, x, a: a*x*y*z
            >>> integrate.tplquad(f, 0, 1, 0, 1, 0, 1, args=(1,))
                (0.125, 5.527033708952211e-15)
            >>> integrate.tplquad(f, 0, 1, 0, 1, 0, 1, args=(3,))
                (0.375, 1.6581101126856635e-14)
    
            Compute the three-dimensional Gaussian Integral, which is the integral of
            the Gaussian function :math:`f(x,y,z) = e^{-(x^{2} + y^{2} + z^{2})}`, over
            :math:`(-\infty,+\infty)`. That is, compute the integral
            :math:`\iiint^{+\infty}_{-\infty} e^{-(x^{2} + y^{2} + z^{2})} \,dz
            \,dy\,dx`.
    
            >>> f = lambda x, y, z: np.exp(-(x ** 2 + y ** 2 + z ** 2))
            >>> integrate.tplquad(f, -np.inf, np.inf, -np.inf, np.inf, -np.inf, np.inf)
                (5.568327996830833, 4.4619078828029765e-08)
    
        trapezoid(y, x=None, dx=1.0, axis=-1)
            Integrate along the given axis using the composite trapezoidal rule.
    
            If `x` is provided, the integration happens in sequence along its
            elements - they are not sorted.
    
            Integrate `y` (`x`) along each 1d slice on the given axis, compute
            :math:`\int y(x) dx`.
            When `x` is specified, this integrates along the parametric curve,
            computing :math:`\int_t y(t) dt =
            \int_t y(t) \left.\frac{dx}{dt}\right|_{x=x(t)} dt`.
    
            Parameters
            ----------
            y : array_like
                Input array to integrate.
            x : array_like, optional
                The sample points corresponding to the `y` values. If `x` is None,
                the sample points are assumed to be evenly spaced `dx` apart. The
                default is None.
            dx : scalar, optional
                The spacing between sample points when `x` is None. The default is 1.
            axis : int, optional
                The axis along which to integrate.
    
            Returns
            -------
            trapezoid : float or ndarray
                Definite integral of `y` = n-dimensional array as approximated along
                a single axis by the trapezoidal rule. If `y` is a 1-dimensional array,
                then the result is a float. If `n` is greater than 1, then the result
                is an `n`-1 dimensional array.
    
            See Also
            --------
            cumulative_trapezoid, simpson, romb
    
            Notes
            -----
            Image [2]_ illustrates trapezoidal rule -- y-axis locations of points
            will be taken from `y` array, by default x-axis distances between
            points will be 1.0, alternatively they can be provided with `x` array
            or with `dx` scalar.  Return value will be equal to combined area under
            the red lines.
    
            References
            ----------
            .. [1] Wikipedia page: https://en.wikipedia.org/wiki/Trapezoidal_rule
    
            .. [2] Illustration image:
                   https://en.wikipedia.org/wiki/File:Composite_trapezoidal_rule_illustration.png
    
            Examples
            --------
            Use the trapezoidal rule on evenly spaced points:
    
            >>> import numpy as np
            >>> from scipy import integrate
            >>> integrate.trapezoid([1, 2, 3])
            4.0
    
            The spacing between sample points can be selected by either the
            ``x`` or ``dx`` arguments:
    
            >>> integrate.trapezoid([1, 2, 3], x=[4, 6, 8])
            8.0
            >>> integrate.trapezoid([1, 2, 3], dx=2)
            8.0
    
            Using a decreasing ``x`` corresponds to integrating in reverse:
    
            >>> integrate.trapezoid([1, 2, 3], x=[8, 6, 4])
            -8.0
    
            More generally ``x`` is used to integrate along a parametric curve. We can
            estimate the integral :math:`\int_0^1 x^2 = 1/3` using:
    
            >>> x = np.linspace(0, 1, num=50)
            >>> y = x**2
            >>> integrate.trapezoid(y, x)
            0.33340274885464394
    
            Or estimate the area of a circle, noting we repeat the sample which closes
            the curve:
    
            >>> theta = np.linspace(0, 2 * np.pi, num=1000, endpoint=True)
            >>> integrate.trapezoid(np.cos(theta), x=np.sin(theta))
            3.141571941375841
    
            ``trapezoid`` can be applied along a specified axis to do multiple
            computations in one call:
    
            >>> a = np.arange(6).reshape(2, 3)
            >>> a
            array([[0, 1, 2],
                   [3, 4, 5]])
            >>> integrate.trapezoid(a, axis=0)
            array([1.5, 2.5, 3.5])
            >>> integrate.trapezoid(a, axis=1)
            array([2.,  8.])
    
    DATA
        __all__ = ['AccuracyWarning', 'BDF', 'DOP853', 'DenseOutput', 'Integra...
    
    FILE
        /opt/anaconda3/envs/CompPhys/lib/python3.12/site-packages/scipy/integrate/__init__.py
    
    


The `integrate.quad(func, a, b)` integrates the function in the interval of [a,b].
And returns the values and the absolute error.
```py
    y : float
        The integral of func from `a` to `b`.
    abserr : float
        An estimate of the absolute error in the result.
```


```python
help(integrate.quad)
```

    Help on function quad in module scipy.integrate._quadpack_py:
    
    quad(func, a, b, args=(), full_output=0, epsabs=1.49e-08, epsrel=1.49e-08, limit=50, points=None, weight=None, wvar=None, wopts=None, maxp1=50, limlst=50, complex_func=False)
        Compute a definite integral.
    
        Integrate func from `a` to `b` (possibly infinite interval) using a
        technique from the Fortran library QUADPACK.
    
        Parameters
        ----------
        func : {function, scipy.LowLevelCallable}
            A Python function or method to integrate. If `func` takes many
            arguments, it is integrated along the axis corresponding to the
            first argument.
    
            If the user desires improved integration performance, then `f` may
            be a `scipy.LowLevelCallable` with one of the signatures::
    
                double func(double x)
                double func(double x, void *user_data)
                double func(int n, double *xx)
                double func(int n, double *xx, void *user_data)
    
            The ``user_data`` is the data contained in the `scipy.LowLevelCallable`.
            In the call forms with ``xx``,  ``n`` is the length of the ``xx``
            array which contains ``xx[0] == x`` and the rest of the items are
            numbers contained in the ``args`` argument of quad.
    
            In addition, certain ctypes call signatures are supported for
            backward compatibility, but those should not be used in new code.
        a : float
            Lower limit of integration (use -numpy.inf for -infinity).
        b : float
            Upper limit of integration (use numpy.inf for +infinity).
        args : tuple, optional
            Extra arguments to pass to `func`.
        full_output : int, optional
            Non-zero to return a dictionary of integration information.
            If non-zero, warning messages are also suppressed and the
            message is appended to the output tuple.
        complex_func : bool, optional
            Indicate if the function's (`func`) return type is real
            (``complex_func=False``: default) or complex (``complex_func=True``).
            In both cases, the function's argument is real.
            If full_output is also non-zero, the `infodict`, `message`, and
            `explain` for the real and complex components are returned in
            a dictionary with keys "real output" and "imag output".
    
        Returns
        -------
        y : float
            The integral of func from `a` to `b`.
        abserr : float
            An estimate of the absolute error in the result.
        infodict : dict
            A dictionary containing additional information.
        message
            A convergence message.
        explain
            Appended only with 'cos' or 'sin' weighting and infinite
            integration limits, it contains an explanation of the codes in
            infodict['ierlst']
    
        Other Parameters
        ----------------
        epsabs : float or int, optional
            Absolute error tolerance. Default is 1.49e-8. `quad` tries to obtain
            an accuracy of ``abs(i-result) <= max(epsabs, epsrel*abs(i))``
            where ``i`` = integral of `func` from `a` to `b`, and ``result`` is the
            numerical approximation. See `epsrel` below.
        epsrel : float or int, optional
            Relative error tolerance. Default is 1.49e-8.
            If ``epsabs <= 0``, `epsrel` must be greater than both 5e-29
            and ``50 * (machine epsilon)``. See `epsabs` above.
        limit : float or int, optional
            An upper bound on the number of subintervals used in the adaptive
            algorithm.
        points : (sequence of floats,ints), optional
            A sequence of break points in the bounded integration interval
            where local difficulties of the integrand may occur (e.g.,
            singularities, discontinuities). The sequence does not have
            to be sorted. Note that this option cannot be used in conjunction
            with ``weight``.
        weight : float or int, optional
            String indicating weighting function. Full explanation for this
            and the remaining arguments can be found below.
        wvar : optional
            Variables for use with weighting functions.
        wopts : optional
            Optional input for reusing Chebyshev moments.
        maxp1 : float or int, optional
            An upper bound on the number of Chebyshev moments.
        limlst : int, optional
            Upper bound on the number of cycles (>=3) for use with a sinusoidal
            weighting and an infinite end-point.
    
        See Also
        --------
        dblquad : double integral
        tplquad : triple integral
        nquad : n-dimensional integrals (uses `quad` recursively)
        fixed_quad : fixed-order Gaussian quadrature
        simpson : integrator for sampled data
        romb : integrator for sampled data
        scipy.special : for coefficients and roots of orthogonal polynomials
    
        Notes
        -----
        For valid results, the integral must converge; behavior for divergent
        integrals is not guaranteed.
    
        **Extra information for quad() inputs and outputs**
    
        If full_output is non-zero, then the third output argument
        (infodict) is a dictionary with entries as tabulated below. For
        infinite limits, the range is transformed to (0,1) and the
        optional outputs are given with respect to this transformed range.
        Let M be the input argument limit and let K be infodict['last'].
        The entries are:
    
        'neval'
            The number of function evaluations.
        'last'
            The number, K, of subintervals produced in the subdivision process.
        'alist'
            A rank-1 array of length M, the first K elements of which are the
            left end points of the subintervals in the partition of the
            integration range.
        'blist'
            A rank-1 array of length M, the first K elements of which are the
            right end points of the subintervals.
        'rlist'
            A rank-1 array of length M, the first K elements of which are the
            integral approximations on the subintervals.
        'elist'
            A rank-1 array of length M, the first K elements of which are the
            moduli of the absolute error estimates on the subintervals.
        'iord'
            A rank-1 integer array of length M, the first L elements of
            which are pointers to the error estimates over the subintervals
            with ``L=K`` if ``K<=M/2+2`` or ``L=M+1-K`` otherwise. Let I be the
            sequence ``infodict['iord']`` and let E be the sequence
            ``infodict['elist']``.  Then ``E[I[1]], ..., E[I[L]]`` forms a
            decreasing sequence.
    
        If the input argument points is provided (i.e., it is not None),
        the following additional outputs are placed in the output
        dictionary. Assume the points sequence is of length P.
    
        'pts'
            A rank-1 array of length P+2 containing the integration limits
            and the break points of the intervals in ascending order.
            This is an array giving the subintervals over which integration
            will occur.
        'level'
            A rank-1 integer array of length M (=limit), containing the
            subdivision levels of the subintervals, i.e., if (aa,bb) is a
            subinterval of ``(pts[1], pts[2])`` where ``pts[0]`` and ``pts[2]``
            are adjacent elements of ``infodict['pts']``, then (aa,bb) has level l
            if ``|bb-aa| = |pts[2]-pts[1]| * 2**(-l)``.
        'ndin'
            A rank-1 integer array of length P+2. After the first integration
            over the intervals (pts[1], pts[2]), the error estimates over some
            of the intervals may have been increased artificially in order to
            put their subdivision forward. This array has ones in slots
            corresponding to the subintervals for which this happens.
    
        **Weighting the integrand**
    
        The input variables, *weight* and *wvar*, are used to weight the
        integrand by a select list of functions. Different integration
        methods are used to compute the integral with these weighting
        functions, and these do not support specifying break points. The
        possible values of weight and the corresponding weighting functions are.
    
        ==========  ===================================   =====================
        ``weight``  Weight function used                  ``wvar``
        ==========  ===================================   =====================
        'cos'       cos(w*x)                              wvar = w
        'sin'       sin(w*x)                              wvar = w
        'alg'       g(x) = ((x-a)**alpha)*((b-x)**beta)   wvar = (alpha, beta)
        'alg-loga'  g(x)*log(x-a)                         wvar = (alpha, beta)
        'alg-logb'  g(x)*log(b-x)                         wvar = (alpha, beta)
        'alg-log'   g(x)*log(x-a)*log(b-x)                wvar = (alpha, beta)
        'cauchy'    1/(x-c)                               wvar = c
        ==========  ===================================   =====================
    
        wvar holds the parameter w, (alpha, beta), or c depending on the weight
        selected. In these expressions, a and b are the integration limits.
    
        For the 'cos' and 'sin' weighting, additional inputs and outputs are
        available.
    
        For finite integration limits, the integration is performed using a
        Clenshaw-Curtis method which uses Chebyshev moments. For repeated
        calculations, these moments are saved in the output dictionary:
    
        'momcom'
            The maximum level of Chebyshev moments that have been computed,
            i.e., if ``M_c`` is ``infodict['momcom']`` then the moments have been
            computed for intervals of length ``|b-a| * 2**(-l)``,
            ``l=0,1,...,M_c``.
        'nnlog'
            A rank-1 integer array of length M(=limit), containing the
            subdivision levels of the subintervals, i.e., an element of this
            array is equal to l if the corresponding subinterval is
            ``|b-a|* 2**(-l)``.
        'chebmo'
            A rank-2 array of shape (25, maxp1) containing the computed
            Chebyshev moments. These can be passed on to an integration
            over the same interval by passing this array as the second
            element of the sequence wopts and passing infodict['momcom'] as
            the first element.
    
        If one of the integration limits is infinite, then a Fourier integral is
        computed (assuming w neq 0). If full_output is 1 and a numerical error
        is encountered, besides the error message attached to the output tuple,
        a dictionary is also appended to the output tuple which translates the
        error codes in the array ``info['ierlst']`` to English messages. The
        output information dictionary contains the following entries instead of
        'last', 'alist', 'blist', 'rlist', and 'elist':
    
        'lst'
            The number of subintervals needed for the integration (call it ``K_f``).
        'rslst'
            A rank-1 array of length M_f=limlst, whose first ``K_f`` elements
            contain the integral contribution over the interval
            ``(a+(k-1)c, a+kc)`` where ``c = (2*floor(|w|) + 1) * pi / |w|``
            and ``k=1,2,...,K_f``.
        'erlst'
            A rank-1 array of length ``M_f`` containing the error estimate
            corresponding to the interval in the same position in
            ``infodict['rslist']``.
        'ierlst'
            A rank-1 integer array of length ``M_f`` containing an error flag
            corresponding to the interval in the same position in
            ``infodict['rslist']``.  See the explanation dictionary (last entry
            in the output tuple) for the meaning of the codes.
    
    
        **Details of QUADPACK level routines**
    
        `quad` calls routines from the FORTRAN library QUADPACK. This section
        provides details on the conditions for each routine to be called and a
        short description of each routine. The routine called depends on
        `weight`, `points` and the integration limits `a` and `b`.
    
        ================  ==============  ==========  =====================
        QUADPACK routine  `weight`        `points`    infinite bounds
        ================  ==============  ==========  =====================
        qagse             None            No          No
        qagie             None            No          Yes
        qagpe             None            Yes         No
        qawoe             'sin', 'cos'    No          No
        qawfe             'sin', 'cos'    No          either `a` or `b`
        qawse             'alg*'          No          No
        qawce             'cauchy'        No          No
        ================  ==============  ==========  =====================
    
        The following provides a short description from [1]_ for each
        routine.
    
        qagse
            is an integrator based on globally adaptive interval
            subdivision in connection with extrapolation, which will
            eliminate the effects of integrand singularities of
            several types.
        qagie
            handles integration over infinite intervals. The infinite range is
            mapped onto a finite interval and subsequently the same strategy as
            in ``QAGS`` is applied.
        qagpe
            serves the same purposes as QAGS, but also allows the
            user to provide explicit information about the location
            and type of trouble-spots i.e. the abscissae of internal
            singularities, discontinuities and other difficulties of
            the integrand function.
        qawoe
            is an integrator for the evaluation of
            :math:`\int^b_a \cos(\omega x)f(x)dx` or
            :math:`\int^b_a \sin(\omega x)f(x)dx`
            over a finite interval [a,b], where :math:`\omega` and :math:`f`
            are specified by the user. The rule evaluation component is based
            on the modified Clenshaw-Curtis technique
    
            An adaptive subdivision scheme is used in connection
            with an extrapolation procedure, which is a modification
            of that in ``QAGS`` and allows the algorithm to deal with
            singularities in :math:`f(x)`.
        qawfe
            calculates the Fourier transform
            :math:`\int^\infty_a \cos(\omega x)f(x)dx` or
            :math:`\int^\infty_a \sin(\omega x)f(x)dx`
            for user-provided :math:`\omega` and :math:`f`. The procedure of
            ``QAWO`` is applied on successive finite intervals, and convergence
            acceleration by means of the :math:`\varepsilon`-algorithm is applied
            to the series of integral approximations.
        qawse
            approximate :math:`\int^b_a w(x)f(x)dx`, with :math:`a < b` where
            :math:`w(x) = (x-a)^{\alpha}(b-x)^{\beta}v(x)` with
            :math:`\alpha,\beta > -1`, where :math:`v(x)` may be one of the
            following functions: :math:`1`, :math:`\log(x-a)`, :math:`\log(b-x)`,
            :math:`\log(x-a)\log(b-x)`.
    
            The user specifies :math:`\alpha`, :math:`\beta` and the type of the
            function :math:`v`. A globally adaptive subdivision strategy is
            applied, with modified Clenshaw-Curtis integration on those
            subintervals which contain `a` or `b`.
        qawce
            compute :math:`\int^b_a f(x) / (x-c)dx` where the integral must be
            interpreted as a Cauchy principal value integral, for user specified
            :math:`c` and :math:`f`. The strategy is globally adaptive. Modified
            Clenshaw-Curtis integration is used on those intervals containing the
            point :math:`x = c`.
    
        **Integration of Complex Function of a Real Variable**
    
        A complex valued function, :math:`f`, of a real variable can be written as
        :math:`f = g + ih`.  Similarly, the integral of :math:`f` can be
        written as
    
        .. math::
            \int_a^b f(x) dx = \int_a^b g(x) dx + i\int_a^b h(x) dx
    
        assuming that the integrals of :math:`g` and :math:`h` exist
        over the interval :math:`[a,b]` [2]_. Therefore, ``quad`` integrates
        complex-valued functions by integrating the real and imaginary components
        separately.
    
    
        References
        ----------
    
        .. [1] Piessens, Robert; de Doncker-Kapenga, Elise;
               Überhuber, Christoph W.; Kahaner, David (1983).
               QUADPACK: A subroutine package for automatic integration.
               Springer-Verlag.
               ISBN 978-3-540-12553-2.
    
        .. [2] McCullough, Thomas; Phillips, Keith (1973).
               Foundations of Analysis in the Complex Plane.
               Holt Rinehart Winston.
               ISBN 0-03-086370-8
    
        Examples
        --------
        Calculate :math:`\int^4_0 x^2 dx` and compare with an analytic result
    
        >>> from scipy import integrate
        >>> import numpy as np
        >>> x2 = lambda x: x**2
        >>> integrate.quad(x2, 0, 4)
        (21.333333333333332, 2.3684757858670003e-13)
        >>> print(4**3 / 3.)  # analytical result
        21.3333333333
    
        Calculate :math:`\int^\infty_0 e^{-x} dx`
    
        >>> invexp = lambda x: np.exp(-x)
        >>> integrate.quad(invexp, 0, np.inf)
        (1.0, 5.842605999138044e-11)
    
        Calculate :math:`\int^1_0 a x \,dx` for :math:`a = 1, 3`
    
        >>> f = lambda x, a: a*x
        >>> y, err = integrate.quad(f, 0, 1, args=(1,))
        >>> y
        0.5
        >>> y, err = integrate.quad(f, 0, 1, args=(3,))
        >>> y
        1.5
    
        Calculate :math:`\int^1_0 x^2 + y^2 dx` with ctypes, holding
        y parameter as 1::
    
            testlib.c =>
                double func(int n, double args[n]){
                    return args[0]*args[0] + args[1]*args[1];}
            compile to library testlib.*
    
        ::
    
           from scipy import integrate
           import ctypes
           lib = ctypes.CDLL('/home/.../testlib.*') #use absolute path
           lib.func.restype = ctypes.c_double
           lib.func.argtypes = (ctypes.c_int,ctypes.c_double)
           integrate.quad(lib.func,0,1,(1))
           #(1.3333333333333333, 1.4802973661668752e-14)
           print((1.0**3/3.0 + 1.0) - (0.0**3/3.0 + 0.0)) #Analytic result
           # 1.3333333333333333
    
        Be aware that pulse shapes and other sharp features as compared to the
        size of the integration interval may not be integrated correctly using
        this method. A simplified example of this limitation is integrating a
        y-axis reflected step function with many zero values within the integrals
        bounds.
    
        >>> y = lambda x: 1 if x<=0 else 0
        >>> integrate.quad(y, -1, 1)
        (1.0, 1.1102230246251565e-14)
        >>> integrate.quad(y, -1, 100)
        (1.0000000002199108, 1.0189464580163188e-08)
        >>> integrate.quad(y, -1, 10000)
        (0.0, 0.0)
    



```python
integrate.quad(lambda x: sin(x)/x, 0, 1)
```




    (0.9460830703671831, 1.0503632079297089e-14)




```python
def square(x):
    return x**2

integrate.quad(square, 0, 1)
```




    (0.33333333333333337, 3.700743415417189e-15)



# Classes
A `class` can have attributes (variables) and methods (functions).  
* Each class method should have an argument, `self`, as its first argument that act as a self-reference.
* Some class method names have special meaning,
    * `__init__`: Constructors
    * `__str__` : A string representation of the class to be printed
    * `__repr__`: Representation of the class
    * `__call__`: Functor, called when the instance is "called as a function
* For each class member and method that starts with `__` is private, if not, it is public.




```python
class Point:
    """
    Simple class for representing a point in a Cartesian coordinate system.
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y
        print('You inputted x and y as', x, y)
    
    def translate(self, dx, dy):
        """
        Translate the point by dx and dy in the x and y direction.
        """
        self.x += dx
        self.y += dy
        self.t = 1.0

    def __str__(self):
        return ("Point at [%.2f, %.2f]" % (self.x, self.y))
    
    def __call__(self, z):
        return self.x, self.y, z
```


```python
q = Point(2., 3.0)
q.translate(1, 1)
print(q.x, q.y, q.t)
print(q)
q(4.5) # call the object as a function 
```

    You inputted x and y as 2.0 3.0
    3.0 4.0 1.0
    Point at [3.00, 4.00]





    (3.0, 4.0, 4.5)




```python
p = Point(1., 2.)   # You inputted x and y as 1.0 2.0
print(p)            # Point at [1.00, 2.00]
p.translate(1,1)    
print(p)            # Point at [2.00, 3.00]
print(p(1.0))       # (2.0, 3.0, 1.0)
print(p.x, p.y)     # 2.0 3.0
```

    You inputted x and y as 1.0 2.0
    Point at [1.00, 2.00]
    Point at [2.00, 3.00]
    (2.0, 3.0, 1.0)
    2.0 3.0


Some other special methods includes `__add__`, `__len__`, etc.

# Modules
Modules are used to packages program code and data for reuse at the highest-level of a program,  it provides self-contained namespaces that minimize name conflicts. 

Modularity in Python is:
1. variables, functions
2. classes, which combine variables and functions
3. modules, which combine classes, variables, and functions.

Module is a python file(`*.py`) or a module created by compiler (`*.so`).


`__name__` is a special variable, its value is set to `__main__` if the program is exectured directly as in the example below. If the module is imported instead, the `__name__` will be set into whatever the actual name of the module is, such as mymodule.


```python
%%file mymodule.py
# mymodule.py
"""
Example of a python module. Contains a variable called my_variable,a function called my_function, and a class called MyClass
"""
my_variable = 0

def my_function():
    """
    Example function
    """
    return my_variable

class MyClass:
    """
    Example class.
    """
    def __init__(self):
        self.variable = my_variable

    def set_variable(self, new_value):
        """
        Set self.variable to a new value
        """
        self.variable = new_value
    
    def get_variable(self):
        return self.variable

print(__name__)

if  __name__ == '__main__':
    # Code here only runs when the script is executed directly.
    m = MyClass()
    m.set_variable(1.0)
    print(m.get_variable())
```

    Overwriting mymodule.py


* `!`: prefix allows you to run shell commands directly from a Jupyter Notebook.
* `cat` is a Unix/Linux shell command that displays the contents of a file. It stands for "concatenate and display."


```python
!cat mymodule.py
```

    # mymodule.py
    """
    Example of a python module. Contains a variable called my_variable,a function called my_function, and a class called MyClass
    """
    my_variable = 0
    
    def my_function():
        """
        Example function
        """
        return my_variable
    
    class MyClass:
        """
        Example class.
        """
        def __init__(self):
            self.variable = my_variable
    
        def set_variable(self, new_value):
            """
            Set self.variable to a new value
            """
            self.variable = new_value
        
        def get_variable(self):
            return self.variable
    
    print(__name__)
    
    if  __name__ == '__main__':
        # Code here only runs when the script is executed directly.
        m = MyClass()
        m.set_variable(1.0)
        print(m.get_variable())



```python
import mymodule as my
import scipy as sy
help(my)
```

    mymodule
    Help on module mymodule:
    
    NAME
        mymodule - Example of a python module. Contains a variable called my_variable,a function called my_function, and a class called MyClass
    
    CLASSES
        builtins.object
            MyClass
    
        class MyClass(builtins.object)
         |  Example class.
         |
         |  Methods defined here:
         |
         |  __init__(self)
         |      Initialize self.  See help(type(self)) for accurate signature.
         |
         |  get_variable(self)
         |
         |  set_variable(self, new_value)
         |      Set self.variable to a new value
         |
         |  ----------------------------------------------------------------------
         |  Data descriptors defined here:
         |
         |  __dict__
         |      dictionary for instance variables
         |
         |  __weakref__
         |      list of weak references to the object
    
    FUNCTIONS
        my_function()
            Example function
    
    DATA
        my_variable = 0
    
    FILE
        /Users/brianfooo/Documents/SCIENCE UST/2. Course Archive/PHYS/PHYS3142/Rutgers488/3 Basic Python/mymodule.py
    
    



```python
m = my.MyClass()
m.set_variable(10)
m.get_variable()
```




    10



# Exercise: List of prime numbers
Write a python code that will create a **list** of the first 1000 prime numbers.


```python
import math
import time
```

This is my first attempt.


```python
def isPrime(n):
    isprime = True
    for i in range(2,n,1):
        if (n % i == 0):
            # print('--  ', i)
            isprime = False
            break
        else:
            # print('++  ', i)
            isprime = True
    return isprime
```


```python
# Try try
for num in range(2,10,1):
    print(num, isPrime(num))
```

    2 True
    3 True
    4 False
    5 True
    6 False
    7 True
    8 False
    9 False



```python
ls = []
num = 2
ctr = 0
# print(ctr, ls)

t0 = time.time()

while ctr < 1000:
    if isPrime(num):
        ls.append(num)
        ctr += 1
        # print(ctr, ls[ctr-1])
    num += 1

t1 = time.time()

print('clock timed: ', t1-t0, 's.')
print(ctr)
print(ls)


```

    clock timed:  0.697023868560791 s.
    1000
    [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997, 1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069, 1087, 1091, 1093, 1097, 1103, 1109, 1117, 1123, 1129, 1151, 1153, 1163, 1171, 1181, 1187, 1193, 1201, 1213, 1217, 1223, 1229, 1231, 1237, 1249, 1259, 1277, 1279, 1283, 1289, 1291, 1297, 1301, 1303, 1307, 1319, 1321, 1327, 1361, 1367, 1373, 1381, 1399, 1409, 1423, 1427, 1429, 1433, 1439, 1447, 1451, 1453, 1459, 1471, 1481, 1483, 1487, 1489, 1493, 1499, 1511, 1523, 1531, 1543, 1549, 1553, 1559, 1567, 1571, 1579, 1583, 1597, 1601, 1607, 1609, 1613, 1619, 1621, 1627, 1637, 1657, 1663, 1667, 1669, 1693, 1697, 1699, 1709, 1721, 1723, 1733, 1741, 1747, 1753, 1759, 1777, 1783, 1787, 1789, 1801, 1811, 1823, 1831, 1847, 1861, 1867, 1871, 1873, 1877, 1879, 1889, 1901, 1907, 1913, 1931, 1933, 1949, 1951, 1973, 1979, 1987, 1993, 1997, 1999, 2003, 2011, 2017, 2027, 2029, 2039, 2053, 2063, 2069, 2081, 2083, 2087, 2089, 2099, 2111, 2113, 2129, 2131, 2137, 2141, 2143, 2153, 2161, 2179, 2203, 2207, 2213, 2221, 2237, 2239, 2243, 2251, 2267, 2269, 2273, 2281, 2287, 2293, 2297, 2309, 2311, 2333, 2339, 2341, 2347, 2351, 2357, 2371, 2377, 2381, 2383, 2389, 2393, 2399, 2411, 2417, 2423, 2437, 2441, 2447, 2459, 2467, 2473, 2477, 2503, 2521, 2531, 2539, 2543, 2549, 2551, 2557, 2579, 2591, 2593, 2609, 2617, 2621, 2633, 2647, 2657, 2659, 2663, 2671, 2677, 2683, 2687, 2689, 2693, 2699, 2707, 2711, 2713, 2719, 2729, 2731, 2741, 2749, 2753, 2767, 2777, 2789, 2791, 2797, 2801, 2803, 2819, 2833, 2837, 2843, 2851, 2857, 2861, 2879, 2887, 2897, 2903, 2909, 2917, 2927, 2939, 2953, 2957, 2963, 2969, 2971, 2999, 3001, 3011, 3019, 3023, 3037, 3041, 3049, 3061, 3067, 3079, 3083, 3089, 3109, 3119, 3121, 3137, 3163, 3167, 3169, 3181, 3187, 3191, 3203, 3209, 3217, 3221, 3229, 3251, 3253, 3257, 3259, 3271, 3299, 3301, 3307, 3313, 3319, 3323, 3329, 3331, 3343, 3347, 3359, 3361, 3371, 3373, 3389, 3391, 3407, 3413, 3433, 3449, 3457, 3461, 3463, 3467, 3469, 3491, 3499, 3511, 3517, 3527, 3529, 3533, 3539, 3541, 3547, 3557, 3559, 3571, 3581, 3583, 3593, 3607, 3613, 3617, 3623, 3631, 3637, 3643, 3659, 3671, 3673, 3677, 3691, 3697, 3701, 3709, 3719, 3727, 3733, 3739, 3761, 3767, 3769, 3779, 3793, 3797, 3803, 3821, 3823, 3833, 3847, 3851, 3853, 3863, 3877, 3881, 3889, 3907, 3911, 3917, 3919, 3923, 3929, 3931, 3943, 3947, 3967, 3989, 4001, 4003, 4007, 4013, 4019, 4021, 4027, 4049, 4051, 4057, 4073, 4079, 4091, 4093, 4099, 4111, 4127, 4129, 4133, 4139, 4153, 4157, 4159, 4177, 4201, 4211, 4217, 4219, 4229, 4231, 4241, 4243, 4253, 4259, 4261, 4271, 4273, 4283, 4289, 4297, 4327, 4337, 4339, 4349, 4357, 4363, 4373, 4391, 4397, 4409, 4421, 4423, 4441, 4447, 4451, 4457, 4463, 4481, 4483, 4493, 4507, 4513, 4517, 4519, 4523, 4547, 4549, 4561, 4567, 4583, 4591, 4597, 4603, 4621, 4637, 4639, 4643, 4649, 4651, 4657, 4663, 4673, 4679, 4691, 4703, 4721, 4723, 4729, 4733, 4751, 4759, 4783, 4787, 4789, 4793, 4799, 4801, 4813, 4817, 4831, 4861, 4871, 4877, 4889, 4903, 4909, 4919, 4931, 4933, 4937, 4943, 4951, 4957, 4967, 4969, 4973, 4987, 4993, 4999, 5003, 5009, 5011, 5021, 5023, 5039, 5051, 5059, 5077, 5081, 5087, 5099, 5101, 5107, 5113, 5119, 5147, 5153, 5167, 5171, 5179, 5189, 5197, 5209, 5227, 5231, 5233, 5237, 5261, 5273, 5279, 5281, 5297, 5303, 5309, 5323, 5333, 5347, 5351, 5381, 5387, 5393, 5399, 5407, 5413, 5417, 5419, 5431, 5437, 5441, 5443, 5449, 5471, 5477, 5479, 5483, 5501, 5503, 5507, 5519, 5521, 5527, 5531, 5557, 5563, 5569, 5573, 5581, 5591, 5623, 5639, 5641, 5647, 5651, 5653, 5657, 5659, 5669, 5683, 5689, 5693, 5701, 5711, 5717, 5737, 5741, 5743, 5749, 5779, 5783, 5791, 5801, 5807, 5813, 5821, 5827, 5839, 5843, 5849, 5851, 5857, 5861, 5867, 5869, 5879, 5881, 5897, 5903, 5923, 5927, 5939, 5953, 5981, 5987, 6007, 6011, 6029, 6037, 6043, 6047, 6053, 6067, 6073, 6079, 6089, 6091, 6101, 6113, 6121, 6131, 6133, 6143, 6151, 6163, 6173, 6197, 6199, 6203, 6211, 6217, 6221, 6229, 6247, 6257, 6263, 6269, 6271, 6277, 6287, 6299, 6301, 6311, 6317, 6323, 6329, 6337, 6343, 6353, 6359, 6361, 6367, 6373, 6379, 6389, 6397, 6421, 6427, 6449, 6451, 6469, 6473, 6481, 6491, 6521, 6529, 6547, 6551, 6553, 6563, 6569, 6571, 6577, 6581, 6599, 6607, 6619, 6637, 6653, 6659, 6661, 6673, 6679, 6689, 6691, 6701, 6703, 6709, 6719, 6733, 6737, 6761, 6763, 6779, 6781, 6791, 6793, 6803, 6823, 6827, 6829, 6833, 6841, 6857, 6863, 6869, 6871, 6883, 6899, 6907, 6911, 6917, 6947, 6949, 6959, 6961, 6967, 6971, 6977, 6983, 6991, 6997, 7001, 7013, 7019, 7027, 7039, 7043, 7057, 7069, 7079, 7103, 7109, 7121, 7127, 7129, 7151, 7159, 7177, 7187, 7193, 7207, 7211, 7213, 7219, 7229, 7237, 7243, 7247, 7253, 7283, 7297, 7307, 7309, 7321, 7331, 7333, 7349, 7351, 7369, 7393, 7411, 7417, 7433, 7451, 7457, 7459, 7477, 7481, 7487, 7489, 7499, 7507, 7517, 7523, 7529, 7537, 7541, 7547, 7549, 7559, 7561, 7573, 7577, 7583, 7589, 7591, 7603, 7607, 7621, 7639, 7643, 7649, 7669, 7673, 7681, 7687, 7691, 7699, 7703, 7717, 7723, 7727, 7741, 7753, 7757, 7759, 7789, 7793, 7817, 7823, 7829, 7841, 7853, 7867, 7873, 7877, 7879, 7883, 7901, 7907, 7919]


Second attempt.
* Check up to square root.
* Ignore even numbers after 2



```python
import math

def isPrime(n):
    if n < 2: # Numbers less than 2 are not prime
        return False
    if n == 2: # Two is the only even prime number
        return True
    for i in range(3,int(math.sqrt(n))+1, 2): # in steps of 2
        if (n % i == 0):
            return False
    return True
```


```python
for num in range(20):
    print(num, isPrime(num))
```

Final implementation.


```python
def isPrime(n):
    """
    Check if a number is prime.
    """
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True

def find_primes(limit):
    """
    Find the first `limit` prime numbers.
    """
    primes = []
    num = 2
    while len(primes) < limit:
        if isPrime(num):
            primes.append(num)
        num = num + 1 if num == 2 else num + 2  # Skip even numbers after 2
    return primes
```


```python
t0 = time.time()
primes = find_primes(1000)
t1 = time.time()

print('clock timed: ', t1-t0, 's.')
print(len(primes))
print(primes[999])
```

    clock timed:  0.008285999298095703 s.
    1000
    7919



```python
primes = find_primes(100000)
print(len(primes))
print(primes[99999])
```

    100000
    1299709

