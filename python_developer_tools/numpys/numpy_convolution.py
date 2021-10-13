# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:9/28/2021 11:14 AM
# @File:numpy_convolution
import numpy as np


# ----------------------------------------------------------------------

def pad_to(a, N):
    "pads to NxN on the right and bottom"
    h, w = a.shape[:2]
    return np.pad(a, ((0, N - h), (0, N - w)))


def roll2(a, *, x=0, y=0):  # keyword-only arguments
    a = np.roll(a, axis=0, shift=y)
    a = np.roll(a, axis=1, shift=x)
    return a


def convolve(f, g):
    ff = np.fft.fft2(f)
    gg = np.fft.fft2(g)
    ffgg = ff * gg
    fg = np.fft.ifft2(ffgg)
    return fg


def print_array(a):
    "prints signed single digit integers, best not give it anything else"
    a = a.copy()
    a[np.isclose(a, 0)] = 0  # suppress negative zeros

    def fmt(val):
        if val == 0:  # isclose took care of that
            return " _"
        else:
            return f"{val: 1.0f}"  # space means - or blank sign

    for row in a:
        print(" ".join(fmt(el) for el in row))

    print()


# ----------------------------------------------------------------------

np.set_printoptions(suppress=True, sign=' ')

N = 10

print("observe how the signs work")

# diagonal
# f = np.eye(N, dtype=np.float32)

f = np.zeros((N, N), dtype=np.float32)

# a cross
f[3:5, 1:7] = 1
f[2:8, 3:5] = 1

# a little dot, note what the convolution makes of it
f[8, 7] = 1

print_array(f)

# g = np.float32([[0, 1], [0, 0]]) # translate +x
# g = np.float32([[0, 0], [1, 0]]) # translate +y

# some simple filters
# g = np.float32([[-1,  0,  0]])
# g = np.float32([[-1, +1]])
g = np.float32([[-1, 0, +1]])
# try [-1, 0, 0] to see how an "identity" (negated) looks like vs the other filters
# it does appear as if each element of f scales a copy of g, then
# all those scaled copies overlay on each other and sum "through" (superposition)

# sum of scaled copies:
# image:       _  _  1  1  _  _  _ :  first and second are zero, then two ones
#             -------------------- : "instances" of [-1, 0, +1]
#              0  0  0             : * 0
#                 0  0  0          : * 0
#                   -1  0 +1       : * 1
#                      -1  0 +1    : * 1
#                          0  0  0 : * 0
#             -------------------- : summed/integrated
# sum:         0  0 -1 -1 +1 +1  0

# equivalently the "moving window" idea:
# image:       _  _  _  1  1  _  _  _
#             ----------------------- : use [-1, 0, +1] reversed, then dot product
#             +1  0 -1                : =>  0
#                +1  0 -1             : => -1
#                   +1  0 -1          : => -1
#                      +1  0 -1       : => +1
#                         +1  0 -1    : => +1
#                            +1  0 -1 : =>  0
#             -----------------------
# result             0 -1 -1 +1 +1  0


# try transposing to get the filter in y direction
# g = g.T

g = pad_to(g, N)  # pad to NxN with zeros

# g = roll2(g, x=-1) # use this to shift around

print_array(g)

fg = convolve(f, g)
print_array(np.real(fg))

print()
print("prewitt is separable")

f = np.float32([[-1, 0, +1]])  # gradient filter
g = np.float32([  # a box blur
    [1],
    [1],  # use 2 here to get sobel (closer to gaussian blur)
    [1]
])

f = pad_to(f, N)
g = pad_to(g, N)
print_array(f)
print_array(g)

fg = convolve(f, g)

print_array(np.real(fg))

print()
print("convolution of two other filters")

f = np.float32([[-1, 0, +1]])
g = np.float32([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
])

f = pad_to(f, N)
g = pad_to(g, N)
print_array(f)
print_array(g)

fg = convolve(f, g)

print_array(np.real(fg))