# This code is an implementation of the Jonker-Volgenant algorithm for the Linear Assignment Problem.
# It can be used to solve the weighted bipartite matching problem.
#
# This code began as an effort to port the matlab code available at:
# http://www.mathworks.com/matlabcentral/fileexchange/26836-lapjv-jonker-volgenant-algorithm-for-linear-assignment-problem-v3-0
#
#
# Reference:
# R. Jonker and A. Volgenant, "A shortest augmenting path algorithm for
# dense and spare linear assignment problems", Computing, Vol. 38, pp.
# 325-340, 1987.
#
# Code by Tomer Levinboim
# September 2013.
#
# Possible problems:
# This code does not support Inf/-Inf entries,
# and was not test with non-square matrices.

#cython: boundscheck=False
#cython: wraparound=False
cimport numpy as np
import numpy as np

def cy_lapjv(np.ndarray[double, ndim=2] costMat, resolution=None):
    costMat = np.mat(costMat)
    if resolution is None:
        maxcost = np.min([1e-16, np.max(costMat)])
        resolution = 1e-16

    cdef:
        int numfree
        size_t rdim, cdim, dim
        size_t i, imin
        int k
        int j1, j2, i0
        #int r,imin
        # int imin
        size_t loopcnt = 0
        int prvnumfree
        double umin, usubmin
        double diff_min


    rdim = costMat.shape[0]
    cdim = costMat.shape[1]
    cdef double M = np.min(costMat)
    if rdim > cdim:
        costMat = costMat.T
        rdim = costMat.shape[0]
        cdim = costMat.shape[1]
        swapf = True
    else:
        swapf = False

    dim = cdim
    costMat = np.vstack((costMat, 2*M*ones(cdim-rdim, cdim)))
    maxcost = np.max(costMat)*dim + 1
    # TODO: handle inf and nan

    cdef np.ndarray[double, ndim=1] v = zeros(1, dim)
    cdef np.ndarray[double, ndim=1] x, xh
    cdef np.ndarray[np.int32_t, ndim=1] rowsol = zeros(1, dim, dtype='i4') - 1
    cdef np.ndarray[np.int32_t, ndim=1] colsol = zeros(dim, 1, dtype='i4') - 1
    cdef np.ndarray[np.int32_t, ndim=1] free

    if np.std(costMat, ddof=1) < np.mean(costMat):
        numfree = -1
        free = zeros(dim, 1, dtype='i4')
        matches = zeros(dim, 1, dtype='i4')
        for j in reversed(xrange(dim)):
            v[j], imin = min(asVector(costMat[:, j]))
            if not matches[imin]:
                rowsol[imin] = j
                colsol[j] = imin
            elif v[j] < v[rowsol[imin]]:
                j1 = rowsol[imin]
                rowsol[imin] = j
                colsol[j] = imin
                colsol[j1] = -1
            else:
                colsol[j] = -1  # row already assigned, column not assigned.

            matches[imin] += 1

        # Reduction transfer from unassigned to assigned row
        for i in xrange(dim):
            if not matches[i]:      # fill list of unassigned 'free' rows.
                numfree += 1
                free[numfree] = i
            elif matches[i] == 1:   # transfer reduction from rows that are assigned once.
                j1 = rowsol[i]
                x = asVector(costMat[i, :] - v)
                x[j1] = maxcost
                v[j1] -= min(x)[0]
    else:
        numfree = dim-2
        v1, r = min(costMat)
        free = np.array(range(dim), dtype='i4')
        _, c = min(v1)
        imin = r[c]
        j = c
        rowsol[imin] = j
        colsol[j] = imin
        # matches(imin)=1
        #free[imin] = []
        free = np.delete(free, imin)
        x = asVector(costMat[imin, :]-v)
        x[j] = maxcost
        v[j] -= min(x)[0]

    # Augmenting reduction of unassigned rows

    while loopcnt < 2:
        loopcnt += 1
        # scan all free rows
        # in some cases, a free row may be replaced with another one to be scaed next
        k = -1
        prvnumfree = numfree
        numfree = -1    # start list of rows still free after augmenting row reduction.

        while k < prvnumfree:
            k += 1
            i = free[k]
            # find minimum and second minimum reduced cost over columns
            x = asVector(costMat[i, :]) - v

            umin, j1 = min(x)

            x[j1] = maxcost
            usubmin, j2 = min(x)
            i0 = colsol[j1]
            diff_min = usubmin - umin
            if diff_min > resolution:
                # change the reduction of the minmum column to increase the
                # minimum reduced cost in the row to the subminimum.
                v[j1] -= diff_min
            else:  # minimum and subminimum equal.
                if i0 >= 0:  # minimum column j1 is assigned.
                    # swap columns j1 and j2, as j2 may be unassigned.
                    j1 = j2
                    i0 = colsol[j2]

            rowsol[i] = j1
            colsol[j1] = i

            if i0 >= 0:  # minimum column j1 assigned easier
                if diff_min > resolution:
                    # put in current k, and go back to that k.
                    # continue augmenting path i - j1 with i0.
                    free[k] = i0
                    k -= 1
                else:
                    # no further augmenting reduction possible
                    # store i0 in list of free rows for next phase.
                    numfree += 1
                    free[numfree] = i0

    cdef size_t f, up, low, last
    cdef double minh, h
    cdef np.ndarray[np.int32_t, ndim=1] collist
    cdef int freerow
    cdef np.ndarray[np.int32_t, ndim=1] pred, k0

    for f in xrange(numfree+1):
        freerow = free[f]  # start row of augmenting path
        # Dijkstra shortest path algorithm.
        # runs until unassigned column added to shortest path tree.
        d = asVector(costMat[freerow, :]) - v
        pred = freerow * ones(dim, dtype='i4')
        collist = np.array(range(dim), dtype='i4')
        low = 0  # columns in 1...low-1 are ready, now none.
        up = 0  #  columns in low...up-1 are to be scaed for current minimum, now none.
        # columns in up+1...dim are to be considered later to find new minimum,
        # at this stage the list simply contains all columns.
        unassignedfound = False
        while not unassignedfound:
            if up == low:   # no more columns to be scaned for current minimum.
                last = low-1
                # scan columns for up...dim to find all indices for which new minimum occurs.
                # store these indices between low+1...up (increasing up).
                minh = d[collist[up]]
                up += 1
                for k in xrange(up, dim):
                    j = collist[k]
                    h = d[j]
                    if h <= minh:
                        if h < minh:
                            up = low
                            minh = h

                        # new index with same minimum, put on index up, and extend list.
                        collist[k] = collist[up]
                        collist[up] = j
                        up += 1

                # check if any of the minimum columns happens to be unassigned.
                # if so, we have an augmenting path right away.
                for k in xrange(low, up):
                    if colsol[collist[k]] < 0:
                        endofpath = collist[k]
                        unassignedfound = True
                        break
            if not unassignedfound:
                # update 'distances' between freerow and all unscanned columns,
                # via next scanned column.
                j1 = collist[low]
                low += 1
                i = colsol[j1] #line 215
                x = asVector(costMat[i, :])-v
                h = x[j1] - minh
                xh = x-h
                k0 = np.array(range(up, dim), dtype='i4')
                j = collist[k0]
                vf0 = xh<d
                vf = vf0[j]
                vj = j[vf]
                vk = k0[vf]
                pred[vj] = i
                v2 = xh[vj]
                d[vj] = v2
                vf = v2 == minh  # new column found at same minimum value
                q2 = vj[vf]
                k2 = vk[vf]
                cf = colsol[q2] < 0
                if np.any(cf):  # unassigned, shortest augmenting path is complete.
                    i2 = find(cf, 1)[0]
                    endofpath = q2[i2]
                    unassignedfound = True
                else:
                    i2 = len(cf)+1
                # add to list to be scaned right away
                for k in range(0, i2-1):
                    collist[k2[k]] = collist[up]
                    collist[up] = q2[k]
                    up += 1

        # update column prices
        q1 = collist[1:last+1]  # TL: q1 used to be j1
        v[q1] = v[q1] + d[q1] - minh
        # reset row and column assignments along the alternating path
        while 1:
            i = pred[endofpath]
            colsol[endofpath] = i
            q1 = endofpath
            endofpath = rowsol[i]
            rowsol[i] = q1
            if i==freerow:
                break

    rowsol = rowsol[range(rdim)]
    u = diag(costMat[:, rowsol])-v[rowsol].T
    u = u[range(rdim)]
    v = v[range(cdim)]
    cost = sum(u)+sum(v[rowsol])
    costMat = submatrix(costMat, range(rdim), range(cdim))
    costMat = costMat - u[:, ones(cdim, dtype=int)] - v[ones(rdim, dtype=int), :]
    if swapf:
        costMat = costMat.T
        t = u.T
        u = v.T
        v = t
    if cost > maxcost:
        cost = float('inf')

    return rowsol, cost, v, u, costMat


# matlab like functions

def min(A):
    if len(A.shape) == 2:
        N = A.shape[1]
        I = np.argmin(A, axis=0)
        return asVector(A[I, range(0, N)]), asVectorInt(I)
    elif len(A.shape) == 1:
        i = np.argmin(A)
        return A[i], i
    else:
        return A, 0


def ones(x, y=1, dtype=float):
    return np.ones((x, y), dtype=dtype)

def sum(v):
    return np.sum(v, axis=0)


def diag(A):
    return np.diag(A)


def find(v, k=None):
    if k is None:
        k = len(v)
    I = []
    for i in xrange(len(v)):
        if v[i] != 0:
            I.append(i)
            if len(I) == k:
                break
    return I


def zeros(unsigned int N, unsigned int D=1, dtype=float):
    if N == 1 or D == 1:
        return np.zeros(N*D, dtype=dtype)
    else:
        return np.zeros((N, D), dtype=dtype)


def submatrix(np.ndarray[double, ndim=2] A, rows, cols):  # this is quite slow for some reason, maybe since a new array is allocated.
    I = np.ix_(rows, cols)
    return A[I]

def asVectorInt(np.ndarray[np.int32_t, ndim=2] M):
    return np.array(M)[0]

def asVector(np.ndarray[double, ndim=2] M):
    return np.array(M)[0]