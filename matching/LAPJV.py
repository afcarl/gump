# This function solves the Linear Assignment Problem by implementing the Jonker-Volgenant algorithm
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

import numpy as np

# comment the following imports to remove cython dependency.
import pyximport
pyximport.install(setup_args={'include_dirs':[np.get_include()]})
from cyLAPJV import cy_lapjv


def fast_lapjv(costMat, resolution=None):
    return cy_lapjv(costMat, resolution)

# python implementation
def lapjv(costMat, resolution=None):
    costMat = np.mat(costMat)
    if resolution is None:
        maxcost = np.min([1e-16, np.max(costMat)])
        resolution = 1e-16

    rdim, cdim = costMat.shape
    M = np.min(costMat)
    if rdim > cdim:
        costMat = costMat.T
        rdim, cdim = costMat.shape
        swapf = True
    else:
        swapf = False

    dim = cdim
    costMat = np.vstack((costMat, 2*M*ones(cdim-rdim, cdim)))
    maxcost = np.max(costMat)*dim + 1
    # TODO: handle inf and nan

    v = zeros(1, dim)
    rowsol = zeros(1, dim, int) - 1
    colsol = zeros(dim, 1, int) - 1
    if np.std(costMat, ddof=1) < np.mean(costMat):
        numfree = -1
        free = zeros(dim, 1)
        matches = zeros(dim, 1)
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
        free = range(dim)
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
    loopcnt = 0
    while loopcnt < 2:
        loopcnt += 1
        # scan all free rows
        # in some cases, a free row may be replaced with another one to be scaned next
        k = -1
        prvnumfree = numfree
        numfree = -1    # start list of rows still free after augmenting row reduction.

        while k < prvnumfree:
            k += 1
            i = free[k]
            # find minimum and second minimum reduced cost over columns
            x = asVector(costMat[i, :] - v)

            [umin, j1] = min(x)

            x[j1] = maxcost
            [usubmin, j2] = min(x)
            i0 = colsol[j1]
            if usubmin - umin > resolution:
                # change the reduction of the minmum column to increase the
                # minimum reduced cost in the row to the subminimum.
                v[j1] -= (usubmin - umin)
            else:  # minimum and subminimum equal.
                if i0 >= 0:  # minimum column j1 is assigned.
                    # swap columns j1 and j2, as j2 may be unassigned.
                    j1 = j2
                    i0 = colsol[j2]

            rowsol[i] = j1
            colsol[j1] = i

            if i0 >= 0:  # minimum column j1 assigned easier
                if usubmin - umin > resolution:
                    # put in current k, and go back to that k.
                    # continue augmenting path i - j1 with i0.
                    free[k] = i0
                    k -= 1
                else:
                    # no further augmenting reduction possible
                    # store i0 in list of free rows for next phase.
                    numfree += 1
                    free[numfree] = i0
    for f in xrange(numfree+1):
        freerow = free[f]  # start row of augmenting path
        # Dijkstra shortest path algorithm.
        # runs until unassigned column added to shortest path tree.
        d = asVector(costMat[freerow, :] - v)
        pred = asVector(freerow * ones(1, dim))
        collist = np.array(range(dim))
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
                x = asVector(costMat[i, :]-v)
                h = x[j1] - minh
                xh = x-h
                k0 = np.array(range(up, dim))
                j = collist[k0]
                vf0 = xh<d
                vf = vf0[j]
                vj = j[vf]
                vk = k0[vf]
                pred[vj] = i
                v2 = xh[vj]
                d[vj] = v2
                vf = v2 == minh  # new column found at same minimum value
                j2 = vj[vf]
                k2 = vk[vf]
                cf = colsol[j2] < 0
                if np.any(cf):  # unassigned, shortest augmenting path is complete.
                    i2 = find(cf, 1)[0]
                    endofpath = j2[i2]
                    unassignedfound = True
                else:
                    i2 = len(cf)+1
                # add to list to be scaned right away
                for k in range(0, i2-1):
                    collist[k2[k]] = collist[up]
                    collist[up] = j2[k]
                    up += 1

        # update column prices
        j1=collist[1:last+1]
        v[j1] = v[j1] + d[j1] - minh
        # reset row and column assignments along the alternating path
        while 1:
            i = pred[endofpath]
            colsol[endofpath] = i
            j1 = endofpath
            endofpath = rowsol[i]
            rowsol[i] = j1
            if [i==freerow]:
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

def assert_perm(pi):
    print pi + 1,
    N = len(pi)
    u = set(pi)
    T = len(u) == N and np.min(pi) == 0 and np.max(pi) == (N-1)


# matlab-like functions

def min(A):
    if len(A.shape) == 2:
        N = A.shape[1]
        I = np.argmin(A, axis=0)
        return asVector(A[I, range(0, N)]), asVector(I)
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


def zeros(N, D=1, dtype=float):
    if N == 1 or D == 1:
        return np.zeros(N*D, dtype=dtype)
    else:
        return np.zeros((N, D), dtype=dtype)


def submatrix(A, rows, cols):
    I = np.ix_(rows, cols)
    return A[I]


def asVector(M):
    return np.array(M)[0]


def testAgainstMunkres(N=100, d=100, round=False):
    # compare Hungarian (munkres) algoritm to LAPJV
    import munkres
    success = 0
    for i in xrange(N):
        A = np.random.randn(d,d) / np.random.randn(d,d)
        if round:
            A = A.round()
        if i % 2 == 0: # test some branh of the code
            A = A - np.mean(A)
        [rowsol, cost, v, u, costMat] = cy_lapjv(A)
        E = munkres.munkres(A)
        rowsol_munkres = np.nonzero(E)[1]
        cost_munkres = (A[E]).sum()
        acc = np.abs(cost - cost_munkres)
        if acc < 1e-8 and np.all(rowsol_munkres==rowsol):
            success += 1
        else:
            print i
            print 'failed with accurracy', acc
            print 'munkres:', cost_munkres
            print 'LAPJV:', cost
            print rowsol_munkres - rowsol
            print rowsol
            print A
            asd
    print success, 'out of' , N


def testPyCySpeed():
    # compare the python and cython implementations
    import time
    d = 500
    A = np.random.randn(d,d) / np.random.randn(d,d)

    t0 = time.time()
    [rowsol, cost, v, u, costMat] = fast_lapjv(A)
    elapsed = time.time() - t0
    print 'cython', elapsed, cost

    t1 = time.time()
    [rowsol, cost, v, u, costMat] = lapjv(A)
    elapsed = time.time() - t0
    print 'python', elapsed, cost


if __name__ == '__main__':
    import sys
    N = 1
    np.random.seed(int(sys.argv[1]))
    testAgainstMunkres(1000, 30)
    #testPyCySpeed()
