#export
def E_step(X,MU,SIGMA,PI,k):
    """
    E-step - Expectation
    Calculate responsibility for each
    of the data points, for the given
    MU, SIGMA and PI.

    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    MU = numpy.ndarray[numpy.ndarray[float]] - k x n
    SIGMA = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - k x n x n
    PI = numpy.ndarray[float] - k x 1
    k = int

    returns:
    responsibility = numpy.ndarray[numpy.ndarray[float]] - k x m
    """
    m, n = X.shape
    k, _ = MU.shape

    # R = np.zeros((k,m))
    R_ = np.zeros((k,m))
#     import pdb
    # pdb.set_trace()
    PI = np.reshape(PI, (1,k))
    # for _k in range(k):
    #     # for _m in range(m):
    #     #     R[_k, _m] = prob(X[_m, :], MU[_k, :], SIGMA[_k, :, :])
    #     R[_k, :] = PI[0,_k]*np.apply_along_axis(prob, axis=1, arr=X, mu=MU[_k,:], sigma=SIGMA[_k,:,:])
    # R /= np.sum(R, axis=0)

#     import pdb
    X_mu = np.repeat(X[None, :, :], repeats=k, axis=0) - MU[:,None,:]
    probs = np.zeros((k, m))
    for _k in range(k):
        X_muk = X_mu[_k, :, :][:, None, :]
        sigmak = np.linalg.inv(SIGMA[_k, :, :])

        X_sigma = np.dot(X_muk, sigmak)
        X_exp = (1.0 / (2*np.pi)**(n/2.0)*np.linalg.det(sigmak)**(0.5)) * np.exp(np.sum(X_muk * X_sigma, axis=2) * (-0.5))
        probs[_k, :] = X_exp.T
    R_ = PI.T * probs
    R_ /= np.sum(R_, axis=0)
    # pdb.set_trace()
    return R_

########## DON'T WRITE ANY CODE OUTSIDE THE FUNCTION! ################
##### CODE BELOW IS USED FOR RUNNING LOCAL TEST DON'T MODIFY IT ######
%time tests.GMMTests().test_gmm_e_step(E_step)
################ END OF LOCAL TEST CODE SECTION ######################
