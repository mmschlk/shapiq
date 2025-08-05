import numpy as np
from shapiq.utils import powerset


if __name__ == "__main__":
    # all sets of size 1 and 2
    n = 7
    #Q = []

    MAX_ORDER = 3
    Q = []
    for S in powerset(set(range(n)),max_size=MAX_ORDER):
        Q.append(S)
    print('Basis Q:', Q)

    m = 20

    A = np.zeros((m, len(Q)))

    samples = []
    weights = []

    print('Sampled Subsets:')
    for idx in range(m//2):
        weight = np.random.rand()
        #weight = 1
        if idx == 0:
            random_S = []
        else:
            random_size = np.random.randint(1, n)
            random_S = np.random.choice(n, random_size, replace=False)
        complement_S = [i for i in range(n) if i not in random_S]
        print(random_S, complement_S, sep='\t')
        samples.append(random_S)
        samples.append(complement_S)
        weights.append(weight**2)
        weights.append(weight**2)
        for q in Q:
            if set(q).issubset(set(random_S)):
                A[2*idx, Q.index(q)] = weight
            if set(q).issubset(set(complement_S)):
                A[2*idx+1, Q.index(q)] = weight
    #print('A')
    #print(A)

    b = np.random.rand(m)
    b = A @ np.random.rand(len(Q))

    M = np.zeros((n+1, len(Q)))
    M[0,0] = 1
    for q in Q:
        for i in q:
            M[i+1, Q.index(q)] = 1 /len(q)


    #tilde_xn = np.linalg.lstsq(A[:,:n],b)[0]
    tilde_xn = np.linalg.lstsq(A[:,:n+1],b)[0]
    tilde_xQ = np.linalg.lstsq(A,b)[0]

    print('tilde_xn', tilde_xn)
    print('shap tilde xQ', M @ tilde_xQ)

    AtA = A[:,:n+1].T @ A[:,:n+1]
    AtA_inv = np.linalg.pinv(A[:,:n+1].T @ A[:,:n+1])
    AtB = A[:,:n+1].T @ A[:,n+1:]

    print('M_end')
    print(M[:,n+1:])

    M_prime = AtA_inv @ AtB

    print('M_prime')
    print(M_prime.round(10))

    print('AtA @ M_prime')
    print((AtA @ M_prime).round(10))

    print('AtB')
    print(AtB.round(10))

    def row_guess(S, samples, weights):
        numerator = 0
        denominator = 0
        for sample, weight in zip(samples, weights):
            # If |S cap T| = 1
            denominator += 2 * weight
            if len(set(S).intersection(set(sample))) == 1:
                numerator -= weight
        return numerator / denominator

    def row_guess_k(S, samples, weights):
        numerator = 0
        denominator = 0
        for sample, weight in zip(samples, weights):
            denominator += weight
            for k in range(1,len(S)):
                if len(set(S).intersection(set(sample))) == k:
                    numerator -= weight * k / len(S)
        return numerator / denominator


    for S in Q:
        print()
        if len(S) >= 2:
            guess = row_guess_k(S, samples, weights)
            print(i,S, guess)

    buildAtAM_prime = np.zeros((n+1, len(Q)-(n+1)))
    for idx, S in enumerate(Q):
        if len(S) == 2:
            for i in range(n):
                guess = 0
                for sample, weight in zip(samples, weights):
                    if i in sample:
                        guess += weight * M_prime[0, idx-(n+1)]
                        for j in S:
                            if j in sample:
                                guess += weight * 1 / len(S)
                buildAtAM_prime[i+1, idx-(n+1)] = guess

    #print(buildAtAM_prime)
    #print(AtA @ M_prime)

    assert np.allclose(AtA @ M_prime, AtB)