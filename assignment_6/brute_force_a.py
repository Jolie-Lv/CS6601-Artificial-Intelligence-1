import numpy as np
import hashlib

def _solve(train_samples, hashes, full=False):
    l0, l1, l2 = len(train_samples[0]), len(train_samples[1]), len(train_samples[2])
    if full:
        l0 += 1
        l1 += 1
        l2 += 1

    for p0 in range(1, l0):
        for p1 in range(1,l1):
            for p2 in range(1,l2):
                partition = np.array(train_samples[0][0:p0] + train_samples[1][0:p1] + train_samples[2][0:p2])
                mu = np.around(np.mean(partition), 3)
                sigma = np.around(np.std(partition), 3)
                # print(mu, sigma)
                mu_hash = hashlib.sha256(str.encode(str(mu))).hexdigest()
                sigma_hash = hashlib.sha256(str.encode(str(sigma))).hexdigest()
                if mu_hash == hashes[0] and sigma_hash == hashes[1]:
                    return (p0, p1, p2), (mu, sigma)

def _remove_elements(train_samples, partition):
    for idx in range(len(train_samples)):
        train_samples[idx] = train_samples[idx][partition[idx]:]
    return train_samples

def bf_solver(train_samples, target_hashes):
    first_partition, params1 = _solve(train_samples, target_hashes['1'])
    print("first partition:", first_partition, "params: ", params1)

    # remove elements after partition
    train_samples = _remove_elements(train_samples, first_partition)

    second_partition, params2 = _solve(train_samples, target_hashes['2'])
    print("second partition:", second_partition, "params: ", params2)

    # remove elements after partition
    train_samples = _remove_elements(train_samples, second_partition)

    third_partition, params3 = _solve(train_samples, target_hashes['3'], full=True)
    print("third partition:", third_partition, "params: ", params3)

    t0 = np.around(1.0/np.sum(first_partition), 3)
    t1 = np.around(1.0/np.sum(second_partition), 3)
    t2 = np.around(1.0/np.sum(third_partition), 3)

    print("Transition Probs:")
    print(1-t0, t0)
    print(1-t1, t1)
    print(1-t2, t2)

if __name__ == "__main__":
    # BUY train samples
    B = [[36, 44, 52, 56, 49, 44],
         [42, 46, 54, 62, 68, 65, 60, 56],
         [42, 40, 41, 43, 52, 55, 59, 60, 55, 47]]
    b_hashes = {}
    b_hashes['1'] = ["fc12f049f5c759702b7bcd27d461fb57a4c9176bebfab3e4d15426a74c911d03",
                      "9b62d9c6eac8cbacdc2ccdfed1d60feb0716e2b39f5b94eac4bc69f803697ede"]
    b_hashes['2'] = ["7a696b9ae0bc3323ce647c690106a78287ec2d5ce24ee5d11f48168bdb1a5dbb",
                      "8f3ff2d53dd528ebf1cccbb60667e2a1c0906da993de765634c01e6b5c85b34a"]
    b_hashes['3'] = ["c308fb57f5aa803bbcddeecac1e547d2b7010018758e72252cb7ceca298e4dbf",
                      "69f775cb8dc0f5d96d0c78826f813fc17b99018aee95f8d34e30f7e3f46743ba"]

    C = [[47, 39, 32, 34, 36, 42, 42, 42, 34, 25],
         [35, 35, 43, 46, 52, 52, 56, 49, 45],
         [28, 35, 46, 46, 48, 43, 43, 40]]

    c_hashes = {}
    c_hashes['1'] = ["ece665f5d82dd6570657a9b11736924a97e327adc5ff314be890b2e565193f44",
                      "224edb71a15e864dff50b2224ca79bb5eb5179e4b287bdf3a54f6abec1f5be3e"]
    c_hashes['2'] = ["e63f3d3e0ce127bb0afcca123bc00babd29820b15f53a7f9b6a31534a4fb0597",
                      "8f65223004a75f44404f485a1e84090699acef51f39de9411d6d9b377ae859a5"]
    c_hashes['3'] = ["51489ee602434160b5c1cfc98a781353eb98db3b0fee064b951ba5baa4c9a014",
                      "6031bf9944ad15cdfcb096f4432643b7c097da0f179e7d584a016724d9338c98"]


    H = [[37, 36, 32, 26, 26, 25, 23, 22, 21, 39, 48, 60, 70, 74, 77],
         [50, 50, 49, 47, 39, 39, 38, 38, 50, 56, 61, 67, 67, 67, 67],
         [45, 43, 44, 43, 40, 35, 36, 37, 39, 45, 60, 68, 66, 72, 72, 75]]

    h_hashes = {}
    h_hashes['1'] = ["bbb4004ad949f0be888a1923e336d84485cbbace6641a94e09f6380fbc52b9ae",
                      "17c40ca95ab8e9107a4157365cb34646c64447a9f39cb4447176a736036495b3"]
    h_hashes['2'] = ["9f9ac2449c421664c29f3e534b384d10900fe65fc2726941569a614a801f4b47",
                      "616a46cf184e50b2ff1debd938a19b3f112c2704f07985a3fe13f849bec48288"]
    h_hashes['3'] = ["6d2e4be9b46ce8375256cf2bc5e2eb4c38a0fe2c6ae02f32a1e1955305cf3809",
                      "966d64084414dc3ce0e395a8ed417665a82b21e6f9858e4168d3578585042cc4"]


    print("Solving BUY")
    bf_solver(B, b_hashes)

    print("Solving CAR")
    bf_solver(C, c_hashes)

    print("Solving HOUSE")
    bf_solver(H, h_hashes)
