import numpy as np
Y_test_mid = []
Y_set = np.loadtxt('dabase_test/antigen_mapping.txt')
for row in range(253):
    for column in range(253):
        if row == column:
            continue
        else:
            value = Y_set[row, column]
            Y_test_mid.append(value)
Y_test = np.array(Y_test_mid)