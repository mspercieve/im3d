import numpy as np

bin = {}

NUM_ORI_BIN = 6
ORI_BIN_WIDTH = float(2 * np.pi / NUM_ORI_BIN) # 60 degrees width for each bin.
# orientation bin ranges from -np.pi to np.pi.
bin['ori_bin'] = [[(i - NUM_ORI_BIN / 2) * ORI_BIN_WIDTH, (i - NUM_ORI_BIN / 2 + 1) * ORI_BIN_WIDTH] for i
                    in range(NUM_ORI_BIN)]

print(bin)