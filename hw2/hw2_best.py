import sys
import numpy as np
from xgboost import XGBClassifier

import utils

if __name__ == '__main__':
    tr_x, tr_y, tt_x = utils.load_data(sys.argv[1], sys.argv[2], sys.argv[3])
    #tr_x, tt_x = utils.normalize(tr_x, tt_x)

    model = XGBClassifier()
    model.fit(tr_x, tr_y)

    pred = model.predict(tt_x)
    pred = np.around(pred)
   
    outfile = open(sys.argv[4], 'w')
    outfile.write('id,label\n')
    for i, v in enumerate(pred):
        outfile.write('%d,%d\n' % (i + 1, v))
    outfile.close()
