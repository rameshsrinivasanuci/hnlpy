# Mariel Tisby
# 8/15/19

'''
HNL Project

This project is the main function for sorting through HDDM rt data 
includes:
	~ CSV file outputs with subject data
	~ MAT file output with indices of removed trials and original trial order
	~ Removal of NAN trials
	~ cutoff function using EWMAV

'''

import LoadMat as LM
import EWMAV_PY as EM

goal_dir = '/data/pdmattention/task3/'

data = LM.load_data(goal_dir)
data1 = EM.EWMAV(data)

# CSV function 
# complete = create_CSV(data1)

# if complete == True:
# 	print('EWMAV Rt Sorting Complete')
