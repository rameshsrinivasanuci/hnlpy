# imports
import diffusion

debugging = True

### globals ###

#global variable for different task
taskNum = 3
basedir = 'data/pdmattention/'
# global variable for level of analysis
# lvlAnalysis indicates if we are running analysis on test data or all data
# 1 = test
# 2 = all
lvlAnalysis = 1

def main():

	# reconstructs base directory path based on whether we are looking for the data directory
	# and the task we are analyzig
	path = diffusion.get_paths(basedir, taskNum)
		# path_type represents the base path we are looking for; where we store analyzed data or 
		# retrieve raw data
	
	# returns a list of subject IDs based on wheter or not this is testing analysis
	# or analysis of all subjects
	subIDs= diffusion.choose_subs(lvlAnalysis, path)
	preprocessing_main(subIDs, path)




def preprocessing_main(subIDs, path):
	for xx in range(len(subIDs)):
		currentSub = subIDs[xx]
		BehPath, EEGPath = diffusion.path_reconstruct(currentSub, path, taskNum)
		BehInd, EEGInd = diffusion.readFiles(BehPath, EEGPath)
		OverlapInd = diffusion.find_overlapIndices(BehInd, EEGInd)
		sub_data = diffusion.extract_data(OverlapInd, currentSub, path, taskNum)
		diffusion.writeCSV(currentSub, sub_data)


def debug():
	debug_path = '/data/pdmattention/task3/'
	currentSub = 's195_ses2_' 
	BehPath, EEGPath = diffusion.path_reconstuct(currentSub, debug_path, taskNum)
	BehInd, EEGInd = diffusion.ReadFiles(BehPath, EEGPath, taskNum, currentSub)
	OverlapInd = diffusion.find_overlapIndices(BehInd, EEGInd)
	sub_data = diffusion.extract_data(OverlapInd, currentSub, debug_path, taskNum)
	
	if debugging == True:
		print(BehPath, EEGPath)
		print("BehInd: ", BehInd)
		print("EEGInd: ", EEGInd)
		print(OverlapInd)

	# for adding more colums to data, read in all data again with the additional column

if __name__ == "__main__":
	debug()


