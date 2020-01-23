# imports
import diffusion

### globals ###
# global variables for different paths
data_path = 1
store_path = 2
#global variable for different task
taskNum = 3

# global variable for level of analysis
# lvlAnalysis indicates if we are running analysis on test data or all data
# 1 = test
# 2 = all
lvlAnalysis = 1

def preprocessing_main():

	# constructs path for retrieving raw data
	path = get_paths(data_path, task)
	allFiles, allIDs, testFiles, testIDs = get_files(path)
	files, IDs= PickAnalysisLvl(lvlAnalysis, allFiles, allIDs, testFiles, testIDs)

	for xx in range(len(files)):
		currentSub = files[xx]
		rts, conditions, corrects = ReadFiles(currentSub, path)

