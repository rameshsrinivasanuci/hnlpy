# imports
import diffusion

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
	path = get_paths(basedir, taskNum)
		# path_type represents the base path we are looking for; where we store analyzed data or 
		# retrieve raw data
	
	# returns a list of subject IDs based on wheter or not this is testing analysis
	# or analysis of all subjects
	subIDs= choose_subs(lvlAnalysis, path)
	preprocessing_main(subIDs, path)




def preprocessing_main(subIDs, path):
	for xx in range(len(subIDs)):
		currentSub = subIDs[xx]
		BehPath, EEGPath = path_reconstruct(currentSub, path, taskNum)
		BehInd, EEGInd = readFiles(beh_file, eeg_file)
		OverlapInd = find_overlapIndices(BehInd, EEGInd)
		sub_data = extract_data(OverlapInd, xx)
		writeCSV(currentSub, sub_data)


if __name__ == "__main__":
	main()


