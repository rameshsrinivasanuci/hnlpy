# Mariel Tisby 
# HNL Project: Sorting Reaction Time -- Exponetially Weighted Moving Average Filter
# 7.18.19


'''
First sort RTs from short to long
'''


'''
mar's notes:

zip_longest -- sorts all maybe??
'''

class rt_sort:

	def __init__(self, rts):
		self.rts = rts 

	def sortRTData(self)
		sorted(range(len(self.rts), key = lambda k:self.rts[k]))
		return self.rts 