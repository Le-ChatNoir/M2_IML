# Use to equilibrate everything
def equilibring(data_images, data_results):
	temp = zip(data_images, data_results)
	#Create empty dictionnary
	counter_array={}
	result=[]
	nb_type_x = 0
	# Counting how many time every type appears
	for row in temp:
		# If key in dict
		if row[1] in counter_array:
			counter_array[row[1]] += 1
		else:
			#Create new key in dic using the type found number
			counter_array[row[1]] = 1
			
	#Printing the dict of things found
	print(counter_array.items())
	
	# Finding the one result with the minimal in the values of the dict
	minimumkey = min(counter_array.keys(), key=(lambda k: counter_array[k]))
	print("Minimal value found is ", counter_array[minimumkey], " (", minimumkey, ")")
	
	# Initialize every counter of removal
	minimal_counter = {}
	for key in counter_array:
		minimal_counter[key] = counter_array[key] - counter_array[minimumkey]
		if minimal_counter[key] == minimumkey:
			minimal_counter[key] = 0
	
	#Print how many to remove of each
	print("Removing: ")
	print(minimal_counter.items())
	
	# Cloning to a new array
	temp = zip(data_images, data_results)
	for row in temp:
		if minimal_counter.get(row[1]) == 0:
			result.append(row)
		else:
			minimal_counter[row[1]] -= 1
			
	
	#Replacing it to to regular arrays
	data_images, data_results = zip(*result)
	return data_images, data_results
