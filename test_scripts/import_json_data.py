import json
from itertools import combinations
import sys

json_data = str(sys.argv[1])
output_file = str(sys.argv[2])

# print item names
def arrayPrint(array):
  for item in array:
    print " - ", item

# Open the json file
test_json = json.load(open(json_data))
print "\nOutfits in this dataset: ", len(test_json)

## Loop through the dataset and generate outfit combinations
#for image_set in test_json:
#  print "------------------------------------------ \n"
#  set_id = image_set["set_id"]
#  set_name = image_set["name"]
#  set_likes = image_set["likes"]
#  set_items = image_set["items"]
#  set_item_names = []

#  # Get the item names in the set
#  for item in set_items:
#    set_item_names.append(item["name"])

for i in range(0, 3):
  outfit = test_json[i]
  items = outfit["items"]

  # This array will be the basis for item combination generator
  indices = range(0, len(items))
  
  # Iterate through different combinations (combinations of 1, 2, ... len-1)
  for nr_of_input_items in range(1, len(indices)):
    # Get the combination indicies
    combo_idxs_array = list(combinations(indices, nr_of_input_items))

    # Get the combos
    for input_idxs in combo_idxs_array:
      output_idxs = list(set(indices) - set(input_idxs))
      print "input idxs:", list(input_idxs), "| output idxs:", output_idxs
    
  print "--------------"

#idxs = [0, 1, 2, 3, 4, 5, 6]
#n_combos = 0

#for i in range(1, len(idxs)):
#  combos = list(combinations(idxs, i))
#  n = len(combos)
#  n_combos += n
#  # print "Amount of combinations with", i, "pairs is", n 
#  # print combos

#print "Combinations in total:", n_combos

#print "Amount of training combinations (#outfits x #combinations):", n_combos*len(test_json)
