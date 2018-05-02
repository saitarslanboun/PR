"""
Usage: $ python create_outfit_combos input_json_file.json output_json_file.json
"""

import json
from itertools import combinations
import sys
import io

try:
    to_unicode = unicode
except NameError:
    to_unicode = str

json_data_path_in = str(sys.argv[1])
json_data_path_out = str(sys.argv[2])

# print item names
def arrayPrint(array):
  for item in array:
    print " - ", item

# Open the json file
test_json = json.load(open(json_data_path_in))
max_outfits = len(test_json)
print "\nOutfits in this dataset: ", max_outfits

# Open the output json file and write to it
with io.open(json_data_path_out, 'w') as json_out:

  # Initialize the json data list
  output_json_data_list = []

  for i in range(0, max_outfits):
    outfit = test_json[i]
    outfit_id = outfit["set_id"]
    items = outfit["items"]
    item_indices = []

    # Get the actual indicies of the items
    for item in items:
      item_indices.append(item["index"])
    
    # Iterate through different combinations (combinations of 1, 2, ... len-1)
    for nr_of_input_items in range(1, len(item_indices)):

      # Get the combination indicies
      combo_idxs_array = list(combinations(item_indices, nr_of_input_items))

      # Get the combos
            

      for input_idxs in combo_idxs_array:
        output_idxs = list(set(item_indices) - set(input_idxs))
        # print "input idxs:", list(input_idxs), "| output idxs:", output_idxs
      
        # Write the output to a json file
        output_json_data = {}
        output_json_data["outfit_id"] = outfit_id
        output_json_data["input_ids"] = list(input_idxs)
        output_json_data["output_ids"] = output_idxs
        
        # Append the outfit combo
        output_json_data_list.append(output_json_data)

  str_ = json.dumps(output_json_data_list, sort_keys=False, indent=2)
  json_out.write(to_unicode(str_))
        

