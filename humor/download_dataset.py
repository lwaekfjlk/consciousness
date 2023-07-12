from datasets import load_dataset

# load train/val/test splits for each task
dset = load_dataset("jmhessel/newyorker_caption_contest", "matching")
dset = load_dataset("jmhessel/newyorker_caption_contest", "ranking")
dset = load_dataset("jmhessel/newyorker_caption_contest", "explanation")

# load in the "from pixels" setting
dset = load_dataset("jmhessel/newyorker_caption_contest", "ranking_from_pixels")

# we ran in cross val (split 0 is default ^ ) so you can load splits 1/2/3/4 this way the 4th data split
dset = load_dataset("jmhessel/newyorker_caption_contest", "explanation_4")
# ... or split 1...
dset = load_dataset("jmhessel/newyorker_caption_contest", "explanation_from_pixels_1")