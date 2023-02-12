import os

partimagenet_to_grouped_partimagenet = {}

ANIMAL_HEAD = 10
ANIMAL_BODY = 11
ANIMAL_ARM = 12
ANIMAL_LEG = 13
ANIMAL_TAIL = 14

# background
partimagenet_to_grouped_partimagenet[0] = 0
# Aeroplane
partimagenet_to_grouped_partimagenet[1] = 1
partimagenet_to_grouped_partimagenet[2] = 2
partimagenet_to_grouped_partimagenet[3] = 3
partimagenet_to_grouped_partimagenet[4] = 4
partimagenet_to_grouped_partimagenet[5] = 5
# Bicycle
partimagenet_to_grouped_partimagenet[6] = 6
partimagenet_to_grouped_partimagenet[7] = 7
partimagenet_to_grouped_partimagenet[8] = 8
partimagenet_to_grouped_partimagenet[9] = 9
# Biped
partimagenet_to_grouped_partimagenet[10] = ANIMAL_HEAD # animal head
partimagenet_to_grouped_partimagenet[11] = ANIMAL_BODY # animal body
partimagenet_to_grouped_partimagenet[12] = ANIMAL_ARM # animal arm
partimagenet_to_grouped_partimagenet[13] = ANIMAL_LEG # animal leg
partimagenet_to_grouped_partimagenet[14] = ANIMAL_TAIL # animal tail
# Bird
partimagenet_to_grouped_partimagenet[15] = ANIMAL_HEAD # animal head
partimagenet_to_grouped_partimagenet[16] = ANIMAL_BODY # animal body
partimagenet_to_grouped_partimagenet[17] = ANIMAL_ARM # animal wing
partimagenet_to_grouped_partimagenet[18] = ANIMAL_LEG # animal leg
partimagenet_to_grouped_partimagenet[19] = ANIMAL_TAIL # animal tail

# Boat
partimagenet_to_grouped_partimagenet[20] = 15
partimagenet_to_grouped_partimagenet[21] = 16

# Bottle
partimagenet_to_grouped_partimagenet[22] = 17
partimagenet_to_grouped_partimagenet[23] = 18

# Car
partimagenet_to_grouped_partimagenet[24] = 19
partimagenet_to_grouped_partimagenet[25] = 20
partimagenet_to_grouped_partimagenet[26] = 21

# Fish
partimagenet_to_grouped_partimagenet[27] = ANIMAL_HEAD # animal head
partimagenet_to_grouped_partimagenet[28] = ANIMAL_BODY # animal body
partimagenet_to_grouped_partimagenet[29] = ANIMAL_ARM # animal fins
partimagenet_to_grouped_partimagenet[30] = ANIMAL_TAIL # animal tail

# Quadruped
partimagenet_to_grouped_partimagenet[31] = ANIMAL_HEAD # animal head
partimagenet_to_grouped_partimagenet[32] = ANIMAL_BODY # animal body
partimagenet_to_grouped_partimagenet[33] = ANIMAL_LEG # animal leg
partimagenet_to_grouped_partimagenet[34] = ANIMAL_TAIL # animal tail

# Reptile
partimagenet_to_grouped_partimagenet[35] = ANIMAL_HEAD # animal head
partimagenet_to_grouped_partimagenet[36] = ANIMAL_BODY # animal body
partimagenet_to_grouped_partimagenet[37] = ANIMAL_LEG # animal leg
partimagenet_to_grouped_partimagenet[38] = ANIMAL_TAIL # animal tail

# Snake
partimagenet_to_grouped_partimagenet[39] = ANIMAL_HEAD # animal head
partimagenet_to_grouped_partimagenet[40] = ANIMAL_BODY # animal body

grouped_partimagenet_id2name = {}
grouped_partimagenet_id2name[0] = 'background'

grouped_partimagenet_id2name[1] = 'Aeroplane_head'
grouped_partimagenet_id2name[2] = 'Aeroplane_body'
grouped_partimagenet_id2name[3] = 'Aeroplane_engine'
grouped_partimagenet_id2name[4] = 'Aeroplane_wing'
grouped_partimagenet_id2name[5] = 'Aeroplane_tail'

grouped_partimagenet_id2name[6] = 'Bicycle_frame' 
grouped_partimagenet_id2name[7] = 'Bicycle_handle'
grouped_partimagenet_id2name[8] = 'Bicycle_seat'
grouped_partimagenet_id2name[9] = 'Bicycle_wheel'

grouped_partimagenet_id2name[10] = 'Animal_head' 
grouped_partimagenet_id2name[11] = 'Animal_body'
grouped_partimagenet_id2name[12] = 'Animal_arm'
grouped_partimagenet_id2name[13] = 'Animal_leg'
grouped_partimagenet_id2name[14] = 'Animal_tail'

grouped_partimagenet_id2name[15] = 'Boat_body'
grouped_partimagenet_id2name[16] = 'Boat_engine'

grouped_partimagenet_id2name[17] = 'Bottle_body'
grouped_partimagenet_id2name[18] = 'Bottle_cap'

grouped_partimagenet_id2name[19] = 'Car_body'
grouped_partimagenet_id2name[20] = 'Car_wheel'
grouped_partimagenet_id2name[21] = 'Car_engine'


partimagenet_id2name = {}   
# partimagenet_id2name[0] = 'background'

partimagenet_id2name[0] = 'Aeroplane_head'
partimagenet_id2name[1] = 'Aeroplane_body'
partimagenet_id2name[2] = 'Aeroplane_engine'
partimagenet_id2name[3] = 'Aeroplane_wing'
partimagenet_id2name[4] = 'Aeroplane_tail'

partimagenet_id2name[5] = 'Bicycle_frame'
partimagenet_id2name[6] = 'Bicycle_handle'
partimagenet_id2name[7] = 'Bicycle_seat'
partimagenet_id2name[8] = 'Bicycle_wheel'

partimagenet_id2name[9] = 'Biped_head'
partimagenet_id2name[10] = 'Biped_body'
partimagenet_id2name[11] = 'Biped_arm'
partimagenet_id2name[12] = 'Biped_leg'
partimagenet_id2name[13] = 'Biped_tail'

partimagenet_id2name[14] = 'Bird_head'
partimagenet_id2name[15] = 'Bird_body'
partimagenet_id2name[16] = 'Bird_wing'
partimagenet_id2name[17] = 'Bird_leg'
partimagenet_id2name[18] = 'Bird_tail'

partimagenet_id2name[19] = 'Boat_body'
partimagenet_id2name[20] = 'Boat_sail'

partimagenet_id2name[21] = 'Bottle_head'
partimagenet_id2name[22] = 'Bottle_body'

partimagenet_id2name[23] = 'Car_body'
partimagenet_id2name[24] = 'Car_wheel'
partimagenet_id2name[25] = 'Car_mirror'

partimagenet_id2name[26] = 'Fish_head'
partimagenet_id2name[27] = 'Fish_body'
partimagenet_id2name[28] = 'Fish_fins'
partimagenet_id2name[29] = 'Fish_tail'

partimagenet_id2name[30] = 'Quadruped_head'
partimagenet_id2name[31] = 'Quadruped_body'
partimagenet_id2name[32] = 'Quadruped_leg'
partimagenet_id2name[33] = 'Quadruped_tail'

partimagenet_id2name[34] = 'Reptile_head'
partimagenet_id2name[35] = 'Reptile_body'
partimagenet_id2name[36] = 'Reptile_leg'
partimagenet_id2name[37] = 'Reptile_tail'

partimagenet_id2name[38] = 'Snake_head'
partimagenet_id2name[39] = 'Snake_body'




# label_dir = '~/data/PartImageNet'
# part_segmentations_path = os.path.join(
#     label_dir, "PartSegmentations", 'All', 'train'
# )
imagenet_labels_filename = 'LOC_synset_mapping.txt'
imagenet_id2name = {}
with open(imagenet_labels_filename, "r") as f:        
    for line in f:
        line = line.strip()
        line_split = line.split()
        imagenet_id2name[line_split[0]] = line_split[1]


PART_IMAGENET_CLASSES = {
    "Quadruped": 4,
    "Biped": 5,
    "Fish": 4,
    "Bird": 5,
    "Snake": 2,
    "Reptile": 4,
    "Car": 3,
    "Bicycle": 4,
    "Boat": 2,
    "Aeroplane": 5,
    "Bottle": 2,
}
PART_IMAGENET_CLASSES = dict(sorted(PART_IMAGENET_CLASSES.items()))

# IMAGENET_IDS = set()
# # create mapping from imagenet id to partimagenet id
# # partimagenetid_to_imagenetid = {}
# for class_label, part_imagenet_class in enumerate(PART_IMAGENET_CLASSES):
#     with open(f"{part_segmentations_path}/{part_imagenet_class}.txt", "r") as f:
#         filenames = f.readlines()
#         for filename in filenames:
#             filename = filename.strip()
#             imagenet_id = filename.split('/')[0]
#             IMAGENET_IDS.add(imagenet_id)

# IMAGENET_IDS = list(IMAGENET_IDS)
# assert len(IMAGENET_IDS) == 158 # https://arxiv.org/pdf/2112.00933.pdf
# IMAGENET_IDS.sort()

# imagenetclass_to_imagenetid = {}
# imagenetid_to_imagenetclass = {}
# for imagenet_class, imagenet_id in enumerate(IMAGENET_IDS):
#     imagenetid_to_imagenetclass[imagenet_id] = imagenet_class
#     imagenetclass_to_imagenetid[imagenet_class] = imagenet_id



# TODO:
# # saving annotations
# json_object = json.dumps(partimagenet_id2name, indent=4)
# id2name_filename = os.path.join(label_dir, 'part_imagenet_id2name.json')
# with open(id2name_filename, "w") as outfile:
#     outfile.write(json_object)


