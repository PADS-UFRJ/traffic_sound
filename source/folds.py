folds_number = 10

# Inicializando o dicionário para cada fold
folds = [dict() for i in range(folds_number)]

for idx in range(folds_number):
    folds[idx]["name"] = 'fold_' + str(idx)
    folds[idx]["index"] = idx

##########################
# Divisao Leo Mazza
##########################

# folds[0]['train'] = [
#     'M2U00001MPG',
#     'M2U00002MPG',
#     'M2U00005MPG',
# ]
# folds[0]['val'] = [
#     'M2U00003MPG',
#     'M2U00006MPG',
# ]

# folds[6]['train'] = [
#     'M2U00001MPG',
#     'M2U00003MPG',
#     'M2U00006MPG',
# ]
# folds[6]['val'] = [
#     'M2U00008MPG',
# ]

##########################
# Divisao 30/8
##########################

# folds[0]['train'] = [
#     'M2U00005MPG',
#     'M2U00006MPG',
#     'M2U00007MPG',
#     'M2U00008MPG',
#     'M2U00012MPG',
#     'M2U00015MPG',
#     'M2U00016MPG',
#     'M2U00017MPG',
#     'M2U00018MPG',
#     'M2U00019MPG',
#     'M2U00022MPG',
#     'M2U00023MPG',
#     'M2U00024MPG',
#     'M2U00025MPG',
#     'M2U00026MPG',
#     'M2U00027MPG',
#     'M2U00029MPG',
#     'M2U00030MPG',
#     'M2U00031MPG',
#     'M2U00032MPG',
#     'M2U00033MPG',
#     'M2U00035MPG',
#     'M2U00036MPG',
#     'M2U00037MPG',
#     'M2U00039MPG',
#     'M2U00041MPG',
#     'M2U00043MPG',
#     'M2U00045MPG',
#     'M2U00048MPG',
#     'M2U00050MPG',
# ]

# folds[0]['val'] = [
#     'M2U00001MPG',
#     'M2U00002MPG',
#     'M2U00003MPG',
#     'M2U00004MPG',
#     'M2U00014MPG',
#     'M2U00042MPG',
#     'M2U00046MPG',
#     'M2U00047MPG',
# ]


##########################
# Divisao do Matheus Lima
##########################

##########################
# Fold 0

folds[0]["train"] = [
    "M2U00004MPG",
    "M2U00006MPG",
    "M2U00008MPG",
    "M2U00012MPG",
    "M2U00014MPG",
    "M2U00017MPG",
    "M2U00015MPG",
    "M2U00016MPG",
    "M2U00018MPG",
    "M2U00022MPG",
    "M2U00024MPG",
    "M2U00026MPG",
    "M2U00027MPG",
    "M2U00029MPG",
    "M2U00030MPG",
    "M2U00031MPG",
    "M2U00035MPG",
    "M2U00037MPG",
    "M2U00039MPG",
    "M2U00043MPG",
    "M2U00045MPG",
    "M2U00046MPG",
    "M2U00047MPG",
    "M2U00048MPG",
    "M2U00050MPG"
]

folds[0]["val"] = [
    "M2U00001MPG",
    "M2U00002MPG",
    "M2U00003MPG",
    "M2U00005MPG",
    "M2U00007MPG",
    "M2U00019MPG",
    "M2U00023MPG",
    "M2U00025MPG",
    "M2U00032MPG",
    "M2U00033MPG",
    "M2U00036MPG"
]

##########################
# Fold 1

folds[1]["train"] = [
    "M2U00001MPG",
    "M2U00002MPG",
    "M2U00003MPG",
    "M2U00005MPG",
    "M2U00006MPG",
    "M2U00008MPG",
    "M2U00012MPG",
    "M2U00014MPG",
    "M2U00015MPG",
    "M2U00018MPG",
    "M2U00019MPG",
    "M2U00022MPG",
    "M2U00023MPG",
    "M2U00024MPG",
    "M2U00025MPG",
    "M2U00029MPG",
    "M2U00031MPG",
    "M2U00032MPG",
    "M2U00036MPG",
    "M2U00037MPG",
    "M2U00039MPG",
    "M2U00043MPG",
    "M2U00045MPG",
    "M2U00046MPG",
    "M2U00047MPG",
    "M2U00048MPG",
    "M2U00050MPG"
]

folds[1]["val"] = [
    "M2U00004MPG",
    "M2U00007MPG",
    "M2U00017MPG",
    "M2U00016MPG",
    "M2U00026MPG",
    "M2U00027MPG",
    "M2U00030MPG",
    "M2U00033MPG",
    "M2U00035MPG"
]

##########################
# Fold 2

folds[2]["train"] = [
    "M2U00001MPG",
    "M2U00002MPG",
    "M2U00003MPG",
    "M2U00007MPG",
    "M2U00008MPG",
    "M2U00014MPG",
    "M2U00017MPG",
    "M2U00016MPG",
    "M2U00018MPG",
    "M2U00019MPG",
    "M2U00023MPG",
    "M2U00024MPG",
    "M2U00025MPG",
    "M2U00027MPG",
    "M2U00029MPG",
    "M2U00030MPG",
    "M2U00031MPG",
    "M2U00032MPG",
    "M2U00033MPG",
    "M2U00036MPG",
    "M2U00037MPG",
    "M2U00039MPG",
    "M2U00043MPG",
    "M2U00045MPG",
    "M2U00048MPG",
    "M2U00050MPG"
]

folds[2]["val"] = [
    "M2U00004MPG",
    "M2U00005MPG",
    "M2U00006MPG",
    "M2U00012MPG",
    "M2U00015MPG",
    "M2U00022MPG",
    "M2U00026MPG",
    "M2U00035MPG",
    "M2U00046MPG",
    "M2U00047MPG"
]

##########################
# Fold 3

folds[3]["train"] = [
    "M2U00001MPG",
    "M2U00002MPG",
    "M2U00003MPG",
    "M2U00004MPG",
    "M2U00005MPG",
    "M2U00006MPG",
    "M2U00007MPG",
    "M2U00008MPG",
    "M2U00012MPG",
    "M2U00017MPG",
    "M2U00015MPG",
    "M2U00016MPG",
    "M2U00019MPG",
    "M2U00022MPG",
    "M2U00023MPG",
    "M2U00025MPG",
    "M2U00026MPG",
    "M2U00027MPG",
    "M2U00030MPG",
    "M2U00032MPG",
    "M2U00033MPG",
    "M2U00035MPG",
    "M2U00036MPG",
    "M2U00046MPG",
    "M2U00047MPG",
    "M2U00048MPG",
    "M2U00050MPG"
]

folds[3]["val"] = [
    "M2U00014MPG",
    "M2U00018MPG",
    "M2U00024MPG",
    "M2U00029MPG",
    "M2U00031MPG",
    "M2U00037MPG",
    "M2U00039MPG",
    "M2U00043MPG",
    "M2U00045MPG"
]

##########################
# Fold 4

folds[4]["train"] = [
    "M2U00001MPG",
    "M2U00002MPG",
    "M2U00004MPG",
    "M2U00005MPG",
    "M2U00006MPG",
    "M2U00007MPG",
    "M2U00014MPG",
    "M2U00017MPG",
    "M2U00015MPG",
    "M2U00016MPG",
    "M2U00019MPG",
    "M2U00023MPG",
    "M2U00024MPG",
    "M2U00025MPG",
    "M2U00026MPG",
    "M2U00027MPG",
    "M2U00029MPG",
    "M2U00030MPG",
    "M2U00031MPG",
    "M2U00032MPG",
    "M2U00033MPG",
    "M2U00035MPG",
    "M2U00037MPG",
    "M2U00039MPG",
    "M2U00043MPG",
    "M2U00045MPG",
    "M2U00046MPG",
    "M2U00047MPG"
]
folds[4]["val"] = [
    "M2U00003MPG",
    "M2U00008MPG",
    "M2U00012MPG",
    "M2U00018MPG",
    "M2U00022MPG",
    "M2U00036MPG",
    "M2U00048MPG",
    "M2U00050MPG"
]

##########################
# Fold 5

folds[5]["train"] = [
    "M2U00001MPG",
    "M2U00002MPG",
    "M2U00003MPG",
    "M2U00004MPG",
    "M2U00012MPG",
    "M2U00014MPG",
    "M2U00017MPG",
    "M2U00015MPG",
    "M2U00016MPG",
    "M2U00022MPG",
    "M2U00025MPG",
    "M2U00026MPG",
    "M2U00029MPG",
    "M2U00030MPG",
    "M2U00031MPG",
    "M2U00032MPG",
    "M2U00033MPG",
    "M2U00035MPG",
    "M2U00036MPG",
    "M2U00037MPG",
    "M2U00039MPG",
    "M2U00043MPG",
    "M2U00045MPG",
    "M2U00046MPG",
    "M2U00047MPG",
    "M2U00048MPG",
    "M2U00050MPG"
]

folds[5]["val"] = [
    "M2U00005MPG",
    "M2U00006MPG",
    "M2U00007MPG",
    "M2U00008MPG",
    "M2U00018MPG",
    "M2U00019MPG",
    "M2U00023MPG",
    "M2U00024MPG",
    "M2U00027MPG"
]

##########################
# Fold 6

folds[6]["train"] = [
    "M2U00001MPG",
    "M2U00002MPG",
    "M2U00003MPG",
    "M2U00004MPG",
    "M2U00005MPG",
    "M2U00006MPG",
    "M2U00007MPG",
    "M2U00008MPG",
    "M2U00016MPG",
    "M2U00018MPG",
    "M2U00019MPG",
    "M2U00023MPG",
    "M2U00024MPG",
    "M2U00025MPG",
    "M2U00027MPG",
    "M2U00032MPG",
    "M2U00033MPG",
    "M2U00035MPG",
    "M2U00036MPG",
    "M2U00037MPG",
    "M2U00039MPG",
    "M2U00043MPG",
    "M2U00045MPG",
    "M2U00046MPG",
    "M2U00047MPG",
    "M2U00048MPG",
    "M2U00050MPG"
]

folds[6]["val"] = [
    "M2U00012MPG",
    "M2U00014MPG",
    "M2U00017MPG",
    "M2U00015MPG",
    "M2U00022MPG",
    "M2U00026MPG",
    "M2U00029MPG",
    "M2U00030MPG",
    "M2U00031MPG"
]

##########################
# Fold 7

folds[7]["train"] = [
    "M2U00001MPG",
    "M2U00002MPG",
    "M2U00004MPG",
    "M2U00005MPG",
    "M2U00006MPG",
    "M2U00007MPG",
    "M2U00008MPG",
    "M2U00014MPG",
    "M2U00017MPG",
    "M2U00016MPG",
    "M2U00018MPG",
    "M2U00022MPG",
    "M2U00023MPG",
    "M2U00025MPG",
    "M2U00026MPG",
    "M2U00027MPG",
    "M2U00030MPG",
    "M2U00031MPG",
    "M2U00033MPG",
    "M2U00035MPG",
    "M2U00036MPG",
    "M2U00037MPG",
    "M2U00039MPG",
    "M2U00043MPG",
    "M2U00045MPG",
    "M2U00050MPG"
]

folds[7]["val"] = [
    "M2U00003MPG",
    "M2U00012MPG",
    "M2U00015MPG",
    "M2U00019MPG",
    "M2U00024MPG",
    "M2U00029MPG",
    "M2U00032MPG",
    "M2U00046MPG",
    "M2U00047MPG",
    "M2U00048MPG"
]

##########################
# Fold 8

folds[8]["train"] = [
    "M2U00003MPG",
    "M2U00005MPG",
    "M2U00006MPG",
    "M2U00007MPG",
    "M2U00012MPG",
    "M2U00014MPG",
    "M2U00017MPG",
    "M2U00015MPG",
    "M2U00016MPG",
    "M2U00019MPG",
    "M2U00022MPG",
    "M2U00023MPG",
    "M2U00024MPG",
    "M2U00026MPG",
    "M2U00027MPG",
    "M2U00029MPG",
    "M2U00030MPG",
    "M2U00031MPG",
    "M2U00032MPG",
    "M2U00033MPG",
    "M2U00036MPG",
    "M2U00037MPG",
    "M2U00042MPG",
    "M2U00043MPG",
    "M2U00045MPG",
    "M2U00047MPG",
    "M2U00050MPG"
]

folds[8]["val"] = [
    "M2U00001MPG",
    "M2U00002MPG",
    "M2U00004MPG",
    "M2U00008MPG",
    "M2U00018MPG",
    "M2U00025MPG",
    "M2U00035MPG",
    "M2U00039MPG",
    "M2U00041MPG",
    "M2U00046MPG",
    "M2U00048MPG"
]

##########################
# Fold 9

folds[9]["train"] = [
    "M2U00001MPG",
    "M2U00004MPG",
    "M2U00005MPG",
    "M2U00006MPG",
    "M2U00008MPG",
    "M2U00014MPG",
    "M2U00017MPG",
    "M2U00015MPG",
    "M2U00018MPG",
    "M2U00022MPG",
    "M2U00023MPG",
    "M2U00024MPG",
    "M2U00025MPG",
    "M2U00026MPG",
    "M2U00029MPG",
    "M2U00030MPG",
    "M2U00031MPG",
    "M2U00032MPG",
    "M2U00035MPG",
    "M2U00036MPG",
    "M2U00037MPG",
    "M2U00039MPG",
    "M2U00041MPG",
    "M2U00043MPG",
    "M2U00045MPG",
    "M2U00046MPG",
    "M2U00048MPG",
    "M2U00050MPG"
]

folds[9]["val"] = [
    "M2U00002MPG",
    "M2U00003MPG",
    "M2U00007MPG",
    "M2U00012MPG",
    "M2U00016MPG",
    "M2U00019MPG",
    "M2U00027MPG",
    "M2U00033MPG",
    "M2U00042MPG",
    "M2U00047MPG"
]