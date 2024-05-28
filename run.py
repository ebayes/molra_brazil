from molra.main import MOLRA

high_level_taxa=["plant", "palm", "flower", "berries"]
box_threshold = 0.25
text_threshold = 0.6

pipeline = MOLRA(
    detection={
        "box_threshold": box_threshold,
        "text_threshold": text_threshold,
        "classes": high_level_taxa 
    },
    classification=True,
    type="below_canopy",
    input_dir="./input/above_canopy",
    # model="plantnet",
    save_annotations=True,
    cluster=True
)

pipeline.run()



"""


# Below canopy settings
high_level_taxa = ["individual tree"]
box_threshold = 0.25
text_threshold = 0.6

pipeline = MOLRA(
    type="above_canopy",
    input_dir="./input/above_canopy",
    detection={
        "classes": high_level_taxa,
        "box_threshold": box_threshold,
        "text_threshold": text_threshold,
    },
    classification=True,
    high_level_taxa=high_level_taxa,
    save_annotations=True,
    cluster=True,
)

pipeline.run()
"""