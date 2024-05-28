"""
                        if self.model == "plantnet":
                            if classification_results[0] == "plant":
                                classes_path = class_paths["plant"]
                                classes = get_classes(classes_path)
                                self.classification_model = PlantNet(classes=classes)
                                plantnet_results = self.classification_model.identify(crop)
                                print(plantnet_results)
                            elif classification_results[0] == "fern":
                                classes_path = class_paths["fern"]
                                classes = get_classes(classes_path)
                                self.classification_model = PlantNet(classes=classes)
                                plantnet_results = self.classification_model.identify(crop)
                                print(plantnet_results)
                            elif classification_results[0] == "flower":
                                classes_path = class_paths["full"]
                                classes = get_classes(classes_path)
                                self.classification_model = PlantNet(classes=classes, organs="flower")
                                plantnet_results = self.classification_model.identify(crop)
                                print(plantnet_results)
                            elif classification_results[0] == "berries":
                                classes_path = class_paths["full"]
                                classes = get_classes(classes_path)
                                self.classification_model = PlantNet(classes=classes, organs="fruit")
                                plantnet_results = self.classification_model.identify(crop)
                                print(plantnet_results)
                            elif classification_results[0] == "palm":
                                classes_path = class_paths["palm"]
                                classes = get_classes(classes_path)
                                self.classification_model = PlantNet(classes=classes)
                                plantnet_results = self.classification_model.identify(crop)
                                print(plantnet_results)
                            else:
                                classes_path = class_paths["full"]
                                classes = get_classes(classes_path)
                                self.classification_model = PlantNet(classes=classes)
                                plantnet_results = self.classification_model.identify(crop)
                                print(plantnet_results)
"""