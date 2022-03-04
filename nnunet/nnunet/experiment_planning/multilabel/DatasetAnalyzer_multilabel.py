from nnunet.experiment_planning.DatasetAnalyzer import *

class DatasetAnalyzer_multilabel(DatasetAnalyzer):
    def get_modalities_label(self):
        datasetjson = load_json(join(self.folder_with_cropped_data, "dataset.json"))
        modalities = datasetjson["modality_label"]
        modalities = {int(k): modalities[k] for k in modalities.keys()}
        return modalities

    def analyze_dataset(self, collect_intensityproperties=True):
        '''add modalities_label to dataset_properties dict
        '''

        # get all spacings and sizes
        sizes, spacings = self.get_sizes_and_spacings_after_cropping()

        # get all classes and what classes are in what patients
        # class min size
        # region size per class
        classes = self.get_classes()
        all_classes = [int(i) for i in classes.keys() if int(i) > 0]

        # modalities
        modalities = self.get_modalities()
        modalities_label = self.get_modalities_label()

        # collect intensity information
        if collect_intensityproperties:
            intensityproperties = self.collect_intensity_properties(len(modalities))
        else:
            intensityproperties = None

        # size reduction by cropping
        size_reductions = self.get_size_reduction_by_cropping()

        dataset_properties = dict()
        dataset_properties['all_sizes'] = sizes
        dataset_properties['all_spacings'] = spacings
        dataset_properties['all_classes'] = all_classes
        dataset_properties['modalities'] = modalities  # {idx: modality name}
        dataset_properties['modalities_label'] = modalities_label
        dataset_properties['intensityproperties'] = intensityproperties
        dataset_properties['size_reductions'] = size_reductions  # {patient_id: size_reduction}

        save_pickle(dataset_properties, join(self.folder_with_cropped_data, "dataset_properties.pkl"))
        return dataset_properties