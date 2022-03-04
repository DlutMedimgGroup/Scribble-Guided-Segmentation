from multiprocessing import Queue
from time import sleep, time
from multiprocessing import Event
from multiprocessing import Process
import SimpleITK as sitk
import numpy as np
import GeodesicDis
from batchgenerators.utilities.file_and_folder_operations import *
from threadpoolctl import threadpool_limits


def generate(case_all_data, output, properties):
    data_file_name = properties['data_file_name']
    input_data_name = properties['input_data_name']
    epoch_int = properties['epoch_int']
    cache_path = properties['cache_path']
    num_modalities_label = properties['num_modalities_label']

    input_array = case_all_data[:-num_modalities_label, :, :, :]
    # Output Intermediate File
    originlabel_filename = data_file_name[:-4] + "-originlabel.npy"
    if epoch_int == 0:
        originlabel = output.argmax(0).astype(np.int16)                    
        np.save(originlabel_filename, originlabel)
        # debug
        maybe_mkdir_p(cache_path)
        image_input_filename = join(cache_path, str(epoch_int) + "-" + input_data_name + "-image.nii.gz")
        image_image = sitk.GetImageFromArray(input_array[0, :, :, :])
        sitk.WriteImage(image_image, image_input_filename)
        seed_filename = join(cache_path, str(epoch_int) + "-" + input_data_name + "-seed.nii.gz")
        seed_array = case_all_data[-1, :, :, :].astype(np.int16).copy()
        seed_array = seed_array+1
        seed_image = sitk.GetImageFromArray(seed_array)
        sitk.WriteImage(seed_image, seed_filename)
        
        # netout prop
        for c in range(3):
            prop_image = sitk.GetImageFromArray(output[c])
            prop_filename = join(cache_path, str(epoch_int) + "-" + input_data_name + "-netout_prop-" + str(c) +".nii.gz")
            sitk.WriteImage(prop_image, prop_filename)

    else:
        originlabel = np.load(originlabel_filename, "r")
    netout_filename = join(cache_path, str(epoch_int) + "-" + input_data_name + "-netout.nii.gz")
    netout_image = sitk.GetImageFromArray(output.argmax(0).astype(np.int16))
    sitk.WriteImage(netout_image, netout_filename)

    # Geodesic Distance
    generater = GeodesicDis.GeoDisScibble()
    generater.SetInputImage((input_array[0, :, :, :]))
    generater.SetInputSeedMap(case_all_data[-1, :, :, :].astype(np.int))
    generater.SetOriginLabelMap(originlabel)
    generater.SetSpacing([1, 1, 1])
    generater.SetOrigin([0, 0, 0])
    generater.SetPropertyMap(output)
    generater.SetProperties(0.01, 1, 0.1, 5)
    generater.SetSortPeriod(10000)
    generater.DebugOff()
    generater.Generate()

    # output
    rndst = np.random.RandomState(1234)
    fakelabel = np.empty([2]+list(input_array.shape[1:]), dtype=np.int16)
    fakelabel[0] = generater.GetToughLabelMap()
    fakelabel[1] = generater.GetConfidenceMap() # In ConfidenceMap, Nagetive value means unfocus area
    geodesic_dis = np.empty([3]+list(input_array.shape[1:]), dtype=np.float16)
    geodesic_dis = generater.GetGeodesicDis()

    
    focus = np.argwhere(fakelabel[1] > 0)
    if len(focus) > 0:
        target_num_samples = min(10000, len(focus))
        target_num_samples = max(target_num_samples, int(np.ceil(len(focus) * 0.01)))
        focus_selected = focus[rndst.choice(len(focus), target_num_samples, replace=False)]
    else:
        focus_selected = []

    labels = list(np.unique(fakelabel[0]))
    class_locs = {}    
    for c in labels:
        if c == 0:
            continue
        all_locs = np.argwhere(fakelabel[0] == c)
        if len(all_locs) == 0:
            class_locs[c] = []
            continue
        target_num_samples = min(10000, len(all_locs))
        target_num_samples = max(target_num_samples, int(np.ceil(len(all_locs) * 0.01)))
        selected = all_locs[rndst.choice(len(all_locs), target_num_samples, replace=False)]
        class_locs[c] = selected

    pkl_dict = {"focus_area": focus_selected, "class_locations": class_locs}
    pkl_filename = data_file_name[:-4] + "-pkl.pkl"
    save_pickle(pkl_dict, pkl_filename)
    fakelabel[1] = np.abs(fakelabel[1])
    output_filename = data_file_name[:-4] + "-fakelabel.npy"
    np.save(output_filename, fakelabel)

    # debug
    fakelabel_filename = join(cache_path, str(epoch_int) + "-" + input_data_name + "-fakelabel.nii.gz")
    toughlabel_image = sitk.GetImageFromArray(fakelabel[0])
    sitk.WriteImage(toughlabel_image, fakelabel_filename)
    confidence_filename = join(cache_path, str(epoch_int) + "-" + input_data_name + "-confidence.nii.gz")
    confidence_image = sitk.GetImageFromArray(fakelabel[1])
    sitk.WriteImage(confidence_image, confidence_filename)
    for c in labels:
        geodesic_image = sitk.GetImageFromArray(geodesic_dis[c])
        geodesic_filename = join(cache_path, str(epoch_int) + "-" + input_data_name + "-geodes-" + str(c) +".nii.gz")
        sitk.WriteImage(geodesic_image, geodesic_filename)

    del generater

def producer(queue, thread_id, abort_event, wait_time: float = 0.02):
    try:
        while True:
            if not abort_event.is_set():
                if queue.qsize() > 0:
                    # print("start to calculate geodesic: " + str(queue.qsize()) + " id: " + str(thread_id))
                    item = queue.get()
                    if item == 'finish':
                        break
                    case_all_data = item[0]
                    output = item[1]
                    properties = item[2]                    
                    generate(case_all_data, output, properties)
                    # print("finished: " + str(thread_id))
                else:
                    sleep(wait_time)
            else:
                return
    except KeyboardInterrupt:
        abort_event.set()
        return
    except Exception as e:
        print("Exception in background worker %d:\n" % thread_id, e)
        abort_event.set()
        return

class MultiThreadedGeodesic(object):
    """ Calulate and save geodesic fake label multi threadedly
    """

    def __init__(self, num_processes=4, num_cached=4, timeout=10, wait_time=0.02):
        super().__init__()
        self.num_processes = num_processes
        self.num_cached = num_cached
        self.timeout = timeout
        self.wait_time = wait_time

        self.out_queue = Queue(self.num_cached)
        self.abort_event = Event()
        self._processes = []

    def add_result(self, item):
        not_finished = True
        while(not_finished):
            try:
                if self.abort_event.is_set():
                    # print('abort event is set')
                    return
                if not self.out_queue.full():
                    self.out_queue.put(item)
                    not_finished = False
                    # print('add reslut: ' + str(self.out_queue.qsize()))
                else:
                    sleep(self.wait_time)
                    continue
            except KeyboardInterrupt:
                self.abort_event.set()
                raise KeyboardInterrupt
    
    def start(self):
        if len(self._processes) != self.num_processes:
            self.stop()
            self.abort_event.clear()
        else:
            print("MultiThreadedGenerator Warning: start() has been called but workers are already running")

        with threadpool_limits(limits=1, user_api="blas"):
            for i in range(self.num_processes):
                self._processes.append(Process(target=producer, args=(
                    self.out_queue, i, self.abort_event)))
                self._processes[-1].daemon = True
                self._processes[-1].start()
            # self.out_queue = Queue(self.num_cached)

    def stop(self):
        self.abort_event.set()
        if len(self._processes) != 0:
            [i.terminate() for i in self._processes]
        # if self.out_queue != None:
        #     self.out_queue.close()
        #     self.out_queue.join_thread()
        #     self.out_queue = None

    def wait_finish(self):
        not_finished = True
        while(not_finished):
            if not self.out_queue.empty():
                sleep(self.wait_time)
            else:
                for i in range(self.num_processes):
                    self.out_queue.put('finish')
                [i.join() for i in self._processes]
                not_finished = False



