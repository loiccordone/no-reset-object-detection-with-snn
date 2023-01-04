import os
import tqdm

import torch
from torch.utils.data import Dataset

import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
from prophesee_utils.io.psee_loader import PSEELoader

def continuous_collate_fn(batch):
    samples = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    samples = [torch.stack([*s]) for s in zip(*samples)]
    targets = [t for t in zip(*targets)]
    
    return [samples, targets]

class DetectionDataset(Dataset):
    def __init__(self, args, mode="train", prefix=""):
        self.tbin = args.tbin
        self.C, self.T = 2 * args.tbin, args.T
        self.sample_size = args.sample_size
        self.h, self.w = args.image_shape
        self.quantization_size = [args.sample_size // args.T, *args.spatial_q]

        self.prefix = prefix
        self.dataset = args.path[-4:].lower()
        self.synchro = prefix.startswith('sync')
        
        save_file_name = f"{prefix}_{mode}_{self.sample_size/1000}_{self.quantization_size[0]/1000}ms_2c_{self.tbin}tbin.pt"
        save_file = os.path.join(args.path,save_file_name)
        
        if os.path.isfile(save_file):
            print(f"Loading {save_file}...")
            self.samples = torch.load(save_file)
            print("File loaded.")
        else:
            print(f"Building {save_file}...")
            data_dir = os.path.join(args.path, mode)
            self.samples = self.build_dataset(data_dir, save_file)
            torch.save(self.samples, save_file)
            print(f"Done! File saved as {save_file}")
            
    def __getitem__(self, index):
        return self.samples[index]
    
    def __len__(self):
        return len(self.samples)
        
    def build_dataset(self, data_dir, save_file):
        raise NotImplementedError("The method build_dataset has not been implemented.")
        
    def create_sample(self, video, boxes):
        raise NotImplementedError("The method create_sample has not been implemented.")
        
    def create_targets(self, boxes):
        torch_boxes = torch.from_numpy(structured_to_unstructured(boxes[['x', 'y', 'w', 'h']], dtype=np.float32))
        
        # keep only last instance of every object per target
        _,unique_indices = np.unique(np.flip(boxes['track_id']), return_index=True) # keep last unique objects
        unique_indices = np.flip(-(unique_indices+1))
        torch_boxes = torch_boxes[[*unique_indices]]
        
        torch_boxes[:, 2:] += torch_boxes[:, :2] # implicit conversion to xyxy
        torch_boxes[:, 0::2].clamp_(min=0, max=self.w)
        torch_boxes[:, 1::2].clamp_(min=0, max=self.h)
        
        # valid idx = width and height of GT bbox aren't 0
        valid_idx = (torch_boxes[:,2]-torch_boxes[:,0] != 0) & (torch_boxes[:,3]-torch_boxes[:,1] != 0)
        torch_boxes = torch_boxes[valid_idx, :]
        
        torch_labels = torch.from_numpy(boxes['class_id']).to(torch.long)
        torch_labels = torch_labels[[*unique_indices]]
        torch_labels = torch_labels[valid_idx]
        
        return {'boxes': torch_boxes, 'labels': torch_labels}
    
    def create_data(self, events):
        if events.size == 0:
            coords = torch.zeros((0,3), dtype=torch.int32)
            feats = torch.zeros((0,self.C), dtype=bool)
        else:
            events['t'] -= events['t'][0]
            events['t'] = events['t'].clip(min=0, max=self.sample_size-1)
            feats = torch.nn.functional.one_hot(torch.from_numpy(events['p']).to(torch.long), self.C)

            coords = torch.from_numpy(
                structured_to_unstructured(events[['t', 'y', 'x']], dtype=np.int32))
            coords = torch.floor(coords/torch.tensor(self.quantization_size))
                
        quantized_h, quantized_w = self.h // self.quantization_size[1], self.w // self.quantization_size[2]
        coords[:, 1].clamp_(min=0, max=quantized_h-1)
        coords[:, 2].clamp_(min=0, max=quantized_w-1)
    
#         # To reproduce MinkowskiEngine coalescing
#         coords, inverse_indices = torch.unique_consecutive(coords, return_inverse=True, dim=0)
#         feats = feats[torch.unique(inverse_indices),:]

        sparse_tensor = torch.sparse_coo_tensor(
            coords.t().to(torch.int32), 
            feats,
            (self.T, quantized_h, quantized_w, self.C),
        )

        sparse_tensor = sparse_tensor.coalesce().to(torch.bool)
            
        return sparse_tensor

class ContinuousSparseDetectionDataset(DetectionDataset):
    def __init__(self, args, mode="train", prefix="continuous"):
        super().__init__(args, mode, prefix)
            
    def build_dataset(self, path, save_file):
        # Remove duplicates (.npy and .dat)
        files = [os.path.join(path, time_seq_name[:-9]) for time_seq_name in os.listdir(path)
                      if time_seq_name[-3:] == 'npy']

        print('Building the Dataset')
        pbar = tqdm.tqdm(total=len(files), unit='File', unit_scale=True)
        samples = []
        count = 0
        for i, file_name in enumerate(files):
            print(f"Processing {file_name}...")
            events_file = file_name + '_td.dat'
            video = PSEELoader(events_file)

            boxes_file = file_name + '_bbox.npy'
            boxes = np.load(boxes_file)
            # Rename 'ts' in 't' if needed (Prophesee GEN1)
            boxes.dtype.names = [dtype if dtype != "ts" else "t" for dtype in boxes.dtype.names]

            samples.append([*self.create_sample(video, boxes)])
            pbar.update(1)

        pbar.close()
        torch.save(samples, save_file)
        print(f"Done! File saved as {save_file}")
        return samples
        
    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)
    
    def create_sample(self, video, boxes, synchro=False):
        boxes['t'] = np.floor(boxes['t']/self.sample_size)
        
        one_clip_tensors, one_clip_targets = [], []
        curr_idx = 0
        empty_boxes = None
        empty_boxes_done = False
        while not video.done:
            try:
                events = video.load_delta_t(self.sample_size)
            except IndexError:
                break
            curr_boxes = boxes[boxes['t'] == curr_idx]
            if not empty_boxes_done and curr_boxes.size == 0:
                empty_boxes = curr_boxes
                empty_boxes_done = True
            
            one_clip_tensors.append(self.create_data(events))
            one_clip_targets.append(self.create_targets(curr_boxes))
            curr_idx +=1

        # Synchro every clip on their first target
        if self.synchro:
            total_nb = len(one_clip_tensors)
            first_target = 0
            for t, targets in enumerate(one_clip_targets):
                if targets['boxes'].numel() != 0:
                    first_target = t
                    break

            one_clip_tensors = one_clip_tensors[first_target:]
            one_clip_targets = one_clip_targets[first_target:]
            
            # pad with empty events and empty targets
            for _ in range(first_target):
                one_clip_tensors.append(self.create_data(np.zeros(0)))
                one_clip_targets.append(self.create_targets(empty_boxes))

        return one_clip_tensors, one_clip_targets
    
class SingleFilesDetectionDataset(DetectionDataset):
    def __init__(self, args, mode="train", prefix=""):
        self.tbin = args.tbin
        self.C, self.T = 2 * args.tbin, args.T
        self.sample_size = args.sample_size
        self.h, self.w = args.image_shape
        self.quantization_size = [args.sample_size // args.T, *args.spatial_q]

        self.prefix = prefix
        self.dataset = args.path[-4:].lower()
        self.synchro = prefix.startswith('sync')
        
        save_dir_name = f"{mode}_{self.sample_size/1000}_{self.quantization_size[0]/1000}ms_2c_{self.tbin}tbin"
        self.save_dir = os.path.join(args.path, save_dir_name)
        
        if len(os.listdir(self.save_dir)) > 0:
            print("Dataset found.")
            self.samples = [os.path.join(self.save_dir, one_file) for one_file in os.listdir(self.save_dir) if one_file.endswith('.pt')]
        else:
            print(f"Building {self.save_dir}...")
            data_dir = os.path.join(args.path, mode)
            self.samples = self.build_dataset(data_dir, self.save_dir)
            print(f"Done! Files saved in {self.save_dir}.")
            
    def build_dataset(self, path, save_dir):
        # Remove duplicates (.npy and .dat)
        files = [os.path.join(path, time_seq_name[:-9]) for time_seq_name in os.listdir(path)
                      if time_seq_name[-3:] == 'npy']

        pbar = tqdm.tqdm(total=len(files), unit='File', unit_scale=True)
        list_files = []
        count = 0
        for i, file_name in enumerate(files):
            print(f"Processing {file_name}...")
            events_file = file_name + '_td.dat'
            video = PSEELoader(events_file)

            boxes_file = file_name + '_bbox.npy'
            boxes = np.load(boxes_file)
            # Rename 'ts' in 't' if needed (Prophesee GEN1)
            boxes.dtype.names = [dtype if dtype != "ts" else "t" for dtype in boxes.dtype.names]
            
            file_name = file_name.split("/")[-1]
            file_name = os.path.join(save_dir, file_name + ".pt")
            print(file_name)
            
            torch.save([*self.create_sample(video, boxes)], file_name)
            list_files.append(file_name)
            pbar.update(1)
            
            if i == 0:
                break

        pbar.close()
        return list_files
        
    def __getitem__(self, index):
        return torch.load(self.samples[index])

    def __len__(self):
        return len(self.samples)
    
    def create_sample(self, video, boxes, synchro=False):
        boxes['t'] = np.floor(boxes['t']/self.sample_size)
        
        one_clip_tensors, one_clip_targets = [], []
        curr_idx = 0
        empty_boxes = None
        empty_boxes_done = False
        while not video.done:
            try:
                events = video.load_delta_t(self.sample_size)
            except IndexError:
                break
            curr_boxes = boxes[boxes['t'] == curr_idx]
            if not empty_boxes_done and curr_boxes.size == 0:
                empty_boxes = curr_boxes
                empty_boxes_done = True
            
            one_clip_tensors.append(self.create_data(events))
            one_clip_targets.append(self.create_targets(curr_boxes))
            curr_idx +=1

        # Synchro every clip on their first target
        if self.synchro:
            total_nb = len(one_clip_tensors)
            first_target = 0
            for t, targets in enumerate(one_clip_targets):
                if targets['boxes'].numel() != 0:
                    first_target = t
                    break

            one_clip_tensors = one_clip_tensors[first_target:]
            one_clip_targets = one_clip_targets[first_target:]
            
            # pad with empty events and empty targets
            for _ in range(first_target):
                one_clip_tensors.append(self.create_data(np.zeros(0)))
                one_clip_targets.append(self.create_targets(empty_boxes))

        return one_clip_tensors, one_clip_targets