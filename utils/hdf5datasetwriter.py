import h5py
import os

'''
Based on Deep Learning for Computer Vision: Vol 2.
Adrian Rosebrock. 2019
'''

class HDF5DatasetWriter:
    def __init__(self, dims, outputPath, dataKey="images", bufSize=1000):
        if os.path.exists(outputPath):
            raise ValueError("The supplied 'outputPath' already "
                             "exists and cannot be overwritten. Manually "
                             "delete the file before continuing", outputPath)
        # open the HDF5 database for writing and create two datasets:
        # one to store the images/features and another for class labels
        self.db = h5py.File(outputPath, "w")
        self.data = self.db.create_dataset(dataKey, dims, dtype="float")
        self.labels = self.db.create_dataset("labels", (dims[0],),
                                             dtype = "int")
        # store buffer size, then initialize the buffer itself
        # along with the index into the datasets
        self.bufSize = bufSize
        self.buffer = {"data": [], "labels": []}
        self.idx = 0
        """When using VGG16 architecture and taking the outputs after 
        the final pool layer, the dim is 512 x 7 x 7 = 25088. Therefore
        dims = (N, 25088) where N is total number of images in the dataset"""

    def add(self, rows, labels):
        # add the rows and labels to the buffer
        self.buffer["data"].extend(rows)
        self.buffer["labels"].extend(labels)

        # check to see if the buffer needs to be flushed to disk
        if len(self.buffer["data"]) >= self.bufSize:
            self.flush()

    def flush(self):
        # write the buffers to disk then reset the buffer
        i = self.idx + len(self.buffer["data"])
        self.data[self.idx:i] = self.buffer["data"]
        self.labels[self.idx:i] = self.buffer["labels"]
        self.idx = i
        self.buffer = {"data": [], "labels": []}

    def storeClassLabels(self, classLabels):
        # create a dataset to store actual class label names
        dt = h5py.special_dtype(vlen=str)
        labelSet = self.db.create_dataset("label_names",
                                          (len(classLabels),), dtype = dt)
        labelSet[:] = classLabels

    def close(self):
        # check t see if there are any entries in buffer
        # that need to be flushed
        if len(self.buffer["data"]) > 0:
            self.flush()

        self.db.close()
