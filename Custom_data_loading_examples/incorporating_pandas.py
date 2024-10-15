'''The first example is of having a csv file like following, that contains 
file name, label(class) and an extra operation indicator and depending 
on this extra operation flag we do some operation on the image'''

class CustomDatasetFromImages(Dataset):
    def __init__(self, csv_path):
        self.to_tensor = transforms.ToTensor()
        self.data_info = pd.read_csv(csv_path, header=None)
        self.image_arr = np.asarray(self.data_info.iloc[:,0])
        self.label_arr = np.asarray(self.data_info.iloc[:,1])
        self.operation_arr = np.asarray(self.data_info.iloc[:,2])

        self.data_len = len(self.data_info.index)

    
    def __getitem__(self,index):
        single_image_name = self.image_arr[index]

        img_as_img = Image.open(single_image_name)
        some_operation = self.operation_arr[index]

        if some_operation:
            #Do some operation
            pass

        #transform image to tensor
        img_as_tensor = self.to_tensor(img_as_img)

        single_image_name = self.label_arr[index]

        return(img_as_tensor, single_image_name)
    
    def __len__(self):
        return self.data_len
    
if __name__ == "__main__":
    custom = CustomDatasetFromImages(...)


#---------------------------------------------------------------------
#Incorporating pandas with more logic 
#another example might be reading an image from CSV where the value of each pixel is listed in a column.

class CustomDatasetFromCSV(Dataset):
    def __init__(self, csv_path, height, width, transforms=None):
        self.data = pd.read_csv(csv_path)
        self.labels = np.asarray(self.data.iloc[:,0])
        self.height = height 
        self.width = width 
        self.transforms = transform

    def _getitem__(self, index):
        single_image_label = self.labels[index]
        img_as_np = np.asarray(self.data.iloc[index][1:]).reshape(28,28).astype('uint8')
        # Convert image from numpy array to PIL image, mode 'L' is for grayscale
        img_as_img = Image.fromarray(img_as_np)
        img_as_img = img_as_img.convert('L')

        if self.transforms is not None:
            img_as_tensor = self.trasnforms(img_as_img)

        return (img_as_tensor, single_image_label)
    
    def __len__(self):
        return len(self.data.index)
    

if __name__ == "__main__":
    transformations = transforms.Compose([trasnforms.ToTensor()])
    customdatset = CustomDatasetFromImages(...)

     # Define data loader
    mn_dataset_loader = torch.utils.data.DataLoader(dataset=custom_mnist_from_csv,
                                                    batch_size=10,
                                                    shuffle=False)
    
    for images, labels in mn_dataset_loader:
        # Feed the data to the model

         

