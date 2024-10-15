import semantic_data_loader
import pandas as pd
import torch 
from torch.utilis.data import Dataset
from skimage import io


#so this following data loader is for an example in which we have a folder of images and a csv file in which the 
#the names of the images are stored in one column and the outcome in the next 

class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform 


    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self,index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = io.imread(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))

        if self.transform:
            image = self.transform(image)
    

        return (image, y_label)

    

#now we can call this class in our main code the following way 

from Custom_Data_Loader import CustomDataset

dataset = CustomDataset(csv_file = "file_name", root_dir= "file_name", transform = transforms.ToTensor())

train_set, test_set = torch.utilis.data.random_split(dataset, [20000,5000])
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

