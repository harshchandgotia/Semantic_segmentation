from torch.utilis.data.dataset import Dataset 
from torchvision import transformers
from skimage import io
import semantic_data_loader

class MyCustomDataset(Dataset):
    def __init__(self, ---, transforms=None):
        #stuff
        ---
        self.transforms = transforms 

        def __getitem__(self, index):
            #Example 
            img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
            image = io.imread(img_path)
            y_label = torch.tensor(int(self.annotations.iloc[index, 1]))

            
            data = #some data read from a file or image 
            if self.transforms is not None:
                data = self.transforms(data)
            #if the tranforms variable is not None 
            #it applies the operations in the transforms 
            return(img, label)
        
        def __len__(self):
            return count 
        
if __name__ == "__main__":
    transformations = transforms.Compose([transforms.CenterCrop(100), transforms.ToTensor()])
    custom_dataset = MyCustomDataset(..., transformations)


#-------------------------------------------------------------------------
#Another way to use transformers is to define it individually


class MyCustomDataset(Dataset):
    def  __init__(self, ...):
        #Stuff
        ---
        self.center_crop = transforms.CenterCrop(100)
        self.to_tensor = transforms.ToTensor()
        #OR we can just write
        self.transformations = transforms.Compose([transforms.CenterCrop(100), trasnforms.ToTensor()])

    def __getitem__(self, index):
        data = #Some data

        data = self.center_crop(data)
        data = self.to_tensor(data)

        #OR we can call

        data = self.transformations(data)

    def __len__(self):
        return count
    
if __name__ == '__main__':
    custom_dataset = MyCustomDataset(...)
