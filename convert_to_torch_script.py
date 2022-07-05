import torch
import torchvision
from model.backbone import sfnet64_deprecated
from torch.nn import DataParallel
import torch.nn as nn
from torchvision import transforms as T
from PIL import Image

class uberFace(nn.Module):

    def __init__(self, model_path, input_shape=(3, 112, 112)):
        super(uberFace, self).__init__()
        self.model = sfnet64_deprecated()
        self.model = DataParallel(self.model)
        self.model.to(torch.device("cpu"))
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

        self.input_shape = input_shape
        self.transforms = T.Compose([
            T.Resize(self.input_shape[1:]),
        ])

    def forward(self, x):
        x = self.transforms(x)
        out = self.model(x)
        out_flipped = self.model(torch.fliplr(x))
        return torch.cat((out, out_flipped), dim=1)


if __name__ == "__main__":

    test_model_path = r'/Users/ahmet.tuysuzoglu/projects/det_face/opensphere/saved_models/20220621_165040/models/backbone_60000.pth'
    model = sfnet64_deprecated()
    model = DataParallel(model)
    model.to(torch.device("cpu"))
    model.load_state_dict(torch.load(test_model_path, map_location=torch.device('cpu')))

    example = torch.rand(1, 3, 112, 112)
    traced_script_module = torch.jit.trace(model, example)

    # output = traced_script_module(torch.ones(20, 3, 112, 112))
    # print(output.shape)

    image_path = r'/Users/ahmet.tuysuzoglu/projects/face/data/lfw/lfw-deepfunneled/Aaron_Sorkin/Aaron_Sorkin_0001.jpg'
    data = Image.open(image_path)
    data = T.ToTensor()(data)
    uberFace_instance = uberFace(test_model_path)
    #output = uberFace_instance(data)
    data = data[None, :]
    traced_script_module = torch.jit.trace(uberFace_instance, data)
    output = traced_script_module(data)
    print(output.shape)

