import torch
import torchvision
from model.backbone import sfnet64
from torch.nn import DataParallel
import torch.nn as nn
from torchvision import transforms as T
from PIL import Image
from collections import OrderedDict
import torchvision.transforms.functional as F

class uberFace(nn.Module):

    def __init__(self, config):
        super(uberFace, self).__init__()
        args = {
            "in_channel": 3,
            "channels": [64, 128, 256, 512],
            "out_channel": 512}
        self.model = sfnet64(**args)
        # self.model = DataParallel(self.model)
        self.model.to(torch.device("cpu"))

        # state_dict = torch.load(config['model_path'], map_location=torch.device('cpu'))
        # new_state_dict = OrderedDict()
        # for k, v in state_dict.items():
        #     if 'module' not in k:
        #         k = 'module.' + k
        #     else:
        #         k = k.replace('features.module.', 'module.features.')
        #     new_state_dict[k] = v

        state_dict = torch.load(config['model_path'], map_location=torch.device('cpu'))
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[7:]
            # else:
            #     k = k.replace('features.module.', 'module.features.')
            new_state_dict[k] = v

        self.model.load_state_dict(new_state_dict)
        #self.model.load_state_dict(torch.load(config['model_path'], map_location=torch.device('cpu')))
        self.model.eval()
        self.input_shape = config['input_shape']

    def forward(self, x):
        x = F.resize(x, self.input_shape[1:])
        x = (x - 127.5)/127.5
        out = self.model(x)
        return out


if __name__ == "__main__":

    image_path = r'/Users/ahmet.tuysuzoglu/projects/face/data/lfw/lfw-deepfunneled/Aaron_Sorkin/Aaron_Sorkin_0001.jpg'
    data = Image.open(image_path)
    data = T.PILToTensor()(data)
    data = data[None, :]

    with open(image_path, 'rb') as f:
        data_buffer = torch.tensor(torch.frombuffer(f.read(), dtype=torch.uint8))

    img_decoded = torchvision.io.decode_image(data_buffer, mode=torchvision.io.ImageReadMode.RGB)

    test_model_path = \
        r'/Users/ahmet.tuysuzoglu/projects/det_face/opensphere/saved_models/20221111_081942/models/backbone_350000.pth'
    save_path = r'./saved_models/tsmodels/recognition_v1_local.pt'

    dummy_data = torch.rand(1, 3, 112, 112)

    config = {'model_path': test_model_path, 'input_shape': [3, 112, 112]}
    uberFace_instance = uberFace(config)
    output_native = uberFace_instance(data)

    traced_script_module = torch.jit.script(uberFace_instance)
    traced_script_module.save(save_path)
    output_traced = traced_script_module(data)

    print('here')
    print(torch.norm(output_native.squeeze() - output_traced.squeeze()))
    # output = traced_script_module(torch.ones(20, 3, 112, 112))
    # print(output.shape)

