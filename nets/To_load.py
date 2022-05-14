from nets.yolo import YoloBody
import torch
import numpy as np


def weights_init(net, init_type='normal', init_gain = 0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)

backboneweights_path = '../model_data/CSPdarknet53_tiny_backbone_weights.pth'
model_path = '../model_data/yolov4_tiny_weights_voc.pth'  # 不加注意力机制的整个网络的权重
model_path_1 = '../model_data/yolov4_tiny_weights_voc_SE.pth'  # 加注意力机制的整个网络的权重


'''
    注意，使用注意力机制的两种形式
    （1）eca0 = eca_block(512)
        out0_branch= eca0(out0_branch)
    （2）self.attention = eca_block(512)
        out0_branch = self.attention(out0_branch)    
    整个注意力机制是进入了网络中的，会进行反向传播更新参数
'''

anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
model = YoloBody(anchors_mask, 80, phi=1) # 以voc数据集20个类为例。  没加载任何权重   不使用注意力机制，phi=0。若要使用se注意力，phi=1。
weights_init(model) # normal初始化权重

# 加载主干权重？
load_backbone = True
if load_backbone: # 通过这种方式加载主干权重
    model.backbone.load_state_dict(torch.load(backboneweights_path))
    print('load backbone_weight done')



# 如果想加载整个网络权重的话，有两种代码方式 1 or 2
style = 2
'''更加推荐第二种方式进行整个网络的权重加载'''

if style == 1:
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path)
    load_key, no_load_key, temp_dict = [], [], {}
    for k, v in pretrained_dict.items():
        if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
            temp_dict[k] = v
            load_key.append(k)
        else:
            no_load_key.append(k)
    model_dict.update(temp_dict)
    model.load_state_dict(model_dict)
    # ------------------------------------------------------#
    #   显示没有匹配上的Key
    # ------------------------------------------------------#
    print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
    print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
    print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")
elif style == 2:
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path, map_location='cuda')
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print('load nets weights done')
else:
    pass

"""
这是为什么呢？为什么呢？为什么呢？为什么呢？为什么呢？为什么呢？为什么呢？为什么呢？为什么呢？为什么呢？为什么呢？为什么呢？为什么呢？为什么呢？

case1：加载的是voc数据集训练出来的权重，且是没有加注意力机制的

    yolobody里面是20，没有注意力机制，方式1：Successful Load Key Num: 118，Fail To Load Key num: 0
                                  方式2：load nets weights done
    yolobody里面是80，没有注意力机制，方式1：Successful Load Key Num: 114，Fail To Load Key num: 4
                                  方式2：load nets weights done
    yolobody里面是20，有注意力机制，方式1：Successful Load Key Num: 118，Fail To Load Key num: 0
                                  方式2：load nets weights done
    yolobody里面是80，有注意力机制，方式1：Successful Load Key Num: 114，Fail To Load Key num: 4
                                  方式2：load nets weights done    
                  
                                  
case2：加载的是voc数据集训练出来的权重，且加了注意力机制

    yolobody里面是20，没有注意力机制，方式1：Successful Load Key Num: 118，Fail To Load Key num: 0
                                  方式2：load nets weights done
    yolobody里面是80，没有注意力机制，方式1：Successful Load Key Num: 114，Fail To Load Key num: 4
                                  方式2：load nets weights done
    yolobody里面是20，有注意力机制，方式1：Successful Load Key Num: 118，Fail To Load Key num: 0
                                  方式2：load nets weights done
    yolobody里面是80，有注意力机制，方式1：Successful Load Key Num: 114，Fail To Load Key num: 4
                                  方式2：load nets weights done   
"""




