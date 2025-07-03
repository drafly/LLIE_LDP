import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import logging
from torchvision.utils import save_image
from guided_diffusion.nn import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)

class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        # use_checkpoint=True,
        use_new_attention_order=True,
        # use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels  # 输入通道数
        if num_head_channels == -1:
            self.num_heads = num_heads  # 如果未指定，每个头的通道数为num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels  # 计算头数

        self.use_checkpoint = use_checkpoint  # 是否使用梯度检查点
        self.norm = normalization(channels)  # 归一化层
        self.qkv = conv_nd(1, channels, channels * 3, 1)  # QKV投影
        if use_new_attention_order:
            # 在拆分头部之前拆分qkv
            self.attention = QKVAttention(self.num_heads)
        else:
            # 在拆分qkv之前拆分头部
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))  # 输出投影

        # 用于将通道数压缩回原始通道数的卷积层
        # self.compress_conv = nn.Conv2d(channels * 2, channels, kernel_size=1)

    def forward(self, x):
        # 使用梯度检查点来降低内存占用

        return checkpoint(self._forward, (x, ), self.parameters(), True)

    def _forward(self, x):
        dtype = x.device  # 记录输入的设备（CPU或GPU）
        x = x.to(torch.float32)  # 转换为浮点数格式

        b, c, *spatial = x.shape  # 获取批大小、通道数和空间维度
        x = x.reshape(b, c, -1)  # 将x重塑为 [N, C, H*W] 形式
        qkv = self.qkv(self.norm(x))  # 归一化后进行QKV投影
        h = self.attention(qkv)  # 计算注意力
        h = self.proj_out(h)  # 进行输出投影
        x =  (x + h).reshape(b, c, *spatial)  # 将注意力结果加回原始输入，并重塑

        # 压缩回原始通道数
        # x = self.compress_conv(x)

        return x.to(dtype)  # 返回与原始输入相同设备的数据

def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape  # 从y的第一个元素中提取批大小b、通道数c和空间维度
    num_spatial = int(np.prod(spatial))  # 计算空间维度的总数，即H * W

    # 我们执行两次矩阵乘法，操作数相同。
    # 第一次计算权重矩阵，第二次计算值向量的组合。
    matmul_ops = 2 * b * (num_spatial ** 2) * c  # 计算所需的乘法和加法运算数

    model.total_ops += torch.DoubleTensor([matmul_ops])  # 将计算的操作数添加到模型的总操作数中


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/output heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads  # 初始化时定义头的数量

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
                     输入张量，包含查询、键和值的组合。
        :return: an [N x (H * C) x T] tensor after attention.
                 输出张量，包含经过注意力机制处理后的结果。
        """
        bs, width, length = qkv.shape  # 获取输入张量的批量大小、宽度和长度
        assert width % (3 * self.n_heads) == 0  # 确保宽度可以被头数和三分之一整除
        ch = width // (3 * self.n_heads)  # 计算每个头的通道数

        # 将输入张量重塑为包含 Q、K 和 V 的三个部分
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)

        # 计算缩放因子
        scale = 1 / math.sqrt(math.sqrt(ch))

        # 计算注意力权重，使用爱因斯坦求和约定（einsum）
        weight = torch.einsum("bct,bcs->bts", q * scale, k * scale)

        # 对权重应用 softmax，得到注意力分布
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)

        # 使用注意力权重对值进行加权求和
        a = torch.einsum("bts,bcs->bct", weight, v)

        # 重塑输出为原始的批量大小和头数
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)  # 静态方法，用于计算该模块的FLOPs


class QKVAttention(nn.Module):
    """
    一个执行QKV注意力机制的模块，并以不同的顺序进行分割。
    """

    def __init__(self, n_heads):
        """
        初始化QKVAttention模块。

        :param n_heads: 注意力头的数量。
        """
        super().__init__()
        self.n_heads = n_heads  # 设置注意力头的数量

    def forward(self, qkv):
        """
        应用QKV注意力机制。

        :param qkv: 一个形状为[N x (3 * H * C) x T]的张量，包含查询（Q）、键（K）和值（V）。
        :return: 一个形状为[N x (H * C) x T]的张量，经过注意力处理后的输出。
        """
        bs, width, length = qkv.shape  # 获取输入张量的批量大小、宽度和长度
        assert width % (3 * self.n_heads) == 0  # 确保宽度可以被头数整除
        ch = width // (3 * self.n_heads)  # 每个头的通道数

        # 将qkv张量拆分为Q、K和V
        q, k, v = qkv.chunk(3, dim=1)  # 沿着宽度维度分割成三部分

        scale = 1 / math.sqrt(math.sqrt(ch))  # 计算缩放因子，防止点积过大
        # 计算查询和键之间的相似度权重
        weight = torch.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # 使用爱因斯坦求和约定计算点积

        # 对权重应用softmax函数，得到概率分布
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        # 计算加权值
        a = torch.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))

        # 返回处理后的张量，调整形状以匹配输入格式
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        """
        计算模型在前向传播中的浮点运算数量。

        :param model: 当前模型。
        :param _x: 输入张量（未使用）。
        :param y: 输出张量（未使用）。
        :return: 浮点运算数量。
        """
        return count_flops_attn(model, _x, y)


class UNetSeeInDark(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, **kwargs):
        super(UNetSeeInDark, self).__init__()
        self.resid = kwargs.pop("resid", False)
        self.with_photon = kwargs.get('with_photon', False)
        self.concat_origin = kwargs.get('concat_origin', False)
        self.adaptive_res_and_x0 = kwargs.get('adaptive_res_and_x0', False)
        self.in_channels = in_channels
        times = 1
        if self.with_photon:
            times += 1
        if self.concat_origin:
            times += 1

            
        if self.resid: print("[i] predict noise instead of clean image")#logging.info("predict noise instead of clean image")
        #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.conv1_1 = nn.Conv2d(in_channels*times, 32, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.attention3 = AttentionBlock(128, num_heads=4)  # 下采样第三层
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        
        self.conv4_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.attention4 = AttentionBlock(256, num_heads=4)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        
        self.conv5_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.attention5 = AttentionBlock(512, num_heads=4)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        
        self.upv6 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv6_1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.attention6 = AttentionBlock(256, num_heads=4)
        self.conv6_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        
        self.upv7 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv7_1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.attention7 = AttentionBlock(128, num_heads=4)
        self.conv7_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        
        self.upv8 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv8_1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv8_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        
        self.upv9 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv9_1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv9_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        
        
        if self.adaptive_res_and_x0:
            print("[i] Using adaptive_res_and_x0. ")
            out_channels = out_channels * 2 + 1
            self.out_channels = out_channels
        
        self.conv10_1 = nn.Conv2d(32, out_channels, kernel_size=1, stride=1)
        

            
    def forward(self, x):
        conv1 = self.lrelu(self.conv1_1(x))
        conv1 = self.lrelu(self.conv1_2(conv1))
        pool1 = self.pool1(conv1)
        
        conv2 = self.lrelu(self.conv2_1(pool1))
        conv2 = self.lrelu(self.conv2_2(conv2))
        pool2 = self.pool1(conv2)
        
        conv3 = self.lrelu(self.conv3_1(pool2))
        conv3 = self.lrelu(self.conv3_2(conv3))
        # 在下采样第三层应用注意力机制
        conv3 = self.attention3(conv3)
        pool3 = self.pool1(conv3)

        
        conv4 = self.lrelu(self.conv4_1(pool3))
        conv4 = self.lrelu(self.conv4_2(conv4))
        # 在下采样第三层应用注意力机制
        conv4 = self.attention4(conv4)
        pool4 = self.pool1(conv4)
        
        conv5 = self.lrelu(self.conv5_1(pool4))
        conv5 = self.attention5(conv5)
        conv5 = self.lrelu(self.conv5_2(conv5))
        
        up6 = self.upv6(conv5)
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.lrelu(self.conv6_1(up6))
        conv6 = self.attention6(conv6)
        conv6 = self.lrelu(self.conv6_2(conv6))
        
        up7 = self.upv7(conv6)
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.lrelu(self.conv7_1(up7))
        conv7 = self.attention7(conv7)
        conv7 = self.lrelu(self.conv7_2(conv7))
        
        up8 = self.upv8(conv7)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.lrelu(self.conv8_1(up8))
        conv8 = self.lrelu(self.conv8_2(conv8))
        
        up9 = self.upv9(conv8)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.lrelu(self.conv9_1(up9))
        conv9 = self.lrelu(self.conv9_2(conv9))
        
        conv10= self.conv10_1(conv9)
        # out = nn.functional.pixel_shuffle(conv10, 2)
        
        if self.adaptive_res_and_x0:
            mask = torch.sigmoid(conv10[:,[0],:,:])
            ## mask  = conv10[:,[0],:,:]  + (conv10[:,[0],:,:].clip(0, 1) - conv10[:,[0],:,:]).detach()
            # mask = 1 -  conv10[:,[0],:,:].clip(0,1)
            
            conv10 = conv10[:,1:]
            out = conv10[:,:self.out_channels//2]
            resid = conv10[:, self.out_channels//2:]
            out_by_resid = (resid+x[:, :self.in_channels]).clip(0,1)
            out_final = out.clip(0,1) * mask + out_by_resid * (1-mask)
            metadata = {"out_by_resid": out_by_resid, "out":out, "mask": mask}
            
        else:
            out = conv10
            if self.resid:
                out = x - out 
            out_final = out.clip(0, 1)        
            metadata = None
        return out_final, metadata
    
        # iter = 1 
        # save_image(out[:,:3], f"d_x0_{iter}.jpg")
        # save_image(resid[:,:3], f"d_res_{iter}.jpg")
        # save_image((resid+x[:, :self.in_channels])[:,:3], f"d_res_x{iter}.jpg")
        # save_image(mask, f"d_mask{iter}.jpg")
        # save_image(conv10[:,:self.out_channels//2][:,:3], f"d_x{iter}.jpg")
        # save_image(x[:, :3], f"d_in{iter}.jpg")
        # save_image(out_final[:, :3], f"d_final{iter}.jpg")
        
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.02)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)

    def lrelu(self, x):
        outt = torch.max(0.2*x, x)
        return outt
