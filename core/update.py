import torch
import torch.nn as nn
import torch.nn.functional as F


class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=2):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, output_dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


# class DepthHead(nn.Module):
#     def __init__(self, input_dim=128+256, hidden_dim=512, output_dim=1):
#         super(DepthHead, self).__init__()
#         self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
#         self.conv2 = nn.Conv2d(hidden_dim, output_dim, 3, padding=1)
#         self.relu = nn.ReLU(inplace=True)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x, d):
#         x = torch.cat((x, d), dim=1)
#         return self.sigmoid(self.conv2(self.relu(self.conv1(x))))

# class DepthHead(nn.Module):
#     def __init__(self, input_dim=128+256, output_dim=1):
#         super(DepthHead, self).__init__()
#         self.upconv1 = nn.ConvTranspose2d(in_channels=384, out_channels=128, kernel_size=4, stride=2, padding=1)
#         self.conv1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)  # Regular convolution for refinement
#         self.relu1 = nn.ReLU()

#         self.upconv2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
#         self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
#         self.relu2 = nn.ReLU()

#         self.upconv3 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)
#         self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
#         self.relu3 = nn.ReLU()

#         self.final_conv = nn.Conv2d(32, output_dim, kernel_size=3, stride=1, padding=1)

#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x, d):
#         x = torch.cat((x, d), dim=1)

#         # Upsample 1: 30x40 -> 60x80
#         x = self.upconv1(x)
#         x = self.conv1(x)
#         x = self.relu1(x)
        
#         # Upsample 2: 60x80 -> 120x160
#         x = self.upconv2(x)
#         x = self.conv2(x)
#         x = self.relu2(x)
        
#         # Upsample 3: 120x160 -> 240x320
#         x = self.upconv3(x)
#         x = self.conv3(x)
#         x = self.relu3(x)

#         # Final layer to get output of size Bx1x240x320
#         x = self.sigmoid(self.final_conv(x))
        
#         return x


# class DepthMaskHead(nn.Module):
#     def __init__(self, output_dim=1, output_size=(296,512)):
#         super(DepthMaskHead, self).__init__()
#         self.output_size = output_size
        
#         # self.fuserOne = torch.nn.MultiheadAttention(embed_dim=64, num_heads=1, batch_first=True)
#         # self.fuserTwo = torch.nn.MultiheadAttention(embed_dim=64, num_heads=1, batch_first=True)
#         self.fuserThree = torch.nn.MultiheadAttention(embed_dim=96, num_heads=1, batch_first=True)
#         self.fuserFour = torch.nn.MultiheadAttention(embed_dim=128, num_heads=1, batch_first=True)
#         self.fuserFive = torch.nn.MultiheadAttention(embed_dim=256, num_heads=1, batch_first=True)
        
        
#         # self.netScoreOne = torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)
#         # self.netScoreTwo = torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)
#         self.netScoreThr = torch.nn.Conv2d(in_channels=96, out_channels=1, kernel_size=1, stride=1, padding=0)
#         self.netScoreFou = torch.nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0)
#         self.netScoreFiv = torch.nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0)

#         self.netCombine = torch.nn.Sequential(
#             torch.nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1, padding=0),
#             torch.nn.Sigmoid()
#         )
    
#     def forward(self, fmap1_one, fmap1_two, fmap1_three, fmap1_four, fmap1_five, fmap2_one, fmap2_two, fmap2_three, fmap2_four, fmap2_five):
        
#         # fmap_one = torch.cat((fmap1_one,fmap2_one),dim=1)
#         # fmap_two = torch.cat((fmap1_two,fmap2_two),dim=1)
#         # fmap_three = torch.cat((fmap1_three,fmap2_three),dim=1)
#         # fmap_four = torch.cat((fmap1_four,fmap2_four),dim=1)
#         # fmap_five = torch.cat((fmap1_five,fmap2_five),dim=1)
#         B, _, H, W = fmap1_one.shape
#         # fmap_one, _ = self.fuserOne(fmap2_one.view(B, -1, H*W).permute(0, 2, 1), fmap1_one.view(B, -1, H*W).permute(0, 2, 1), fmap1_one.view(B, -1, H*W).permute(0, 2, 1))
#         # fmap_two, _ = self.fuserTwo(fmap2_two.view(B, -1, H*W).permute(0, 2, 1), fmap1_two.view(B, -1, H*W).permute(0, 2, 1), fmap1_two.view(B, -1, H*W).permute(0, 2, 1))
#         fmap_three, _ = self.fuserThree(fmap2_three.view(B, -1, H//2*W//2).permute(0, 2, 1), fmap1_three.view(B, -1, H//2*W//2).permute(0, 2, 1), fmap1_three.view(B, -1, H//2*W//2).permute(0, 2, 1))
#         fmap_four, _ = self.fuserFour(fmap2_four.view(B, -1, H//4*W//4).permute(0, 2, 1), fmap1_four.view(B, -1, H//4*W//4).permute(0, 2, 1), fmap1_four.view(B, -1, H//4*W//4).permute(0, 2, 1))
#         fmap_five, _ = self.fuserFive(fmap2_five.view(B, -1, H//4*W//4).permute(0, 2, 1), fmap1_five.view(B, -1, H//4*W//4).permute(0, 2, 1), fmap1_five.view(B, -1, H//4*W//4).permute(0, 2, 1))

#         # fmap_one = self.netScoreOne(fmap_one.permute(0, 2, 1).view(B, -1, H, W))
#         # fmap_two = self.netScoreTwo(fmap_two.permute(0, 2, 1).view(B, -1, H, W))
#         fmap_three = self.netScoreThr(fmap_three.permute(0, 2, 1).view(B, -1, H//2, W//2))
#         fmap_four = self.netScoreFou(fmap_four.permute(0, 2, 1).view(B, -1, H//4, W//4))
#         fmap_five = self.netScoreFiv(fmap_five.permute(0, 2, 1).view(B, -1, H//4, W//4))

#         # fmap_one = torch.nn.functional.interpolate(input=fmap_one, size=self.output_size, mode='bilinear', align_corners=False)
#         # fmap_two = torch.nn.functional.interpolate(input=fmap_two, size=self.output_size, mode='bilinear', align_corners=False)
#         fmap_three = torch.nn.functional.interpolate(input=fmap_three, size=self.output_size, mode='bilinear', align_corners=False)
#         fmap_four = torch.nn.functional.interpolate(input=fmap_four, size=self.output_size, mode='bilinear', align_corners=False)
#         fmap_five = torch.nn.functional.interpolate(input=fmap_five, size=self.output_size, mode='bilinear', align_corners=False)

#         return self.netCombine(torch.cat([ fmap_three, fmap_four, fmap_five], 1))

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

# class DepthMaskHead(nn.Module):
#     def __init__(self, input_dim=324, output_dim=2):
#         super(DepthMaskHead, self).__init__()
        
#         self.Up_conv4 = conv_block(ch_in=128+input_dim, ch_out=256)

#         self.Up3 = up_conv(ch_in=256,ch_out=128)
#         self.Up_conv3 = conv_block(ch_in=128+96, ch_out=128)
        
#         self.Up2 = up_conv(ch_in=128,ch_out=64)
#         self.Up_conv2 = conv_block(ch_in=64+64, ch_out=64)
        
#         self.Up1 = up_conv(ch_in=64,ch_out=32)

#         self.Conv_1x1 = nn.Conv2d(32,output_dim,kernel_size=1,stride=1,padding=0)


#     def forward(self, corr, fmap_two, fmap_three, fmap_four):
#         d4 = torch.cat((fmap_four,corr),dim=1)  # 128+324
#         d4 = self.Up_conv4(d4)                  # 256
#         d3 = self.Up3(d4)

#         d3 = torch.cat((fmap_three,d3),dim=1)   # 128+96
#         d3 = self.Up_conv3(d3)                  # 128
#         d2 = self.Up2(d3)

#         d2 = torch.cat((fmap_two,d2),dim=1)     # 64+64
#         d2 = self.Up_conv2(d2)                  # 64
#         d1 = self.Up1(d2)                       # 32

#         # d1 = self.softmax(self.Conv_1x1(d1)) 
#         d1 = self.Conv_1x1(d1)

#         return d1
    

class DepthMaskHead(nn.Module):
    def __init__(self, input_dim=324, output_dim=2):
        super(DepthMaskHead, self).__init__()
        self.output_dim = output_dim
        
        self.Up_conv5 = conv_block(ch_in=256+input_dim, ch_out=256)
        self.netScore4 = nn.Conv2d(in_channels=256, out_channels=output_dim, kernel_size=1, stride=1, padding=0)

        self.Up_conv4 = conv_block(ch_in=128+256, ch_out=256)
        # self.Up_conv4 = conv_block(ch_in=128+input_dim, ch_out=256)
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.netScore3 = nn.Conv2d(in_channels=128, out_channels=output_dim, kernel_size=1, stride=1, padding=0)
        
        self.Up_conv3 = conv_block(ch_in=128+96, ch_out=128)
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.netScore2 = nn.Conv2d(in_channels=64, out_channels=output_dim, kernel_size=1, stride=1, padding=0)
        
        self.Up_conv2 = conv_block(ch_in=64+64, ch_out=64)
        self.Up1 = up_conv(ch_in=64,ch_out=32)
        self.netScore1 = nn.Conv2d(in_channels=32, out_channels=output_dim, kernel_size=1, stride=1, padding=0)

        self.Up_conv1 = conv_block(ch_in=64+64, ch_out=32)
        self.Up0 = up_conv(ch_in=32,ch_out=32)
        self.netScore0 = nn.Conv2d(in_channels=32, out_channels=output_dim, kernel_size=1, stride=1, padding=0)

        self.netCombine = nn.Conv2d(in_channels=10, out_channels=output_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, corr, fmap_one, fmap_two, fmap_three, fmap_four, fmap_five):
        d5 = torch.cat((fmap_five,corr),dim=1)  # 256+324
        d4 = self.Up_conv5(d5)                  # 512
        score4 = self.netScore4(d4) 

        d4 = torch.cat((fmap_four,d4),dim=1)    # 128+256
        # d4 = torch.cat((fmap_four,corr),dim=1)  # 128+324
        d4 = self.Up_conv4(d4)                  # 256
        d3 = self.Up3(d4)                       # 128
        score3 = self.netScore3(d3)

        d3 = torch.cat((fmap_three,d3),dim=1)   # 128+96
        d3 = self.Up_conv3(d3)                  # 128
        d2 = self.Up2(d3)                       # 64
        score2 = self.netScore2(d2)

        d2 = torch.cat((fmap_two,d2),dim=1)     # 64+64
        d2 = self.Up_conv2(d2)                  # 64
        d1 = self.Up1(d2)                       # 32
        score1 = self.netScore1(d1)
        
        d1 = torch.cat((fmap_one,d2),dim=1)     # 64+64
        d1 = self.Up_conv1(d1)                  # 32
        d0 = self.Up0(d1)                       # 32                 
        score0 = self.netScore0(d0)             

        output_size = score1.shape[2:]
        score4 = torch.nn.functional.interpolate(input=score4, size=output_size, mode='bilinear', align_corners=False)
        score3 = torch.nn.functional.interpolate(input=score3, size=output_size, mode='bilinear', align_corners=False)
        score2 = torch.nn.functional.interpolate(input=score2, size=output_size, mode='bilinear', align_corners=False)

        score = self.netCombine(torch.cat([score4, score3, score2, score1, score0], 1))

        return score



class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))

        h = (1-z) * h + z * q
        return h

class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))


    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q

        return h

class BasicMotionEncoder(nn.Module):
    def __init__(self, args):
        super(BasicMotionEncoder, self).__init__()
        cor_planes = args.corr_levels * (2*args.corr_radius + 1)**2
        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64+192, 128-2, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)

class BasicUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=128, input_dim=128):
        super(BasicUpdateBlock, self).__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(args)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128+hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256, output_dim=2)


        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0))

    def forward(self, net, inp, corr, flow, upsample=True):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)

        net = self.gru(net, inp)

        delta_flow = self.flow_head(net)

        # scale mask to balence gradients
        mask = .25 * self.mask(net)

        return net, mask, delta_flow

class MTUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=128, input_dim=128):
        super(MTUpdateBlock, self).__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(args)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128+hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256, output_dim=2)

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0))

    def forward(self, net, inp, corr, flow, upsample=True):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)

        net = self.gru(net, inp)

        delta_flow = self.flow_head(net)

        # scale mask to balence gradients
        mask = .25 * self.mask(net)

        return net, mask, delta_flow
