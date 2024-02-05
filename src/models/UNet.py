import torch 
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels

		self.enc1 = self.double_conv(in_channels, 64)
		self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.enc2 = self.double_conv(64, 128)
		self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.enc3 = self.double_conv(128, 256)
		self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.enc4 = self.double_conv(256, 512)
		self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

		self.bottleneck = self.double_conv(512, 1024)

		self.up1 = self.up_conv(1024, 512)
		self.dec1 = self.double_conv(1024, 512)
		self.up2 = self.up_conv(512, 256)
		self.dec2 = self.double_conv(512, 256)
		self.up3 = self.up_conv(256, 128)
		self.dec3 = self.double_conv(256, 128)
		self.up4 = self.up_conv(128, 64)
		self.dec4 = self.double_conv(128, 64)

		self.final = nn.Conv2d(64, 1, 1, 1, 0)
		self.sigmoid = nn.Sigmoid()
		
	def forward(self, x):
		enc1 = self.enc1(x)
		pool1 = self.pool1(enc1)
		enc2 = self.enc2(pool1)
		pool2 = self.pool2(enc2)
		enc3 = self.enc3(pool2)
		pool3 = self.pool3(enc3)
		enc4 = self.enc4(pool3)
		pool4 = self.pool4(enc4)

		bottleneck = self.bottleneck(pool4)

		up1 = self.up1(bottleneck)
		dec1 = self.dec1(torch.concat((enc4, up1), dim=1))
		up2 = self.up2(dec1)
		dec2 = self.dec2(torch.concat((enc3, up2), dim=1))
		up3 = self.up3(dec2)
		dec3 = self.dec3(torch.concat((enc2, up3), dim=1))
		up4 = self.up4(dec3)
		dec4 = self.dec4(torch.concat((enc1, up4), dim=1))
		
		final = self.final(dec4)
		pred = self.sigmoid(final)

		return pred


	def double_conv(self, in_channels, out_channels):
		return nn.Sequential(
			nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
			 kernel_size=3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
			 kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
		)
	
	def up_conv(self, in_channels, out_channels):
		return nn.Sequential(
			nn.Upsample(scale_factor=2, mode='nearest'),
			nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
			 kernel_size=3, stride=1, padding=1)
		)
	
if __name__ == '__main__':
	print('passed')