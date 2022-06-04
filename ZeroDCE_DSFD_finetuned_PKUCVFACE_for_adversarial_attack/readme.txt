需要安装以下Python包：
	easydict
	torchvision
	opencv
	(可能还有别的，如果报错了再补装吧233)

需要安装以下软件：
	ffmpeg

文件说明：
	api_addnoise.py：标准API
	api_denoise.py：在标准API基础上，采用去噪来（试图）防御adversarial attack
	api_addnoise.py：在标准API基础上，采用加噪来（试图）防御adversarial attack
	
	img目录下给出四张样例图，包括两张有人脸的和两张没有人脸的。
		face2.png中人脸比face1.png要更难识别。
		noface1.png中的物体比noface2.png中的物体要更易混淆为人脸。

使用方式：
	把你们的代码放在该目录下（即和api_***.py在同一层目录中），然后：
	from api_standard import ApiClass # 或者 from api_denoise import ApiClass
	                                  # 或者 from api_addnoise import ApiClass
	
	至于ApiClass的使用方式，见api_addnoise.py里最后的示例代码。三套API的使用方式完全一样。
	
	进行攻击只需使用ApiClass.get_confidence函数即可（这个函数会在内部进行提亮和人脸检测，
	所以传进去的图片是暗光图片，在进行attack的时候也请在暗光图片上做改动）。另一个ApiCl
	ass.lowlight_enhance函数是用来在做ppt时，给图片提亮以便放进ppt的。
	
	get_confidence返回的值，表示它认为图中以多大置信度（0到1）存在人脸。