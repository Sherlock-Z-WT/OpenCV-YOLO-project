import cv2
import os

class VideoProcessor:
    def __init__(self,video_path,output_dir='output'):
        #初始化方法，接收文件路径和输出目录
        self.cap = cv2.VideoCapture(video_path)#打开视频文件
        if not self.cap.isOpened():#如果视频文件无法发开，抛出异常
            raise ValueError(f'cannot open the video:{video_path}')
        
        #获取视频的宽度高度和帧率
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))#获取视频宽度
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))#获取视频高度
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)#获取FPS

        #创建输出目录
        os.makedirs(output_dir,exist_ok=True)#exist_ok=True表示如果目录存在不会报错
        #设置输出视频的保存路径在输出目录中
        output_path = os.path.join(output_dir,'result.mp4')
        #设置视频编码方式，mp4v是mp4格式的编码器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        #创建VideoWriter对象，用于写入视频文件，输出路径，编码器，帧率，尺寸
        self.out = cv2.VideoWriter(output_path,fourcc,self.fps,(self.width,self.height))

        print(f'outputfile:{output_path}')#输出文件保存路径

    def read_frame(self):
        #读取视频的每一帧
        return self.cap.read()#返回一个布尔值和帧图像
    
    def write_frame(self,frame):
        #将处理后的帧写入到视频文件
        self.out.write(frame) #写入帧到输出视频文件

    def release(self):
        #释放视频文件
        self.cap.release()#释放视频文件的资源
        self.out.release()#释放视频写入的资源
        