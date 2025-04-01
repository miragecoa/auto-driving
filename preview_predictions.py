#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import cv2
import torch
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser(description='预览车辆检测模型预测结果')
    parser.add_argument('--weights', type=str, required=True, 
                        help='模型权重路径')
    parser.add_argument('--data_path', type=str, default='vehicle_dataset', 
                        help='数据集路径')
    parser.add_argument('--img_size', type=int, default=640, 
                        help='输入图像尺寸')
    parser.add_argument('--conf_thres', type=float, default=0.25, 
                        help='置信度阈值')
    parser.add_argument('--iou_thres', type=float, default=0.45, 
                        help='NMS IOU阈值')
    parser.add_argument('--device', default='', 
                        help='cuda设备, 例如 0 或 0,1,2,3 或 cpu')
    parser.add_argument('--output_dir', default='preview_results', 
                        help='预览结果保存目录')
    parser.add_argument('--max_images', type=int, default=10,
                        help='最多预览多少张图片')
    parser.add_argument('--use_native', action='store_true',
                        help='使用YOLOv5原生推理方法')
    
    return parser.parse_args()

def check_yolov5_exists():
    """检查YOLOv5是否存在"""
    yolov5_dir = os.path.join(os.getcwd(), 'yolov5')
    if not os.path.exists(yolov5_dir):
        print(f"未找到YOLOv5目录: {yolov5_dir}")
        print("请先克隆YOLOv5仓库: git clone https://github.com/ultralytics/yolov5.git")
        return False
    return True

def load_model(weights, device):
    """加载训练好的模型"""
    try:
        # 添加YOLOv5目录到路径
        yolov5_dir = os.path.join(os.getcwd(), 'yolov5')
        sys.path.append(yolov5_dir)
        
        print(f"使用YOLOv5路径: {yolov5_dir}")
        
        # 使用YOLOv5的模型加载函数
        # 先尝试导入models.experimental模块
        try:
            from models.experimental import attempt_load
            
            # 检查权重文件是否存在
            if not os.path.exists(weights):
                print(f"警告: 权重文件不存在: {weights}")
                print(f"尝试在YOLOv5预训练模型中查找: {os.path.join(yolov5_dir, weights)}")
                if os.path.exists(os.path.join(yolov5_dir, weights)):
                    weights = os.path.join(yolov5_dir, weights)
            
            print(f"加载权重: {weights}")
            
            # 尝试不同的参数调用方式
            try:
                # 新版本YOLOv5可能不使用map_location参数
                model = attempt_load(weights)
                print("使用不带map_location参数的attempt_load()")
            except TypeError:
                # 如果上面失败，尝试使用device参数
                try:
                    model = attempt_load(weights, device=device)
                    print("使用device参数的attempt_load()")
                except TypeError:
                    # 如果仍然失败，尝试使用旧方式
                    model = attempt_load(weights, map_location=device)
                    print("使用map_location参数的attempt_load()")
            
            return model
            
        except ImportError:
            # 尝试新版本的导入方式
            print("尝试使用YOLOv5的替代导入方式...")
            import torch
            print(f"加载权重: {weights}")
            model = torch.hub.load(yolov5_dir, 'custom', path=weights, source='local')
            print("使用torch.hub.load加载模型成功")
            return model
        
    except ImportError as e:
        print(f"导入错误: {e}")
        print("确保YOLOv5安装正确，执行: pip install -r yolov5/requirements.txt")
        return None
    except Exception as e:
        print(f"加载模型错误: {e}")
        print("尝试使用更基本的PyTorch加载方式...")
        try:
            import torch
            print(f"使用torch.load直接加载模型: {weights}")
            model = torch.load(weights, map_location=device)
            if isinstance(model, dict) and 'model' in model:
                model = model['model']
            model.to(device)
            model.eval()
            print("使用torch.load加载模型成功")
            return model
        except Exception as e2:
            print(f"所有加载尝试都失败: {e2}")
            return None

def select_device(device=''):
    # 导入YOLOv5的设备选择函数
    try:
        # 添加YOLOv5目录到路径
        yolov5_dir = os.path.join(os.getcwd(), 'yolov5')
        sys.path.append(yolov5_dir)
        from utils.torch_utils import select_device as yolo_select_device
        return yolo_select_device(device)
    except:
        # 备用方案，创建自己的设备选择逻辑
        import torch
        
        # 设置设备
        cuda = device.lower() != 'cpu' and torch.cuda.is_available()
        if cuda and device:
            # 如果指定了设备，检查是否有效
            try:
                device_ids = [int(x) for x in device.split(',') if x]
                if device_ids:
                    device = f'cuda:{device_ids[0]}'
                else:
                    device = 'cuda:0'
            except:
                device = 'cuda:0'
        else:
            device = 'cuda:0' if cuda else 'cpu'
            
        return torch.device(device)

def generate_previews(model, data_path, img_size, conf_thres, iou_thres, device, output_dir, max_images):
    """生成预览图"""
    # 设置设备
    device = select_device(device)
    half = device.type != 'cpu'  # 半精度
    
    # 设置模型
    if model is not None:
        model = model.to(device)
        if half:
            model.half()  # 半精度
        model.eval()  # 设置为评估模式
        try:
            stride = int(model.stride.max())  # 模型步长
        except:
            try:
                stride = max(int(model.stride), 32) if hasattr(model, 'stride') else 32
            except:
                stride = 32
    else:
        print("错误: 模型加载失败")
        return
    
    # 导入必要的YOLOv5模块
    yolov5_dir = os.path.join(os.getcwd(), 'yolov5')
    sys.path.append(yolov5_dir)
    
    # 尝试导入各种可能的函数
    try:
        # 尝试直接导入
        from utils.general import non_max_suppression, scale_coords
        print("成功导入 non_max_suppression 和 scale_coords")
    except ImportError:
        try:
            # 新版本可能有不同路径
            from utils.nms import non_max_suppression
            from utils.augmentations import scale_coords
            print("从替代路径导入 non_max_suppression 和 scale_coords")
        except ImportError:
            # 如果仍然无法导入，直接实现简单版本的NMS
            print("无法导入NMS，使用简单实现")
            
            def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
                """缩放边界框坐标"""
                if ratio_pad is None:  # 从img0_shape计算
                    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain = old / new
                    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
                else:
                    gain = ratio_pad[0][0]
                    pad = ratio_pad[1]

                coords[:, [0, 2]] -= pad[0]  # x padding
                coords[:, [1, 3]] -= pad[1]  # y padding
                coords[:, :4] /= gain
                coords[:, [0, 2]] = coords[:, [0, 2]].clamp(0, img0_shape[1])  # x1, x2
                coords[:, [1, 3]] = coords[:, [1, 3]].clamp(0, img0_shape[0])  # y1, y2
                return coords
            
            def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False, max_det=300):
                """简易版非极大值抑制"""
                import torch
                
                # 获取超过置信度阈值的框
                xc = prediction[..., 4] > conf_thres  # 置信度
                
                # 设置
                min_wh, max_wh = 2, 4096  # (像素) 最小和最大盒长宽
                max_nms = 30000  # 进入NMS的最大框数
                time_limit = 10.0  # 超时
                redundant = True  # 冗余检测要求
                multi_label &= prediction.shape[2] > 5  # 多标签每个框(多标签需要特殊后处理)
                
                output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
                for xi, x in enumerate(prediction):  # 逐张图像处理
                    # 应用约束
                    x = x[xc[xi]]  # 置信度
                    
                    # 如果没有框，继续
                    if not x.shape[0]:
                        continue
                    
                    # 置信度
                    x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
                    
                    # 获取最高置信度的类别
                    box = x[:, :4]
                    conf, j = x[:, 5:].max(1, keepdim=True)
                    x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
                    
                    # 过滤
                    if classes is not None:
                        x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]
                    
                    # 检查数量
                    n = x.shape[0]  # 框数量
                    if not n:  # 如果没有框，继续
                        continue
                    
                    # NMS
                    c = x[:, 5:6] * (0 if agnostic else max_wh)  # 类别*最大宽高
                    boxes, scores = x[:, :4] + c, x[:, 4]  # 框，分数
                    i = torch.ops.torchvision.nms(boxes, scores, iou_thres)  # NMS
                    output[xi] = x[i]
                
                return output
    
    # 导入绘图函数
    try:
        from utils.plots import plot_one_box
        print("成功导入 plot_one_box")
    except ImportError:
        # 如果无法导入，定义一个简单的绘图函数
        print("无法导入plot_one_box，使用简单实现")
        
        def plot_one_box(x, img, color=(128, 128, 128), label=None, line_thickness=3):
            """画一个边界框"""
            tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # 线条/字体粗细
            color = color or [np.random.randint(0, 255) for _ in range(3)]
            c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
            cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
            if label:
                tf = max(tl - 1, 1)  # 字体粗细
                t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
                c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # 填充
                cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取图像路径
    img_dir = os.path.join(data_path, 'images')
    if not os.path.exists(img_dir):
        print(f"错误: 未找到图像目录: {img_dir}")
        return
    
    img_files = [os.path.join(img_dir, f) for f in os.listdir(img_dir) 
                if f.endswith('.jpg') or f.endswith('.png')]
    
    # 限制图像数量
    if len(img_files) > max_images:
        print(f"限制预览图像数量为 {max_images} (共有 {len(img_files)} 张图像)")
        img_files = img_files[:max_images]
    
    # 获取类别名称
    try:
        names = model.module.names if hasattr(model, 'module') else model.names
        # 检查是否只有一个类别 - 这是我们期望的
        if len(names) == 1 and names[0].lower() == 'vehicle':
            print(f"模型只包含预期的单一类别: {names}")
        else:
            # 如果模型包含多个类别，我们强制只使用'vehicle'
            print(f"警告: 模型包含多个类别 {names}，将强制使用'vehicle'标签")
            names = ['vehicle']  # 强制设置为单一vehicle类别
    except:
        # 如果无法获取模型名称，使用默认值
        names = ['vehicle']
        print(f"无法获取类别名称，使用默认值: {names}")
    
    colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in range(1)]  # 只为vehicle类别创建一个颜色
    
    for img_path in tqdm(img_files, desc="正在生成预览"):
        # 读取图像
        img0 = cv2.imread(img_path)  # 原始图像
        if img0 is None:
            print(f"无法读取图像: {img_path}")
            continue
        
        # 转换图像
        img = letterbox(img0, img_size, stride=stride)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        # 推理
        with torch.no_grad():
            try:
                pred = model(img)[0]
            except Exception as e:
                print(f"模型推理错误: {e}")
                pred = torch.zeros((1, 0, 6), device=device)
        
        # 非极大值抑制
        try:
            pred = non_max_suppression(pred, conf_thres, iou_thres)
        except Exception as e:
            print(f"非极大值抑制错误: {e}")
            pred = [torch.zeros((0, 6), device=device)]
        
        # 处理检测结果
        for i, det in enumerate(pred):
            im0 = img0.copy()
            
            # 添加标题
            base_filename = os.path.basename(img_path)
            cv2.putText(im0, f"File: {base_filename}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 添加检测数量信息
            if len(det):
                # 重新缩放框到原始图像大小
                try:
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                except Exception as e:
                    print(f"坐标缩放错误: {e}")
                
                # 绘制结果
                try:
                    for *xyxy, conf, cls in reversed(det):
                        # 忽略类别信息，始终使用'vehicle'标签
                        label = f'vehicle {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[0], line_thickness=3)
                except Exception as e:
                    print(f"绘制边界框错误: {e}")
                
                # 添加检测计数
                cv2.putText(im0, f"Vehicles: {len(det)}", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(im0, "No vehicles detected", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # 保存预览图
            output_filename = os.path.join(output_dir, f"preview_{base_filename}")
            cv2.imwrite(output_filename, im0)
            
    print(f"预览图像已保存到: {output_dir}")

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """从YOLOv5代码中提取的letterbox函数"""
    # 检查图像是否为None
    if img is None:
        raise ValueError("输入图像为None")
    
    # 检查图像形状
    if not isinstance(img, np.ndarray) or img.ndim != 3:
        raise ValueError(f"输入图像必须是3D numpy数组，当前: {type(img)}")
    
    # 确保形状是合适的
    shape = img.shape[:2]  # 当前形状 [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    
    # 确保新形状是有效的
    if not all(s > 0 for s in new_shape):
        raise ValueError(f"新形状必须是正数，当前: {new_shape}")
    
    try:
        # 尺度比例 (新 / 旧)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # 仅缩小，不放大
            r = min(r, 1.0)

        # 计算填充
        ratio = r, r  # 宽度、高度比例
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # 最小矩形
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # 拉伸
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # 宽度、高度比例

        dw /= 2  # 分为两部分
        dh /= 2

        if shape[::-1] != new_unpad:  # 调整大小
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # 加边框
        
        return img, ratio, (dw, dh)
    
    except Exception as e:
        print(f"letterbox处理错误: {e}")
        # 简单回退：直接调整大小并返回
        img_resized = cv2.resize(img, (new_shape[1], new_shape[0]), interpolation=cv2.INTER_LINEAR)
        return img_resized, (1.0, 1.0), (0.0, 0.0)

def try_native_yolov5_prediction(weights, data_path, img_size, conf_thres, iou_thres, device, output_dir, max_images):
    """尝试使用YOLOv5原生方法进行预测"""
    try:
        import subprocess
        import shutil
        
        print("尝试使用YOLOv5原生方法进行预测...")
        
        # 检查YOLOv5目录
        yolov5_dir = os.path.join(os.getcwd(), 'yolov5')
        if not os.path.exists(yolov5_dir):
            print("YOLOv5目录不存在，无法使用原生方法")
            return False
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取图像路径
        img_dir = os.path.join(data_path, 'images')
        if not os.path.exists(img_dir):
            print(f"图像目录不存在: {img_dir}")
            return False
        
        # 限制图像数量
        img_files = [os.path.join(img_dir, f) for f in os.listdir(img_dir) 
                    if f.endswith('.jpg') or f.endswith('.png')]
        
        if len(img_files) > max_images:
            print(f"限制预览图像数量为 {max_images} (共有 {len(img_files)} 张图像)")
            selected_img_files = img_files[:max_images]
        else:
            selected_img_files = img_files
        
        # 创建临时目录存放选定的图像
        temp_dir = os.path.join(os.getcwd(), 'temp_preview_images')
        os.makedirs(temp_dir, exist_ok=True)
        
        # 复制选定的图像到临时目录
        for img_file in selected_img_files:
            shutil.copy(img_file, temp_dir)
        
        # 构建YOLOv5 detect.py命令
        cmd = [
            sys.executable,  # 当前Python解释器
            os.path.join(yolov5_dir, 'detect.py'),
            f'--weights={weights}',
            f'--source={temp_dir}',
            f'--img-size={img_size}',
            f'--conf-thres={conf_thres}',
            f'--iou-thres={iou_thres}',
            f'--save-txt',
            f'--save-conf',
            f'--project={os.path.dirname(output_dir)}',
            f'--name={os.path.basename(output_dir)}',
            f'--single-cls',  # 强制只使用一个类别
            f'--classes=0'    # 只显示索引为0的类别(vehicle)
        ]
        
        if device:
            cmd.append(f'--device={device}')
        
        # 执行YOLOv5 detect.py
        print(f"执行命令: {' '.join(cmd)}")
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            universal_newlines=True
        )
        
        # 实时输出进度
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        
        # 获取返回码
        return_code = process.poll()
        
        # 清理临时目录
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        if return_code != 0:
            error_output = process.stderr.read()
            print(f"YOLOv5 detect.py执行失败: {error_output}")
            return False
        
        print("YOLOv5原生推理完成")
        return True
    
    except Exception as e:
        print(f"使用YOLOv5原生方法时出错: {e}")
        return False

def main():
    args = parse_arguments()
    
    print("=== 车辆检测模型预览 ===")
    print(f"模型权重: {args.weights}")
    print(f"数据集路径: {args.data_path}")
    
    # 检查YOLOv5是否存在
    if not check_yolov5_exists():
        return
    
    # 如果选择使用原生方法，先尝试原生方法
    if args.use_native:
        success = try_native_yolov5_prediction(
            args.weights,
            args.data_path,
            args.img_size,
            args.conf_thres,
            args.iou_thres,
            args.device,
            args.output_dir,
            args.max_images
        )
        if success:
            print(f"成功使用YOLOv5原生方法生成预测结果到: {args.output_dir}")
            return
        else:
            print("YOLOv5原生方法失败，将尝试使用自定义方法...")
    
    # 设置设备
    device = select_device(args.device)
    print(f"使用设备: {device}")
    
    # 加载模型
    print("加载模型...")
    model = load_model(args.weights, device)
    
    if model is not None:
        # 生成预览图
        print("开始生成预览图像...")
        generate_previews(
            model, 
            args.data_path, 
            args.img_size, 
            args.conf_thres, 
            args.iou_thres, 
            args.device,
            args.output_dir,
            args.max_images
        )
    else:
        print("模型加载失败,请检查权重路径或YOLOv5安装")

if __name__ == "__main__":
    main() 