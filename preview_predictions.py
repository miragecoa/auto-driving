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
    parser.add_argument('--conf_thres', type=float, default=0.7, 
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
    parser.add_argument('--debug_level', type=int, default=1, choices=[0, 1, 2],
                        help='调试信息级别: 0=无, 1=基本信息, 2=详细信息')
    parser.add_argument('--log_file', type=str, default='',
                        help='保存调试日志的文件路径，为空则不保存')
    parser.add_argument('--bbox_scale', type=float, default=1.0,
                        help='边界框缩放因子, <1.0缩小框, >1.0放大框')
    
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

def generate_previews(model, data_path, img_size, conf_thres, iou_thres, device, output_dir, max_images, debug_level=1, log_file=None, bbox_scale=1.0):
    """生成预览图"""
    
    # 设置日志文件
    log_fh = None
    if log_file:
        try:
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            log_fh = open(log_file, 'w', encoding='utf-8')
            print(f"调试日志将保存到: {log_file}")
        except Exception as e:
            print(f"打开日志文件出错: {e}")
    
    def log_print(*args, **kwargs):
        """同时打印到控制台和日志文件"""
        print(*args, **kwargs)
        if log_fh:
            # 将打印内容写入日志文件
            print(*args, **kwargs, file=log_fh)
    
    # 确保置信度阈值设置正确
    log_print(f"设置置信度阈值为: {conf_thres} (只显示置信度 > {conf_thres} 的检测结果)")

    # 设置设备
    device = select_device(device)
    half = device.type != 'cpu'  # 半精度
    
    log_print(f"使用设备: {device}, 半精度: {half}")
    
    # 设置模型
    if model is not None:
        model = model.to(device)
        if half:
            model.half()  # 半精度
        model.eval()  # 设置为评估模式
        try:
            stride = int(model.stride.max())  # 模型步长
            log_print(f"模型步长: {stride}")
        except:
            try:
                stride = max(int(model.stride), 32) if hasattr(model, 'stride') else 32
                log_print(f"未能获取模型步长，使用默认值: {stride}")
            except:
                stride = 32
                log_print(f"使用固定步长: {stride}")
    else:
        log_print("错误: 模型加载失败")
        if log_fh:
            log_fh.close()
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
                if debug_level >= 2:
                    log_print(f"调用scale_coords: img1_shape={img1_shape}, img0_shape={img0_shape}")
                    
                if ratio_pad is None:  # 从img0_shape计算
                    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain = old / new
                    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
                    if debug_level >= 2:
                        log_print(f"  计算的gain={gain}, pad={pad}")
                else:
                    gain = ratio_pad[0][0]
                    pad = ratio_pad[1]
                    if debug_level >= 2:
                        log_print(f"  提供的gain={gain}, pad={pad}")

                # 复制坐标以避免修改原始数据
                coords_copy = coords.clone() if isinstance(coords, torch.Tensor) else coords.copy()
                
                # 调试原始坐标
                if debug_level >= 2:
                    if len(coords) > 0:
                        log_print(f"  原始坐标示例: {coords[0]}")
                
                # 去除填充
                coords_copy[:, [0, 2]] -= pad[0]  # x padding
                coords_copy[:, [1, 3]] -= pad[1]  # y padding
                
                # 调试去除填充后的坐标
                if debug_level >= 2:
                    if len(coords) > 0:
                        log_print(f"  去除填充后坐标示例: {coords_copy[0]}")
                
                # 根据缩放比例调整坐标
                coords_copy[:, :4] /= gain
                
                # 调试缩放后的坐标
                if debug_level >= 2:
                    if len(coords) > 0:
                        log_print(f"  缩放后坐标示例: {coords_copy[0]}")
                
                # 限制在边界内
                coords_copy[:, [0, 2]] = coords_copy[:, [0, 2]].clamp(0, img0_shape[1])  # x1, x2
                coords_copy[:, [1, 3]] = coords_copy[:, [1, 3]].clamp(0, img0_shape[0])  # y1, y2
                
                # 调试最终坐标
                if debug_level >= 2:
                    if len(coords) > 0:
                        log_print(f"  最终坐标示例: {coords_copy[0]}")
                
                return coords_copy
            
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
                    # 原始坐标格式是 [中心x, 中心y, 宽度, 高度]
                    # 先把格式转为 [x1, y1, x2, y2]
                    box = x[:, :4].clone()
                    
                    # 将 [中心x, 中心y, 宽度, 高度] 转为 [x1, y1, x2, y2]
                    box_converted = torch.zeros_like(box)
                    box_converted[:, 0] = box[:, 0] - box[:, 2] / 2  # x1 = cx - w/2
                    box_converted[:, 1] = box[:, 1] - box[:, 3] / 2  # y1 = cy - h/2
                    box_converted[:, 2] = box[:, 0] + box[:, 2] / 2  # x2 = cx + w/2
                    box_converted[:, 3] = box[:, 1] + box[:, 3] / 2  # y2 = cy + h/2
                    
                    # 使用转换后的坐标
                    conf, j = x[:, 5:].max(1, keepdim=True)
                    x = torch.cat((box_converted, conf, j.float()), 1)[conf.view(-1) > conf_thres]
                    
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
            log_print(f"模型只包含预期的单一类别: {names}")
        else:
            # 如果模型包含多个类别，我们强制只使用'vehicle'
            log_print(f"警告: 模型包含多个类别 {names}，将强制使用'vehicle'标签")
            # 打印所有类别ID和名称的映射
            if debug_level >= 1:
                log_print("类别ID映射:")
                for i, name in enumerate(names):
                    log_print(f"  ID {i}: {name}")
            names = ['vehicle']  # 强制设置为单一vehicle类别
    except:
        # 如果无法获取模型名称，使用默认值
        names = ['vehicle']
        log_print(f"无法获取类别名称，使用默认值: {names}")
    
    colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in range(1)]  # 只为vehicle类别创建一个颜色
    
    if bbox_scale != 1.0:
        log_print(f"应用边界框缩放因子: {bbox_scale} (将对所有检测框进行缩放)")
    
    for img_path in tqdm(img_files, desc="正在生成预览"):
        # 读取图像
        img0 = cv2.imread(img_path)  # 原始图像
        if img0 is None:
            log_print(f"无法读取图像: {img_path}")
            continue
        
        # 记录原始图像尺寸
        original_shape = img0.shape
        if debug_level >= 1:
            log_print(f"图像 {os.path.basename(img_path)} 原始尺寸: {original_shape}")
        
        # 转换图像
        try:
            letterboxed_img, ratio, pad = letterbox(img0, img_size, stride=stride)
            if debug_level >= 2:
                log_print(f"  调整后尺寸: {letterboxed_img.shape}, 缩放比例: {ratio}, 填充: {pad}")
            
            img = letterboxed_img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            img = np.ascontiguousarray(img)
        except Exception as e:
            log_print(f"图像预处理错误: {e}")
            continue
        
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        # 推理
        with torch.no_grad():
            try:
                pred = model(img)[0]
                # 打印原始预测结果的形状
                if debug_level >= 2:
                    log_print(f"\n图像 {os.path.basename(img_path)} 原始预测形状: {pred.shape}")
                    # 如果预测维度 > 5，说明模型预测了多个类别
                    if pred.shape[-1] > 6:
                        log_print(f"  注意: 模型预测了 {pred.shape[-1] - 5} 个类别")
            except Exception as e:
                log_print(f"模型推理错误: {e}")
                pred = torch.zeros((1, 0, 6), device=device)
        
        # 非极大值抑制
        try:
            # 尝试只检测ID为0的类别（vehicle）
            vehicle_classes = [0]  # 车辆类别ID
            
            # 检查NMS前的预测格式
            if debug_level >= 2 and len(pred[0]) > 0:
                log_print(f"NMS前的预测格式示例: {pred[0][0]}")
                
            # 非极大值抑制
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes=vehicle_classes)
            
            # 检查NMS后的预测格式 
            if debug_level >= 2 and len(pred[0]) > 0:
                log_print(f"NMS后的预测格式示例: {pred[0][0]}")
                
            if debug_level >= 1:
                log_print(f"  NMS后检测到 {len(pred[0])} 个物体")
                
            # 额外确保所有检测结果均满足置信度要求
            for i in range(len(pred)):
                if len(pred[i]) > 0:
                    # 获取原始检测数量
                    orig_count = len(pred[i])
                    # 仅保留置信度大于等于阈值的检测结果
                    high_conf_detections = pred[i][pred[i][:, 4] >= conf_thres]
                    pred[i] = high_conf_detections
                    # 如果有被过滤的结果，显示日志
                    if debug_level >= 1 and len(high_conf_detections) < orig_count:
                        log_print(f"  应用置信度过滤: {orig_count} -> {len(high_conf_detections)} 个物体 (要求置信度 >= {conf_thres})")
                
        except Exception as e:
            log_print(f"非极大值抑制错误: {e}")
            pred = [torch.zeros((0, 6), device=device)]
        
        # 处理检测结果
        for i, det in enumerate(pred):
            im0 = img0.copy()
            
            # 添加标题
            base_filename = os.path.basename(img_path)
            cv2.putText(im0, f"File: {base_filename}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 添加置信度阈值信息
            cv2.putText(im0, f"Confidence threshold: {conf_thres}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            
            # 添加边界框缩放信息
            if bbox_scale != 1.0:
                cv2.putText(im0, f"Bounding box scale: {bbox_scale:.2f}", (10, 85), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
                y_offset = 110  # 调整下一行文本的垂直位置
            else:
                y_offset = 90  # 保持原来的垂直位置

            # 添加检测数量信息
            if len(det):
                # 重新缩放框到原始图像大小
                try:
                    # 保存原始检测结果以便调试
                    if debug_level >= 2:
                        log_print("原始检测结果（缩放前）:")
                        for j, (*xyxy, conf, cls) in enumerate(det):
                            # 注意：检查xyxy格式，确认是否需要转换
                            if len(xyxy) == 4:
                                x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
                                
                                # 检查是否为中心点+宽高格式 (如果x2比x1小或y2比y1小，可能是宽高而不是坐标)
                                if x2 < x1 or y2 < y1:
                                    log_print(f"  物体 {j+1} 检测到中心点+宽高格式: [{float(xyxy[0])},{float(xyxy[1])},{float(xyxy[2])},{float(xyxy[3])}]")
                                    
                                    # 转换为角点坐标 (x1,y1,x2,y2)
                                    cx, cy, w, h = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
                                    x1, y1 = cx - w/2, cy - h/2
                                    x2, y2 = cx + w/2, cy + h/2
                                    det[j, 0] = x1
                                    det[j, 1] = y1
                                    det[j, 2] = x2
                                    det[j, 3] = y2
                                    
                                    log_print(f"  物体 {j+1} 转换后坐标: [{x1},{y1},{x2},{y2}]")
                                else:
                                    log_print(f"  物体 {j+1} 原始坐标: [{float(xyxy[0])},{float(xyxy[1])},{float(xyxy[2])},{float(xyxy[3])}]")
                            else:
                                log_print(f"  物体 {j+1} 原始坐标: {xyxy}")
                    
                    # 缩放坐标到原始图像大小
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    
                    # 打印缩放后的结果以便调试
                    if debug_level >= 2:
                        log_print("缩放后检测结果:")
                        for j, (*xyxy, conf, cls) in enumerate(det):
                            log_print(f"  物体 {j+1} 缩放后坐标: [{float(xyxy[0])},{float(xyxy[1])},{float(xyxy[2])},{float(xyxy[3])}]")
                    
                except Exception as e:
                    log_print(f"坐标缩放错误: {e}")
                    log_print(f"错误详情: {str(e)}")
                    import traceback
                    log_print(traceback.format_exc())
                
                # 绘制结果
                try:
                    if debug_level >= 1:
                        log_print(f"\n检测结果详情 ({len(det)} 个物体):")
                    
                    # 检测框的面积和尺寸信息
                    bbox_areas = []
                    for *xyxy, conf, cls in det:
                        # 确保坐标顺序正确 (x1,y1,x2,y2)，x1<x2, y1<y2
                        x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
                        
                        # 应用边界框缩放
                        if bbox_scale != 1.0:
                            # 计算中心点
                            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                            # 计算宽高
                            w, h = x2 - x1, y2 - y1
                            # 应用缩放
                            w *= bbox_scale
                            h *= bbox_scale
                            # 重新计算坐标
                            x1, y1 = cx - w/2, cy - h/2
                            x2, y2 = cx + w/2, cy + h/2
                            # 确保不超出图像边界
                            x1 = max(0, x1)
                            y1 = max(0, y1)
                            x2 = min(im0.shape[1], x2)
                            y2 = min(im0.shape[0], y2)
                        
                        # 如果坐标顺序不正确，则交换
                        if x1 > x2:
                            x1, x2 = x2, x1
                        if y1 > y2:
                            y1, y2 = y2, y1
                        
                        # 重新计算宽度和高度
                        width = x2 - x1
                        height = y2 - y1
                        area = width * height
                        
                        # 使用正确顺序的坐标
                        xyxy = [x1, y1, x2, y2]
                        bbox_areas.append((xyxy, conf, cls, area, width, height))
                    
                    # 按面积排序
                    bbox_areas.sort(key=lambda x: x[3], reverse=True)
                    
                    for i, (xyxy, conf, cls, area, width, height) in enumerate(bbox_areas):
                        # 打印边界框详情
                        if debug_level >= 1:
                            cls_id = int(cls.item()) if hasattr(cls, 'item') else int(cls)
                            # 尝试获取原始类别名称
                            try:
                                original_names = model.module.names if hasattr(model, 'module') else model.names
                                original_class_name = original_names[cls_id] if cls_id < len(original_names) else f"未知类别{cls_id}"
                                log_print(f"  物体 {i+1}: 类别ID={cls_id}, 类别名称={original_class_name}, 置信度={conf:.4f}, 坐标=[{int(xyxy[0])},{int(xyxy[1])},{int(xyxy[2])},{int(xyxy[3])}], 尺寸=[{int(width)}x{int(height)}], 面积={int(area)}px²")
                            except:
                                log_print(f"  物体 {i+1}: 类别ID={cls_id}, 置信度={conf:.4f}, 坐标=[{int(xyxy[0])},{int(xyxy[1])},{int(xyxy[2])},{int(xyxy[3])}], 尺寸=[{int(width)}x{int(height)}], 面积={int(area)}px²")
                        
                        # 忽略类别信息，始终使用'vehicle'标签
                        label = f'vehicle {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[0], line_thickness=3)
                except Exception as e:
                    log_print(f"绘制边界框错误: {e}")
                
                # 添加检测计数
                cv2.putText(im0, f"Vehicles: {len(det)}", (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(im0, "No vehicles detected", (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # 保存预览图
            output_filename = os.path.join(output_dir, f"preview_{base_filename}")
            cv2.imwrite(output_filename, im0)
            
    log_print(f"预览图像已保存到: {output_dir}")
    
    # 关闭日志文件
    if log_fh:
        log_fh.close()

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

def try_native_yolov5_prediction(weights, data_path, img_size, conf_thres, iou_thres, device, output_dir, max_images, bbox_scale=1.0):
    """尝试使用YOLOv5原生方法进行预测"""
    try:
        import subprocess
        import shutil
        
        print("尝试使用YOLOv5原生方法进行预测...")
        print(f"设置置信度阈值为: {conf_thres} (只显示置信度 > {conf_thres} 的检测结果)")
        if bbox_scale != 1.0:
            print(f"注意: 使用原生YOLOv5推理时不支持边界框缩放 {bbox_scale}，此参数将被忽略")
        
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
    print(f"调试级别: {args.debug_level}")
    if args.log_file:
        print(f"调试日志: {args.log_file}")
    if args.bbox_scale != 1.0:
        print(f"边界框缩放: {args.bbox_scale}")
    
    # 更新README文件中的说明
    try:
        update_readme_for_debug_level()
    except Exception as e:
        print(f"更新README时出错: {e}")
    
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
            args.max_images,
            args.bbox_scale
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
            args.max_images,
            args.debug_level,
            args.log_file,
            args.bbox_scale
        )
    else:
        print("模型加载失败,请检查权重路径或YOLOv5安装")

def update_readme_for_debug_level():
    """更新README_preview.md文件，添加debug_level参数的说明"""
    readme_path = "README_preview.md"
    if not os.path.exists(readme_path):
        print(f"README文件不存在: {readme_path}")
        return
    
    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # 检查是否已经有必要的参数说明
    new_params = ["--debug_level", "--log_file", "--bbox_scale"]
    missing_params = [param for param in new_params if param not in content]
    
    if not missing_params:
        print("README中已有所有必要的参数说明")
        return
    
    # 在参数说明部分添加缺失的参数
    params_section = "## 参数说明"
    
    if params_section in content:
        updated_content = content
        
        if "--debug_level" not in content:
            updated_content = updated_content.replace(
                "- `--use_native`: 使用YOLOv5原生推理方法（推荐用于新版本YOLOv5）",
                "- `--use_native`: 使用YOLOv5原生推理方法（推荐用于新版本YOLOv5）\n- `--debug_level`: 调试信息级别 (0=无, 1=基本信息, 2=详细信息), 默认为1"
            )
        
        if "--log_file" not in updated_content:
            updated_content = updated_content.replace(
                "- `--debug_level`: 调试信息级别 (0=无, 1=基本信息, 2=详细信息), 默认为1",
                "- `--debug_level`: 调试信息级别 (0=无, 1=基本信息, 2=详细信息), 默认为1\n- `--log_file`: 保存调试日志的文件路径，为空则不保存"
            )
        
        if "--bbox_scale" not in updated_content:
            # 尝试在log_file参数后添加
            if "--log_file" in updated_content:
                updated_content = updated_content.replace(
                    "- `--log_file`: 保存调试日志的文件路径，为空则不保存",
                    "- `--log_file`: 保存调试日志的文件路径，为空则不保存\n- `--bbox_scale`: 边界框缩放因子 (<1.0缩小框, >1.0放大框), 默认为1.0"
                )
            # 如果没有log_file参数，则在debug_level参数后添加
            elif "--debug_level" in updated_content:
                updated_content = updated_content.replace(
                    "- `--debug_level`: 调试信息级别 (0=无, 1=基本信息, 2=详细信息), 默认为1",
                    "- `--debug_level`: 调试信息级别 (0=无, 1=基本信息, 2=详细信息), 默认为1\n- `--bbox_scale`: 边界框缩放因子 (<1.0缩小框, >1.0放大框), 默认为1.0"
                )
            # 如果其他位置都找不到，则在最后一个已知参数后添加
            else:
                updated_content = updated_content.replace(
                    "- `--max_images`: 最多预览的图片数量，默认为10",
                    "- `--max_images`: 最多预览的图片数量，默认为10\n- `--bbox_scale`: 边界框缩放因子 (<1.0缩小框, >1.0放大框), 默认为1.0"
                )
        
        # 添加调试示例
        examples_section = "## 示例"
        new_examples = []
        
        if "--debug_level" in missing_params:
            new_examples.append("\n5. 使用详细调试信息:\n   ```bash\n   python preview_predictions.py --weights runs/train/vehicle_detection/weights/best.pt --debug_level 2\n   ```")
        
        if "--log_file" in missing_params:
            new_examples.append("\n6. 保存调试日志到文件:\n   ```bash\n   python preview_predictions.py --weights runs/train/vehicle_detection/weights/best.pt --debug_level 2 --log_file debug_log.txt\n   ```")
        
        if "--bbox_scale" in missing_params:
            new_examples.append("\n7. 缩小边界框(解决框太大问题):\n   ```bash\n   python preview_predictions.py --weights runs/train/vehicle_detection/weights/best.pt --bbox_scale 0.5\n   ```")
        
        if new_examples and examples_section in updated_content:
            examples_text = "".join(new_examples)
            updated_content = updated_content.replace(
                "## 输出结果",
                f"{examples_text}\n\n## 输出结果"
            )
        
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(updated_content)
        
        print(f"已更新 {readme_path} 添加参数说明")
    else:
        print(f"无法在 {readme_path} 中找到参数说明部分")

if __name__ == "__main__":
    main() 