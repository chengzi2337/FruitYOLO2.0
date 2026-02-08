# YOLOv8s 训练脚本 - 实时详细输出版本
# 在前台运行 yolo 训练，实时显示所有输出

# 设置 UTF-8 编码
chcp 65001 | Out-Null
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = 'utf-8'

Write-Host "正在启动 YOLOv8s 训练..." -ForegroundColor Green
Write-Host "=====================================`n" -ForegroundColor Cyan

# 激活 conda 环境
Write-Host "激活 conda pytorch 环境..." -ForegroundColor Yellow
D:/software/anaconda/Scripts/activate
conda activate pytorch

# 训练参数
$model = 'yolov8s.pt'
# 使用增强后的数据集 (5886张训练图, 362张验证图, 382张测试图)
$data = 'Dataset_resplit_aug/data.yaml'
# 可选: 使用重划分数据集 (2943张训练图, 362张验证图, 382张测试图)
# $data = 'Dataset_resplit/data.yaml'
# 可选: 使用原始数据集 (3436张训练图, 251张验证图)
# $data = 'Dataset_Original/data.yaml'
$epochs = 200
$imgsz = 640
$batch = 16
$device = 0

Write-Host "`n训练参数:" -ForegroundColor Cyan
Write-Host "  模型: $model" -ForegroundColor White
Write-Host "  数据集: $data" -ForegroundColor White
Write-Host "  总轮次: $epochs" -ForegroundColor White
Write-Host "  图像大小: $imgsz" -ForegroundColor White
Write-Host "  批次大小: $batch" -ForegroundColor White
Write-Host "  设备: GPU $device`n" -ForegroundColor White

# 检查数据集
if(-not (Test-Path $data)) {
    Write-Host "错误: 数据集配置文件不存在: $data" -ForegroundColor Red
    exit 1
}

Write-Host "开始训练（前台运行，实时输出）..." -ForegroundColor Green
Write-Host "=====================================`n" -ForegroundColor Cyan

# 前台执行训练命令，保留所有实时输出
yolo train model=$model data=$data epochs=$epochs imgsz=$imgsz batch=$batch device=$device verbose=True

Write-Host "`n=====================================`n" -ForegroundColor Cyan
Write-Host "训练完成!" -ForegroundColor Green
