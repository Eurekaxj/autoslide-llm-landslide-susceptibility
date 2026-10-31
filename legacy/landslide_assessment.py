import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, 
                           QHBoxLayout, QLabel, QLineEdit, QPushButton, QFileDialog, 
                           QComboBox, QSpinBox, QDoubleSpinBox, QProgressBar, QMessageBox,
                           QGroupBox, QFormLayout, QTextEdit)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import pandas as pd
import numpy as np
from datetime import datetime

class DatasetPreparationWorker(QThread):
    """数据集准备的工作线程"""
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(bool, str)

    def __init__(self, params):
        super().__init__()
        self.params = params

    def run(self):
        try:
            from ArcgisTools import create_random_points, extract_values_to_points, calculate_lon_lat, export_to_excel
            
            self.status.emit("开始创建随机点...")
            random_points = create_random_points(self.params['run_num'])
            if not random_points:
                self.finished.emit(False, "创建随机点失败")
                return

            self.status.emit("开始提取属性值...")
            points_with_attrs = extract_values_to_points(random_points)
            if not points_with_attrs:
                self.finished.emit(False, "属性提取失败")
                return

            self.status.emit("计算经纬度...")
            if not calculate_lon_lat(points_with_attrs):
                self.finished.emit(False, "经纬度计算失败")
                return

            self.status.emit("导出数据集...")
            excel_output = export_to_excel(points_with_attrs, self.params['run_num'])
            if not excel_output:
                self.finished.emit(False, "数据集导出失败")
                return

            self.finished.emit(True, "数据集准备完成")
        except Exception as e:
            self.finished.emit(False, f"发生错误: {str(e)}")

class ModelTrainingWorker(QThread):
    """模型训练的工作线程"""
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(bool, str)

    def __init__(self, params):
        super().__init__()
        self.params = params

    def run(self):
        try:
            model_type = self.params['model_type']
            if model_type == 'XGBoost':
                from XGBoost_pram import train_model
            elif model_type == 'CatBoost':
                from CatBoost_param import train_model
            elif model_type == 'RandomForest':
                from RandomForest_param import train_model
            elif model_type == 'LR':
                from LR_param import train_model
            elif model_type == 'MLP':
                from MLP_param import train_model
            elif model_type == 'SVM':
                from SVM_param import train_model
            
            self.status.emit(f"开始训练{model_type}模型...")
            result = train_model(self.params)
            
            if result['success']:
                self.finished.emit(True, "模型训练完成")
            else:
                self.finished.emit(False, result['message'])
        except Exception as e:
            self.finished.emit(False, f"发生错误: {str(e)}")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        """初始化主窗口UI"""
        self.setWindowTitle('滑坡易发性评价系统')
        self.setGeometry(100, 100, 800, 600)

        # 创建主标签页
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # 创建三个主要模块的标签页
        self.file_management_tab = QWidget()
        self.dataset_preparation_tab = QWidget()
        self.model_design_tab = QWidget()

        self.setupFileManagementTab()
        self.setupDatasetPreparationTab()
        self.setupModelDesignTab()

        self.tabs.addTab(self.file_management_tab, "文件管理")
        self.tabs.addTab(self.dataset_preparation_tab, "数据集准备")
        self.tabs.addTab(self.model_design_tab, "模型设计")

    def setupFileManagementTab(self):
        """设置文件管理标签页"""
        layout = QVBoxLayout()

        # 滑坡点数据集选择
        landslide_group = QGroupBox("滑坡点数据集")
        landslide_layout = QHBoxLayout()
        self.landslide_path = QLineEdit()
        landslide_btn = QPushButton("选择文件")
        landslide_btn.clicked.connect(lambda: self.selectFile(self.landslide_path, "Excel Files (*.xls *.xlsx)"))
        landslide_layout.addWidget(self.landslide_path)
        landslide_layout.addWidget(landslide_btn)
        landslide_group.setLayout(landslide_layout)

        # 影响因子图层选择
        factors_group = QGroupBox("影响因子图层")
        factors_layout = QVBoxLayout()
        self.factors_list = QTextEdit()
        factors_btn = QPushButton("选择文件")
        factors_btn.clicked.connect(self.selectFactorFiles)
        factors_layout.addWidget(self.factors_list)
        factors_layout.addWidget(factors_btn)
        factors_group.setLayout(factors_layout)

        layout.addWidget(landslide_group)
        layout.addWidget(factors_group)
        layout.addStretch()

        self.file_management_tab.setLayout(layout)

    def setupDatasetPreparationTab(self):
        """设置数据集准备标签页"""
        layout = QVBoxLayout()

        # 非滑坡点采样范围选择
        sample_group = QGroupBox("非滑坡点采样设置")
        sample_layout = QFormLayout()
        
        self.sample_area_path = QLineEdit()
        sample_area_btn = QPushButton("选择文件")
        sample_area_btn.clicked.connect(lambda: self.selectFile(self.sample_area_path, "Shapefile (*.shp)"))
        sample_area_layout = QHBoxLayout()
        sample_area_layout.addWidget(self.sample_area_path)
        sample_area_layout.addWidget(sample_area_btn)
        
        self.sample_count = QSpinBox()
        self.sample_count.setRange(1, 1000)
        self.sample_count.setValue(348)
        
        self.sample_distance = QSpinBox()
        self.sample_distance.setRange(100, 10000)
        self.sample_distance.setValue(2000)
        self.sample_distance.setSuffix(" m")
        
        self.sample_rounds = QSpinBox()
        self.sample_rounds.setRange(1, 100)
        self.sample_rounds.setValue(100)
        
        self.output_dir = QLineEdit()
        output_dir_btn = QPushButton("选择目录")
        output_dir_btn.clicked.connect(self.selectOutputDir)
        output_dir_layout = QHBoxLayout()
        output_dir_layout.addWidget(self.output_dir)
        output_dir_layout.addWidget(output_dir_btn)

        sample_layout.addRow("采样范围:", sample_area_layout)
        sample_layout.addRow("采样点数量:", self.sample_count)
        sample_layout.addRow("最小间距:", self.sample_distance)
        sample_layout.addRow("采样轮次:", self.sample_rounds)
        sample_layout.addRow("输出目录:", output_dir_layout)
        
        sample_group.setLayout(sample_layout)

        # 进度显示
        progress_group = QGroupBox("进度")
        progress_layout = QVBoxLayout()
        self.dataset_progress = QProgressBar()
        self.dataset_status = QLabel("就绪")
        progress_layout.addWidget(self.dataset_progress)
        progress_layout.addWidget(self.dataset_status)
        progress_group.setLayout(progress_layout)

        # 控制按钮
        control_layout = QHBoxLayout()
        self.start_dataset_btn = QPushButton("开始处理")
        self.start_dataset_btn.clicked.connect(self.startDatasetPreparation)
        control_layout.addWidget(self.start_dataset_btn)

        layout.addWidget(sample_group)
        layout.addWidget(progress_group)
        layout.addLayout(control_layout)
        layout.addStretch()

        self.dataset_preparation_tab.setLayout(layout)

    def setupModelDesignTab(self):
        """设置模型设计标签页"""
        layout = QVBoxLayout()

        # 模型选择和参数设置
        model_group = QGroupBox("模型设置")
        model_layout = QFormLayout()
        
        self.model_type = QComboBox()
        self.model_type.addItems(['XGBoost', 'CatBoost', 'RandomForest', 'LR', 'MLP', 'SVM'])
        self.model_type.currentTextChanged.connect(self.updateParamFields)
        
        model_layout.addRow("选择模型:", self.model_type)
        
        # 创建参数输入区域
        self.param_group = QGroupBox("模型参数")
        self.param_layout = QFormLayout()
        self.param_group.setLayout(self.param_layout)
        
        model_group.setLayout(model_layout)

        # 数据集选择
        data_group = QGroupBox("数据集选择")
        data_layout = QFormLayout()
        
        self.train_data_path = QLineEdit()
        train_data_btn = QPushButton("选择文件")
        train_data_btn.clicked.connect(lambda: self.selectFile(self.train_data_path, "Excel Files (*.xls *.xlsx)"))
        train_data_layout = QHBoxLayout()
        train_data_layout.addWidget(self.train_data_path)
        train_data_layout.addWidget(train_data_btn)
        
        self.predict_data_path = QLineEdit()
        predict_data_btn = QPushButton("选择文件")
        predict_data_btn.clicked.connect(lambda: self.selectFile(self.predict_data_path, "Excel Files (*.xls *.xlsx)"))
        predict_data_layout = QHBoxLayout()
        predict_data_layout.addWidget(self.predict_data_path)
        predict_data_layout.addWidget(predict_data_btn)
        
        data_layout.addRow("训练数据:", train_data_layout)
        data_layout.addRow("预测数据:", predict_data_layout)
        data_group.setLayout(data_layout)

        # 进度显示
        progress_group = QGroupBox("进度")
        progress_layout = QVBoxLayout()
        self.model_progress = QProgressBar()
        self.model_status = QLabel("就绪")
        progress_layout.addWidget(self.model_progress)
        progress_layout.addWidget(self.model_status)
        progress_group.setLayout(progress_layout)

        # 控制按钮
        control_layout = QHBoxLayout()
        self.start_model_btn = QPushButton("开始训练")
        self.start_model_btn.clicked.connect(self.startModelTraining)
        control_layout.addWidget(self.start_model_btn)

        layout.addWidget(model_group)
        layout.addWidget(self.param_group)
        layout.addWidget(data_group)
        layout.addWidget(progress_group)
        layout.addLayout(control_layout)
        layout.addStretch()

        self.model_design_tab.setLayout(layout)
        
        # 初始化默认模型的参数字段
        self.updateParamFields(self.model_type.currentText())

    def selectFile(self, line_edit, file_filter):
        """通用文件选择对话框"""
        file_path, _ = QFileDialog.getOpenFileName(self, "选择文件", "", file_filter)
        if file_path:
            line_edit.setText(file_path)

    def selectFactorFiles(self):
        """选择多个影响因子文件"""
        files, _ = QFileDialog.getOpenFileNames(self, "选择影响因子文件", "", "TIF Files (*.tif)")
        if files:
            self.factors_list.setText("\n".join(files))

    def selectOutputDir(self):
        """选择输出目录"""
        directory = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if directory:
            self.output_dir.setText(directory)

    def updateParamFields(self, model_type):
        """根据选择的模型类型更新参数输入字段"""
        # 清除现有的参数字段
        for i in reversed(range(self.param_layout.count())): 
            self.param_layout.itemAt(i).widget().setParent(None)

        # 根据模型类型添加相应的参数字段
        if model_type == 'XGBoost':
            params = {
                'n_estimators': (100, 1000, 100),
                'max_depth': (2, 10, 3),
                'learning_rate': (0.01, 0.3, 0.1),
                'subsample': (0.5, 1.0, 0.8),
                'colsample_bytree': (0.5, 1.0, 0.8),
                'min_child_weight': (1, 10, 5)
            }
        elif model_type == 'CatBoost':
            params = {
                'iterations': (100, 1000, 100),
                'depth': (2, 10, 3),
                'learning_rate': (0.01, 0.3, 0.1),
                'l2_leaf_reg': (1, 10, 3)
            }
        # ... 其他模型的参数设置 ...

        # 创建参数输入字段
        for param_name, (min_val, max_val, default) in params.items():
            if isinstance(default, int):
                spin_box = QSpinBox()
            else:
                spin_box = QDoubleSpinBox()
            
            spin_box.setRange(min_val, max_val)
            spin_box.setValue(default)
            spin_box.setObjectName(f"param_{param_name}")
            
            self.param_layout.addRow(param_name + ":", spin_box)

    def startDatasetPreparation(self):
        """开始数据集准备过程"""
        # 检查输入
        if not self.validateDatasetInputs():
            return

        # 准备参数
        params = {
            'sample_area': self.sample_area_path.text(),
            'sample_count': self.sample_count.value(),
            'sample_distance': self.sample_distance.value(),
            'output_dir': self.output_dir.text(),
            'run_num': self.sample_rounds.value()
        }

        # 创建工作线程
        self.dataset_worker = DatasetPreparationWorker(params)
        self.dataset_worker.progress.connect(self.updateDatasetProgress)
        self.dataset_worker.status.connect(self.updateDatasetStatus)
        self.dataset_worker.finished.connect(self.onDatasetFinished)

        # 禁用开始按钮
        self.start_dataset_btn.setEnabled(False)
        
        # 开始处理
        self.dataset_worker.start()

    def startModelTraining(self):
        """开始模型训练过程"""
        # 检查输入
        if not self.validateModelInputs():
            return

        # 获取当前模型的参数值
        params = {
            'model_type': self.model_type.currentText(),
            'train_data': self.train_data_path.text(),
            'predict_data': self.predict_data_path.text()
        }
        
        # 添加模型特定的参数
        for i in range(self.param_layout.rowCount()):
            param_widget = self.param_layout.itemAt(i, QFormLayout.FieldRole).widget()
            param_name = param_widget.objectName().replace('param_', '')
            params[param_name] = param_widget.value()

        # 创建工作线程
        self.model_worker = ModelTrainingWorker(params)
        self.model_worker.progress.connect(self.updateModelProgress)
        self.model_worker.status.connect(self.updateModelStatus)
        self.model_worker.finished.connect(self.onModelFinished)

        # 禁用开始按钮
        self.start_model_btn.setEnabled(False)
        
        # 开始训练
        self.model_worker.start()

    def validateDatasetInputs(self):
        """验证数据集准备的输入"""
        if not os.path.exists(self.sample_area_path.text()):
            QMessageBox.warning(self, "输入错误", "请选择有效的采样范围文件")
            return False
        if not self.output_dir.text():
            QMessageBox.warning(self, "输入错误", "请选择输出目录")
            return False
        return True

    def validateModelInputs(self):
        """验证模型训练的输入"""
        if not os.path.exists(self.train_data_path.text()):
            QMessageBox.warning(self, "输入错误", "请选择训练数据文件")
            return False
        if not os.path.exists(self.predict_data_path.text()):
            QMessageBox.warning(self, "输入错误", "请选择预测数据文件")
            return False
        return True

    def updateDatasetProgress(self, value):
        """更新数据集准备进度"""
        self.dataset_progress.setValue(value)

    def updateDatasetStatus(self, message):
        """更新数据集准备状态"""
        self.dataset_status.setText(message)

    def updateModelProgress(self, value):
        """更新模型训练进度"""
        self.model_progress.setValue(value)

    def updateModelStatus(self, message):
        """更新模型训练状态"""
        self.model_status.setText(message)

    def onDatasetFinished(self, success, message):
        """数据集准备完成的处理"""
        self.start_dataset_btn.setEnabled(True)
        if success:
            QMessageBox.information(self, "完成", message)
        else:
            QMessageBox.warning(self, "错误", message)

    def onModelFinished(self, success, message):
        """模型训练完成的处理"""
        self.start_model_btn.setEnabled(True)
        if success:
            QMessageBox.information(self, "完成", message)
        else:
            QMessageBox.warning(self, "错误", message)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_()) 