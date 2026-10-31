import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QLabel, QLineEdit, QPushButton, QFileDialog, 
                           QComboBox, QSpinBox, QDoubleSpinBox, QProgressBar, QMessageBox,
                           QGroupBox, QFormLayout)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import pandas as pd
import numpy as np
from datetime import datetime

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
        self.setWindowTitle('滑坡易发性评价系统 - 模型训练')
        self.setGeometry(100, 100, 800, 600)

        # 创建中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # 模型选择和参数设置
        model_group = QGroupBox("模型设置")
        model_layout = QFormLayout()
        
        self.model_type = QComboBox()
        self.model_type.addItems(['XGBoost', 'CatBoost', 'RandomForest', 'LR', 'MLP', 'SVM'])
        self.model_type.currentTextChanged.connect(self.updateParamFields)
        
        model_layout.addRow("选择模型:", self.model_type)
        model_group.setLayout(model_layout)

        # 创建参数输入区域
        self.param_group = QGroupBox("模型参数")
        self.param_layout = QFormLayout()
        self.param_group.setLayout(self.param_layout)

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

        # 添加所有组件到主布局
        layout.addWidget(model_group)
        layout.addWidget(self.param_group)
        layout.addWidget(data_group)
        layout.addWidget(progress_group)
        layout.addLayout(control_layout)
        layout.addStretch()

        # 初始化默认模型的参数字段
        self.updateParamFields(self.model_type.currentText())

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
        elif model_type == 'RandomForest':
            params = {
                'n_estimators': (100, 1000, 100),
                'max_depth': (2, 10, 3),
                'min_samples_split': (2, 20, 5),
                'min_samples_leaf': (1, 10, 2)
            }
        elif model_type == 'LR':
            params = {
                'C': (0.001, 10.0, 1.0),
                'max_iter': (100, 1000, 100)
            }
        elif model_type == 'MLP':
            params = {
                'hidden_layer_sizes': (10, 200, 100),
                'alpha': (0.0001, 0.01, 0.001),
                'learning_rate_init': (0.001, 0.1, 0.01),
                'max_iter': (100, 1000, 200)
            }
        elif model_type == 'SVM':
            params = {
                'C': (0.1, 10.0, 1.0),
                'gamma': (0.01, 1.0, 0.1),
                'kernel': ['rbf', 'linear', 'poly']
            }

        # 创建参数输入字段
        for param_name, param_config in params.items():
            if isinstance(param_config, tuple):
                min_val, max_val, default = param_config
                if isinstance(default, int):
                    spin_box = QSpinBox()
                else:
                    spin_box = QDoubleSpinBox()
                    spin_box.setDecimals(4)
                
                spin_box.setRange(min_val, max_val)
                spin_box.setValue(default)
                spin_box.setObjectName(f"param_{param_name}")
                self.param_layout.addRow(f"{param_name}:", spin_box)
            elif isinstance(param_config, list):
                combo_box = QComboBox()
                combo_box.addItems(param_config)
                combo_box.setObjectName(f"param_{param_name}")
                self.param_layout.addRow(f"{param_name}:", combo_box)

    def selectFile(self, line_edit, file_filter):
        """通用文件选择对话框"""
        file_path, _ = QFileDialog.getOpenFileName(self, "选择文件", "", file_filter)
        if file_path:
            line_edit.setText(file_path)

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
            if isinstance(param_widget, (QSpinBox, QDoubleSpinBox)):
                params[param_name] = param_widget.value()
            elif isinstance(param_widget, QComboBox):
                params[param_name] = param_widget.currentText()

        # 创建工作线程
        self.model_worker = ModelTrainingWorker(params)
        self.model_worker.progress.connect(self.updateModelProgress)
        self.model_worker.status.connect(self.updateModelStatus)
        self.model_worker.finished.connect(self.onModelFinished)

        # 禁用开始按钮
        self.start_model_btn.setEnabled(False)
        
        # 开始训练
        self.model_worker.start()

    def validateModelInputs(self):
        """验证模型训练的输入"""
        if not os.path.exists(self.train_data_path.text()):
            QMessageBox.warning(self, "输入错误", "请选择训练数据文件")
            return False
        if not os.path.exists(self.predict_data_path.text()):
            QMessageBox.warning(self, "输入错误", "请选择预测数据文件")
            return False
        return True

    def updateModelProgress(self, value):
        """更新模型训练进度"""
        self.model_progress.setValue(value)

    def updateModelStatus(self, message):
        """更新模型训练状态"""
        self.model_status.setText(message)

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