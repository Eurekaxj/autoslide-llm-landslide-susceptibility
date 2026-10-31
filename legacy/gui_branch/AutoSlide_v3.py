# -*- coding: utf-8 -*-
import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont,QIcon
import subprocess
import os
import threading

class FileManagementTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        # 页面顶部说明
        self.top_label = QLabel('Automated Landslide Susceptibility Evaluation and Data Processing')
        self.top_label.setFont(QFont("Times New Roman", 16))
        self.top_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.top_label)
        # 滑坡点数据集
        self.landslide_xls_label = QLabel('Landslide dataset(.xls/.xlsx):')
        self.landslide_xls_path = QLineEdit()
        self.landslide_xls_btn = QPushButton('Select file')
        self.landslide_xls_btn.clicked.connect(self.select_landslide_xls)
        h1 = QHBoxLayout()
        h1.addWidget(self.landslide_xls_label)
        h1.addWidget(self.landslide_xls_path)
        h1.addWidget(self.landslide_xls_btn)
        layout.addLayout(h1)
        # 影响因子图层（多行文本输入.tif路径 字段名）
        self.factor_lyr_label = QLabel('Select influencing factor layer(.tif):')
        self.factor_lyr_text = QTextEdit()
        self.factor_lyr_text.setPlaceholderText('Fill in "path" on each line,such as：\nD:/Data/DEM.tif')
        layout.addWidget(self.factor_lyr_label)
        layout.addWidget(self.factor_lyr_text)
        self.setLayout(layout)

    def select_landslide_xls(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Select landslide dataset', '', 'Excel files (*.xls *.xlsx)')
        if path:
            self.landslide_xls_path.setText(path)


class DatasetPreparationThread(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool)

    def __init__(self, python27_path, arcgis_tool_path, params):
        super().__init__()
        self.python27_path = python27_path
        self.arcgis_tool_path = arcgis_tool_path
        self.params = params
        self.process = None
        self._is_running = True
        self.lock = threading.Lock()  # 添加锁

    def run(self):
        try:
            cmd = [
                self.python27_path,
                '-u',  # 无缓冲输出
                self.arcgis_tool_path,
                '--output_dir', self.params['output_dir'],
                '--constraining_shp', self.params['constraining_shp'],
                '--num_points', str(self.params['num_points']),
                '--min_distance', str(self.params['min_distance']),
                '--input_rasters', self.params['input_rasters'],
                '--template_xls', self.params['template_xls'],
                '--loop', str(self.params['loop'])
            ]

            self.log_signal.emit(f"Execute: {' '.join(cmd)}")

            # 设置工作目录为脚本所在目录
            working_dir = os.path.dirname(self.arcgis_tool_path)

            # 创建子进程
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1,  # 行缓冲
                universal_newlines=True,
                cwd=working_dir,
                # shell=True,  # 去除该参数
                env=os.environ,
                encoding='utf-8'  # 强制utf-8解码
            )

            # 创建单独线程来读取输出
            def reader(pipe, pipe_name):
                try:
                    with pipe:
                        for line in iter(pipe.readline, ''):
                            if line and self._is_running:
                                decoded_line = line.strip()
                                with self.lock:  # 使用锁保证日志输出顺序
                                    self.log_signal.emit(f"[{pipe_name}] {decoded_line}")
                except Exception as e:
                    with self.lock:  # 使用锁保证日志输出顺序
                        self.log_signal.emit(f"Read {pipe_name} Fail: {str(e)}")

            # 启动输出读取线程
            import threading
            stdout_thread = threading.Thread(target=reader, args=(self.process.stdout, "stdout"))
            stderr_thread = threading.Thread(target=reader, args=(self.process.stderr, "stderr"))
            stdout_thread.daemon = True
            stderr_thread.daemon = True
            stdout_thread.start()
            stderr_thread.start()

            # 等待进程结束
            while self._is_running:
                if self.process.poll() is not None:
                    break
                self.msleep(100)

            # 确保读取线程完成
            stdout_thread.join(timeout=0.1)
            stderr_thread.join(timeout=0.1)

            return_code = self.process.returncode
            self.log_signal.emit(f"Thread ends, return code: {return_code}")
            self.finished_signal.emit(return_code == 0)

        except Exception as e:
            self.log_signal.emit(f"Thread execution error: {str(e)}")
            self.finished_signal.emit(False)

    def stop(self):
        self._is_running = False
        if self.process and self.process.poll() is None:
            self.process.terminate()

class DatasetPreparationTab(QWidget):
    def __init__(self, file_tab=None, main_window=None):
        super().__init__()
        self.file_tab = file_tab  # 传入FileManagementTab实例
        self.main_window = main_window  # 传入主窗口实例
        layout = QVBoxLayout()
        # 页面顶部说明
        self.top_label = QLabel('Automated Landslide Susceptibility Evaluation and Data Processing')
        self.top_label.setFont(QFont("Times New Roman", 16))
        self.top_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.top_label)
        # 非滑坡点采样范围
        self.nolandslide_shp_label = QLabel('Non-landslide sampling area(.shp):')
        self.nolandslide_shp_path = QLineEdit()
        self.nolandslide_shp_btn = QPushButton('Select file')
        self.nolandslide_shp_btn.clicked.connect(self.select_nolandslide_shp)
        h1 = QHBoxLayout()
        h1.addWidget(self.nolandslide_shp_label)
        h1.addWidget(self.nolandslide_shp_path)
        h1.addWidget(self.nolandslide_shp_btn)
        layout.addLayout(h1)

        # 采样数量
        self.num_label = QLabel('Quantity of non-landslide points:')
        self.num_spin = QSpinBox()
        self.num_spin.setRange(1, 100000)
        h2 = QHBoxLayout()
        h2.addWidget(self.num_label)
        h2.addWidget(self.num_spin)
        layout.addLayout(h2)

        # 点间距
        self.dist_label = QLabel('Minimum spatial separation distance between adjacent points(m):')
        self.dist_spin = QSpinBox()
        self.dist_spin.setRange(0, 10000)
        self.dist_spin.setSingleStep(1)  # 设置步长为1，从而不能为小数
        h3 = QHBoxLayout()
        h3.addWidget(self.dist_label)
        h3.addWidget(self.dist_spin)
        layout.addLayout(h3)

        # 循环次数
        self.loop_label = QLabel('Loop execution count:')
        self.loop_spin = QSpinBox()
        self.loop_spin.setRange(1, 1000)
        h4 = QHBoxLayout()
        h4.addWidget(self.loop_label)
        h4.addWidget(self.loop_spin)
        layout.addLayout(h4)

        # 输出目录
        self.output_label = QLabel('Output directory:')
        self.output_path = QLineEdit()
        self.output_btn = QPushButton('Select directory')
        self.output_btn.clicked.connect(self.select_output_dir)
        h5 = QHBoxLayout()
        h5.addWidget(self.output_label)
        h5.addWidget(self.output_path)
        h5.addWidget(self.output_btn)
        layout.addLayout(h5)

        # 将QLineEdit改为QTextEdit以便显示多行日志
        self.log_label = QLabel('Log:')
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(150)
        layout.addWidget(self.log_label)
        layout.addWidget(self.log_text)

        # 执行按钮
        self.run_btn = QPushButton('Start batch processing')
        self.run_btn.clicked.connect(self.run_arcgis_tool)
        layout.addWidget(self.run_btn)
        self.setLayout(layout)
        self.thread = None

    def select_nolandslide_shp(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Select non-landslide sampling area', '', 'Shapefile (*.shp)')
        if path:
            self.nolandslide_shp_path.setText(path)

    def select_output_dir(self):
        path = QFileDialog.getExistingDirectory(self, 'Select output directory')
        if path:
            self.output_path.setText(path)

    def append_log(self, msg):
        # 确保在主线程中更新UI
        QApplication.processEvents()
        self.log_text.append(msg)
        # 自动滚动到底部
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def finish_log(self, success):
        if success:
            self.append_log("Success！")
        else:
            self.append_log("Fail，Please check the above log！")

        # 确保UI更新
        QApplication.processEvents()

    def get_input_rasters(self):
        # 从FileManagementTab获取多值提取的栅格路径（.tif路径，每行一个，合并为分号分隔）
        if self.file_tab:
            lines = self.file_tab.factor_lyr_text.toPlainText().splitlines()
            rasters = [line.strip() for line in lines if line.strip()]
            return ';'.join(rasters)
        return ''

    def get_template_xls(self):
        # 从FileManagementTab获取模板Excel路径
        if self.file_tab:
            return self.file_tab.landslide_xls_path.text()
        return ''

    def run_arcgis_tool(self):
        # 1. 收集参数
        params = {
            'output_dir': self.output_path.text(),
            'constraining_shp': self.nolandslide_shp_path.text(),
            'num_points': self.num_spin.value(),
            'min_distance': self.dist_spin.value(),
            'input_rasters': self.get_input_rasters(),
            'template_xls': self.get_template_xls(),
            'loop': self.loop_spin.value()
        }
        # 2. 检查参数完整性
        for k, v in params.items():
            if not v:
                self.log_text.setText('Parameter {} cannot be empty'.format(k))
                return

        # 3. 配置Python2.7路径，需按自己的电脑实际情况配置！！！
        python27_path = r'C:\Python27\ArcGIS10.8\python.exe'
        # 配置ArcgisTool.py路径
        def get_app_dir():
            if getattr(sys, 'frozen', False):
                # 打包后
                return os.path.dirname(sys.executable)
            else:
                # 源代码运行
                return os.path.dirname(os.path.abspath(__file__))

        arcgis_tool_path = os.path.join(get_app_dir(), 'ArcgisTool.py')

        # arcgis_tool_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'ArcgisTool.py'))
        #
        # # 获取脚本绝对路径
        # script_dir = os.path.dirname(os.path.abspath(__file__))
        # arcgis_tool_path = os.path.join(script_dir, 'ArcgisTool.py')


        if not os.path.exists(arcgis_tool_path):
            self.log_text.setText(f'Error: Cannot find ArcgisTool.py in {arcgis_tool_path}')
            return

        # 4. 启动线程
        self.log_text.setText('Start execute...')
        self.thread = DatasetPreparationThread(python27_path, arcgis_tool_path, params)
        self.thread.log_signal.connect(self.append_log)
        self.thread.finished_signal.connect(self.finish_log)
        self.thread.start()

        if self.main_window:
            self.main_window.threads.append(self.thread)

class ModelDesignThread(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool)

    def __init__(self, python3_path, script_path, params):
        super().__init__()
        self.python3_path = python3_path
        self.script_path = script_path
        self.params = params

    def run(self):
        cmd = [
            self.python3_path,
            self.script_path,
            '--result_dir', self.params['result_dir'],
            '--predictres_dir', self.params['predictres_dir'],
            '--prediction_file', self.params['prediction_file'],
            '--train_files', self.params['train_files'],
            '--param_grid', self.params['param_grid'],
            '--features', self.params['features'],
            '--label_col', self.params['label_col']
        ]
        try:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True,creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            for line in iter(process.stdout.readline, ''):
                if line:
                    self.log_signal.emit(line.strip())
            process.stdout.close()
            process.wait()
            if process.returncode != 0:
                self.log_signal.emit(f'Subprocess Exited Abnormally, Return Code: {process.returncode}')
            self.finished_signal.emit(process.returncode == 0)
        except Exception as e:
            self.log_signal.emit('Execution Error: {}'.format(str(e)))
            self.finished_signal.emit(False)

class ModelDesignTab(QWidget):
    def __init__(self, main_window=None):
        super().__init__()
        self.main_window = main_window
        layout = QVBoxLayout()
        # 页面顶部说明
        self.top_label = QLabel('Automated Landslide Susceptibility Evaluation and Data Processing')
        self.top_label.setFont(QFont("Times New Roman", 16))
        self.top_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.top_label)
        # 选择机器学习方法
        self.model_label = QLabel('Select machine learning method:')
        self.model_combo = QComboBox()
        self.model_combo.addItems(['XGBoost_v3', 'XGBoost', 'CatBoost','SVM','LR','RF','MLP'])
        # 超参数范围
        self.param_label = QLabel('Hyperparameter search ranges:')
        self.param_edit = QTextEdit()
        self.param_edit.setPlaceholderText('one parameter type per line,such as:\nparam1=1,2,3\nparam2=0.1,0.2')
        self.param_edit.setFixedHeight(80)
        layout.addWidget(self.model_label)
        layout.addWidget(self.model_combo)
        layout.addWidget(self.param_label)
        layout.addWidget(self.param_edit)
        # 训练/验证数据集（多文件）
        self.train_label = QLabel('Select taining and testing set(.xls/.xlsx):')
        self.train_text = QTextEdit()
        self.train_text.setPlaceholderText('Fill in "path" on each line,such as：\nD:/Data/Dataset1.xls')
        self.train_text.setFixedHeight(80)
        self.train_btn = QPushButton('Select file')
        self.train_btn.clicked.connect(self.select_train_files)
        h2 = QHBoxLayout()
        h2.addWidget(self.train_label)
        h2.addWidget(self.train_btn)
        layout.addLayout(h2)
        layout.addWidget(self.train_text)
        # 预测数据集（只允许csv）
        self.pred_label = QLabel('Prediction set(.xls/.xlsx/.csv):')
        self.pred_path = QLineEdit()
        self.pred_btn = QPushButton('Select file')
        self.pred_btn.clicked.connect(self.select_pred_file)
        h3 = QHBoxLayout()
        h3.addWidget(self.pred_label)
        h3.addWidget(self.pred_path)
        h3.addWidget(self.pred_btn)
        layout.addLayout(h3)
        # features输入
        self.feat_label = QLabel('Features(Comma-separated or one per Line):')
        self.feat_text = QTextEdit()
        self.feat_text.setPlaceholderText('DEM,slope,aspect,... or one per Line')
        self.feat_text.setFixedHeight(50)
        layout.addWidget(self.feat_label)
        layout.addWidget(self.feat_text)
        # 输出目录
        self.output_label = QLabel('Output directory:')
        self.output_path = QLineEdit()
        self.output_btn = QPushButton('Select directory')
        self.output_btn.clicked.connect(self.select_output_dir)
        h4 = QHBoxLayout()
        h4.addWidget(self.output_label)
        h4.addWidget(self.output_path)
        h4.addWidget(self.output_btn)
        layout.addLayout(h4)
        # 日志
        self.log_label = QLabel('Log:')
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(120)
        layout.addWidget(self.log_label)
        layout.addWidget(self.log_text)
        # 执行按钮
        self.run_btn = QPushButton('Start training,testing,and prediction')
        self.run_btn.clicked.connect(self.run_model)
        layout.addWidget(self.run_btn)
        self.setLayout(layout)
        self.thread = None

    def select_train_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, 'Select training and testing set', '', 'Excel Files (*.xls *.xlsx);;CSV Files (*.csv)')
        if paths:
            self.train_text.setPlainText('\n'.join(paths))

    def select_pred_file(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Select prediction set', '', 'Excel Files (*.xls *.xlsx);;CSV Files (*.csv)')
        if path:
            self.pred_path.setText(path)

    def select_output_dir(self):
        path = QFileDialog.getExistingDirectory(self, 'Select output directory')
        if path:
            self.output_path.setText(path)

    def append_log(self, msg):
        QApplication.processEvents()
        self.log_text.append(msg)
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def finish_log(self, success):
        if success:
            self.append_log('Success！')
        else:
            self.append_log('Fail，Please check the above log！')
        QApplication.processEvents()

    def run_model(self):
        # 1. 收集参数
        train_files = [line.strip() for line in self.train_text.toPlainText().splitlines() if line.strip()]
        if not train_files:
            self.append_log('Taining and testing set cannot be empty')
            return
        features = self.feat_text.toPlainText().replace('\n', ',').replace('，', ',').replace(' ', '').strip(',')
        if not features:
            self.append_log('Feature cannot be empty')
            return
        param_grid = self.param_edit.toPlainText().replace('\n', ';')
        params = {
            'result_dir': self.output_path.text(),
            'predictres_dir': self.output_path.text(),
            'prediction_file': self.pred_path.text(),
            'train_files': ';'.join(train_files),
            'param_grid': param_grid,
            'features': features,
            'label_col': 'landslide'
        }
        # 检查输出目录
        if not params['result_dir']:
            self.append_log('Output directory cannot be empty')
            return
        if not params['prediction_file']:
            self.append_log('Prediction set cannot be empty')
            return

        # 配置Python3.11路径，需按自己的电脑实际情况配置！！！
        python3_path = r'D:\Anoconda\envs\env3.11\python.exe'
        model_name = self.model_combo.currentText()
        # 配置模型脚本运行路径
        def get_app_dir():
            if getattr(sys, 'frozen', False):
                # 打包后
                return os.path.dirname(sys.executable)
            else:
                # 源码执行
                return os.path.dirname(os.path.abspath(__file__))

        script_path = os.path.abspath(os.path.join(get_app_dir(), f'{model_name}.py'))

        # script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), f'{model_name}.py'))

        self.append_log(f'Execute files: {params["train_files"]}')
        self.thread = ModelDesignThread(python3_path, script_path, params)
        self.thread.log_signal.connect(self.append_log)
        self.thread.finished_signal.connect(self.finish_log)
        self.thread.start()
        if self.main_window:
            self.main_window.threads.append(self.thread)

class ChatGPTTab(QWidget):
    def __init__(self):
        super().__init__()
        # 空白Tab，点击时自动跳转
    def showEvent(self, event):
        import webbrowser
        webbrowser.open('https://chat.openai.com/')
        super().showEvent(event)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('AutoSlide')
        self.resize(800, 600)

        # 添加图标设置
        if getattr(sys, 'frozen', False):
            # 打包后
            icon_path = sys._MEIPASS if hasattr(sys, '_MEIPASS') else os.path.dirname(sys.executable)
        else:
            # 源码执行
            icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'autoslide_icon.ico')

        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        self.tabs = QTabWidget()
        self.threads = []  # 线程池
        self.file_tab = FileManagementTab()
        self.dataset_tab = DatasetPreparationTab(file_tab=self.file_tab, main_window=self)
        self.model_tab = ModelDesignTab(main_window=self)
        self.chatgpt_tab = ChatGPTTab()
        self.tabs.addTab(self.file_tab, 'File management')
        self.tabs.addTab(self.dataset_tab, 'Dataset preparation')
        self.tabs.addTab(self.model_tab, 'Model design')
        self.tabs.addTab(self.chatgpt_tab, 'ChatGPT')
        self.setCentralWidget(self.tabs)

    def closeEvent(self, event):
        # 关闭前终止所有线程
        for thread in self.threads:
            if hasattr(thread, 'stop'):
                thread.stop()
            if thread.isRunning():
                thread.quit()
                thread.wait(1000)
        event.accept()

def main():
    app = QApplication(sys.argv)
    # 设置字体格式
    font = QFont("Times New Roman", 14)
    app.setFont(font)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main() 