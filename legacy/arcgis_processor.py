# -*- coding: utf-8 -*-
import os
import sys
import arcpy
import threading
import Tkinter as tk
import ttk
import tkFileDialog as filedialog
import tkMessageBox as messagebox
import os


class DatasetPreparationThread(threading.Thread):
    """数据集准备的工作线程"""

    def __init__(self, params, callback):
        threading.Thread.__init__(self)  # Python 2.7 初始化方式
        self.params = params
        self.callback = callback
        self._stop = threading.Event()

    def run(self):
        try:
            from ArcgisTools import create_random_points, extract_values_to_points, calculate_lon_lat, export_to_excel

            self.callback("status", u"开始创建随机点...")
            random_points = create_random_points(self.params['run_num'])
            if not random_points:
                self.callback("finished", False, u"创建随机点失败")
                return

            self.callback("status", u"开始提取属性值...")
            points_with_attrs = extract_values_to_points(random_points)
            if not points_with_attrs:
                self.callback("finished", False, u"属性提取失败")
                return

            self.callback("status", u"计算经纬度...")
            if not calculate_lon_lat(points_with_attrs):
                self.callback("finished", False, u"经纬度计算失败")
                return

            self.callback("status", u"导出数据集...")
            excel_output = export_to_excel(points_with_attrs, self.params['run_num'])
            if not excel_output:
                self.callback("finished", False, u"数据集导出失败")
                return

            self.callback("finished", True, u"数据集准备完成")
        except Exception as e:
            self.callback("finished", False, u"发生错误: " + str(e))

    def stop(self):
        self._stop.set()


class MainWindow(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)  # Python 2.7 初始化方式
        self.title(u'滑坡数据处理系统')
        self.geometry('800x600')
        self.resizable(True, True)

        # 创建主notebook
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(expand=True, fill='both', padx=5, pady=5)

        # 创建两个主要模块的标签页
        self.file_management_tab = ttk.Frame(self.notebook)
        self.dataset_preparation_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.file_management_tab, text=u'文件管理')
        self.notebook.add(self.dataset_preparation_tab, text=u'数据集准备')

        self.setupFileManagementTab()
        self.setupDatasetPreparationTab()

    def setupFileManagementTab(self):
        """设置文件管理标签页"""
        # 滑坡点数据集选择
        landslide_frame = ttk.LabelFrame(self.file_management_tab, text=u'滑坡点数据集')
        landslide_frame.pack(fill='x', padx=5, pady=5)

        self.landslide_path = ttk.Entry(landslide_frame)
        self.landslide_path.pack(side='left', fill='x', expand=True, padx=5, pady=5)

        landslide_btn = ttk.Button(landslide_frame, text=u'选择文件',
                                   command=lambda: self.selectFile(self.landslide_path,
                                                                   [('Excel Files', '*.xls;*.xlsx')]))
        landslide_btn.pack(side='right', padx=5, pady=5)

        # 影响因子图层选择
        factors_frame = ttk.LabelFrame(self.file_management_tab, text=u'影响因子图层')
        factors_frame.pack(fill='both', expand=True, padx=5, pady=5)

        self.factors_list = tk.Text(factors_frame, height=10)
        self.factors_list.pack(fill='both', expand=True, padx=5, pady=5)

        factors_btn = ttk.Button(factors_frame, text=u'选择文件',
                                 command=self.selectFactorFiles)
        factors_btn.pack(padx=5, pady=5)

    def setupDatasetPreparationTab(self):
        """设置数据集准备标签页"""
        # 非滑坡点采样设置
        sample_frame = ttk.LabelFrame(self.dataset_preparation_tab, text=u'非滑坡点采样设置')
        sample_frame.pack(fill='x', padx=5, pady=5)

        # 采样范围选择
        ttk.Label(sample_frame, text=u'采样范围:').grid(row=0, column=0, padx=5, pady=5)
        self.sample_area_path = ttk.Entry(sample_frame)
        self.sample_area_path.grid(row=0, column=1, sticky='ew', padx=5, pady=5)
        sample_area_btn = ttk.Button(sample_frame, text=u'选择文件',
                                     command=lambda: self.selectFile(self.sample_area_path,
                                                                     [('Shapefile', '*.shp')]))
        sample_area_btn.grid(row=0, column=2, padx=5, pady=5)

        # 采样点数量
        ttk.Label(sample_frame, text=u'采样点数量:').grid(row=1, column=0, padx=5, pady=5)
        self.sample_count = tk.Spinbox(sample_frame, from_=1, to=1000)
        self.sample_count.delete(0, 'end')
        self.sample_count.insert(0, '348')
        self.sample_count.grid(row=1, column=1, sticky='ew', padx=5, pady=5)

        # 最小间距
        ttk.Label(sample_frame, text=u'最小间距(m):').grid(row=2, column=0, padx=5, pady=5)
        self.sample_distance = tk.Spinbox(sample_frame, from_=100, to=10000)
        self.sample_distance.delete(0, 'end')
        self.sample_distance.insert(0, '2000')
        self.sample_distance.grid(row=2, column=1, sticky='ew', padx=5, pady=5)

        # 采样轮次
        ttk.Label(sample_frame, text=u'采样轮次:').grid(row=3, column=0, padx=5, pady=5)
        self.sample_rounds = tk.Spinbox(sample_frame, from_=1, to=100)
        self.sample_rounds.delete(0, 'end')
        self.sample_rounds.insert(0, '100')
        self.sample_rounds.grid(row=3, column=1, sticky='ew', padx=5, pady=5)

        # 输出目录
        ttk.Label(sample_frame, text=u'输出目录:').grid(row=4, column=0, padx=5, pady=5)
        self.output_dir = ttk.Entry(sample_frame)
        self.output_dir.grid(row=4, column=1, sticky='ew', padx=5, pady=5)
        output_dir_btn = ttk.Button(sample_frame, text=u'选择目录',
                                    command=self.selectOutputDir)
        output_dir_btn.grid(row=4, column=2, padx=5, pady=5)

        # 配置grid列权重
        sample_frame.columnconfigure(1, weight=1)

        # 状态显示
        status_frame = ttk.LabelFrame(self.dataset_preparation_tab, text=u'状态')
        status_frame.pack(fill='x', padx=5, pady=5)

        self.status_label = ttk.Label(status_frame, text=u'就绪')
        self.status_label.pack(padx=5, pady=5)

        # 控制按钮
        self.start_btn = ttk.Button(self.dataset_preparation_tab, text=u'开始处理',
                                    command=self.startDatasetPreparation)
        self.start_btn.pack(pady=10)

    def selectFile(self, entry_widget, file_types):
        """通用文件选择对话框"""
        file_path = filedialog.askopenfilename(filetypes=file_types)
        if file_path:
            entry_widget.delete(0, tk.END)
            entry_widget.insert(0, file_path)

    def selectFactorFiles(self):
        """选择多个ArcGIS图层文件(.lyr)"""
        # Python 2.7版本
        import tkMessageBox

        # 选择.lyr文件
        files = filedialog.askopenfilenames(
            filetypes=[
                ('ArcGIS图层文件', '*.lyr'),
                ('所有文件', '*.*')
            ],
            title="选择ArcGIS图层文件(.lyr)"
        )

        if not files:
            return

        valid_files = []

        for file_path in files:
            # 检查文件扩展名
            ext = os.path.splitext(file_path)[1].lower()
            if ext == '.lyr':
                valid_files.append(file_path)
            else:
                messagebox.showerror("警告", "文件 %s 不是.lyr图层文件" % os.path.basename(file_path))

        if valid_files:
            self.factors_list.delete('1.0', tk.END)
            self.factors_list.insert('1.0', '\n'.join(valid_files))

    def selectOutputDir(self):
        """选择输出目录"""
        directory = filedialog.askdirectory()
        if directory:
            self.output_dir.delete(0, tk.END)
            self.output_dir.insert(0, directory)

    def validateInputs(self):
        """验证输入"""
        if not os.path.exists(self.sample_area_path.get()):
            messagebox.showerror(u'错误', u'请选择有效的采样范围文件')
            return False
        if not self.output_dir.get():
            messagebox.showerror(u'错误', u'请选择输出目录')
            return False
        return True

    def startDatasetPreparation(self):
        """开始数据集准备过程"""
        if not self.validateInputs():
            return

        # 准备参数
        params = {
            'sample_area': self.sample_area_path.get(),
            'sample_count': int(self.sample_count.get()),
            'sample_distance': int(self.sample_distance.get()),
            'output_dir': self.output_dir.get(),
            'run_num': int(self.sample_rounds.get())
        }

        # 禁用开始按钮
        self.start_btn.config(state='disabled')

        # 创建并启动工作线程
        self.worker = DatasetPreparationThread(params, self.threadCallback)
        self.worker.start()

    def threadCallback(self, msg_type, *args):
        """线程回调函数"""
        if msg_type == "status":
            self.status_label.config(text=args[0])
        elif msg_type == "finished":
            success, message = args
            self.start_btn.config(state='normal')
            if success:
                messagebox.showinfo(u'完成', message)
            else:
                messagebox.showerror(u'错误', message)


if __name__ == '__main__':
    app = MainWindow()
    app.mainloop()