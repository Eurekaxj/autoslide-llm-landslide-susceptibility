# worker_thread.py
# -*- coding: utf-8 -*-
import threading
import time
from ArcgisTools import create_random_points, extract_values_to_points, calculate_lon_lat, export_to_excel

class DatasetPreparationThread(threading.Thread):
    """数据集准备的工作线程"""

    def __init__(self, params, queue, callback):
        threading.Thread.__init__(self)
        self.params = params
        self.queue = queue
        self.callback = callback
        self._stop = threading.Event()

    def log(self, message):
        self.queue.put(message)

    def run(self):
        from ArcgisTools import (
            setup_arcpy_environment,
            create_random_points,
            extract_values_to_points,
            calculate_lon_lat,
            export_to_excel
        )

        # 设置 arcpy 环境
        setup_arcpy_environment(self.params['output_dir'])

        # 使用用户指定的影响因子图层路径
        raster_paths = self.params['factor_files']
        template_path = self.params['landslide_path']

        # 正式流程
        self.log(u"开始创建随机点...")
        random_points = create_random_points(self.params['run_num'], self.params, self.log)
        if not random_points:
            self.callback("finished", False, u"创建随机点失败")
            return

        self.log(u"开始提取属性值...")
        if not extract_values_to_points(random_points, raster_paths, self.log):
            self.callback("finished", False, u"属性提取失败")
            return

        self.log(u"开始计算经纬度...")
        if not calculate_lon_lat(random_points, self.log):
            self.callback("finished", False, u"经纬度计算失败")
            return

        self.log(u"导出数据集...")
        excel_output = export_to_excel(random_points, self.params['run_num'], template_path, self.params['output_dir'],
                                       self.log)
        if not excel_output:
            self.callback("finished", False, u"数据集导出失败")
            return

        self.callback("finished", True, u"数据集准备完成")

    def stop(self):
        self._stop.set()
